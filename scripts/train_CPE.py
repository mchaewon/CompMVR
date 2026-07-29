import os
import math
import argparse
import random
import numpy as np
import logging
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from data.dataset import Dataset
from cpe.trainer import CPETrainer
from tqdm import tqdm
import wandb
# os.environ["WANDB_MODE"] = "disabled"
# wandb.login()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def custom_collate_fn(batch):
    img_L_arrays = [item.pop('img_L_array') for item in batch]
    
    collated = torch.utils.data.dataloader.default_collate(batch)
    collated['img_L_array'] = img_L_arrays  # list of numpy arrays
    
    return collated

def main(json_path='options/cpe.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--val_num_samples', type=int, default=500,
                        help='validation random sampling num')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    val_num_samples = parser.parse_args().val_num_samples

    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    init_iter, init_path_G = 0, None
    opt['path']['pretrained_netG'] = init_path_G
    current_step = init_iter
    border = 0

    option.save(opt)
    opt = option.dict_to_nonedict(opt)

    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    wandb.init(project="CPE", name=f'comp_basic')

    dataset_type = opt['datasets']['train']['dataset_type']

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            # ── Codec Balanced Sampler ──────────────────────────────
            codec_map  = {"AVC": 0, "HEVC": 1, "AV1": 2, "VP9": 3}  
            codec_list = [pair["codec"] for pair in train_set.pairs]
            labels     = [codec_map.get(c, -1) for c in codec_list]

            class_counts  = torch.zeros(len(codec_map), dtype=torch.float32)
            for l in labels:
                if l >= 0:
                    class_counts[l] += 1

            class_weights  = 1.0 / (class_counts + 1e-8)
            sample_weights = torch.tensor(
                [class_weights[l].item() if l >= 0 else 0.0 for l in labels],
                dtype=torch.float32
            )
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )

            counts = {c: int(class_counts[i].item()) for c, i in codec_map.items()}

            train_loader = DataLoader(train_set,
                                      batch_size=dataset_opt['dataloader_batch_size'],
                                      sampler=sampler,
                                      num_workers=dataset_opt['dataloader_num_workers'],
                                      drop_last=True,
                                      pin_memory=True, collate_fn=custom_collate_fn)
        elif phase == 'test':
            test_set = Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    trainer = CPETrainer(opt)

    if opt['merge_bn'] and current_step > opt['merge_bn_startpoint']:
        trainer.merge_bnorm_test()

    trainer.init_train()

    max_train_steps = 200000
    progress_bar = tqdm(range(0, max_train_steps), initial=0, desc="Steps")

    for epoch in range(1000000):  # keep running
        for i, train_data in enumerate(train_loader):
            current_step += 1
            if dataset_type == 'dnpatch' and current_step % 20000 == 0:  # for 'train400'
                train_loader.dataset.update_data()

            trainer.update_learning_rate(current_step)

            trainer.feed_data(train_data)

            
            G_loss, QP_loss, codec_loss = trainer.optimize_parameters(current_step)
            wandb.log({'epoch': epoch, 'G_loss': G_loss, 'QP_loss': QP_loss, 'codec_loss': codec_loss})

            progress_bar.update(1)
            info = {"G_loss": G_loss.item(), "QP_loss": QP_loss.item(), 'codec_loss': codec_loss.item()}
            progress_bar.set_postfix(**info)

            if opt['merge_bn'] and opt['merge_bn_startpoint'] == current_step:
                trainer.merge_bnorm_train()
                trainer.print_network()

            if current_step % opt['train']['checkpoint_print'] == 0:
                logs = trainer.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, trainer.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            if current_step % opt['train']['checkpoint_save'] == 0:
                logger.info('Saving the trainer.')
                trainer.save(current_step)

            if current_step >= max_train_steps:  
                break
            if current_step % opt['train']['checkpoint_test'] == 0:

                n_total = len(test_set)
                if val_num_samples > 0 and n_total > val_num_samples:
                    val_indices = random.sample(range(n_total), val_num_samples)
                else:
                    val_indices = list(range(n_total))
 
                val_subset = torch.utils.data.Subset(test_set, val_indices)
                val_loader = DataLoader(val_subset, batch_size=1,
                                        shuffle=False, num_workers=1,
                                        drop_last=False, pin_memory=True)
                # ────────────────────────────────────────────────────

                avg_psnr = 0.0
                avg_ssim = 0.0
                avg_psnrb = 0.0
                idx = 0
                qp_abs_errors = []
                codec_correct = 0
                codec_total   = 0
                print(f"Test dataset size: {len(val_loader)}")
                for test_data in val_loader:
                    idx += 1
                    print(idx)
                    image_name_ext = os.path.basename(test_data['H_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)

                    trainer.feed_data(test_data)
                    trainer.test()

                    visuals = trainer.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])

                    qp_pred_norm  = float(visuals['QP'])         
                    qp_pred_display = qp_pred_norm * 55.0      

                    qp_gt = float(test_data['qp'][0]) if 'qp' in test_data else None
                    if qp_gt is not None:
                        qp_gt_display = qp_gt * 55.0         
                        qp_abs_errors.append(abs(qp_pred_display - qp_gt_display))


                    codec_pred_idx = visuals['codec'].argmax().item()
                    codec_map = {0: 'AVC', 1: 'HEVC', 2: 'AV1', 3: 'VP9'}
                    codec_str = codec_map.get(codec_pred_idx, 'Unknown')
                    codec_gt_idx = None
                    if 'codec' in test_data:
                        codec_gt_raw = test_data['codec'][0]
                        if hasattr(codec_gt_raw, 'argmax'):
                            codec_gt_idx = int(codec_gt_raw.argmax().item())
                        else:
                            codec_gt_idx = int(codec_gt_raw)
                        codec_correct += int(codec_pred_idx == codec_gt_idx)
                        codec_total   += 1

                    save_img_path = os.path.join(img_dir, '{:s}.png'.format(img_name))
                    util.imsave(E_img, save_img_path)

                    current_psnr = util.calculate_psnr(E_img, H_img, border=border)

                    avg_psnr += current_psnr

                    current_ssim = util.calculate_ssim(E_img, H_img, border=border)

                    avg_ssim += current_ssim

                    current_psnrb = util.calculate_psnrb(H_img, E_img, border=border)
                    avg_psnrb += current_psnrb

                    qp_gt_str = f'{qp_gt_display:<4.1f}' if qp_gt is not None else 'N/A'
                    codec_gt_str = codec_map.get(codec_gt_idx, 'N/A') if codec_gt_idx is not None else 'N/A'

                    logger.info(
                        '{:->4d}--> {:>10s} | PSNR : {:<4.2f}dB | SSIM : {:<4.3f}dB | PSNRB : {:<4.2f}dB'.format(
                            idx, image_name_ext, current_psnr, current_ssim, current_psnrb))
                    logger.info(
                                '  QP pred: {:<4.2f}  gt: {:>4s} | Codec pred: {:>4s}  gt: {:>4s}'.format(
                                    qp_pred_display, qp_gt_str, codec_str, codec_gt_str))

                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx
                avg_psnrb = avg_psnrb / idx

                avg_qp_mae     = float(np.mean(qp_abs_errors)) if qp_abs_errors else float('nan')
                codec_accuracy = (codec_correct / codec_total * 100.0) if codec_total > 0 else float('nan')


                # testing log
                logger.info(
                            '<epoch:{:3d}, iter:{:8,d}> '
                            'PSNR:{:<.2f}dB  SSIM:{:<.3f}  PSNRB:{:<.2f}dB  '
                            'QP_MAE:{:<.3f}  Codec_Acc:{:<.1f}%\n'.format(
                                epoch, current_step,
                                avg_psnr, avg_ssim, avg_psnrb,
                                avg_qp_mae, codec_accuracy))
                wandb.log({'epoch': epoch, 'PSNR': avg_psnr, 'SSIM': avg_ssim, 'PSNRB': avg_psnrb, 'QP_MAE': avg_qp_mae,  'Codec_Acc(%)':  codec_accuracy})
        if current_step >= max_train_steps:
            break

    logger.info('Saving the final trainer.')
    trainer.save('latest')
    logger.info('End of training.')


if __name__ == '__main__':
    main()
