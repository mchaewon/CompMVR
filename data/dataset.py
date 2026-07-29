import os
import re
import json
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
import torchvision.transforms.functional as F

_FRAME_RE = re.compile(r"^frame_(\d{4})(?:_QP\d+)?$") 
def _frame_id_from_path(p: str) -> int:
    stem = os.path.splitext(os.path.basename(p))[0] 
    m = _FRAME_RE.match(stem)
    if not m:
        raise ValueError(f"Unexpected filename format (expected frame_0001 or frame_0001_QP37): {p}")
    return int(m.group(1))

def _qp_to_int(qp_str):
    if qp_str is None:
        return None
    if isinstance(qp_str, str) and qp_str.startswith("QP"):
        return int(qp_str[2:])
    if isinstance(qp_str, (int, float)):
        return int(qp_str)
    return None



class Dataset(data.Dataset):

    def __init__(self, opt):
        super(Dataset, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt.get('n_channels') else 3
        self.patch_size = opt['H_size'] if opt.get('H_size') else 64
        self.normalize = opt.get("normalize", False)
 

        self.paths_H = []
        for root in opt['dataroot_H']:
            self.paths_H.extend(util.get_image_paths(root))
        raw_map = {}
        for h in self.paths_H:
            h_path = h["path"]
            scene = h.get("scene", None)
            seq_idx = h.get("seq_idx", None)
 
            raw_map[(scene, seq_idx)] = h_path

        self.paths_L = []
        for root in opt['dataroot_L']:
            self.paths_L.extend(util.get_image_paths(root))

        self.pairs = []
        miss = 0
        for l in self.paths_L:
            l_path = l["path"]
            codec = l.get("codec", None)
            scene = l.get("scene", None)
            qp_str = l.get("qp", None)

            seq_idx = l.get("seq_idx", None)
            fid = l.get("frame_id", None)

            h_path = raw_map.get((scene, seq_idx), None)
            if h_path is None:
                miss += 1
                continue

            self.pairs.append({
                "H_path": h_path,
                "L_path": l_path,
                "codec": codec,
                "scene": scene,
                "qp": _qp_to_int(qp_str),
                "frame_id": fid,   
                "seq_idx": seq_idx  
            })

        if miss > 0:
            print(f"[WARN] RAW matching failure")

        self.pairs.sort(
            key=lambda x: (
                str(x["scene"]),
                str(x["codec"]),
                (x["qp"] if x["qp"] is not None else -1),
                x["seq_idx"] if x["seq_idx"] is not None else -1
            )
        )
        VALID_QPS = {20, 26, 32, 38, 44, 50}
        before = len(self.pairs)
        self.pairs = [p for p in self.pairs if p["qp"] in VALID_QPS]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]

        H_path = pair["H_path"]
        L_path = pair["L_path"]
        codec  = pair["codec"]
        scene  = pair["scene"]
        qp     = pair["qp"]      # int or None
        # -------------------------------------
        # read images
        # -------------------------------------
        img_H = util.imread_uint(H_path, self.n_channels)
        img_L = util.imread_uint(L_path, self.n_channels)
        img_L_array = np.array(img_L)
        

        if qp is None:
            noise_level = torch.FloatTensor([0.0])
        else:
            noise_level = torch.FloatTensor([min(float(qp) / 55.0, 1.0)])

        codec_map = {"AVC": 0, "HEVC": 1, "AV1": 2, "VP9": 3}
        codec_int = codec_map.get(codec, -1)


        # -------------------------------------
        # train-time cropping/augmentation
        # -------------------------------------
        if self.opt['phase'] == 'train':
            H, W = img_H.shape[:2]

            if H < self.patch_size or W < self.patch_size:
                pad_h = max(0, self.patch_size - H)
                pad_w = max(0, self.patch_size - W)
                img_H = np.pad(img_H, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                img_L = np.pad(img_L, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                H, W = img_H.shape[:2]
            
            # ---------------------------------
            # randomly crop the patch
            # ---------------------------------
            self.patch_size_plus8 = self.patch_size + 8

            # ---------------------------------
            # randomly crop the patch
            # ---------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size_plus8))
            rnd_w = random.randint(0, max(0, W - self.patch_size_plus8))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size_plus8, rnd_w:rnd_w + self.patch_size_plus8, :]
            patch_L = img_L[rnd_h:rnd_h + self.patch_size_plus8, rnd_w:rnd_w + self.patch_size_plus8, :]


            # ---------------------------------
            # augmentation - flip, rotate
            # ---------------------------------
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)
            patch_L = util.augment_img(patch_L, mode=mode)

            img_H = patch_H.copy()
            img_L = patch_L.copy()

            H2, W2 = img_H.shape[:2]
            if random.random() > 0.5:
                rnd_h2 = random.randint(0, max(0, H2 - self.patch_size))
                rnd_w2 = random.randint(0, max(0, W2 - self.patch_size))
            else:
                rnd_h2, rnd_w2 = 0, 0

            img_H = img_H[rnd_h2:rnd_h2 + self.patch_size, rnd_w2:rnd_w2 + self.patch_size]
            img_L = img_L[rnd_h2:rnd_h2 + self.patch_size, rnd_w2:rnd_w2 + self.patch_size]
            
        img_L, img_H = util.uint2tensor3(img_L), util.uint2tensor3(img_H)
        

        if self.normalize:
            img_L = F.normalize(img_L, mean=[0.5], std=[0.5])
            img_H = F.normalize(img_H, mean=[0.5], std=[0.5])

         
        # print(self.opt['phase'])
        return {
            'L': img_L,
            'H': img_H,
            'qp': noise_level,
            'codec':  codec_int,
            'scene': scene,
            'frame_id': pair["frame_id"],
            'L_path': L_path,
            'H_path': H_path,
            'img_L_array': img_L_array
        }
 
class MultiViewDataset(Dataset):
    def __init__(self, opt, num_views: int = 4):
        super().__init__(opt)
        self.num_views = num_views

        # {scene: {seq_idx: [pair_idx, ...]}}
        self.scene_frame_groups = {}
        for i, pair in enumerate(self.pairs):
            scene   = str(pair["scene"])
            seq_idx = pair["seq_idx"]
            if scene not in self.scene_frame_groups:
                self.scene_frame_groups[scene] = {}
            if seq_idx not in self.scene_frame_groups[scene]:
                self.scene_frame_groups[scene][seq_idx] = []
            self.scene_frame_groups[scene][seq_idx].append(i)

        # sample = (scene, [seq_idx_0, seq_idx_1, ..., seq_idx_{V-1}])
        self.samples = []
        for scene, frame_dict in self.scene_frame_groups.items():
            sorted_frames = sorted(frame_dict.keys())
            n_frames = len(sorted_frames)

            if n_frames < num_views:
                continue
            for start in range(0, n_frames - num_views + 1):
                frame_window = sorted_frames[start:start + num_views]
                self.samples.append((scene, frame_window))

        print(f"[MultiViewDataset] {len(self.samples)} samples "
              f"({num_views} views/sample)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        scene, frame_window = self.samples[index]

        aug_params = self._sample_aug_params_from_image(
            self.pairs[
                self.scene_frame_groups[scene][frame_window[0]][0]
            ]["H_path"]
        )

        images_L, images_H, img_L_arrays = [], [], []
        qp_values, codec_ids = [], []
        codec_map = {"AVC": 0, "HEVC": 1, "AV1": 2, "VP9": 3}

        for seq_idx in frame_window:
            pair_indices = self.scene_frame_groups[scene][seq_idx]
            chosen_idx   = random.choice(pair_indices)
            pair         = self.pairs[chosen_idx]

            img_H, img_L = self._load_and_aug(pair, aug_params)

            img_L_t = util.uint2tensor3(img_L)
            img_H_t = util.uint2tensor3(img_H)
            if self.normalize:
                img_L_t = F.normalize(img_L_t, mean=[0.5], std=[0.5])
                img_H_t = F.normalize(img_H_t, mean=[0.5], std=[0.5])

            images_L.append(img_L_t)
            images_H.append(img_H_t)
            img_L_arrays.append(
                torch.from_numpy(img_L.copy())
            )

            qp = pair["qp"]
            qp_norm = min(float(qp) / 55.0, 1.0) if qp is not None else 35.0 / 55.0
            qp_values.append(qp_norm)
            codec_ids.append(codec_map.get(pair["codec"], 0))

        return {
            "images_L":    torch.stack(images_L),
            "images_H":    torch.stack(images_H),
            "img_L_array": torch.stack(img_L_arrays),
            "qp_values":   torch.tensor(qp_values, dtype=torch.float32),
            "codec_ids":   torch.tensor(codec_ids,  dtype=torch.long),
            "scene_id":    scene,
            "frame_ids":   torch.tensor(
                               [int(f) if isinstance(f, (int, np.integer))
                                else 0 for f in frame_window],
                               dtype=torch.long),
            "L":           images_L[0],
            "H":           images_H[0],
            "L_path":      self.pairs[
                               self.scene_frame_groups[scene][frame_window[0]][0]
                           ]["L_path"],
            "H_path":      self.pairs[
                               self.scene_frame_groups[scene][frame_window[0]][0]
                           ]["H_path"],
        }
 
    def _sample_aug_params_from_image(self, h_path: str) -> dict:
        if self.opt['phase'] != 'train':
            return {}
 
        img = util.imread_uint(h_path, self.n_channels)
        H, W = img.shape[:2]
 
        if H < self.patch_size or W < self.patch_size:
            H = max(H, self.patch_size)
            W = max(W, self.patch_size)
 
        ps8 = self.patch_size + 8
 
        rnd_h = random.randint(0, max(0, H - ps8))
        rnd_w = random.randint(0, max(0, W - ps8))
 
        # flip/rotate mode
        aug_mode = random.randint(0, 7)
 
        use_rand_crop2 = random.random() > 0.5
        if use_rand_crop2:
            rnd_h2 = random.randint(0, max(0, ps8 - self.patch_size))
            rnd_w2 = random.randint(0, max(0, ps8 - self.patch_size))
        else:
            rnd_h2, rnd_w2 = 0, 0
 
        return {
            "rnd_h":          rnd_h,
            "rnd_w":          rnd_w,
            "rnd_h2":         rnd_h2,
            "rnd_w2":         rnd_w2,
            "aug_mode":       aug_mode,
            "use_rand_crop2": use_rand_crop2,
        }
 
    def _load_and_aug(self, pair: dict, aug_params: dict):
        img_H = util.imread_uint(pair["H_path"], self.n_channels)
        img_L = util.imread_uint(pair["L_path"], self.n_channels)
 
        if self.opt['phase'] == 'train':
            H, W = img_H.shape[:2]
 
            if H < self.patch_size or W < self.patch_size:
                pad_h = max(0, self.patch_size - H)
                pad_w = max(0, self.patch_size - W)
                img_H = np.pad(img_H, ((0,pad_h),(0,pad_w),(0,0)), mode='reflect')
                img_L = np.pad(img_L, ((0,pad_h),(0,pad_w),(0,0)), mode='reflect')
 
            ps8 = self.patch_size + 8
 
            rnd_h = aug_params["rnd_h"]
            rnd_w = aug_params["rnd_w"]
            img_H = img_H[rnd_h:rnd_h+ps8, rnd_w:rnd_w+ps8, :]
            img_L = img_L[rnd_h:rnd_h+ps8, rnd_w:rnd_w+ps8, :]
 
            img_H = util.augment_img(img_H, mode=aug_params["aug_mode"])
            img_L = util.augment_img(img_L, mode=aug_params["aug_mode"])
 
            rnd_h2 = aug_params["rnd_h2"]
            rnd_w2 = aug_params["rnd_w2"]
            img_H = img_H[rnd_h2:rnd_h2+self.patch_size, rnd_w2:rnd_w2+self.patch_size]
            img_L = img_L[rnd_h2:rnd_h2+self.patch_size, rnd_w2:rnd_w2+self.patch_size]
 
        return img_H, img_L
 
 
 
# ============================================================
# multiview_collate_fn
# ============================================================
 
def multiview_collate_fn(batch):
    str_keys = {"scene_id", "L_path", "H_path"}
    collated = {}
 
    for key in batch[0].keys():
        if key in str_keys:
            collated[key] = [item[key] for item in batch]
        else:
            collated[key] = torch.stack([item[key] for item in batch])
 
    return collated
 