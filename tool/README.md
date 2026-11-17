# Data Preparation
## Train Set Preparation
Our training set is constructed from **DTU, LLFF, Mip-NeRF 360, and Tanks and Temples**. The original datasets can be downloaded from the following sources:

| Dataset | Original Data |
| --- | --- |
| DTU | [Download](https://roboimagedata.compute.dtu.dk/?page_id=36) |
| LLFF | [Download](https://github.com/fyusion/llff) |
| Mip-NeRF 360 | [Download](https://jonbarron.info/mipnerf360/) |
| Tanks and Temples | [Download](https://www.tanksandtemples.org/download/) |

The scenes used for training and validation are listed in [tool/train_list.txt](./tool/train_list.txt) and [tool/val_list.txt](./tool/val_list.txt), respectively.

### Naming Conventions

Organize the training data using the following naming conventions:

| Component | Format | Example |
| --- | --- | --- |
| Compressed scene folder | `{codec}_{scene_name}_QP{n}` | `AV1_bicycle_QP20` |
| Compressed frame | `frame_{index:04d}_QP{n}.png` | `frame_0001_QP20.png` |
| GT scene folder | `{scene_name}` | `bicycle` |
| GT frame | `frame_{index:04d}.png` | `frame_0001.png` |

Frame indices start at **1** and use **four-digit zero padding**. Each compressed frame must correspond to the GT frame with the same scene name and frame index. Scene names must be unique across the training datasets.

### Directory Structure

```text
dataset/
└── trainset/
    ├── comp/
    │   ├── AV1_bicycle_QP20/
    │   │   └── frame/
    │   │       ├── frame_0001_QP20.png
    │   │       ├── frame_0002_QP20.png
    │   │       ├── frame_0003_QP20.png
    │   │       └── ...
    │   ├── AV1_bicycle_QP22/
    │   │   └── frame/
    │   │       ├── frame_0001_QP22.png
    │   │       ├── frame_0002_QP22.png
    │   │       └── ...
    │   └── ...
    └── gt/
        ├── bicycle/
        │   ├── frame_0001.png
        │   ├── frame_0002.png
        │   ├── frame_0003.png
        │   └── ...
        └── ...
```

Store compressed frames separately for each codec, scene, and QP combination. Store GT frames only once per scene, shared across all compression settings.

## Test Set Preparation
The list covering part of the RealEstate10K and CO3Dv2 subset used in our evaluation are available in [Realestate10K_test_list.txt](./Realestate10K_test_list.txt) and [CO3Dv2_test_list.txt](./CO3Dv2_test_list.txt).

For detailed compression settings, please refer to **Section 3, “Compression protocol”**, of the [supplementary material](../figs/docs/CompMVR_supple.pdf).

### 1. Dataset Downloads

| Dataset | Original Data | Our Compressed Test Subset |
| --- | --- | --- |
| CO3Dv2 | [Download](https://ai.meta.com/datasets/co3d-downloads/) | [Download](https://drive.google.com/file/d/1BoKVZSFg-O_xO_qoAytIPg2fjKuOSZTQ/view?usp=sharing) |
| Free (F2-NeRF) | [Download](https://www.dropbox.com/scl/fo/x392wbggkavb4vnq8nadw/AEYQylTYNAZ_HjnN-OWANWA?rlkey=8yj7d7sygysb3egci13b1i8lq&e=1&dl=0) | — |
| RealEstate10K | [Download (Hugging Face mirror)](https://huggingface.co/datasets/DavidYan2001/RealEstate10K) | — |


You can download the provided compressed test subsets or prepare them from the source data using the instructions below.

### 2. Dataset Attribution and Licenses

**CO3Dv2.** Our compressed test subset is derived from [CO3Dv2](https://github.com/facebookresearch/co3d) by Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. The data are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). We selected evaluation sequences and applied the preprocessing and compression procedures described in this guide and the supplementary material. Please retain the original attribution and license notices and comply with the noncommercial-use restriction.

<!-- **Free (F2-NeRF).** The Free dataset was introduced by Peng Wang et al. in [F2-NeRF](https://totoro97.github.io/projects/f2-nerf/). Please follow the terms supplied with the original dataset.  -->

### 3. Input layout

Put the original RGB frames under `raw_data/{dataset}/{scene}/`. 

Example paths:

```text
raw_data/CO3Dv2/sceneA/00000001.png
raw_data/CO3Dv2/sceneA/00000002.png
raw_data/CO3Dv2/sceneB/00000001.png
raw_data/Free/grass/00000001.png
raw_data/RealEstate10K/scene001/00000001.png
```

Input frames should have identical, even dimensions within each scene. The script reads the first frame's dimensions and reduces odd dimensions by one pixel when preparing YUV420p; it does not change the copied GT images. Use even dimensions to keep GT and reconstructed frame sizes aligned.

`raw_data/compressed` and the selected output directory are excluded from RGB discovery.

### 4. Run

From the directory containing `compression.py`:

```bash
# Preview discovered scenes and planned compression jobs.
python compression.py --dry-run

# Process all datasets and scenes under raw_data.
python compression.py --input ./raw_data --output ./testset --qp-values 27,42 --backend software --workers 4
```

Intermediate files and the job log are stored under `{input}/compressed`, so the default run uses `raw_data/compressed`.

| Reference/output label | AVC QP | HEVC QP | AV1 CRF (software) | VP9 CRF (software) |
| --- | --- | --- | --- | --- |
| QP27 | 27 | 27 | 32 | 33 |
| QP42 | 42 | 42 | 50 | 52 |

### 5. Intermediate files and logs

The encoded video, reconstructed YUV, and logs using the following structure:

```text
raw_data/
└── compressed/
    ├── job_log.json
    └── AV1/
        ├── QP27/
        │   └── CO3Dv2/
        │       ├── sceneA
        │       │    ├── sceneA_input.yuv
        │       │    ├── sceneA.mkv
        │       │    ├── sceneA_images.txt
        │       │    ├── sceneA_to_yuv.txt
        │       │    ├── sceneA_encode.txt
        │       │    ├── sceneA_decode.txt
        │       │    └── sceneA_vstats.log
        │       └── ...   
        └── QP42/
            └── CO3Dv2/
                └── ...
```

The same structure is created for Free and RealEstate10K.

| Output | Description |
| --- | --- |
| `.mp4` / `.mkv` / `.webm` | Encoded video |
| `.yuv` | Reconstructed video decoded from the encoded video |
| `_encode.txt` | Encoding log |
| `_decode.txt` | Decoding log |
| `_vstats.log` | FFmpeg frame statistics |
| `job_log.json` | Dataset-level job status and output paths |

Encoded video extensions are `.mp4` for AVC/HEVC, `.mkv` for AV1, and `.webm` for VP9. Additional QP JSON files are saved when QP information can be extracted.


### 6. Final PNG Directory Structure

```text
dataset/
└── testset/
    ├── GT/
    │   ├── CO3Dv2/
    │   │   └── sceneA/
    │   │       ├── 00000001.png
    │   │       ├── 00000002.png
    │   │       └── ...
    │   ├── Free/
    │   │   └── grass/
    │   │       └── ...
    │   └── RealEstate10K/
    │       └── scene001/
    │           └── ...
    ├── AVC/
    │   ├── QP27/
    │   │   ├── CO3Dv2/
    │   │   │   └── sceneA/
    │   │   │       ├── 00000001.png
    │   │   │       ├── 00000002.png
    │   │   │       └── ...
    │   │   ├── Free/
    │   │   │   └── ...
    │   │   └── RealEstate10K/
    │   │       └── ...
    │   └── QP42/
    │       └── ...
    ├── HEVC/
    │   └── ...
    ├── AV1/
    │   └── ...
    └── VP9/
        └── ...
```

All output frames use eight-digit filenames starting from `00000001.png`.
Matching filenames identify corresponding GT and reconstructed frames.

The export script stops if an output scene directory is already nonempty
to prevent overwriting existing files.