# Data Preparation

## Test Set Preparation
A list covering part of the RealEstate10K subset used in our evaluation is available in [Realestate10K_test_list.txt](./Realestate10K_test_list.txt).

For detailed compression settings, please refer to **Section 3, “Compression protocol”**, of the supplementary material available on our [project page](https://mchaewon.github.io/CompMVR-project-page/).

### 1. Dataset Downloads

| Dataset | Original Data | Our Compressed Test Subset |
| --- | --- | --- |
| CO3Dv2 | [Download](https://ai.meta.com/datasets/co3d-downloads/) | [Download](CO3DV2_COMPRESSED_DOWNLOAD_URL) |
| Free (F2-NeRF) | [Download](https://www.dropbox.com/scl/fo/x392wbggkavb4vnq8nadw/AEYQylTYNAZ_HjnN-OWANWA?rlkey=8yj7d7sygysb3egci13b1i8lq&e=1&dl=0) | — |
| RealEstate10K | [Download (Hugging Face mirror)](https://huggingface.co/datasets/DavidYan2001/RealEstate10K) | — |

<!-- Add the compressed Free subset link after confirming that
redistribution permission applies to the dataset:
[Download](https://drive.google.com/file/d/1sS8NDAKN48zIsS-DbQBHijab7hHgXdbZ/view?usp=sharing)
-->

You can download the provided compressed test subsets or prepare them from the source data using the instructions below.

### 2. Dataset Attribution and Licenses

**CO3Dv2.** Our compressed test subset is derived from [CO3Dv2](https://github.com/facebookresearch/co3d) by Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. The data are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). We selected evaluation sequences and applied the preprocessing and compression procedures described in this guide and the supplementary material. Please retain the original attribution and license notices and comply with the noncommercial-use restriction.

**Free (F2-NeRF).** The Free dataset was introduced by Peng Wang et al. in [F2-NeRF](https://totoro97.github.io/projects/f2-nerf/). Please follow the terms supplied with the original dataset. 

### 3. Input Directory Structure

Prepare the original videos as **8-bit YUV420p raw YUV files**.
Each filename must follow this format:

```text
{scene_name}_{width}x{height}_{num_frames}.yuv
```

Place all scene files directly inside each dataset directory. Nested YUV directories are not supported.

```text
raw_data/
└── input_yuv/
    ├── CO3Dv2/
    │   ├── sceneA_640x480_100.yuv
    │   └── sceneB_640x480_150.yuv
    ├── Free/
    │   └── grass_1920x1080_100.yuv
    ├── RealEstate10K/
    │   └── scene001_1280x720_100.yuv
    └── original_rgb/                  # Optional
        ├── CO3Dv2/
        │   ├── sceneA/
        │   │   ├── frame_0001.png
        │   │   ├── frame_0002.png
        │   │   └── ...
        │   └── sceneB/
        │       └── ...
        ├── Free/
        │   └── grass/
        │       └── ...
        └── RealEstate10K/
            └── scene001/
                └── ...
```

Original RGB frames are optional. When available, `export_frame.py`
automatically detects the scene folders and uses these images as GT:

- PNG files are copied without re-encoding.
- JPG/JPEG files are converted to PNG.
- Frames are sorted by natural filename order.
- The frame count, resolution, and ordering must match the corresponding YUV video.
- If no RGB frames are found, GT PNGs are extracted from the original YUV.
- If multiple RGB candidate folders are found for a scene, the script stops rather than selecting one automatically.

### 4. Run Compression and PNG Export

The preparation requires Python 3.9 or later, `tqdm`, and FFmpeg with the `libx264`, `libx265`, `libsvtav1`, and `libvpx-vp9` encoders. Pillow is also required when using original RGB frames.

```bash
pip install tqdm Pillow
```

Run the preparation script from the directory containing `compression.py`, `export_frame.py`, and `prepare_testset.sh`:

```bash
bash prepare_testset.sh
```

The script performs the following steps for each dataset:

1. Compress all input scenes using the selected codecs and quality settings.
2. Decode the compressed videos into reconstructed YUV files.
3. Check the compression job log for failed jobs.
4. Export reconstructed YUV frames as PNGs.
5. Copy original RGB frames as GT, or extract GT PNGs from the original YUV when RGB frames are unavailable.

The example script uses AVC, HEVC, AV1, and VP9 with quality values of 27 and 42. Adjust `QP_VALUES` in `prepare_dataset.sh` to select other settings.

### 5. Intermediate Compression Outputs

`compression.py` saves the encoded video, reconstructed YUV, and logs using the following structure:

```text
raw_data/
└── compressed/
    └── CO3Dv2/
        ├── job_log.json
        ├── AV1_sceneA_QP27/
        │   └── comp_result_QP27/
        │       ├── sceneA_640x480_100.mkv
        │       ├── sceneA_640x480_100.yuv
        │       ├── sceneA_640x480_100_encode.txt
        │       ├── sceneA_640x480_100_decode.txt
        │       └── sceneA_640x480_100_vstats.log
        ├── AV1_sceneA_QP42/
        │   └── comp_result_QP42/
        │       └── ...
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

`export_frame.py` saves reconstructed frames by codec, quality setting,
dataset, and scene. GT frames are saved once per scene.

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