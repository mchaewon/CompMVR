# CompMVR

<p align="center">
  <!-- <a href="#">
    <img src="https://img.shields.io/badge/Paper-CVPR%202026-red" alt="Paper">
  </a> -->
  <a href="https://diffusion-sr.github.io/FiDeSR/">
    <img src="https://img.shields.io/badge/Project-Page-blue" alt="Project">
  </a>
</p>

<p align="center">
  ⭐ <b>Accepted by SIGGRAPH-ASIA 2026</b>
</p>

## 🔥 News
- **2026-06** SIGGRAPH-ASIA 2026 Accepted

## TODO

- [ ] Release the main code 
- [ ] Release pre-trained checkpoints

## 📌 Framework
<p align="center">
  <img src="figs/framework.png" width="90%">
</p>

## ⚙️ Dependencies & Installation
```bash
git clone https://github.com/mchaewon/CompMVR.git
cd CompMVR
```

**Docker** (recommended):

```bash
docker build -t compmvr:latest .
docker run -it --gpus '"device=0"' --shm-size 16G \
  -v $(pwd):/compmvr -v /path/to/compmvr/data:/dataset \
  compmvr:latest /bin/bash
```

**Conda** 

```bash
conda create -n compmvr python=3.10
conda activate compmvr

```
**Install**
```bash
pip install -r requirements.txt
```

## ⚡ Quick Inference

### Step 1: Download the Pretrained Models

Download the following models:

| Model | Description | Link |
|---|---|---|
| SD 2.1-base | Base diffusion model | [Stable Diffusion 2.1-base](https://huggingface.co/Manojb/stable-diffusion-2-1-base) |
| CompMVR | CompMVR stage 1 checkpoint | [compmvr_s1.pkl](#) |
| CompMVR | CompMVR stage 2 checkpoint | [compmvr_s2.pkl](#) |

### Step 2: Prepare the test datasets

Download testsets from #.

### Step 3: Run Inference

```bash
bash scripts/main_test.sh
```

## 🖼️ Results

### Visual Comparison

<p align="center">
  <img src="#" width="95%">
</p>

## License

This project is released under the Apache 2.0 license.

## Acknowledgments

Our project builds upon [CODiff](https://github.com/jp-guo/codiff). We sincerely thank the authors for their awesome work.

## Citations

```bash
#
```
