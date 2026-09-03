# CompMVR: Compression-Aware Multi-View Restoration Using Diffusion Models for Geometrically Consistent 3D Reconstruction
Official PyTorch implementation of the paper "CompMVR: Compression-Aware Multi-View Restoration Using Diffusion Models for Geometrically Consistent 3D Reconstruction", accepted to SIGGRAPH-ASIA 2026.

[Dong-hwi Kim](https://sites.google.com/view/dlhwi/%ED%99%88)<sup>1,\*</sup>, [Chaewon Moon](https://mchaewon.github.io/)<sup>1,\*</sup>, [Hojun Song](https://hojunking.github.io/webpages/hojunsong),  Dongbeom Kim<sup>1</sup>, Junyeong Jang<sup>1</sup>, [Aro Kim](https://ar0kim.github.io/)<sup>1</sup>, Gahyeon Kim<sup>1</sup>, [Heejung Choi](https://scholar.google.com/citations?view_op=list_works&hl=ko&user=zocGdPwAAAAJ)<sup>1</sup>, Jehee Kim<sup>1</sup>, Gianella Cravioto<sup>1</sup>, Sohyun Lee<sup>1</sup>, Gyeongjin Choi<sup>1</sup>, EunHye Jeong<sup>1</sup>, [Soo Ye Kim](https://sites.google.com/view/sooyekim)<sup>2,†</sup>, [Jaehyup Lee](https://sites.google.com/view/knuairlab/team/professor?authuser=0)<sup>1,†</sup>, and [Sang-hyo Park](https://sites.google.com/view/knuvi/s-park?authuser=0)<sup>1,†</sup> 

<sup>1</sup> Kyungpook National University, South Korea
<sup>2</sup> Adobe Research, USA

<sup>\*</sup> Equal contribution &nbsp;&nbsp; <sup>†</sup> Corresponding author


<a href="#">
  <img src="https://img.shields.io/badge/Paper-SIGGRAPH_ASIA-red" alt="Paper">
</a>
<a href="https://mchaewon.github.io/CompMVR-project-page/">
  <img src="https://img.shields.io/badge/Project-Page-blue" alt="Project">
</a>

---
## 🔥 News
- **2026-06** SIGGRAPH-ASIA 2026 Accepted
- **2026-09** Code released.

## TODO

- [x] Release the main code 
- [ ] Release pre-trained checkpoints
- [ ] Release datasets

---

## 📌 Framework
<p align="center">
  <img src="figs/main_framework.png" width="90%">
</p>

---

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

## Data preparation

The datasets used for training and evaluation can be downloaded from the links below:

| Split | Download |
|---|---|
| Train set | [Download](https://gofile.me/7Z8aO/N4Dli1Yj5) |
| Test set | [Download](https://gofile.me/7Z8aO/dtTfPA9Bn) |

## ⚡ Quick Inference

### Step 1: Download the Pretrained Models

Download the following models:

| Model | Description | Link |
|---|---|---|
| SD 2.1-base | Base diffusion model | [Stable Diffusion 2.1-base](https://huggingface.co/Manojb/stable-diffusion-2-1-base) |
| CompMVR | CompMVR stage 1 checkpoint | coming soon |
| CompMVR | CompMVR stage 2 checkpoint | coming soon |
<!-- | CompMVR | CompMVR stage 1 checkpoint | [compmvr_s1.pkl](#) |
| CompMVR | CompMVR stage 2 checkpoint | [compmvr_s2.pkl](#) | -->

### Step 2: Run Inference

```bash
bash scripts/main_test.sh
```

---
## 🖼️ Results

### Visual Comparison

#### 2D restoration results
<p align="center">
  <img src="figs/2D_quality.png" width="95%">
</p>

#### NVS results rendered at GT camera poses
<p align="center">
  <img src="figs/3DGS_render.png" width="95%">
</p>

#### Synthesized novel 3D views
<p align="center">
  <img src="figs/3DGS_viewer.png" width="95%">
</p>

---
## License

This project is released under the Apache 2.0 license.

## Acknowledgments

Our project builds upon [CODiff](https://github.com/jp-guo/codiff). We sincerely thank the authors for their awesome work.

## Citations

```bash
@inproceedings
{kim2026compmvr,
  author    = {Kim, Dong-hwi and others},
  title     = {CompMVR: Compression-Aware Multi-View Restoration Using Diffusion Models for Geometrically Consistent 3D Reconstruction},
  booktitle = {SIGGRAPH Asia 2026 Conference Papers},
  year      = {2026},
  address   = {Kuala Lumpur, Malaysia},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3829340.3842283},
  isbn      = {979-8-4007-2842-6}
}
```
