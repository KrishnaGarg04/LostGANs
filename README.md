
#  LostGANs – Layout and Style Reconfigurable GANs for Controllable Image Synthesis

A modern, from-scratch reimplementation of **LostGANs (ICCV 2019)** that synthesizes high-quality images from user-defined layouts and style codes, with robust support for modern PyTorch and CUDA environments. This project goes beyond replicating prior work — we introduce architectural refinements, memory-aware training strategies, and modular code design to improve both usability and performance.

Importantly, we **trained the entire network from scratch without relying on pretrained weights**, ensuring that our results purely reflect the effectiveness of our optimized pipeline. Enhancements such as a redesigned mask regression module, a RoIAlign-based object discriminator, and full dual-path adversarial training have made our version more resilient to mode collapse and layout inconsistencies, especially under limited hardware constraints.

---

##  Team Members

- **Parth Agarwal** – 2310110546  
- **Krishna Garg** – 2310110692  
- **Ayush Tiwari** – 2310110661  
- **Bind Pratap Singh** – 2310110084

---

##  Project Overview

LostGANs is a layout-to-image synthesis framework that allows users to input semantic layouts (bounding boxes and labels) and generate diverse, high-quality images with user-controlled object styles. Unlike typical GANs, LostGANs is designed to:

- Respect object layout explicitly
- Enable one-to-many style reconfigurations
- Operate at object-level granularity with per-object control

Our version emphasizes full reproducibility and modular design. It also supports updated dependencies and custom modules like a new mask regression network.

---

## Technical Architecture

###  Generator Components

- **Style Encoder:** Projects class labels and noise into style vectors per object.
- **Layout Encoder / Mask Predictor:** Transforms coarse bounding boxes into soft spatial masks.
- **Generator Backbone:** ResNet-style blocks with style-conditioned normalization layers.

###  Discriminator Design

- Dual-path discriminator:
  - **Image-Level Path:** Evaluates realism of the entire scene.
  - **Object-Level Path:** Uses **RoIAlign** to focus on object regions and check semantic accuracy.

###  Optimization

- Hinge Loss
- L1 + VGG Perceptual Losses
- Mixed precision support

---

##  Enhancements in Our Implementation

###  Modern Compatibility

- Rewritten using **Python 3.12+**, with full support for **PyTorch ≥ 2.0** and **CUDA 12.8**
- All models, dataloaders, loss functions, and training loops restructured for compatibility
- Legacy APIs replaced with modern equivalents (e.g., torchvision's native RoIAlign)

###  LostGAN-Specific Architecture Enhancements

- Redesigned **mask regression module** (MaskRegressNetv2) for higher-quality shape priors
- Style codes generated per object using a **custom style mapper** (label + noise)
- Discriminator refined with **instance-level realism assessment** using layout-aligned features
- Modular generator allows integration of per-object conditioning at every layer

###  Efficient Memory Utilization

Given hardware limitations, we introduced the following:

- Dynamic memory release with `torch.cuda.empty_cache()`
- **Virtual memory swapping** for extended runtime on 4GB GPUs
- Batch size control logic to adaptively reduce memory spikes
- Intermediate tensor detachment and reduced forward history tracking

### Checkpointing

- Save checkpoints every 3 epochs (adjustable)
- Track training state: epoch, optimizer state, loss values
- Resume from any previous point using `--resume` argument

###  Storage-Aware Development

- Datasets and `.pth` models not committed to repository
- Instructions provided for downloading and preprocessing COCO-Stuff dataset
- Logs and checkpoints are automatically cleared in low-disk environments

---

##  How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Dataset

Download and place COCO-Stuff (128x128 version). Update path in `dataloader.py`.

### 3. Train

```bash
python train.py
```

### 4. Generate Outputs

```bash
python test.py
```

---


##  Project Features

- Modular code (train/test/models/utils clearly separated)
- Dynamic logging with sample generation
- MaskRegressNetv2 for fine-grained spatial control
- Clean YAML config support (WIP)

---

##  Contributions Summary

- Implemented LostGAN from scratch in PyTorch 2.0
- Designed new modules: mask regression, dual discriminator, style injector
- Overcame hardware constraints with memory-aware training
- Evaluated both qualitative and quantitative performance
- Compared different training setups (low-resource vs. HPC)

---


##  Acknowledgments

This project was completed as part of an academic course at **Shiv Nadar University**. We thank the university HPC support staff, peers, and faculty for their technical and research guidance throughout this project.

---

## 📊 References

1. Park et al., "Semantic Image Synthesis with SPADE" – CVPR 2019  
2. Suen et al., "Layout and Style Reconfigurable GANs" – ICCV 2019  
3. He et al., "Mask R-CNN" (RoIAlign) – ICCV 2017  
