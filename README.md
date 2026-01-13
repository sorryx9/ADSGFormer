# ADSGFormer

This repository contains the implementation of **ADSGFormer** (Adaptive Dual-Stream Gated Transformer).
This code is directly related to the manuscript 'Enhancing 3D Human Pose Estimation: An Asymmetric Dual-Stream Gated Transformer Approach' submitted to The Visual Computer. We encourage readers to cite this manuscript if you use this code.

## Structure

- `model/`: Contains the core model components.
  - `sogc_stream.py`: Spatial Stream (SOGC) components.
  - `msote_stream.py`: Temporal Stream (MSOTE) components.
  - `fusion.py`: Gated Fusion module.
  - `adsgformer.py`: Main ADSGFormer model assembly.

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```
##Citation
If you find this work useful, please cite our paper: (Citation details will be updated upon publication)