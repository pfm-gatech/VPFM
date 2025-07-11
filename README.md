# [TOG (SIGGRAPH 2025)] Fluid Simulation on Vortex Particle Flow Maps (VPFM)

by [Sinan Wang](https://sinanw.com), [Junwei Zhou](https://zjw49246.github.io/website/), [Fan Feng](https://sking8.github.io/), [Zhiqi Li](https://zhiqili-cg.github.io/), [Yuchen Sun](https://yuchen-sun-cg.github.io/), [Duowen Chen](https://cdwj.github.io), [Greg Turk](https://faculty.cc.gatech.edu/~turk/), and [Bo Zhu](https://faculty.cc.gatech.edu/~bozhu/)

Our paper and video results can be found at our [project website](https://vpfm.sinanw.com/).

## Cloning This Repository with Submodules

This repository contains an external submodule [`AMGPCG_Pybind_Torch`](https://github.com/swang3081/AMGPCG_Pybind_Torch), located under the `external/` directory.  
To ensure everything builds correctly, make sure to clone the repository **with submodules**:

```bash
git clone --recurse-submodules https://github.com/pfm-gatech/VPFM.git
```

## Installation
Our code is tested on Windows 11 with CUDA 12.3, Python 3.10.9, and Taichi 1.6.0.

To set up the environment, first create a conda environment:

```bash
conda create -n "vpfm_env" python=3.10.9 ipython
conda activate vpfm_env
```

Then, install the requirements with:

```bash
pip install -r requirements.txt
```

Then, follow the instruction from [PyTorch](https://pytorch.org/get-started/locally/) and install pytorch 2.1.2+cu121.

Then, follow the instructions in the [AMGPCG_Pybind_Torch repository](https://github.com/swang3081/AMGPCG_Pybind_Torch) to install the AMGPCG pybind.

## Simulation
To run the plesiosaur example, first download the pose sequence from [here](https://drive.google.com/file/d/1tKjaIca3SLLkKTCBeC9OZrUpJP58YLLN/view?usp=sharing), unzip it under the same folder as the src code, then execute:

```bash
python run_vpfm.py
```

Hyperparameters can be tuned by changing the values in the file `hyperparameters.py`. Different initial conditions can be found in init_conditions.py.

## Visualization
The results will be stored in `logs/[exp_name]/vtks`. We recommend using ParaView to load these `.vti` files as a sequence and visualize them by selecting **Volume** in the Representation drop-down menu.

## Bibliography
If you find our paper or code helpful, consider citing:

```bibtex
@article{wang2025vortex,
title={Fluid Simulation on Vortex Particle Flow Maps},
author={Wang, Sinan and Zhou, Junwei and Feng, Fan and Li, Zhiqi and Sun, Yuchen and Chen, Duowen and Turk, Greg and Zhu, Bo},
journal={ACM Transactions on Graphics (TOG)},
volume={44},
number={4},
pages={1--24},
year={2025},
publisher={ACM New York, NY, USA}
}
```
