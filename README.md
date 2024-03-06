# CUDA Accelerated Object Detection (WIP)

## About

The aim of this project is to implement an object detection pipeline in C++ and CUDA, using image processing concepts, such as Gaussian convolutions, mathematical morphology and connected component labeling. It is a work in progress, with the CUDA implementation currently underway.

## Getting started

This project uses [nlohmann JSON](https://github.com/nlohmann/json), and is included via a git submodule. To clone this repo, run the following commands:

```bash
git clone git@github.com:sudomane/cv-gpu-object-detection.git
cd cv-gpu-object-detection
git submodule init   # Create submodule configuration file
git submodule update # Fetch submodule data
```

Or alternatively,

```bash
git clone --recurse-submodules git@github.com:sudomane/cv-gpu-object-detection.git
```

## Setting up the Python Environment

To create the python environment:
```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Building C++ code

To build the source code:
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```
