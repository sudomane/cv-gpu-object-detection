# CUDA Accelerated Object Detection (WIP)

## About

The aim of this project is to implement an object detection pipeline in C++ and CUDA, using image processing concepts, such as Gaussian convolutions, mathematical morphology and connected component labeling. It is a work in progress, with the CUDA implementation currently underway.

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
