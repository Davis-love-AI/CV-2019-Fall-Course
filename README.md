# CV-2019-Fall-Course

CV 2019 Fall Course Project (Mask R-CNN)


## Installation

run:

```shell
git clone https://github.com/XuehaiPan/CV-2019-Fall-Course.git
cd CV-2019-Fall-Course
git submodule update --init --recursive
```

### Requirements

- Linux or macOS
- Python >= 3.6
- PyTorch 1.3
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
    You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- OpenCV, needed by demo and visualization
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
- pycocotools: `pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'`
- GCC >= 4.9

### Build Detectron2

After having the above dependencies, run:

```shell
cd Detectron2
python setup.py build develop

# or if you are on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop

# or, as an alternative to `setup.py`, do
# pip install [--editable] .
```

Note: you may need to rebuild detectron2 after reinstalling a different build of PyTorch.
