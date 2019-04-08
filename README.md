# Pytorch ROIPooling

Welcome!

This is a generic implementation of ROIpooling operation used in the context of object detection.

## Feats

- Modularized

- JIT compilation with cupy

- Works well with batches of images :wink:

## Getting started

We need the following requirements `cuda`, `pytorch==1.0.1`, `cupy=5.1.0` which we can get most of them from [anaconda.org](http://anaconda.org/) with trusted channels.

1. Install anaconda or miniconda.

    > Skip this if you already have miniconda or anaconda installed in your system.

1. Create a new environment

    `conda create -n pytorch-extensions python=3.7 pytorch cupy -c pytorch`

    > This step creates a conda environment called `pytorch-extensions`. In case, you change the name keep it mind to update the next lines accordingly.

1. `conda activate pytorch-extensions`

1. `python example.py`

    Hopefully everything runs like the breeze.

### Can I use it in Colab?

Sure, take a look at this [notebook](https://colab.research.google.com/drive/1zoZKhWynAwnXJAWGTfOiU2-kbc4FH2EM). It provides a guide for the setup and usage of the `roi_pooling` `Function`.

## LICENSE

[MIT](https://choosealicense.com/licenses/mit/)

We highly appreciate that you leave attribution notes when you copy portions of this codebase in yours.

### Did you like it?

Support me, gimme a :star: in the github banner or invite me a :coffee:/:beer:. If you are in academia, I would appreciate that you cite my research:

```
@article{EscorciaDJGS18,
  author    = {Victor Escorcia and
               Cuong Duc Dao and
               Mihir Jain and
               Bernard Ghanem and
               Cees Snoek},
  title     = {Guess Where? Actor-Supervision for Spatiotemporal Action Localization},
  journal   = {CoRR},
  volume    = {abs/1804.01824},
  year      = {2018},
  url       = {http://arxiv.org/abs/1804.01824},
  archivePrefix = {arXiv},
  eprint    = {1804.01824}
}
```

This implementation was built on top of the legendary Faster-RCNN which you _must_ cite:

```
@article{RenHG017,
  author    = {Shaoqing Ren and
               Kaiming He and
               Ross B. Girshick and
               Jian Sun},
  title     = {Faster {R-CNN:} Towards Real-Time Object Detection with Region Proposal
               Networks},
  journal   = {{IEEE} Trans. Pattern Anal. Mach. Intell.},
  volume    = {39},
  number    = {6},
  pages     = {1137--1149},
  year      = {2017},
  url       = {https://doi.org/10.1109/TPAMI.2016.2577031},
  doi       = {10.1109/TPAMI.2016.2577031}
}
```

This was also possible due to [Chainer](https://chainer.org/), and the easy to follow [pyinn](https://github.com/szagoruyko/pyinn).

## FAQs

### Do I need to buy an anaconda license?

Of course not! You do everything with virtual environments. Indeed, I would be pleased to accept a PR with a recipe for virtual environments.

_Why anaconda?_

In short, due to the last five letters.

### Why another ROIpooling operation?

Well, I tried many C extensions mainly taken from this [repo](https://github.com/longcw/faster_rcnn_pytorch) but those did not fit my purpose of ROIPooling over batches of images.

_Why?_

You can clearly see [here](https://github.com/longcw/faster_rcnn_pytorch/blob/master/faster_rcnn/roi_pooling/src/roi_pooling_cuda.c#L27-L30) that when the batch size is greater than 1, the output is zero.

_Does that mean that they are useless?_

Of course not! I noticed that FastRCNN uses a batch size of 1. Probably, they did not mind to make it more general implementation.

_Why didn't you remove the conditional?_

I tried in one of the repos but it fails. I even removed all the binaries and compiled again but it still returned zeros. Thus, I just moved on and pursue my personal reason:

I was really curious of launching cupy kernels using data from pytorch tensors. It is simply amazing. Moreover, it was a great experience to explore CUDA and pytorch.autograd.
