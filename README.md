# Pytorch ROIPooling

Welcome!

This is a generic implementation of ROIpooling operation used in the context of object detection.

## Feats

- Modularized

- JIT compilation with [pyinn](https://github.com/szagoruyko/pyinn) and [cupy](https://cupy.chainer.org/)

- Works well with batches of images :wink:

## Did you like it?

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

This implementation was built on top of previous amazing work, thus you _must_ cite the following papers:

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

@article{ZagoruykoK17,
  author    = {Sergey Zagoruyko and
               Nikos Komodakis},
  title     = {DiracNets: Training Very Deep Neural Networks Without Skip-Connections},
  journal   = {CoRR},
  volume    = {abs/1706.00388},
  year      = {2017},
  url       = {http://arxiv.org/abs/1706.00388},
  archivePrefix = {arXiv},
  eprint    = {1706.00388}
}
```

This is also possible due to [Chainer](https://chainer.org/) and [PyTorch](https://pytorch.org/).

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

I was really curious of launching cupy kernels using data from pytorch tensors. It is simply amazing. Moreover, it was a great experience to expose myself to CUDA and pytorch.autograd.
