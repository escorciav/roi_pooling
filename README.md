# Pytorch ROIPooling

Welcome!

This is a generic implementation of ROIpooling operation used in the context of object detection.

## Feats

- Modularized

- JIT compilation with pyinn and cupy

- Works well with batches of images :wink:

## Did you like it?

Support me, gimme a :star: in the github banner or invite me a :coffee:/:beer:. If you are in academia, I would appreciate that you cite my research:

```
Actor-Supervision
```

This implementation was built on top of previous amazing work, thus you _must_ cite the following papers:

```
FastRCNN
Sergey work
```

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
