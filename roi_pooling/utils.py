"""utils module adapted from pyinn

Modified work:
-------------------------------------------------------------------------------
Copyright (c) 2019 Victor Escorcia
-------------------------------------------------------------------------------

Original work:
-------------------------------------------------------------------------------
pyinn
Copyright (c) 2017
Licensed under The MIT License [see pyinn/LICENSE for details]
Writter by Sergey Zagoruyko
-------------------------------------------------------------------------------
"""
from collections import namedtuple
import cupy
import torch
from string import Template


Stream = namedtuple('Stream', ['ptr'])


def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'
    elif isinstance(t, torch.cuda.IntTensor):
        return 'int'
    else:
        raise ValueError('WIP. Check pyinn-issue-#10')


@cupy.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)