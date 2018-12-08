from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import torch


def merge_dims(x, a, b):
    shape = tuple(x.shape)
    n = len(shape)
    a, b = _absolute_range(a, b, n)
    merge_shape = shape[a:b]
    y = torch.reshape(x, shape[:a] + (-1,) + shape[b:])
    restore_fn = functools.partial(split_dim, merge_shape=merge_shape)
    return y, restore_fn


def split_dim(y, axis, merge_shape):
    shape = tuple(y.shape)
    return torch.reshape(y, shape[:axis] + merge_shape + shape[axis:][1:])


def _absolute_range(a, b, n):
    if a is None:
        a = 0
    if b is None:
        b = n
    if a < 0:
        a = a + n
    if b < 0:
        b = b + n
    return a, b


def map_images(fn, x):
    x, unflatten_fn = flatten_batch(x, 3)
    y = fn(x)
    y = unflatten_fn(y)
    return y


def flatten_batch(x, n):
    '''Merges all dimensions except last n.'''
    shape = tuple(x.shape)
    batch_shape, rest_shape = shape[:-n], shape[-n:]
    y = torch.reshape(x, (-1,) + rest_shape)
    unflatten_fn = functools.partial(unflatten_batch, batch_shape=batch_shape)
    return y, unflatten_fn


def unflatten_batch(y, batch_shape):
    '''Breaks first dimension into `batch_shape`.'''
    shape = tuple(y.shape)
    rest_shape = shape[1:]
    x = torch.reshape(y, batch_shape + rest_shape)
    return x


def flatten_few_shot_examples(inputs, shuffle=False):
    '''
    Args:
        inputs: [b, k, n, ...]

    Returns:
        inputs_flat: [b, k * n, ...]; same memory as `inputs`
        labels_flat: [b, k * n]; integer in [0, k); on CPU
    '''
    b = inputs.shape[0]
    k = inputs.shape[1]
    n = inputs.shape[2]
    labels = torch.arange(k)
    labels = labels.unsqueeze(-1).expand(b, k, n)
    # Flatten all.
    inputs, _ = merge_dims(inputs, 1, 3)
    labels, _ = merge_dims(labels, 1, 3)
    if shuffle:
        m = k * n
        order = torch.stack([torch.randperm(m) for _ in range(b)])
        labels = torch.gather(labels, 1, order)
        inputs = torch.gather(inputs, 1, unsqueeze_n(order, 3, dim=-1).expand_as(inputs))
    return inputs, labels


class MeanAccumulator(object):

    def __init__(self, total=0, count=0):
        self.total = 0
        self.count = 0

    def add(self, total, count=1):
        self.total += total
        self.count += count

    def mean(self):
        return self.total / self.count


def open_and_read(fname):
    with open(fname, 'r') as f:
        return [line.strip() for line in f.readlines()]


def strtobool(s):
    s = s.lower().strip()
    if s in ('true', 't', 'yes', 'y', '1'):
        return True
    elif s in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise ValueError('cannot cast to bool: "{}"'.format(s))


def cross_entropy(input, target, dim=-1, reduction='elementwise_mean', **kwargs):
    '''
    Args:
        input: [before_dims, n, after_dims]
        target: [before_dims, after_dims]
    '''
    if reduction == 'none':
        raise NotImplementedError('non-reduction is not supported')
    n = len(input.shape)
    if dim < 0:
        dim += n
    if dim != 1:
        dims = [i for i in range(n) if i != dim]
        dims.insert(1, dim)
        input = input.permute(dims)
    return torch.nn.functional.cross_entropy(input, target, **kwargs)


def unsqueeze_n(x, n, dim):
    for i in range(n):
        x = torch.unsqueeze(x, dim)
    return x


def one_hot(n, labels, device=None):
    '''
    Args:
        n: Integer
        labels: [dims]

    Returns:
        [dims, n]
    '''
    return torch.eq(torch.arange(n, device=device), labels.unsqueeze(-1))
