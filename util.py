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


def flatten_few_shot_examples(inputs):
    '''
    Args:
        inputs: [b, k, n, ...]

    Returns:
        inputs_flat: [b, k * n, ...]
        labels_flat: [b, k * n]; integer in [0, k).
    '''
    b = inputs.shape[0]
    k = inputs.shape[1]
    n = inputs.shape[2]
    labels = torch.arange(k)  # .to(inputs.device)
    labels = labels.unsqueeze(-1).expand(b, k, n)
    # Flatten all.
    inputs_flat, _ = merge_dims(inputs, 1, 3)
    labels_flat, _ = merge_dims(labels, 1, 3)
    return inputs_flat, labels_flat


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


def compare_all(similar_fn, train_inputs, test_inputs):
    '''
    Args:
        similar_fn: Maps tensors of size [batch_dims, feature_dims] to [batch_dims, 1]
        train_inputs: [b, k, n, feature_dims]
        test_inputs: [b, m, feature_dims]

    Returns:
        scores: [b, m, k, n]
    '''
    train_inputs = torch.unsqueeze(train_inputs, 1)
    test_inputs = torch.unsqueeze(test_inputs, 2)
    test_inputs = torch.unsqueeze(test_inputs, 2)
    # train_inputs: [b, 1, k, n, ...]
    # test_inputs:  [b, m, 1, 1, ...]
    scores = similar_fn(train_inputs, test_inputs)
    # scores: [b, m, k, n, 1]
    scores = torch.squeeze(scores, 4)
    return scores


def max_per_class(similar_fn, train_inputs, test_inputs):
    '''
    Args:
        See compare_all().

    Returns:
        class_scores: [b, m, k]
    '''
    example_scores = compare_all(similar_fn, train_inputs, test_inputs)
    # example_scores: [b, m, k, n]
    class_scores, _ = torch.max(example_scores, dim=-1, keepdim=False)
    return class_scores


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
