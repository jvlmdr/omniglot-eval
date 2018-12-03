from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import itertools
import numpy as np
import pprint
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets.omniglot as omniglot


def main():
    args = parse_args()

    if args.arch == 'koch':
        embed_dim = 4096
        siamese = Siamese(KochEmbeddingNet(embed_dim), embed_dim)
        image_transform = None
    elif args.arch == 'vinyals':
        embed_dim = 64
        siamese = Siamese(VinyalsEmbeddingNet(embed_dim), embed_dim)
        image_transform = torchvision.transforms.Resize((28, 28))
    else:
        raise ValueError('unknown arch: "{}"'.format(arch))

    predictor = MostSimilar(siamese)
    parameters = list(siamese.parameters())
    print('model parameters:')
    pprint.pprint([x.shape for x in parameters])
    optimizer = torch.optim.SGD(parameters, lr=1e-2, momentum=0.9)

    if args.resplit:
        entire_dataset = load_both_and_merge(
            args.data_dir,
            transform=image_transform,
            download=args.download)
        dataset_train, dataset_test = split_classes(entire_dataset, [0.8, 0.2])
    else:
        dataset_train = from_torchvision(torchvision.datasets.omniglot.Omniglot(
            args.data_dir,
            background=True,
            transform=image_transform,
            download=args.download))
        dataset_test = from_torchvision(torchvision.datasets.omniglot.Omniglot(
            args.data_dir,
            background=False,
            transform=image_transform,
            download=args.download))

    examples = PairSampler(dataset_train,
                           np.random.RandomState(seed=args.train_seed),
                           batch_size=args.batch_size,
                           mode=args.train_sample_mode)
    train(siamese, examples, optimizer, args.num_train_steps)

    for mode in args.test_sample_modes:
        problems = FewShotSampler(dataset_test,
                                  np.random.RandomState(seed=args.test_seed),
                                  mode=mode,
                                  k=args.num_classes,
                                  n_train=args.num_shots,
                                  n_test=1)
        test(predictor, problems, num_problems=args.num_test_problems)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--arch', default='vinyals')
    parser.add_argument('--num_train_steps', type=int, default=int(1e4))
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_test_problems', type=int, default=int(1e3))
    parser.add_argument('-k', '--num_classes', type=int, default=20)
    parser.add_argument('-n', '--num_shots', type=int, default=1)
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('-s', '--resplit', action='store_true')
    parser.add_argument('--train_sample_mode', default='uniform')
    parser.add_argument('--test_sample_modes', nargs='+',
                        default=['uniform', 'within_alphabet'])
    parser.add_argument('--train_seed', type=int, default=0)
    parser.add_argument('--test_mode', default='uniform')
    parser.add_argument('--test_seed', type=int, default=0)
    return parser.parse_args()


def train(model, examples, optimizer, num_steps):
    model.train()

    for i, (im0, im1, target) in zip(range(num_steps), examples):
        # TODO: Move to transform().
        im0, im1, target = torch.tensor(im0), torch.tensor(im1), torch.tensor(target)
        # im0, im1, target = im0.to(device), im1.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(im0, im1)
        loss = nn.functional.binary_cross_entropy_with_logits(output, target)
        loss.backward()
        optimizer.step()
        print('step {:d}, loss {:.3g}'.format(i, loss.item()))


def test(model, problems, num_problems, log_interval=100):
    model.eval()
    accuracy = MeanAccumulator()

    for i, (train_ims, test_ims, _) in zip(range(num_problems), problems):
        train_ims, test_ims = torch.tensor(train_ims), torch.tensor(test_ims)
        # Add batch dimension.
        train_ims = torch.unsqueeze(train_ims, 0)
        test_ims = torch.unsqueeze(test_ims, 0)
        b = train_ims.shape[0]
        n = train_ims.shape[2]
        # Flatten test examples.
        test_ims, gt = flatten_few_shot_examples(test_ims)
        pred = model(train_ims, test_ims)
        is_correct = torch.eq(pred, gt).numpy()
        accuracy.add(np.sum(is_correct), is_correct.size)
        if (i + 1) % log_interval == 0:
            print('steps {}, error rate {:.3g}'.format(i + 1, 1 - accuracy.mean()))

    print('error rate: {:.3g}'.format(1 - accuracy.mean()))


class MeanAccumulator(object):

    def __init__(self, total=0, count=0):
        self.total = 0
        self.count = 0

    def add(self, total, count=1):
        self.total += total
        self.count += count

    def mean(self):
        return self.total / self.count


def flatten_few_shot_examples(inputs, shuffle=False):
    '''
    Args:
        inputs: [b, k, n, ...]

    Returns:
        inputs_flat: [b, k * n, ...]
        labels_flat: [b, k * n, 1]; integer in [0, k).
    '''
    b = inputs.shape[0]
    k = inputs.shape[1]
    n = inputs.shape[2]
    labels = torch.arange(k).unsqueeze(-1).unsqueeze(-1)
    labels = labels.expand(b, k, n, 1)
    # Flatten all.
    inputs_flat, _ = merge_dims(inputs, 1, 3)
    labels_flat, _ = merge_dims(labels, 1, 3)
    return inputs_flat, labels_flat


def from_torchvision(dataset):
    print('loading images...')
    images = []
    labels = []
    for im, label in dataset:
        images.append(im)
        labels.append(label)
    print('converting images...')
    images = [_im2arr(im) for im in images]
    print('done: load images')
    return OmniglotDataset(dataset._alphabets, dataset._characters, images, labels)


class OmniglotDataset(object):
    '''
    Contains all images in memory.
    Builds indices for efficient sampling.
    '''

    def __init__(self, alphabets, characters, images, labels):
        self.alphabets = alphabets
        self.characters = characters
        self.images = images
        self.labels = labels
        self._build_index()

    def _build_index(self):
        if self.alphabets:
            alphabet_chars = {alphabet: [] for alphabet in self.alphabets}
            for char_index, char_name in enumerate(self.characters):
                alphabet, _ = char_name.split('/')
                alphabet_chars[alphabet].append(char_index)
            self.alphabet_chars = alphabet_chars
        else:
            self.alphabet_chars = None

        # Get index for self.images and self.labels.
        char_instances = {char: [] for char in range(len(self.characters))}
        for i, char_index in enumerate(self.labels):
            char_instances[char_index].append(i)
        self.char_instances = char_instances


def _im2arr(x):
    x = np.array(x).astype(np.float32) / 255.0
    assert len(x.shape) == 2
    x = x[None, :, :]
    return x


class PairSampler(torch.utils.data.Sampler):

    def __init__(self, dataset, rand, batch_size, prob_pos=0.5, mode='uniform'):
        self._dataset = dataset
        self._rand = rand
        self._batch_size = batch_size
        self._prob_pos = prob_pos
        self._mode = mode

    def __iter__(self):
        return self

    def __len__(self):
        return 10 ** 9

    def __next__(self):
        if self._mode == 'within_alphabet':
            alphabet = self._rand.choice(self._dataset.alphabets)
            char_candidates = self._dataset.alphabet_chars[alphabet]
        elif self._mode == 'uniform':
            char_candidates = len(self._dataset.characters)
        else:
            raise ValueError('unknown mode: {}'.format(self._mode))

        # TODO: Use BatchSampler like DataLoader does.
        batch_im0 = []
        batch_im1 = []
        batch_label = []
        for i in range(self._batch_size):
            # Choose pair of characters.
            pair = self._rand.choice(char_candidates, 2, replace=False)
            is_pos = (self._rand.random_sample() < self._prob_pos)
            if is_pos:
                # Load two different images from the same class.
                char = pair[0]
                examples = self._rand.choice(self._dataset.char_instances[char], 2, replace=False)
            else:
                # Load one image from each class.
                examples = [self._rand.choice(self._dataset.char_instances[char]) for char in pair]
            im0, im1 = [self._dataset.images[i] for i in examples]
            label = [float(is_pos)]
            batch_im0.append(im0)
            batch_im1.append(im1)
            batch_label.append(label)
        return batch_im0, batch_im1, batch_label


class FewShotSampler(object):

    def __init__(self, dataset, rand, mode, k, n_train, n_test):
        '''
        Builds necessary indices for efficient sampling.
        '''
        self._dataset = dataset
        self._rand = rand
        self._mode = mode
        self._k = k
        self._n_train = n_train
        self._n_test = n_test

    def __iter__(self):
        return self

    def __next__(self):
        '''
        Returns:
            train_ims: [k, n_train, c, h, w]
            test_ims: [k, n_test, c, h, w]
            labels: [k, 1]

        TODO: Should we flatten and shuffle test_ims to eliminate risk of model using location?
        '''
        if self._mode == 'uniform':
            # Choose random subset of k character classes.
            chars = self._rand.choice(len(self._dataset.characters), self._k, replace=False)
        elif self._mode == 'within_alphabet':
            # Choose an alphabet.
            alphabet = self._rand.choice(self._dataset.alphabets)
            # Choose characters.
            subset = self._dataset.alphabet_chars[alphabet]
            chars = self._rand.choice(subset, self._k, replace=False)
        else:
            raise ValueError('unknown mode: "{}"'.format(self._mode))

        # TODO: Support using characters all drawn by same author?
        # Sample a random n instances for each character.
        n = self._n_train + self._n_test
        ims = []
        for char in chars:
            char_ims = []
            instances = self._rand.choice(self._dataset.char_instances[char], n, replace=False)
            for index in instances:
                char_ims.append(self._dataset.images[index])
            ims.append(char_ims)
        ims = np.asarray(ims)

        train_ims = ims[:, :self._n_train]
        test_ims = ims[:, self._n_train:]
        labels = chars[:, None]
        return train_ims, test_ims, labels


def load_both_and_merge(data_dir, **kwargs):
    '''Merge background and evaluation sets into a single OmniglotDataset.'''
    assert 'target_transform' not in kwargs
    a = omniglot.Omniglot(data_dir, background=True, **kwargs)
    b = omniglot.Omniglot(
        data_dir, background=False,
        target_transform=functools.partial(lambda offset, i: offset + i, len(a._characters)),
        **kwargs,
    )
    merge = a + b
    merge._alphabets = a._alphabets + b._alphabets
    merge._characters = a._characters + b._characters
    return from_torchvision(merge)


def split_classes(dataset, p, seed=0):
    '''
    Example:
        train, test = split_classes(dataset, [0.8, 0.2])
    '''
    num_chars = len(dataset.characters)
    num_subsets = len(p)

    chars = np.arange(num_chars)
    rand = np.random.RandomState(seed)
    rand.shuffle(chars)

    p = (1 / np.sum(p)) * np.asfarray(p)
    cdf = np.cumsum(p)
    stops = [0] + [int(round(x)) for x in cdf * num_chars]
    subsets = [chars[stops[i]:stops[i+1]] for i in range(num_subsets)]
    assert sum(map(len, subsets)) == num_chars

    # Split images based on label.
    images = [[] for i in range(num_subsets)]
    labels = [[] for i in range(num_subsets)]
    subset_lookup = {}
    for i in range(num_subsets):
        for char in subsets[i]:
            subset_lookup[char] = i
    for im, label in zip(dataset.images, dataset.labels):
        i = subset_lookup[label]
        images[i].append(im)
        labels[i].append(label)
    # Re-number labels within subset.
    for i in range(num_subsets):
        index_map = _inv_map(subsets[i])
        labels[i] = list(map(index_map.__getitem__, labels[i]))

    names = [[dataset.characters[char] for char in subsets[i]] for i in range(num_subsets)]
    return [OmniglotDataset(None, names[i], images[i], labels[i]) for i in range(num_subsets)]


def _inv_map(x):
    r = {}
    for i, xi in enumerate(x):
        r[xi] = i
    return r


class MostSimilar(nn.Module):
    '''Takes maximum similarity in each class.'''

    def __init__(self, similarity_fn):
        '''
        Args:
            similarity_fn:
                Maps tensors of size [batch_dims, feature_dims] to [batch_dims, 1]
        '''
        super(MostSimilar, self).__init__()
        self.similarity_fn = similarity_fn

    def forward(self, train_inputs, test_inputs):
        '''
        Args:
            train_inputs: [b, k, n, feature_dims]
            test_inputs: [b, m, feature_dims]
        
        The feature_dims part must be broadcastable.

        Returns:
            predictions: [b, m, 1]; integers in [0, k)
        '''
        train_inputs = torch.unsqueeze(train_inputs, 1)
        test_inputs = torch.unsqueeze(test_inputs, 2)
        test_inputs = torch.unsqueeze(test_inputs, 2)
        # train_inputs: [b, 1, k, n, ...]
        # test_inputs:  [b, m, 1, 1, ...]
        match_scores = self.similarity_fn(train_inputs, test_inputs)
        # match_scores: [b, m, k, n, 1]
        class_scores, _ = torch.max(match_scores, dim=3, keepdim=False)
        _, labels = torch.max(class_scores, dim=2, keepdim=False)
        return labels


class Siamese(nn.Module):

    def __init__(self, embed, embed_dim):
        super(Siamese, self).__init__()
        self.embed = embed
        self.postproc = nn.Linear(embed_dim, 1)

    def forward(self, images_a, images_b):
        '''
        Args:
            images_a: [..., c, h, w]
            images_b: [..., c, h, w]
        '''
        images_a, images_b = torch.distributions.utils.broadcast_all(images_a, images_b)
        # Flatten batch to evaluate conv-net.
        feat_a = map_images(self.embed, images_a)
        feat_b = map_images(self.embed, images_b)
        delta = torch.abs(feat_a - feat_b)
        output = self.postproc(delta)
        return output


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


# For fully-connected layers, following pattern in:
# https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py


class VinyalsEmbeddingNet(nn.Module):
    # https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py 

    # TODO: Dimension should be 32 not 28 to cover image well?

    def __init__(self, hidden_channels=64):
        super(VinyalsEmbeddingNet, self).__init__()

        def module(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.conv_net = nn.Sequential(
            # 28
            module(1, hidden_channels),
            # 14 = 28 // 2
            module(hidden_channels, hidden_channels),
            # 7 = 14 // 2
            module(hidden_channels, hidden_channels),
            # 3 = 7 // 2
            module(hidden_channels, hidden_channels),
            # 1 = 3 // 2
        )

    def forward(self, x):
        # TODO: Check input is 28px.
        x = self.conv_net(x)
        # Remove spatial dimensions.
        x = torch.squeeze(x, 2)
        x = torch.squeeze(x, 2)
        return x


class KochEmbeddingNet(nn.Module):

    def __init__(self, output_features=4096):
        super(KochEmbeddingNet, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

        self.conv_net = nn.Sequential(
            # 105
            conv_block(1, 64, kernel_size=10),
            # 96 = 105 - 10 + 1
            nn.MaxPool2d(2),
            # 48 = 96 / 2
            conv_block(64, 128, kernel_size=7),
            # 42 = 46 - 7 + 1
            nn.MaxPool2d(2),
            # 21 = 42 / 2
            conv_block(128, 128, kernel_size=4),
            # 18 = 21 - 4 + 1
            nn.MaxPool2d(2),
            # 9 = 18 / 2
            conv_block(128, 256, kernel_size=4),
            # 6 = 9 - 4 + 1
        )
        self.output_net = nn.Sequential(
            nn.Linear(6 ** 2 * 256, output_features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # TODO: Check input is 105px.
        x = self.conv_net(x)
        x = x.view(x.size(0), 6 ** 2 * 256)
        x = self.output_net(x)
        return x


#def make_same_different(ims):
#    '''Constructs pairs with label same (1) or different (0).
#
#    If original batch contains k classes with n instances,
#    then for positive examples we can obtain 
#
#    then we can compare each instance to the (k - 1) * n instances from
#    different classes.
#
#    Let m = k * n * (k - 1) * n be the number of pairs.
#    When k = 2 and n = 1, we have m = 2.
#
#    Args:
#        ims: [k, n, ...]
#
#    Returns:
#        pairs: [m, 2, ...]
#        labels: [m]
#    '''
#    pair_ims = []
#    labels = []
#    for i in range(k):
#        for p in range(n):
#            for j in range(i + 1, k):
#                for q in range(n):
#                    pair_ims.append([ims[i], ims[j]])


def _two_tuple(x):
    try:
        n = len(x)
    except TypeError:
        return (x, x)
    return tuple(x)


if __name__ == '__main__':
    main()
