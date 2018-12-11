from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import functools
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import omniglot


def from_torchvision(dataset):
    print('loading images...')
    images = []
    labels = []
    for im, label in dataset:
        images.append(im)
        labels.append(label)
    print('done: load images')
    return OmniglotDataset(dataset._alphabets, dataset._characters, images, labels)


class OmniglotDataset(object):
    '''
    Contains all images in memory.
    Builds indices for efficient sampling.
    '''

    def __init__(self, alphabets, characters, images, labels):
        '''
        Args:
            alphabets: List of strings.
            characters: List of strings.
            images: List of images.
            labels: List of character indices.
        '''
        self.alphabets = alphabets
        self.characters = characters
        self.images = images
        self.labels = labels
        self._build_index()

    def _build_index(self):
        '''
        Creates:
            alphabet_chars: Map from string to list of character indices.
            char_instances: Map from character index to list of example indices.
        '''
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


def add_rotations(dataset):
    angles = [0, 90, 180, 270]
    label_angle_pairs = list(itertools.chain.from_iterable(
        ((label, angle) for label in range(len(dataset.characters))) for angle in angles))
    new_label = _inv_map(enumerate(label_angle_pairs))

    labels = list(itertools.chain.from_iterable(
        (new_label[(label, angle)] for label in dataset.labels) for angle in angles))
    images = list(itertools.chain.from_iterable(
        (transforms.functional.rotate(im, angle) for im in dataset.images)
        for angle in angles))
    characters = [_rot_name(dataset.characters[label], angle)
                  for label, angle in label_angle_pairs]
    return OmniglotDataset(dataset.alphabets, characters, images, labels)


def _rot_name(original, angle):
    return original if angle == 0 else (original + '_rot' + str(angle))


class PairSampler(torch.utils.data.Sampler):

    def __init__(self, dataset, rand, batch_size, prob_pos=0.5, sample_mode='uniform', transform=None):
        '''
        Args:
            transform: None, or must convert PIL image to torch Tensor.
        '''
        self._dataset = dataset
        self._rand = rand
        self._batch_size = batch_size
        self._prob_pos = prob_pos
        self._sample_mode = sample_mode
        self._transform = transform or transforms.ToTensor()

    def __iter__(self):
        return self

    def __len__(self):
        return 10 ** 9

    def __next__(self):
        if self._sample_mode == 'within_alphabet':
            if self._dataset.alphabets is None:
                raise RuntimeError('dataset does not support alphabets')
            alphabet = self._rand.choice(self._dataset.alphabets)
            char_candidates = self._dataset.alphabet_chars[alphabet]
        elif self._sample_mode == 'uniform':
            char_candidates = len(self._dataset.characters)
        else:
            raise ValueError('unknown sample mode: {}'.format(self._sample_mode))

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
            im0 = self._transform(im0)
            im1 = self._transform(im1)
            label = torch.tensor([float(is_pos)])
            batch_im0.append(im0)
            batch_im1.append(im1)
            batch_label.append(label)

        batch_im0 = torch.stack(batch_im0)
        batch_im1 = torch.stack(batch_im1)
        batch_label = torch.stack(batch_label)
        return batch_im0, batch_im1, batch_label


class FewShotSampler(object):

    def __init__(self, dataset, rand, sample_mode, batch_size, k, n_train, n_test, transform=None):
        '''
        Builds necessary indices for efficient sampling.

        Args:
            transform: None, or must convert PIL image to torch Tensor.
        '''
        self._dataset = dataset
        self._rand = rand
        self._sample_mode = sample_mode
        self._batch_size = batch_size
        self._k = k
        self._n_train = n_train
        self._n_test = n_test
        self._transform = transform or transforms.ToTensor()

    def __iter__(self):
        return self

    def __next__(self):
        '''
        Returns:
            train_ims: [k, n_train, c, h, w]
            test_ims: [k, n_test, c, h, w]
            labels: [k, 1]

        TODO: Should we flatten and shuffle test_ims to eliminate risk of model using location?

        TODO: Support using characters all drawn by same author?
        '''
        if self._sample_mode == 'uniform':
            # Choose random subset of k character classes.
            chars = self._rand.choice(len(self._dataset.characters), self._k, replace=False)
        elif self._sample_mode == 'within_alphabet':
            if self._dataset.alphabets is None:
                raise RuntimeError('dataset does not support alphabets')
            # Choose an alphabet.
            alphabet = self._rand.choice(self._dataset.alphabets)
            # Choose characters.
            subset = self._dataset.alphabet_chars[alphabet]
            chars = self._rand.choice(subset, self._k, replace=False)
        else:
            raise ValueError('unknown sample mode: "{}"'.format(self._sample_mode))

        # Sample a random n instances for each character.
        n = self._n_train + self._n_test

        # TODO: Neat and simple way to avoid this mess?
        batch_train_ims = []
        batch_test_ims = []
        batch_labels = []

        for i in range(self._batch_size):
            train_ims = []
            test_ims = []
            for char in chars:
                char_ims = []
                instances = self._rand.choice(self._dataset.char_instances[char], n, replace=False)
                for index in instances:
                    im = self._dataset.images[index]
                    im = self._transform(im)
                    char_ims.append(im)
                # Take first `n_train` for training, rest for testing.
                train_ims.append(torch.stack(char_ims[:self._n_train]))
                test_ims.append(torch.stack(char_ims[self._n_train:]))
            train_ims = torch.stack(train_ims)
            test_ims = torch.stack(test_ims)
            labels = torch.tensor(chars).unsqueeze(-1)
            # Add to batch.
            batch_train_ims.append(train_ims)
            batch_test_ims.append(test_ims)
            batch_labels.append(labels)

        batch_train_ims = torch.stack(batch_train_ims)
        batch_test_ims = torch.stack(batch_test_ims)
        batch_labels = torch.stack(batch_labels)
        return batch_train_ims, batch_test_ims, batch_labels


def load_both_and_merge(data_dir, **kwargs):
    '''Merge background and evaluation sets into a single OmniglotDataset.

    Args: For torchvision.datasets.omniglot.Omniglot().
    '''
    assert 'background' not in kwargs
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


def subset_alphabets(dataset, alphabets):
    # Identify subset of characters (by index) which will be kept.
    char_subset = list(itertools.chain.from_iterable(
        dataset.alphabet_chars[a] for a in alphabets))
    new_index = _inv_map(enumerate(char_subset))
    # Take subset of characters.
    characters = [dataset.characters[i] for i in char_subset]
    # Take subset of images where label is in subset.
    char_subset_unordered = set(char_subset)
    images = []
    labels = []
    for image, label in zip(dataset.images, dataset.labels):
        if label in char_subset_unordered:
            images.append(image)
            labels.append(new_index[label])
    return OmniglotDataset(alphabets, characters, images, labels)


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
        index_map = _inv_map(enumerate(subsets[i]))
        labels[i] = list(map(index_map.__getitem__, labels[i]))

    names = [[dataset.characters[char] for char in subsets[i]] for i in range(num_subsets)]
    return [OmniglotDataset(None, names[i], images[i], labels[i]) for i in range(num_subsets)]


def _inv_map(items):
    r = {}
    for i, xi in items:
        r[xi] = i
    return r
