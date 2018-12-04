from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import util


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
        feat_a = util.map_images(self.embed, images_a)
        feat_b = util.map_images(self.embed, images_b)
        delta = torch.abs(feat_a - feat_b)
        output = self.postproc(delta)
        return output


# For fully-connected layers, following pattern in:
# https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py


class VinyalsEmbedding(nn.Module):
    # https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py 

    def __init__(self, hidden_channels=64):
        super(VinyalsEmbedding, self).__init__()

        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.conv_net = nn.Sequential(
            # 28 (ideal 24)
            block(1, hidden_channels),
            # 14 = 28 // 2 (ideal 12 = 24 // 2)
            block(hidden_channels, hidden_channels),
            # 7 = 14 // 2 (ideal 6 = 12 // 2)
            block(hidden_channels, hidden_channels),
            # 3 = 7 // 2 (ideal 3 = 6 // 2)
            # block(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, 3),
            # 1
        )

    def forward(self, x):
        # TODO: Check input is 28px.
        x = self.conv_net(x)
        # Remove spatial dimensions.
        x = torch.squeeze(x, 2)
        x = torch.squeeze(x, 2)
        return x


class KochEmbedding(nn.Module):

    def __init__(self, output_features=4096):
        super(KochEmbedding, self).__init__()

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
