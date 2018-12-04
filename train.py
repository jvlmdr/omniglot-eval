from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pprint
import torch
from torchvision import transforms
from torchvision.datasets import omniglot

import nets
import data
import util


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.arch == 'koch':
        embed_dim = 4096
        embed_fn = nets.KochEmbedding(embed_dim)
        image_pre_transform = None
    elif args.arch == 'vinyals':
        embed_dim = 64
        embed_fn = nets.VinyalsEmbedding(embed_dim)
        image_pre_transform = transforms.Resize((24, 24))
    else:
        raise ValueError('unknown arch: "{}"'.format(arch))

    similar_fn = getattr(nets, args.join)(use_bnorm=args.join_bnorm, n=embed_dim)
    siamese = nets.Siamese(embed_fn, similar_fn)
    siamese.to(device)
    predictor = nets.MostSimilar(siamese)
    parameters = list(siamese.parameters())
    print('model parameters:')
    pprint.pprint([x.shape for x in parameters])
    optimizer = torch.optim.SGD(parameters, lr=1e-2, momentum=0.9)

    dataset_train, dataset_test = load_datasets(args, transform=image_pre_transform)
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.0,)),
    ])

    examples = data.PairSampler(
        dataset_train,
        np.random.RandomState(seed=args.train_seed),
        batch_size=args.batch_size,
        mode=args.train_sample_mode,
        transform=image_transform)
    train(siamese, examples, optimizer, args.num_train_steps, device=device)

    for mode in args.test_sample_modes:
        problems = data.FewShotSampler(
            dataset_test,
            np.random.RandomState(seed=args.test_seed),
            mode=mode,
            k=args.num_classes,
            n_train=args.num_shots,
            n_test=1,
            transform=image_transform)
        test(predictor, problems, num_problems=args.num_test_problems, device=device)


def load_datasets(args, transform=None):
    if args.split == 'vinyals':
        entire_dataset = data.load_both_and_merge(
            args.data_dir,
            transform=transform,
            download=args.download)
        alphabets_train = util.open_and_read('splits/vinyals/trainval.txt')
        alphabets_test = util.open_and_read('splits/vinyals/test.txt')
        dataset_train = data.subset_alphabets(entire_dataset, alphabets_train)
        dataset_test = data.subset_alphabets(entire_dataset, alphabets_test)
    elif args.split == 'mix':
        entire_dataset = data.load_both_and_merge(
            args.data_dir,
            transform=transform,
            download=args.download)
        dataset_train, dataset_test = data.split_classes(entire_dataset, [0.8, 0.2])
    elif args.split == 'lake':
        dataset_train = data.from_torchvision(omniglot.Omniglot(
            args.data_dir,
            background=True,
            transform=transform,
            download=args.download))
        dataset_test = data.from_torchvision(omniglot.Omniglot(
            args.data_dir,
            background=False,
            transform=transform,
            download=args.download))
    else:
        raise ValueError('unknown split: "{}"'.format(args.split))
    return dataset_train, dataset_test


def train(model, examples, optimizer, num_steps, device=None):
    model.train()

    for i, (im0, im1, target) in zip(range(num_steps), examples):
        if device:
            im0, im1, target = im0.to(device), im1.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(im0, im1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target)
        loss.backward()
        optimizer.step()
        print('step {:d}, loss {:.3g}'.format(i, loss.item()))


def test(model, problems, num_problems, device=None, log_interval=100):
    model.eval()
    accuracy = util.MeanAccumulator()

    for i, (train_ims, test_ims, _) in zip(range(num_problems), problems):
        if device:
            train_ims, test_ims = train_ims.to(device), test_ims.to(device)
        b = train_ims.shape[0]
        n = train_ims.shape[2]
        # Flatten test examples.
        test_ims, gt = util.flatten_few_shot_examples(test_ims)
        pred = model(train_ims, test_ims)
        pred, gt = pred.cpu(), gt.cpu()
        is_correct = torch.eq(pred, gt).numpy()
        accuracy.add(np.sum(is_correct), is_correct.size)
        if (i + 1) % log_interval == 0:
            print('steps {}, error rate {:.3g}'.format(i + 1, 1 - accuracy.mean()))

    print('error rate: {:.3g}'.format(1 - accuracy.mean()))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--arch', default='vinyals')
    parser.add_argument('--join', default='WeightedL1')
    parser.add_argument('--join_bnorm', type=util.strtobool, default=True)
    parser.add_argument('--num_train_steps', type=int, default=int(1e4))
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_test_problems', type=int, default=int(1e3))
    parser.add_argument('-k', '--num_classes', type=int, default=20)
    parser.add_argument('-n', '--num_shots', type=int, default=1)
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('-s', '--split', default='lake')
    parser.add_argument('--train_sample_mode', default='uniform')
    parser.add_argument('--test_sample_modes', nargs='+',
                        default=['uniform', 'within_alphabet'])
    parser.add_argument('--train_seed', type=int, default=0)
    parser.add_argument('--test_mode', default='uniform')
    parser.add_argument('--test_seed', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    main()
