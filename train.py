from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import csv
import numpy as np
import pprint
import torch
from torchvision import transforms
from torchvision.datasets import omniglot

import logging
logger = logging.getLogger(__name__)

import nets
import data
import util


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
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
    apply_fn = getattr(nets, args.apply_method)

    model = nets.Siamese(embed_fn, similar_fn)
    model.to(device)
    parameters = list(model.parameters())
    logger.info('model parameters: %s', [x.shape for x in parameters])
    optimizer = torch.optim.SGD(parameters, lr=1e-2, momentum=0.9)

    dataset_train, dataset_test = load_datasets(args, transform=image_pre_transform)
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.0,)),
    ])

    if args.train_mode == 'pair':
        examples = data.PairSampler(
            dataset_train,
            np.random.RandomState(seed=args.train_seed),
            batch_size=args.batch_size,
            sample_mode=args.sample_mode_train,
            transform=image_transform)
    elif args.train_mode == 'softmax':
        examples = data.FewShotSampler(
            dataset_train,
            np.random.RandomState(seed=args.train_seed),
            args.sample_mode_train,
            batch_size=args.batch_size,
            k=args.num_classes_train,
            n_train=args.num_shots_train,
            n_test=1,
            transform=image_transform)
    else:
        raise ValueError('unknown train mode: "{}"'.format(args.train_mode))
    train(device, model, apply_fn, args.train_mode, examples, optimizer, args.num_steps,
          max_num_queries=args.max_num_queries_train)

    def make_config_name(mode, k, n):
        return '{:d}_way_{:d}_shot_{:s}'.format(k, n, mode)

    def configs():
        return ((sample_mode, k, n) for k in args.num_classes
                                    for n in args.num_shots
                                    for sample_mode in args.sample_modes)

    results = collections.OrderedDict()
    for sample_mode, k, n in configs():
        name = make_config_name(sample_mode, k, n)
        problems = data.FewShotSampler(
            dataset_test,
            np.random.RandomState(seed=args.test_seed),
            batch_size=1,
            sample_mode=sample_mode,
            k=k,
            n_train=n,
            n_test=1,
            transform=image_transform)
        results[name] = evaluate(device, model, apply_fn, problems,
                                 num_problems=args.num_test_problems)

    with open('results.txt', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['problem'] + args.sample_modes)
        for k in args.num_classes:
            for n in args.num_shots:
                row_header = '{:d}-way {:d}-shot'.format(k, n)
                error_rates = [results[make_config_name(mode, k, n)]['error']
                               for mode in args.sample_modes]
                percents = ['{:.1f}%'.format(100 * x) for x in error_rates]
                writer.writerow([row_header] + percents)


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


def train(device, model, apply_fn, train_mode, examples, optimizer, num_steps,
          max_num_queries=None):
    model.train()

    for i, example in zip(range(num_steps), examples):
        optimizer.zero_grad()
        if train_mode == 'pair':
            im0, im1, target = example
            im0, im1, target = im0.to(device), im1.to(device), target.to(device)
            output = model(im0, im1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target)
        elif train_mode == 'softmax':
            train_ims, test_ims, _ = example
            # train_ims: [b, k, n, ...]
            # test_ims: [b, k, n', ...]
            test_ims, gt = util.flatten_few_shot_examples(test_ims, shuffle=True)
            if max_num_queries:
                test_ims = test_ims[:, :max_num_queries]
                gt = gt[:, :max_num_queries]
            train_ims, test_ims, gt = train_ims.to(device), test_ims.to(device), gt.to(device)
            # test_ims: [b, m, ...]
            # gt: [b, m]
            scores = apply_fn(model, train_ims, test_ims)
            # scores: [b, m, k]
            loss = util.cross_entropy(scores, gt, dim=-1)
            # Besides the loss, we can obtain the accuracy.
            _, pred = torch.max(scores, -1, keepdim=False)
            is_correct = torch.eq(pred, gt).cpu().numpy()
            acc = np.sum(is_correct) / is_correct.size
        else:
            raise ValueError('unknown train mode: "{}"'.format(train_mode))
        loss.backward()
        optimizer.step()
        logger.info('step %d, loss %.4f', i, loss.item())


def evaluate(device, model, apply_fn, problems, num_problems, log_interval=100):
    model.eval()
    accuracy = util.MeanAccumulator()

    for i, (train_ims, test_ims, _) in zip(range(num_problems), problems):
        # train_ims: [b, k, n, ...]
        # test_ims: [b, k, n', ...]
        test_ims, gt = util.flatten_few_shot_examples(test_ims)
        train_ims, test_ims, gt = train_ims.to(device), test_ims.to(device), gt.to(device)
        # test_ims: [b, m, ...]
        # gt: [b, m]
        scores = apply_fn(model, train_ims, test_ims)
        # scores: [b, m, k]
        _, pred = torch.max(scores, -1, keepdim=False)
        is_correct = torch.eq(pred, gt).cpu().numpy()
        accuracy.add(np.sum(is_correct), is_correct.size)
        if (i + 1) % log_interval == 0:
            logger.info('steps %d, error rate %.3g', i + 1, 1 - accuracy.mean())

    return collections.OrderedDict([
        ('error', 1 - accuracy.mean()),
    ])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--arch', default='vinyals')
    parser.add_argument('--apply_method', default='nearest',
                        choices=['nearest', 'protonet'])
    parser.add_argument('--join', default='Cosine')
    parser.add_argument('--join_bnorm', type=util.strtobool, default=False)
    parser.add_argument('--num_steps', type=int, default=int(1e4))
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_test_problems', type=int, default=int(1e3))
    parser.add_argument('-k', '--num_classes', nargs='+', type=int, default=[20, 5])
    parser.add_argument('-n', '--num_shots', nargs='+', type=int, default=[1, 5])
    parser.add_argument('--train_mode', default='softmax', choices=['pair', 'softmax'])
    parser.add_argument('--num_classes_train', type=int, default=5,
                        help='only when train_mode is softmax')
    parser.add_argument('--num_shots_train', type=int, default=1,
                        help='only when train_mode is softmax')
    parser.add_argument('--max_num_queries_train', type=int, default=None,
                        help='only when train_mode is softmax')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('-s', '--split', default='lake')
    parser.add_argument('--sample_mode_train', default='uniform')
    parser.add_argument('--sample_modes', nargs='+',
                        default=['uniform', 'within_alphabet'])
    parser.add_argument('--train_seed', type=int, default=0)
    parser.add_argument('--test_mode', default='uniform')
    parser.add_argument('--test_seed', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    main()
