# Omniglot Evaluation Protocol

The [Omniglot dataset (Lake et al. 2015)](https://github.com/brendenlake/omniglot) has become a popular dataset for evaluating few-shot learning algorithms.
However, there are some variations in the evaluation protocol.

The N-shot K-way learning problem is: given N examples for each of K classes, classify a novel input.
For Omniglot, each distinct character is a class.
Typical values are K = 5 or 20 and N = 1 or 5.

## Types of evaluation

There are two different ways to obtain few-shot classification problems for testing an algorithm.
We will refer to these as "within-alphabet" and "unstructured" evaluation.
The difference lies in how a random set of K classes is obtained:

- within-alphabet: Choose an alphabet, then choose K characters from that alphabet (without replacement).
- unstructured: Concatenate the characters of all alphabets, then choose K characters (without replacement).
The hierarchy of alphabets and characters is ignored.

Intuitively, we might expect that the unstructured problem is easier, because there is likely to be more variation _between_ alphabets than _within_ alphabets.
(This may seem counter-intuitive since characters within an alphabet must be different from one another, whereas characters across alphabets may be identical. However, a character in one alphabet can have at most one such near-identical match in another alphabet.)

The original Omniglot github repo uses within-alphabet evaluation.
It [defines 20 runs](https://github.com/brendenlake/omniglot/blob/9afc313/python/one-shot-classification/all_runs.zip), each of which comprises 20 training images and 20 testing images from the _same_ alphabet (2 runs for each of the 10 `evaluation` alphabets; see `lake-evaluation/run_alphabets.txt` for the correspondence).
We used within-alphabet evaluation in our [NIPS 2016 paper](https://arxiv.org/abs/1606.05233), although we used many random trials instead of 20 runs.

In contrast, several recent papers have used unstructured evaluation.
For example, in the ProtoNets source code, they [load a list of all characters](https://github.com/jakesnell/prototypical-networks/blob/c9bb4d2/protonets/data/omniglot.py#L115-L124) and then [take a random subset of these characters](https://github.com/jakesnell/prototypical-networks/blob/c9bb4d2/protonets/data/base.py#L39).

### Empirical comparison

Here we train a simple siamese network and compare the numbers obtained using different evaluation methods.
We report results in terms of classification error (lower is better; chance is `1 - 1/K`).
The results support the hypothesis that within-alphabet problems are much harder than unstructured problems.

| | unstructured | within-alphabet |
|---------------|---------|-----------------|
| 20-way 1-shot | 10.7%   | 24.2%           |
| 20-way 5-shot | 3.6%    | 12.8%           |
| 5-way 1-shot  | 3.5%    | 9.5%            |
| 5-way 5-shot  | 1.1%    | 4.2%            |

Hence, while recent papers may seem to be approaching saturation of the Omniglot task, we should remember that there is still room for improvement in the within-alphabet task.

Details for this experiment are as follows (the results can be replicated by running `python train.py --data_dir=data/ --download`).
The embedding network uses more or less the same architecture as the Matching Nets paper (Vinyals et al. 2016).
We input 24px images to a 4-layer conv-net, where each hidden layer has 64 channels.
There are relu, batch-norm and max-pooling (stride 2) operations between each linear layer (but not at the output).
The 64-D embedding vectors are compared using a cosine distance.
During training, each batch contains 16 few-shot problems, each of which contains K = 5 classes with 1 training image and 1 testing image.
We then use the cross-entropy-of-softmax loss with one-hot labels.
To make predictions, we use a 1-NN (nearest neighbour) classifier.


## Dataset splits

The situation is further complicated by the use of different splits.

- The official Omniglot repo defines a `background` set of 30 alphabets and an `evaluation` set of 20 alphabets.
- Koch et al. (2015) refer to a 40/10 split of alphabets.
- The Prototypical Networks repo [contains train/val/test splits](https://github.com/jakesnell/prototypical-networks/tree/c9bb4d2/data/omniglot/splits/vinyals) which partition the 50 alphabets into 33/5/12 alphabets respectively.
These are known as the Vinyals splits, presumably because they were used in the Matching Nets paper.
The `test` set is a subset of the original `evaluation` set.
We list the alphabets in this repository at `splits/vinyals/`.
