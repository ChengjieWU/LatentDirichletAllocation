"""Copyright (c) 2020 Chengjie Wu"""

import multiprocessing
import itertools

import numpy as np

from gblda import GibbsLDA


np.random.seed(0)

# with open("data/pseudo_data.txt", "r") as fp:
with open("data/data.txt", "r") as fp:
    fp_lines = fp.readlines()


def train_lda(argv):
    global fp_lines
    lda = GibbsLDA(*argv, iterations=100, verbose=False)
    lda.fit(fp_lines[1:])
    lda.save_state("output/z_{}_{}_{}.npz".format(*argv))
    return lda


range_n_components = [2, 3, 5, 7, 10, 20]
range_doc_topic_prior = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
range_topic_word_prior = [0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

hyperparameters = list()
for (n_components, doc_topic_prior, topic_word_prior) in itertools.product(
        range_n_components, range_doc_topic_prior, range_topic_word_prior):
    doc_topic_prior *= 1/n_components
    topic_word_prior *= 1/n_components
    hyperparameters.append((n_components, doc_topic_prior, topic_word_prior))

pool = multiprocessing.Pool(processes=200)
LDAs = pool.map(train_lda, hyperparameters)
