"""Copyright (c) 2020 Chengjie Wu"""

import multiprocessing

import numpy as np

from gblda import GibbsLDA


np.random.seed(0)

with open("data/data.txt", "r") as fp:
    fp_lines = fp.readlines()


def train_lda(n_components):
    global fp_lines, mode
    if mode == 1:
        # these two are set according to ab grid search on some k
        # it turns out that 0.5, 0.1 are consistent good choice!
        doc_topic_prior = 1 / n_components * 0.5
        topic_word_prior = 1 / n_components * 0.1
    else:
        doc_topic_prior = 50 / n_components
        topic_word_prior = 0.01
    lda = GibbsLDA(n_components, doc_topic_prior, topic_word_prior,
                   iterations=50, verbose=False)
    lda.fit(fp_lines[1:])
    if mode == 1:
        lda.save_state("output_k/z_{}_{}_{}.npz".format(
            n_components, doc_topic_prior, topic_word_prior))
    else:
        lda.save_state("output_k_2/z_{}_{}_{}.npz".format(
            n_components, doc_topic_prior, topic_word_prior))
    return lda


# mode = 1: generate data when alpha = 0.5/K and beta = 0.1/K
# else: generate data when alpha = 50/K and beta = 0.01
mode = 1

if mode == 1:
    range_n_components = \
        np.concatenate((np.arange(1, 201, 1), np.arange(205, 1000, 5),
                        np.arange(1050, 3000, 50), np.arange(3200, 8000, 200)))
else:
    range_n_components = np.arange(5, 500, 5)

pool = multiprocessing.Pool(processes=200)
LDAs = pool.map(train_lda, range_n_components)
