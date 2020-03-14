"""Copyright (c) 2020 Chengjie Wu"""

import multiprocessing

import numpy as np

from gblda import GibbsLDA


np.random.seed(0)

with open("data/data.txt", "r") as fp:
    fp_lines = fp.readlines()


def train_lda(n_components):
    global fp_lines
    # these two are set according to ab grid search on some k
    # it turns out that 0.5, 0.1 are consistent good choice!
    doc_topic_prior = 1 / n_components * 0.5
    topic_word_prior = 1 / n_components * 0.1
    lda = GibbsLDA(n_components, doc_topic_prior, topic_word_prior,
                   iterations=10, verbose=False)
    lda.fit(fp_lines[1:])
    lda.save_state("output_k/z_{}_{}_{}.npz".format(
        n_components, doc_topic_prior, topic_word_prior))
    return lda


# range_n_components = \
#     np.concatenate((np.arange(1, 201, 1), np.arange(205, 1000, 5),
#                     np.arange(1050, 3000, 50)))
range_n_components = np.arange(3200, 8000, 200)

pool = multiprocessing.Pool(processes=200)
LDAs = pool.map(train_lda, range_n_components)
