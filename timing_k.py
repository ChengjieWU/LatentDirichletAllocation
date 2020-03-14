"""Copyright (c) 2020 Chengjie Wu"""

import multiprocessing
import time

import numpy as np
import matplotlib.pyplot as plt

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
                   iterations=30, verbose=False)
    lda._initializing_corpus(fp_lines[1:])
    lda.loaded = True
    start_time = time.time()
    lda.fit("NOT CARE")
    end_time = time.time()
    return end_time - start_time


range_n_components = np.arange(1, 100, 2)
pool = multiprocessing.Pool(processes=200)
time_values = pool.map(train_lda, range_n_components)

time_values = np.array(time_values)

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1, 1, 1)
ax.plot(range_n_components, time_values,
        color="red", linewidth=2, markersize=4, marker='o', alpha=0.3)
ax.set_xlabel("k")
ax.set_ylabel("time for 30 iteration (s)")
ax.grid(which='major', axis='both',
        linewidth=0.75, linestyle='-',
        color='lightgray')
ax.grid(which='minor', axis='both',
        linewidth=0.25, linestyle='-',
        color='lightgray')
fig.show()
fig.savefig("plots/timing_k.jpg",
            dpi=600, bbox_inches="tight")
