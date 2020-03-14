"""Copyright (c) 2020 Chengjie Wu"""

import multiprocessing
import os

import numpy as np
import matplotlib.pyplot as plt


from gblda import GibbsLDA


if not os.path.exists("output_k"):
    os.mkdir("output_k")

with open("data/data.txt", "r") as fp:
    fp_lines = fp.readlines()


def read_lda(n_components):
    global fp_lines
    doc_topic_prior = 1 / n_components * 0.5
    topic_word_prior = 1 / n_components * 0.1
    lda = GibbsLDA(n_components, doc_topic_prior, topic_word_prior,
                   iterations=100, verbose=False)
    lda.load_state(fp_lines[1:], "output_k/z_{}_{}_{}.npz".format(
        n_components, doc_topic_prior, topic_word_prior))
    return lda


# Read trained LDAs
range_n_components = \
    np.concatenate((np.arange(1, 201, 1), np.arange(205, 1000, 5),
                    np.arange(1050, 3000, 50), np.arange(3200, 8000, 200)))

pool = multiprocessing.Pool(processes=200)
LDAs = pool.map(read_lda, range_n_components)


# plot log likelihood with respect to k
ll_values = np.array([i.ll_best for i in LDAs])
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1, 1, 1)
ax.plot(range_n_components, ll_values,
        color="red", linewidth=2, markersize=4, marker='o', alpha=0.3)
ax.set_xlabel("k")
ax.set_ylabel("log likelihood")
ax.grid(which='major', axis='both',
        linewidth=0.75, linestyle='-',
        color='lightgray')
ax.grid(which='minor', axis='both',
        linewidth=0.25, linestyle='-',
        color='lightgray')
fig.show()
fig.savefig("plots/different_k.jpg",
            dpi=600, bbox_inches="tight")


# plot learning curve of some LDAs
plotting_k = [199, 149, -1]
plotting_colors = ["red", "green", "blue"]

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1, 1, 1)
for i, k in enumerate(plotting_k):
    log = np.copy(LDAs[k].log)
    x = np.arange(1, len(log)+1)
    ax.plot(x, log, label="k={}".format(range_n_components[k]),
            color=plotting_colors[i],
            linewidth=2, markersize=4, marker='o', alpha=0.3)
ax.legend()
ax.set_xlabel("iterations")
ax.set_ylabel("log likelihood")
ax.grid(which='major', axis='both',
        linewidth=0.75, linestyle='-',
        color='lightgray')
ax.grid(which='minor', axis='both',
        linewidth=0.25, linestyle='-',
        color='lightgray')
fig.show()
