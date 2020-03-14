"""Copyright (c) 2020 Chengjie Wu"""

import multiprocessing
import itertools
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from gblda import GibbsLDA


if not os.path.exists("output"):
    os.mkdir("output")

with open("data/data.txt", "r") as fp:
    fp_lines = fp.readlines()


def read_lda(argv):
    global fp_lines
    lda = GibbsLDA(*argv, iterations=100, verbose=True)
    lda.load_state(fp_lines[1:], "output/z_{}_{}_{}.npz".format(*argv))
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

# Read trained LDAs
pool = multiprocessing.Pool(processes=200)
LDAs = pool.map(read_lda, hyperparameters)
LDAs = np.array(LDAs)
LDAs = LDAs.reshape((len(range_n_components), len(range_doc_topic_prior),
                     len(range_topic_word_prior)))

# Find the best alpha, and beta for different Ks
k_best = list()
for i in range(LDAs.shape[0]):
    ldas = LDAs[i, :, :]
    ll_best = -np.inf
    lda_best = None
    for j in range(LDAs.shape[1]):
        for k in range(LDAs.shape[2]):
            if ldas[j, k].ll_best > ll_best:
                ll_best = ldas[j, k].ll_best
                lda_best = ldas[j, k]
    k_best.append(lda_best)

for lda in k_best:
    print(lda.n_components, lda.doc_topic_prior, lda.topic_word_prior,
          lda.ll_best)


# Use k=3 and k=20 as an example, plot graphs according to alpha or beta
# [1, 1, 3] is the best
# k = 1
# best_alpha = 1
# best_beta = 3
k = 5
best_alpha = 1
best_beta = 3

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1, 1, 1)
alphas = np.array(range_doc_topic_prior) * 1/range_n_components[k]
betas = np.array(range_topic_word_prior) * 1/range_n_components[k]
value_on_alphas = np.array(
    [LDAs[k, i, best_beta].ll_best for i in range(len(range_doc_topic_prior))])
value_on_betas = np.array(
    [LDAs[k, best_alpha, i].ll_best for i in range(len(range_topic_word_prior))])
ax.plot(alphas, value_on_alphas, label="beta=1/K*0.1", color="green",
        linewidth=2, markersize=4, marker='o', alpha=0.3)
ax.plot(betas, value_on_betas, label="alpha=1/K*0.5", color="red",
        linewidth=2, markersize=4, marker='o', alpha=0.3)
ax.legend()
ax.set_xlabel("alpha/beta")
ax.set_ylabel("log likelihood")
ax.set_xscale("log")
ax.grid(which='major', axis='both',
        linewidth=0.75, linestyle='-',
        color='lightgray')
ax.grid(which='minor', axis='both',
        linewidth=0.25, linestyle='-',
        color='lightgray')
fig.show()
fig.savefig("plots/ab_{}.jpg".format(range_n_components[k]),
            dpi=600, bbox_inches="tight")


# Use k=3 and k=20 as an example, plot graphs according to alpha and beta
# k = 1
k = 5

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1, 1, 1)
values = np.array(
    [[LDAs[k, i, j].ll_best for j in range(len(range_topic_word_prior))]
     for i in range(len(range_doc_topic_prior))])
im = ax.imshow(values)
ax.set_yticks(np.arange(len(range_doc_topic_prior)))
ax.set_xticks(np.arange(len(range_topic_word_prior)))
ax.set_yticklabels(range_doc_topic_prior)
ax.set_xticklabels(range_topic_word_prior)
ax.set_ylabel("alpha: multiple of K")
ax.set_xlabel("beta: multiple of K")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)
fig.show()
fig.savefig("plots/grid_{}.jpg".format(range_n_components[k]),
            dpi=600, bbox_inches="tight")
