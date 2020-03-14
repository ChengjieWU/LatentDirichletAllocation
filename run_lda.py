"""Copyright (c) 2020 Chengjie Wu"""

import numpy as np
import matplotlib.pyplot as plt

from gblda import GibbsLDA


np.random.seed(0)

# read data
with open("data/data.txt", "r") as fp:
    fp_lines = fp.readlines()

# create a LDA instance
lda = GibbsLDA(n_components=3, doc_topic_prior=0.5/3, topic_word_prior=0.1/3,
               iterations=100, verbose=True)

# fit LDA model
# X is a list of strings, where each string represents a document.
# The documents will be parsed with space to get all words.
lda.fit(X=fp_lines[1:])

# print top 10 representative words in each topic
lda.get_representative_words()

# print log likelihood
print("log likelihood:", lda.ll_best)

# plot learning curve
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1, 1, 1)
log = np.copy(lda.log)
x = np.arange(1, len(log)+1)
ax.plot(x, log, label="k={}".format(3),
        color="blue", linewidth=2, markersize=4, marker='o', alpha=0.3)
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
fig.savefig("plots/learningcurve_{}.jpg".format(3),
            dpi=600, bbox_inches="tight")
