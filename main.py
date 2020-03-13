import numpy as np
import pickle

from gblda import GibbsLDA


np.random.seed(0)

with open("data/data.txt", "r") as fp:
    fp_lines = fp.readlines()

num_docs = int(fp_lines[0])
dict_word2ind = dict()
list_ind2word = list()
corpus = list()

for line in fp_lines[1:]:
    words = line.strip().split(" ")
    document = list()
    for word in words:
        if word not in dict_word2ind:
            dict_word2ind[word] = len(list_ind2word)
            list_ind2word.append(word)
        document.append(dict_word2ind[word])
    corpus.append(document)
assert len(corpus) == num_docs
for word, ind in dict_word2ind.items():
    assert list_ind2word[ind] == word

num_words = len(list_ind2word)

print("Number of documents:", num_docs)
print("Number of words:", num_words)


lda = GibbsLDA(n_components=3, iterations=2000)
lda.fit(fp_lines[1:])

print(lda.theta[0])
with open("save.pkl", "wb") as fp:
    pickle.dump(lda, fp)
