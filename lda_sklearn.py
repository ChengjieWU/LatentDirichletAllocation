from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


# with open("data/data.txt", "r") as fp:
with open("data/pseudo_data.txt", "r") as fp:
    fp_lines = fp.readlines()


cntVector = CountVectorizer()
X = cntVector.fit_transform(fp_lines[1:])

lda = LatentDirichletAllocation(n_components=3)
lda.fit(X)
print(lda.score(X))
print(lda.perplexity(X))
print(lda.components_)

