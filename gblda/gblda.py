import numpy as np
import time


class GibbsLDA:
    def __init__(self, alpha_value=1, beta_value=1, n_components=3, iterations=1000):
        self.n_components = n_components
        self.iterations = iterations
        self.alpha_value = alpha_value
        self.beta_value = beta_value

        self.alpha = None
        self.beta = None
        self.num_docs = None
        self.num_words = None
        self.dict_word2ind = None
        self.list_ind2word = None
        self.corpus = None

        self.n_mk = None
        self.n_kt = None
        self.n_k = None
        self.n_m = None
        self.z_mn = None

        self.theta = None
        self.phi = None
        self.c_mean = None
        self.log = None

    def fit(self, X, y=None):
        self.num_docs = len(X)
        self.dict_word2ind = dict()
        self.list_ind2word = list()
        self.corpus = list()

        max_len = 0
        for line in X:
            words = line.strip().split(" ")
            max_len = max(max_len, len(words))
            document = list()
            for word in words:
                if word not in self.dict_word2ind:
                    self.dict_word2ind[word] = len(self.list_ind2word)
                    self.list_ind2word.append(word)
                document.append(self.dict_word2ind[word])
            self.corpus.append(document)
        assert len(self.corpus) == self.num_docs
        for word, ind in self.dict_word2ind.items():
            assert self.list_ind2word[ind] == word

        self.num_words = len(self.list_ind2word)

        # get alpha and beta
        self.alpha = np.full(shape=(self.n_components,),
                             fill_value=self.alpha_value, dtype=np.float32)
        self.beta = np.full(shape=(self.num_words,),
                            fill_value=self.beta_value, dtype=np.float32)

        # initialization
        self.n_mk = np.zeros(shape=(self.num_docs, self.n_components), dtype=np.int32)
        self.n_kt = np.zeros(shape=(self.n_components, self.num_words), dtype=np.int32)
        self.n_k = np.zeros(shape=(self.n_components,), dtype=np.int32)
        self.n_m = np.zeros(shape=(self.num_docs,), dtype=np.int32)
        self.z_mn = np.zeros(shape=(self.num_docs, max_len), dtype=np.int32)
        for m, dm in enumerate(self.corpus):
            for n, w_mn in enumerate(dm):
                k = np.random.randint(self.n_components)
                self.z_mn[m, n] = k
                self.n_mk[m, k] += 1
                self.n_m[m] += 1
                self.n_kt[k, w_mn] += 1
                self.n_k[k] += 1
        self.theta = np.zeros(shape=(self.num_docs, self.n_components))
        self.phi = np.zeros(shape=(self.n_components, self.num_words))
        self.c_mean = 0
        self.log = list()

        # Gibbs sampling
        # 1. burn in for 1000 iterations
        for _ in range(1000):
            self._gibbs_sampling_iteration()

        # 2. sample z and update theta & phi every 3 transitions
        for it in range(self.iterations):
            start_time = time.time()

            for _ in range(3):
                self._gibbs_sampling_iteration()

            theta = np.zeros(shape=(self.num_docs, self.n_components))
            phi = np.zeros(shape=(self.n_components, self.num_words))
            for m in range(self.theta.shape[0]):
                theta[m, :] = \
                    (self.alpha + self.n_mk[m, :]) / np.sum(self.alpha + self.n_mk[m, :])
            for k in range(self.phi.shape[0]):
                phi[k, :] = \
                    (self.beta + self.n_kt[k, :]) / np.sum(self.beta + self.n_kt[k, :])

            self.c_mean += 1
            self.theta = self.theta + 1 / self.c_mean * (theta - self.theta)
            self.phi = self.phi + 1 / self.c_mean * (phi - self.phi)

            ll = self.log_likelihood(self.corpus)
            self.log.append(ll)

            end_time = time.time()
            print("Iteration", it, ", log-likelihood:", ll,
                  ", time consumed:", end_time - start_time)

    def _gibbs_sampling_iteration(self):
        for m, dm in enumerate(self.corpus):
            for n, w_mn in enumerate(dm):
                k = self.z_mn[m, n]
                self.n_mk[m, k] -= 1
                self.n_m[m] -= 1
                self.n_kt[k, w_mn] -= 1
                self.n_k[k] -= 1
                k = self._conditional_z(
                    self.alpha, self.beta, self.n_mk, self.n_kt, m, w_mn)
                self.z_mn[m, n] = k
                self.n_mk[m, k] += 1
                self.n_m[m] += 1
                self.n_kt[k, w_mn] += 1
                self.n_k[k] += 1

    def _conditional_z(self, alpha, beta, n_mk, n_kt, m, t):
        probability = ((alpha + n_mk[m, :]) / np.sum(alpha + n_mk[m, :])) \
                      * ((beta[t] + n_kt[:, t]) / np.sum(beta[t] + n_kt[:, t]))
        probability /= np.sum(probability)
        assert(np.all(probability >= 0.))
        assert(1.0 - 1e-6 < np.sum(probability) < 1.0 + 1e-6)
        assert(len(probability) == self.n_components)
        return np.random.choice(self.n_components, p=probability)

    def log_likelihood(self, corpus):
        ret = 0.
        for m, dm in enumerate(corpus):
            for n, w_mn in enumerate(dm):
                tp = 0.
                for k in range(self.n_components):
                    tp += self.theta[m, k] * self.phi[k, w_mn]
                ret += np.log(tp)
        return ret

