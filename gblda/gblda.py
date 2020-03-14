"""Copyright (c) 2020 Chengjie Wu"""

import time

import numpy as np


class GibbsLDA:
    def __init__(self, n_components=3, doc_topic_prior=None,
                 topic_word_prior=None, iterations=1000, verbose=True):
        """Latent Dirichlet Allocation with Gibbs sampling

        :param n_components: int, number of topics
        :param doc_topic_prior: float, alpha. If None, 1/n_components*0.5 is set
            by default.
        :param topic_word_prior: float, beta. If None, 1/n_components*0.1 is set
            by default.
        :param iterations: int, iterations of Gibbs sampling
        :param verbose: bool, whether to print intermediate results
        """

        self.loaded = False
        self.verbose = verbose

        self.n_components = n_components
        self.iterations = iterations
        self.doc_topic_prior = \
            doc_topic_prior if doc_topic_prior else 1/n_components*0.5
        self.topic_word_prior = \
            topic_word_prior if topic_word_prior else 1/n_components*0.1

        self.num_docs = None
        self.num_words = None
        self.dict_word2ind = None
        self.list_ind2word = None
        self.corpus = None

        self.alpha = None
        self.beta = None
        self.beta_sum = None

        self.n_mk = None
        self.n_kt = None
        self.n_k = None
        self.n_m = None
        self.z_mn = None        # current z sampling state

        self.theta = None       # best theta
        self.phi = None         # best phi
        self.z_best = None      # best z
        self.ll_best = None     # best ll
        self.log = None         # record log likelihood during training

    def load_state(self, X, file):
        """Load LDA state from file.

        :param X: a list of strings, corpus. Each string represents a document,
            and words are separated with space.
        :param file: string, file to load.
        :return: None.
        """
        self._initializing_corpus(X, file)
        self.loaded = True

    def save_state(self, file):
        """Save LDA state to file.

        Note that we do not save the corpus. When the LDA is loaded next time,
        it MUST read in exactly the same corpus. Behaviors are undefined if not.

        :param file: string, file to save.
        :return: None.
        """
        np.savez(file, z_mn=self.z_mn, theta=self.theta, phi=self.phi,
                 z_best=self.z_best, ll_best=self.ll_best, log=self.log)

    def fit(self, X, y=None):
        """ Train LDA.

        :param X: a list of strings, corpus. Each string represents a document,
            and words are separated with space. If state is loaded, either
            through load_state or a previous call to fit, X is ignored.
        :param y: ignored.
        :return: None.
        """
        if not self.loaded:
            self._initializing_corpus(X)
            self.loaded = True

        if self.verbose:
            print("Before training:", self.log_likelihood())

        # NOTE: we keep the best theta & phi
        for it in range(self.iterations):
            start_time = time.time()

            self._gibbs_sampling_iteration()

            theta = self.calculate_theta(
                self.num_docs, self.n_components, self.alpha, self.n_mk)
            phi = self.calculate_phi(
                self.n_components, self.num_words, self.beta, self.n_kt)
            ll = self.log_likelihood(theta, phi)
            self.log.append(ll)
            if self.verbose:
                print("log likelihood:", self.log_likelihood(theta, phi))

            if self.iterations < 30 or it >= 30:
                # we keep 30 iterations of Gibbs sampling as burning in
                if ll > self.ll_best:
                    self.ll_best = ll
                    self.theta = np.copy(theta)
                    self.phi = np.copy(phi)
                    self.z_best = np.copy(self.z_mn)

            end_time = time.time()
            if self.verbose:
                print("Iteration", it, ", best log-likelihood:", self.ll_best,
                      ", time consumed:", end_time - start_time)

    def _initializing_corpus(self, X, z_file=None):
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

        if self.verbose:
            print("Number of documents:", self.num_docs)
            print("Number of words:", self.num_words)

        # get alpha and beta
        self.alpha = np.full(shape=(self.n_components,),
                             fill_value=self.doc_topic_prior, dtype=np.float32)
        self.beta = np.full(shape=(self.num_words,),
                            fill_value=self.topic_word_prior, dtype=np.float32)
        self.beta_sum = np.sum(self.beta)

        self.n_mk = np.zeros(shape=(self.num_docs, self.n_components), dtype=np.int32)
        self.n_kt = np.zeros(shape=(self.n_components, self.num_words), dtype=np.int32)
        self.n_k = np.zeros(shape=(self.n_components,), dtype=np.int32)
        self.n_m = np.zeros(shape=(self.num_docs,), dtype=np.int32)
        if z_file:
            db = np.load(z_file)
            self.z_mn = np.array(db["z_mn"])
            self.theta = db["theta"]
            self.phi = db["phi"]
            self.z_best = db["z_best"]
            self.ll_best = db["ll_best"]
            self.log = db["log"]
            # initialization
            for m, dm in enumerate(self.corpus):
                for n, w_mn in enumerate(dm):
                    k = self.z_mn[m, n]
                    self.n_mk[m, k] += 1
                    self.n_m[m] += 1
                    self.n_kt[k, w_mn] += 1
                    self.n_k[k] += 1
        else:
            # initialization
            self.z_mn = np.zeros(shape=(self.num_docs, max_len), dtype=np.int32)
            for m, dm in enumerate(self.corpus):
                for n, w_mn in enumerate(dm):
                    k = np.random.randint(self.n_components)
                    self.z_mn[m, n] = k
                    self.n_mk[m, k] += 1
                    self.n_m[m] += 1
                    self.n_kt[k, w_mn] += 1
                    self.n_k[k] += 1
            self.theta = self.calculate_theta(
                self.num_docs, self.n_components, self.alpha, self.n_mk)
            self.phi = self.calculate_phi(
                self.n_components, self.num_words, self.beta, self.n_kt)
            self.z_best = np.copy(self.z_mn)
            self.ll_best = self.log_likelihood()
            self.log = list()

    def _gibbs_sampling_iteration(self):
        """ One Gibbs sampling step.

        :return: None.
        """
        for m, dm in enumerate(self.corpus):
            for n, w_mn in enumerate(dm):
                k = self.z_mn[m, n]
                self.n_mk[m, k] -= 1
                self.n_m[m] -= 1
                self.n_kt[k, w_mn] -= 1
                self.n_k[k] -= 1
                k = self._conditional_z(
                    self.n_components, self.alpha, self.beta,
                    self.n_mk, self.n_kt, m, w_mn, self.beta_sum, self.n_k)
                self.z_mn[m, n] = k
                self.n_mk[m, k] += 1
                self.n_m[m] += 1
                self.n_kt[k, w_mn] += 1
                self.n_k[k] += 1

    @staticmethod
    def _conditional_z(K, alpha, beta, n_mk, n_kt, m, t, beta_sum, n_k):
        """Sample new z_mt using conditional distribution."""
        probability = \
            (alpha + n_mk[m, :]) * ((beta[t] + n_kt[:, t]) / (beta_sum + n_k))
        probability /= np.sum(probability)
        assert(np.all(probability >= 0.))
        assert(1.0 - 1e-6 < np.sum(probability) < 1.0 + 1e-6)
        assert(len(probability) == K)
        return np.random.choice(K, p=probability)

    def calculate_theta(self, M=None, K=None, alpha=None, n_mk=None):
        """ Calculate theta.

        If parameters are not given, then corresponding parameters are picked
        from current LDA.

        :param M: int, number of documents.
        :param K: int, number of topics.
        :param alpha: numpy array of shape [K, ], alpha.
        :param n_mk: numpy array of shape [M, K], n_mk.
        :return: numpy array of shape [M, K], theta.
        """
        M = M if M is not None else self.num_docs
        K = K if K is not None else self.n_components
        alpha = alpha if alpha is not None else self.alpha
        n_mk = n_mk if n_mk is not None else self.n_mk
        theta = n_mk + np.tile(alpha, (M, 1))
        theta /= np.repeat(np.sum(theta, axis=1).reshape((-1, 1)), K, axis=1)
        return theta

    def calculate_phi(self, K=None, V=None, beta=None, n_kt=None):
        """ Calculate phi.

        If parameters are not given, then corresponding parameters are picked
        from current LDA.

        :param K: int, number of topics.
        :param V: int, number of words (size of vocabulary).
        :param beta: numpy array of shape [V, ], beta.
        :param n_kt: numpy array of shape [K, V], n_kt.
        :return: numpy array of shape [K, V], phi.
        """
        K = K if K is not None else self.n_components
        V = V if V is not None else self.num_words
        beta = beta if beta is not None else self.beta
        n_kt = n_kt if n_kt is not None else self.n_kt
        phi = n_kt + np.tile(beta, (K, 1))
        phi /= np.repeat(np.sum(phi, axis=1).reshape((-1, 1)), V, axis=1)
        return phi

    def log_likelihood(self, theta=None, phi=None):
        """ Calculate log likelihood with respect to this LDA's corpus.

        If parameters are not given, then corresponding parameters are picked
        from current LDA.

        :param theta: numpy array of shape [M, K], theta.
        :param phi: numpy array of shape [K, V], phi.
        :return: float, log likelihood.
        """
        theta = theta if theta is not None else self.theta
        phi = phi if phi is not None else self.phi
        ret = 0.
        for m, dm in enumerate(self.corpus):
            for n, w_mn in enumerate(dm):
                tp = 0.
                for k in range(self.n_components):
                    tp += theta[m, k] * phi[k, w_mn]
                ret += np.log(tp)
        return ret

    def get_representative_words(self, phi=None):
        """ Get top 10 representative words in each topic.

        If parameters are not given, then corresponding parameters are picked
        from current LDA.

        :param phi: numpy array of shape [K, V], phi.
        :return: None.
        """
        phi = phi if phi is not None else self.phi
        for i in range(self.n_components):
            print("Topic", i)
            c = np.argsort(self.phi[i, :])
            for j in c[-1:-11:-1]:
                print(self.list_ind2word[j], phi[i, j])
