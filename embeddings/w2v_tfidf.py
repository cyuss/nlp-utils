# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import gensim


# create an iterator over sentences stored in pandas data frame
class MySentences:
    def __init__(self, texts):
        self.texts = texts

    def __iter__(self):
        for line in self.texts:
            yield line.split()

# word2vec class
class Word2vecEmbedding:
    def __init__(self, size=100, window=5, min_count=5, max_vocab_size=None, sample=.001, workers=3, negative=5, alpha=.025):
        self.word2weight = None
        # model parameters
        self.size = size # vectors dimension
        self.window = window # the maximum distance between the current and the predicted word within a sentence
        self.min_count = min_count # ignore all words with total frequency lower than this
        self.max_vocab_size = max_vocab_size # set the vocabulary size
        self.sample = sample # the threshold for configuring which higher-frequency words are randomly downsampled
        self.workers = workers
        self.negative = negative # specifies how many “noise words” should be drawn
        self.alpha = alpha # the initial learning rate

    def fit(self, X, y=None):
        self.sentences = X
        # train word2vec embedding
        self.word2vec = gensim.models.Word2Vec(MySentences(self.sentences), min_count=self.min_count, size=self.size,
                                              window=self.window, max_vocab_size=self.max_vocab_size,
                                              sample=self.sample, negative=self.negative,
                                              alpha=self.alpha, workers=self.workers)
        self.model = dict(zip(self.word2vec.wv.index2word, self.word2vec.wv.vectors))
        # self.dim = len(next(iter(self.model.values())))
        # train tf-idf vectorizer
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # store the coefficients
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()]
        )
        return self

    def transform(self, X, y=None):
        return np.array([
            np.mean([self.model[w] * self.word2weight[w]
                     for w in words if w in self.model] or
                    [np.zeros(self.size)], axis=0)
            for words in MySentences(X)
        ])
    
    def fit_transform(self, X, y=None):
        temp = self.fit(X)
        return temp.transform(X)
    
    def __str__(self):
        return str(self.word2vec)
