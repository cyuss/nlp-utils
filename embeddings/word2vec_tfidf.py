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
class Word2vecTfIdfEmbedding:
    def __init__(self, *args, **kwargs):
        self.word2weight = None
        # model parameters
        self.args = args
        self.kwargs = kwargs
        self.size = size[0] if args else kwargs["size"] # vectors dimension

    def fit(self, X, y=None):
        self.sentences = X
        # train word2vec embedding
        self.word2vec = gensim.models.Word2Vec(MySentences(self.sentences), *self.args, **self.kwargs)
        self.model = dict(zip(self.word2vec.wv.index2word, self.word2vec.wv.vectors))
        # train tf-idf vectorizer
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # store the coefficients
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, # default value for the defaultdict
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()]
        )
        return self

    def transform(self, X, y=None):
        return np.array([
            np.mean([self.model[w] * self.word2weight[w]
                     for w in words if w in self.model] or
                    [np.random.rand(self.size)], axis=0)
            for words in MySentences(X)
        ])
    
    def fit_transform(self, X, y=None):
        temp = self.fit(X)
        return temp.transform(X)
    
    def __str__(self):
        return str(self.word2vec)
