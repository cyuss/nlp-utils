# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import gensim


class MySentences:
    """It represents the dataset documents for better iteration.

    Attributes:
        texts: A sequence of documents.
    """
    def __init__(self, texts):
        self.texts = texts

    def __iter__(self):
        for line in self.texts:
            yield line.split()

# word2vec class
class Word2vecTfIdfEmbedding:
    """Word2Vec embedding weighted by Tf-Idf scores to get a vector 
    representation of a document.

    Attributes:
        word2weight: A dict containing the coefficients/weights of 
            word2vec from tf-idf.
        args: A tuple representing `gensim.models.Word2Vec` parameters.
        kwargs: Same as args but in dict format.
        size: Size of word2vec vectors.
        sentences: A sequence of sentences on which we train the word 
            embedding model.
        word2vec: A `gensim.models.Word2Vec` object.
        model: A dict representing the words of the vocabulary and their 
            vector representation.

    .. note:: 
        In word2vec model there's no data labels.
        The argument `y` is added just for compatibility
        with Scikit-Learn definition pattern in
        :func:`fit`, :func:`transform` and :func:`fit_transform` functions.
    """
    def __init__(self, *args, **kwargs):
        """Inits Word2vecTfIdfEmbedding with the `gensim.models.Word2Vec` parameters."""
        self.word2weight = None
        # model parameters
        self.args = args
        self.kwargs = kwargs
        self.size = size[0] if args else kwargs["size"] # vectors dimension

    def fit(self, X, y=None):
        """Train the word2vec model for document vector representation.

        Args:
            X: A sequence of documents on which we train our model.
            y: A sequence of data labels.

        Returns:
            The Word2vecTfIdfEmbedding object.
        """
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
        """Transform a sequence of documents to vectors.

        Args:
            X: A sequence of documents which we need to transform.
            y: A sequence of data labels.

        Returns:
            A double numpy array of size :attr:`self.size` representing the words.
        """
        return np.array([
            np.mean([self.model[w] * self.word2weight[w]
                     for w in words if w in self.model] or
                    [np.random.rand(self.size)], axis=0)
            for words in MySentences(X)
        ])
    
    def fit_transform(self, X, y=None):
        """Combine `fit` and `transform` functions.

        It trains the model on the given data then transforming
        and returning the training data into vectors.
        
        Args:
            X: A sequence of documents which we need to transform.
            y: A sequence of data labels.

        Returns:
            A double numpy array of size :attr:`self.size` representing the words.
        """
        temp = self.fit(X)
        return temp.transform(X)
    
    def __str__(self):
        return str(self.word2vec)
