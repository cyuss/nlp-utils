# Word Embeddings
While working on *Natural Language Processing*, we usually need a **vector representation** of each word or document, called *word embeddings*. 

Word embeddings are capable of capturing the context of a word in a document, relation with other words, define a semantic and syntactic similarity... etc.

The different classes defined here are different variants of word embeddings that can be used to get a vector representation of a word.

## Word2Vec weighted by TF-IDF
*Word2Vec* is a model created by [Mikolov et al.](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) In Python, we have a good implementation developed by [gensim](https://radimrehurek.com/gensim/).

*TF-IDF* stands for **term frequency-inverse document frequency**. It is a statistical measure used to evaluate how important a word is to a document in a collection of a corpus.

The class `Word2vecEmbedding` define a vector representation of a **document** by weighting the vector representation of each word in that document by its Tf-Idf.

This class follows the same pattern as `Scikit-Learn` models by defining a `fit` and `transform` functions that train the model on the given data.

```python
from embeddings.word2vec_tfidf import Word2vecEmbedding

# data sample
X = ["bonjour tout le monde", 
     "comment allez vous", 
     "je code en python", 
     "bonjour il fait beau",
     "comment vous faites"]
# initialize the model
model = Word2vecEmbedding(size=20, min_count=1, window=1)
# train the model
model.fit(X)
# get the vector representation of a document
print(model.tranform(["Bonjour comment allez vous"]))
```



