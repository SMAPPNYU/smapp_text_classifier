'''
Author: Fridolin Linder
'''
import os
import joblib
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin, BaseEstimator
from gensim.models import FastText

def _check_X(X):
    '''Check if document input has an index'''
    if not hasattr(X, 'index'):
        raise ValueError('X needs index if vectorized from cache')

class CacheError(ValueError):
    '''Basic exception for errors related to the vectorizer cache'''
    def __init__(self, msg=None):
        msg = "Cache not found" if msg is None else msg
        super().__init__(msg)

class CachedVectorizer:
    '''Base class to build cached vectorizer.

    Attributes:
        recompute: bool, if True the cache is ignored, the vectorizer is fitted
            and the feature matrix recomputed
        cache: str, path to the file to cache the vectorizer + feature matrix
    '''
    def __init__(self, cache_dir=None, recompute=None):
        self.cache_dir = cache_dir
        self.recompute = recompute
        self.feature_matrix = None
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)

    def _load_from_cache(self, cache):
        if self.recompute:
            raise CacheError()
        elif os.path.exists(cache):
            cached_self = joblib.load(cache)
            self.__dict__.update(cached_self.__dict__)
        else:
            raise CacheError()

class CachedCountVectorizer(CountVectorizer, CachedVectorizer):
    '''Modification of the sklearn CountVectorizer to allow caching

    The vectorizer dumps itself + the document - term matrix for the set of
    documents it was fitted to to a cache file. When re-used, subsets of the
    initial document can be vectorized by pulling the corresponding rows from
    the full document-term matrix. This avoids re-tokenization and vectorization

    Attributes:
        cache_dir: str, directory to cache the vectorizer to
        ds_name: str, name of the dataset the vectorizer is fitted to
        recompute: bool, ignore the cache if True
        ngram_range: tuple, see parent documentation
        analyzer: str, see parent documentation
        **kwargs: additional arguments passed to CountVectorizer, see parent
            documentation
    '''
    def __init__(self, cache_dir='./', ds_name='test', ngram_range=None,
                 analyzer=None, recompute=False, **kwargs):
        super().__init__(ngram_range=ngram_range, analyzer=analyzer, **kwargs)
        self.cache_dir = cache_dir
        self.ds_name = ds_name
        self.recompute = recompute
        super(CountVectorizer, self).__init__(cache_dir, self.recompute)

    @property
    def cache(self):
        return os.path.join(
            self.cache_dir,
            f'{self.ds_name}_{self.analyzer}_{self.ngram_range}.joblib'
        )

    def transform(self, raw_documents):
        try:
            self._load_from_cache(self.cache)
            _check_X(raw_documents)
            return self.feature_matrix[raw_documents.index, ]
        except CacheError:
            return super().transform(raw_documents)

    def fit_transform(self, raw_documents, y=None):
        try:
            self._load_from_cache(self.cache)
            _check_X(raw_documents)
            return self.feature_matrix[raw_documents.index, ]
        except CacheError:
            self.feature_matrix = super().fit_transform(raw_documents)
            joblib.dump(self, self.cache)
            return self.feature_matrix


class CachedEmbeddingVectorizer(TransformerMixin, BaseEstimator,
                                CachedVectorizer):
    '''Vectorizer to transform a set of text documents into a matrix of document
    embeddings.

    The vectorizer dumps itself + the document - term matrix for the set of
    documents it was fitted to to a cache file. When re-used, subsets of the
    initial document can be vectorized by pulling the corresponding rows from
    the full document-term matrix. This avoids re-tokenization and vectorization

    Attributes:
        embedding_model: tuple(name, path), identifier and path to the embedding
            model. Currently only Fasttext embeddings in Gensim or Facebook
            binary Fasttext format are supported.
        cache_dir: str, directory to cache the vectorizer to.
        ds_name: str, name of the dataset the vectorizer is fitted to.
        pooling_method: str, one of [`mean`, `max`]. Method to combine word
            vectors to a document vector.
        recompute: bool, ignore the cache if True
    '''

    def __init__(self, embedding_model, cache_dir=None,
                 ds_name=None, pooling_method=None,
                 tokenize=None, recompute=False):
        self.embedding_model_name = embedding_model[0]
        self.embedding_model_path = embedding_model[1]
        self.pooling_method = pooling_method
        self.cache_dir = cache_dir
        self.ds_name = ds_name
        self.dimensionality = None
        self.recompute = recompute
        super(BaseEstimator, self).__init__(self.cache_dir, self.recompute)
        if tokenize is None:
            self.tokenize = lambda x: x.split()
        else:
            self.tokenize = tokenize
        self.recompute = recompute

    @property
    def cache(self):
        return (f'{self.ds_name}_{self.embedding_model_name}_'
                f'{self.pooling_method}.pkl')

    def fit(self, X):
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        try:
            self._load_from_cache(self.cache)
            _check_X(X)
            return self.feature_matrix[X.index, ]
        except CacheError:
            em = self._load_embedding_model()
            self.feature_matrix = np.array([self._embed_doc(doc, em) 
                                            for doc in X])
            joblib.dump(self, self.cache)
            return self.feature_matrix

    def _load_embedding_model(self):
        if self.embedding_model_path.endswith('.model'):
            em = FastText.load(self.embedding_model_path)
        else:
            em = FastText.load_fasttext_format(self.embedding_model_path)
        self.dimensionality = em.wv[em.wv.index2word[0]].shape[0]
        return em

    def _get_vector(self, word, model):
        '''Method to retrieve word vector from gensim model that is save
        for words that are not in the vocabulary (for fasttext: words of
        which no character n-gram is in the vocabulary; should be rare)
        '''
        try:
            return model[word]
        except KeyError:
            return np.zeros(self.dimensionality)

    def _embed_doc(self, doc, model):
        '''Calculate the document embedding by pooling the embedding of each
        token in the document
        '''
        tokens = self.tokenize(doc)
        vectors = np.array([self._get_vector(t, model.wv) for t in tokens])
        if self.pooling_method == 'mean':
            doc_vector = vectors.mean(axis=0)
        elif self.pooling_method == 'max':
            doc_vector = vectors.max(axis=0)
        else:
            raise ValueError(f'Unsupported pooling method: '
                             f'{self.pooling_method}')
        return doc_vector
