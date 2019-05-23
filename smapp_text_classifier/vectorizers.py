'''
Author: Fridolin Linder
'''
import os
import logging
import numpy as np
import gensim
import joblib

from gensim.models import FastText
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin, BaseEstimator
from smapp_text_classifier.utilities import timeit

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
    def __init__(self, cache_dir='./', recompute=False):
        self.cache_dir = cache_dir
        self.recompute = recompute
        self.feature_matrix = None
        self.index_mapping = None
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)

    @timeit
    def _load_from_cache(self, cache):
        if self.recompute:
            logging.debug('Not loading due to recompute request')
            raise CacheError()
        elif os.path.exists(cache):
            cached_self = joblib.load(cache)
            del cached_self.recompute
            self.__dict__.update(cached_self.__dict__)
        else:
            logging.debug('Cache not found')
            raise CacheError()

    def get_docs(self, raw_idxs):
        mapped = [self.index_mapping[idx] for idx in raw_idxs]
        return self.feature_matrix[mapped, ]


class CachedCountVectorizer(CountVectorizer, CachedVectorizer):
    '''Modification of the sklearn CountVectorizer to allow caching

    The vectorizer dumps itself + the document - term matrix for the set of
    documents it was fitted to to a cache file. When re-used, subsets of the
    initial document can be vectorized by pulling the corresponding rows from
    the full document-term matrix. This avoids re-tokenization and vectorization

    There are multiple peculiar design choices (e.g. cache as a property) that
    stem from the fact that this class is meant to be used in a sklearn pipeline
    that is cross validated using the sklearn cross-validation tools. I.e.
    during cross validation, the base estimator is cloned (that is, the
    estimator with default paramter settings) and the parameters that are tuned
    are set dynamically after initialization.

    Attributes:
        cache_dir: str, directory to cache the vectorizer to
        ds_name: str, name of the dataset the vectorizer is fitted to
        recompute: bool, ignore the cache if True
        ngram_range: tuple, see parent documentation
        analyzer: str, see parent documentation
        **kwargs: additional arguments passed to CountVectorizer, see parent
            documentation
    '''
    def __init__(self, cache_dir='./', ds_name='test', ngram_range=(1, 1),
                 analyzer='word', recompute=False, **kwargs):
        super().__init__(ngram_range=ngram_range, analyzer=analyzer, **kwargs)
        self.ds_name = ds_name
        super(CountVectorizer, self).__init__(cache_dir, recompute)


    @property
    def cache(self):
        return os.path.join(
            self.cache_dir,
            f'{self.ds_name}_{self.analyzer}_{self.ngram_range}.joblib'
        )

    @timeit
    def transform(self, raw_documents):
        try:
            logging.debug('Transforming from cache')
            self._load_from_cache(self.cache)
            _check_X(raw_documents)
            return self.get_docs(raw_documents.index)
        except CacheError:
            logging.debug('Transforming from scratch')
            return super().transform(raw_documents)

    @timeit
    def fit_transform(self, raw_documents, y=None):
        try:
            logging.debug('Transforming from cache')
            self._load_from_cache(self.cache)
            _check_X(raw_documents)
            return self.get_docs(raw_documents.index)

        except CacheError:
            logging.debug('Transforming from scratch')
            self.feature_matrix = super().fit_transform(raw_documents)
            # keep track of what index location of input maps to which row in
            # the feature matrix
            self.index_mapping = {
                    idx: i for i, idx in enumerate(raw_documents.index)
            }
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

    There are multiple peculiar design choices (e.g. cache as a property) that
    stem from the fact that this class is meant to be used in a sklearn pipeline
    that is cross validated using the sklearn cross-validation tools. I.e.
    during cross validation, the base estimator is cloned (that is, the
    estimator with default paramter settings) and the parameters that are tuned
    are set dynamically after initialization.

    Attributes:
        embedding_model: tuple(name, path), identifier and path to the embedding
            model. Currently only Fasttext embeddings in Gensim or Facebook
            binary Fasttext format are supported.
        cache_dir: str, directory to cache the vectorizer to.
        ds_name: str, name of the dataset the vectorizer is fitted to.
        pooling_method: str, one of [`mean`, `max`]. Method to combine word
            vectors to a document vector.
        recompute: bool, ignore the cache if True
        tokenizer: function that tokenizes a string to a list of tokens
    '''
    def __init__(self, embedding_model, cache_dir=None,
                 ds_name=None, pooling_method=None,
                 tokenizer=None, recompute=False):
        self.embedding_model = embedding_model
        self.pooling_method = pooling_method
        self.ds_name = ds_name
        self.dimensionality = None
        super(BaseEstimator, self).__init__(cache_dir, recompute)
        self.tokenize = tokenizer

    @property
    def cache(self):
        return os.path.join(
            self.cache_dir,
            (f'{self.ds_name}_{self.embedding_model[0]}_'
             f'{self.pooling_method}.pkl')
        )

    def fit(self, X):
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    @timeit
    def transform(self, X, y=None):
        try:
            logging.debug('Transforming from cache')
            self._load_from_cache(self.cache)
            _check_X(X)
            return self.get_docs(X.index)
        except CacheError:
            logging.debug('Transforming from cache')
            em = self._load_embedding_model()
            self.feature_matrix = np.array([self._embed_doc(doc, em)
                                            for doc in X])
            # keep track of what index location of input maps to which row in
            # the feature matrix
            self.index_mapping = {
                    idx: i for i, idx in enumerate(X.index)
            }
            joblib.dump(self, self.cache)
            return self.feature_matrix

    @timeit
    def _load_embedding_model(self):
        # Quick hack to allow passing of pre-loaded models
        # TODO: change naming of attribute and update documentation
        #       this might be a useful feature
        logging.debug('Loading embedding model')
        if isinstance(self.embedding_model[1],
                      gensim.models.fasttext.FastText):
            em = self.embedding_model[1]
        else:
            if self.embedding_model[1].endswith('.model'):
                em = FastText.load(self.embedding_model[1])
            else:
                em = FastText.load_fasttext_format(self.embedding_model[1])
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
