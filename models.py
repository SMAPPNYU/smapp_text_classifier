'''
Abstraction of machine learning model

Authors: Michael Fengyuan Liu, Fridolin Linder
'''

import pickle
import os
import multiprocessing
import scipy as sp
import numpy as np

from gensim.models import FastText
from scipy import sparse
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import chi2
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class Chi2Reducer(TransformerMixin, BaseEstimator):
    '''Estimator that reduces the number of features by
    selecting the top predictive features according to
    chi^2 statistic (see sklearn.feature_selection.chi2)

    max_n_features: Maximum numbers of features to return.
    '''
    def __init__(self, max_n_features):
        self.max_n_features = int(max_n_features)
        self.top_idxs = None

    def fit(self, X, y):
        '''Calculates chi2 statistics for all features in X'''
        c2 = chi2(X, y)

        sorted_features = sorted([(x, i) for i, x in enumerate(c2[0])],
                                 key=lambda x: x[0],
                                 reverse=True)
        top_features = sorted_features[:self.max_n_features]
        self.top_idxs = list(map(lambda x: x[1], top_features))
        return self

    def transform(self, X, y=None):
        '''Returns the top self.max_n_features from X'''
        return X.tocsr()[:, self.top_idxs]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

class DictionaryModel:
    '''
    A high level wrapper around vaderSentiment packageself.

    tokenizer: function that tokenizes string
    model: specifies which dictionary model is applied. Only 'vader' is
        supported in this version.
    labels: list of three label values for positive, negative, neutral
        (in that order)

    Method:
    ---
    predict: return a list of predicted sentiment for each tweet in X.
    '''

    def __init__(self, tokenizer, model='vader', n_jobs=-1,
                 labels=['positive', 'negative', 'neutral']):
        self.tokenizer = tokenizer
        self.labels = labels
        if model == 'vader':
            self.analyzer = SentimentIntensityAnalyzer()
        else:
            raise ValueError('Invalid dictionary method model.')
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()

    def score_document(self, document):
        '''Score a single document'''
        vs = self.analyzer.polarity_scores(document.lower())
        if vs['compound'] > 0:
            return self.labels[0]
        elif vs['compound'] < 0:
            return self.labels[1]
        else:
            return self.labels[2]

    def predict(self, X):
        '''Score a collection of documents in parallel'''
        vader_sentiments = []
        with multiprocessing.Pool(self.n_jobs) as pool:
            vader_sentiments = pool.map(self.score_document, X)

        return vader_sentiments

class PrecomputeVectorizer(CountVectorizer):
    '''
    load precomputed document-term matrix or embedding
    in pipeline instead of calling CountVectorizer in
    each process
    '''

    def __init__(self, dataset, feature_set, cache_dir=None, ngram_range=None,
                 pooling_method=None):
        self.dataset = dataset
        if feature_set == 'char_ngrams':
            self.analyzer = 'char_wb'
        elif feature_set == 'word_ngrams':
            self.analyzer = 'word'
        self.feature_set = feature_set
        self.ngram_range = ngram_range
        self.pooling_method = pooling_method
        self.cache_dir = cache_dir

    def fit_transform(self, rawdocuments, y=None):
        if self.feature_set in ['char_ngrams', 'word_ngrams']:
            # load and concatenate document-term matrices
            mat_path = os.path.join(self.cache_dir, (f'{self.dataset.name}_'
                                                     f'{self.analyzer}_'
                                                     f'{self.ngram_range[0]}'
                                                     f'.npz'))
            X = sparse.load_npz(mat_path)
            X = X[rawdocuments.index, ]
            for i in range(self.ngram_range[0]+1, self.ngram_range[1]+1):
                mat_path = os.path.join(self.cache_dir, (f'{self.dataset.name}_'
                                                         f'{self.analyzer}_'
                                                         f'{i}.npz'))
                X1 = sparse.load_npz(mat_path)
                X1 = X1[rawdocuments.index, ]

                X = sparse.hstack([X, X1])
        elif self.feature_set == 'embeddings':
            mat_path = os.path.join(self.cache_dir, (f'{self.dataset.name}_'
                                                     f'fasttext_'
                                                     f'{self.pooling_method}_'
                                                     f'embedded.p'))
            X = pickle.load(open(mat_path, 'rb'))
            X = X[rawdocuments.index, ]

        return X

    def transform(self, rawdocuments):
        return self.fit_transform(rawdocuments)

class TextClfPipeline:
    '''
    '''
    def __init__(self, dataset, algorithm, feature_set,
                 max_n_features, embedding_model=None,
                 cache_dir='feature_cache/', recompute_features=False):
        self.algorithm = algorithm
        self.dataset = dataset
        self.max_n_features = max_n_features
        self.cache_dir=cache_dir
        self.embedding_model=embedding_model
        self.feature_set=feature_set
        self.recompute_features=recompute_features

        if feature_set == 'char_ngrams':
            self.params = {
                'vectorize__ngram_range': [(3, 3), (3, 4), (3, 5)],
            }
        elif feature_set == 'word_ngrams':
            self.params = {
                'vectorize__ngram_range': [(1, 1), (1, 2), (1, 3)],
            }
        elif feature_set == 'embeddings':
            self.params = {
                'vectorize__pooling_method': ['mean', 'max']
            }

        # classifiers
        if self.algorithm == 'random_forest':
            self.classifier = RandomForestClassifier()
            self.params.update({'clf__n_estimators': sp.stats.randint(10, 100),
                                'clf__max_features': sp.stats.randint(1, 11)})
        elif self.algorithm == 'lasso':
            self.classifier = SGDClassifier(loss='log', penalty='elasticnet',
                                            max_iter=1000, tol=1e-3)
            self.params.update({'clf__alpha': sp.stats.uniform(0.0001, 0.01),
                                'clf__l1_ratio': sp.stats.uniform(0, 1)})
        elif self.algorithm == 'naive_bayes':
            self.classifier = GaussianNB()
            self.params.update({'clf__var_smoothing': sp.stats.uniform(1e-12,
                                                                       1e-6)})
        elif self.algorithm == "svm":
            self.classifier = SVC()
            self.params.update({'clf__gamma': sp.stats.uniform(1e-3, 1e3),
                                'clf__C': sp.stats.uniform(1e-2, 1e2),
                                'clf__kernel': ['linear', 'rbf'],
                                'clf__tol': sp.stats.uniform(1e-1, 1e-5)})
        else:
            raise ValueError("classifiers must be one of ['naive_bayes', 'lass"
                             "o', 'random_forest', 'svm']")

        # Assemble Pipeline
        if feature_set == 'embeddings':
            self.pipeline = Pipeline([
                ('vectorize', PrecomputeVectorizer(dataset=self.dataset, 
                                                   feature_set=feature_set,
                                                   cache_dir=self.cache_dir)),
                #('scale', StandardScaler(copy=True, with_mean=True, 
                #                         with_std=True)),
                ('clf', self.classifier)])
        else:
            self.pipeline = Pipeline([
                ('vectorize', PrecomputeVectorizer(dataset=self.dataset, 
                                                   feature_set=feature_set,
                                                   cache_dir=self.cache_dir)),
                ('reduce', Chi2Reducer(max_n_features=self.max_n_features)),
                #('scale', StandardScaler(copy=True, with_mean=True, 
                #                         with_std=True)),
                ('clf', self.classifier)])

        # Precompute features
        self.precompute_features()

    def precompute_features(self):
        '''
        Precompute and store feature matrices to cache

        Compute the following feature matrices depending on the pipeline feature
        arugment:
        - Word grams (1 to 6 separate file each)
        - Character grams (1 to 6)
        - Embeddings (two matrices with `max` and `mean` pooled embeddings)

        Returns:
        -----------
        Stores one file per feature matrix in `cache_dir`
        '''
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        X = self.dataset.df['text']

        # Precompute bag-of-words/chars features.
        if self.feature_set == 'word_ngrams':
            for ngram in range(1, 4):
                ana = 'word'
                fname = os.path.join(self.cache_dir, 
                                     f'{self.dataset.name}_{ana}_{ngram}.npz')
                if not os.path.isfile(fname) or self.recompute_features:
                    print(f'Precomputing {self.feature_set}...')
                    cv = CountVectorizer(tokenizer=self.dataset.tokenizer,
                                         analyzer=ana, 
                                         ngram_range=(ngram, ngram))
                    transX = cv.fit_transform(X)
                    sparse.save_npz(fname, transX)
        elif self.feature_set == 'char_ngrams':
            for ngram in range(3, 6):
                ana = 'char_wb'
                fname = os.path.join(self.cache_dir, 
                                     f'{self.dataset.name}_{ana}_{ngram}.npz')
                if not os.path.isfile(fname) or self.recompute_features:
                    print(f'Precomputing {self.feature_set}...')
                    cv = CountVectorizer(tokenizer=self.dataset.tokenizer,
                                         analyzer=ana, 
                                         ngram_range=(ngram, ngram))
                    transX = cv.fit_transform(X)
                    sparse.save_npz(fname, transX)

        elif self.feature_set == 'embeddings':
            # TODO: this should not be hardcoded
            em_name = 'fasttext'
            em = None
            # Precomput the embedded documents
            for pooling in ['mean', 'max']:
                fname = os.path.join(
                        self.cache_dir, 
                        (f'{self.dataset.name}_{em_name}_{pooling}_'
                         'embedded.p')
                        )
                if not os.path.isfile(fname) or self.recompute_features:
                    if em is None: 
                        print(f'Loading {em_name} embedding model...')
                        em = FastText.load(self.embedding_model)
                    print(f'Precomputing {pooling}-pooled embeddings...')
                    X_out = []
                    for i, doc in enumerate(X):
                        tokens = self.dataset.tokenizer(doc)
                        vecs = []
                        for t in tokens:
                            try:
                                vecs.append(em.wv[t])
                            # no char ngram of the work can be found
                            except KeyError:
                                continue
                        if pooling == 'mean':
                            X_out.append(np.mean(vecs, axis=0))
                        else:
                            X_out.append(np.max(vecs, axis=0))

                    X_out = np.array(X_out)
                    pickle.dump(X_out, open(fname, 'wb'))
