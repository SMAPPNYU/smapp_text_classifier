import os
import shutil
import pandas as pd
from smapp_text_classifier.vectorizers import (CachedCountVectorizer,
                                               CachedEmbeddingVectorizer)

# Directory for caching during tests
# Warning: if this directory already exists it will be deleted by the test
cache_dir = 'test_cache_dir'

# Test 'documents'
DOCS1 = pd.Series(['hello world', 'hello smapp', 'smapp clf'])
DOCS2 = pd.Series(['first new', 'second smapp', 'new world'])

def vectorizer_test(vec):

    # If for some reason directory exists delete contents
    if os.path.exists(cache_dir):
        for f in os.listdir(cache_dir):
            os.unlink(f)

    # First transform from scratch
    X_first = vec.fit_transform(DOCS1)
    # Second transform from cache
    X_second = vec.fit_transform(DOCS1)

    # Assert that X_first and X_second are the same
    assert (X_first != X_second).sum() == 0

    # Transform of subset
    subset = DOCS1.iloc[:2]
    X_subset = vec.transform(subset)
    assert (X_first[:2, :] != X_subset).sum() == 0

    # Transform of new data
    X_new = vec.transform(DOCS2)
    assert X_new.shape == X_first.shape

    # Clean up
    shutil.rmtree(cache_dir)

def test_count_vectorizer():

    vec = CachedCountVectorizer(
        cache_dir=cache_dir,
        ds_name='test',
        ngram_range=(1, 1),
        analyzer='word'
    )
    vectorizer_test(vec)

#def test_embedding_vectorizer():
#
#    vec = CachedEmbeddingVectorizer(
#        embedding_model_name='glove-wiki-gigaword-50',
#        cache_dir=cache_dir,
#        ds_name='test',
#        pooling_method='mean',
#        recompute=False,
#    )
#    vectorizer_test(vec)
