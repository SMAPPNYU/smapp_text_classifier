import os
import shutil
import pandas as pd
from smapp_text_classifier.vectorizers import (CachedCountVectorizer,
                                               CachedEmbeddingVectorizer)


def test_count_vectorizer():

    # Warning: if this directory already exists it will be deleted by the test
    cache_dir = 'test_cache_dir'

    # If for some reason directory exists delete it
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    vec = CachedCountVectorizer(cache_dir=cache_dir,
                                ds_name='test',
                                ngram_range=(1, 1),
                                analyzer='word',
                                recompute=False)
    docs = pd.Series(['hello world', 'hello smapp', 'smapp clf'])
    # First transform from scratch
    X_first = vec.fit_transform(docs)
    # Second transform from cache
    X_second = vec.fit_transform(docs)

    # Assert that X_first and X_second are the same
    assert (X_first != X_second).sum() == 0

    # Transform of subset
    docs = docs.iloc[:2]
    X_subset = vec.transform(docs)
    assert (X_first[:2, :] != X_subset).sum() == 0

    # Transform of new data
    docs = pd.Series(['first new', 'second smapp', 'new world'])
    X_new = vec.transform(docs)
    assert X_new.shape == X_first.shape

    # Clean up
    shutil.rmtree(cache_dir)
