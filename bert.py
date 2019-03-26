from bert_serving.client import BertClient
from sklearn.base import TransformerMixin, BaseEstimator

class BertEncoder(TransformerMixin, BaseEstimator):
    '''Custom transformer to encode documents using a pretrained BERT model in
    the sklearn pipeline.

    Note: This requires a running bert server:
    https://github.com/hanxiao/bert-as-service

    Arguments:
    ----------
    See bert-as-service documentation for documentation of all arguments
    Methods:
    ----------
    fit,
    transform
    '''
    def __init__(self):
        self.client = BertClient()

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        return np.array(self.client.encode([x for x in X]))

