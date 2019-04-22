import json
import shutil
import copy

import numpy as np
import pandas as pd
import spacy

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

class DataSet:
    '''Data Set for the Sentiment Analysis Experiment + Pipeline

    tokenizer: must contain a `tokenize(string)` method
    '''

    def __init__(self, name, field_mapping,
                 file_subset=None, test_size=0.25,
                 tokenizer=None, na_rm=True, train_input=None,
                 test_input=None, input_=None):

        # Store inputs
        self.name = name
        self.field_mapping = field_mapping
        self.test_size = test_size
        self.file_subset = file_subset
        self.tokenizer = tokenizer
        self.input_ = input_
        self.train_input, self.test_input = train_input, test_input

        if input_ is not None:
            self.df = self.read_transform(input_)
            if na_rm:
                self.df.dropna(inplace=True)
            self.df.reset_index(inplace=True, drop=True)
            self.train_idxs, self.test_idxs = train_test_split(
                    range(len(self)), test_size=test_size)

        elif train_input is not None:
            if input_ is not None:
                raise ValueError('Pleas only specify either `input_` or '
                                 '`train_input`')
            if test_input is None:
                raise ValueError('Please pass data as `input_` if not '
                                 'specifying both `train_input` and '
                                 '`test_input`')
            train_df = self.read_transform(train_input)
            test_df = self.read_transform(test_input)
            self.df = train_df.append(test_df)
            self.df.reset_index(inplace=True, drop=True)
            if na_rm:
                self.df.dropna(inplace=True)
            self.train_idxs = list(range(train_df.shape[0]))
            self.test_idxs = list(range(train_df.shape[0],
                                         train_df.shape[0] + test_df.shape[0]))
        else:
            raise ValueError('Either `input_` or (`train_input`, `test_input`)'
                             ' have to be specified.')
    def __len__(self):
        return self.df.shape[0]

    @property
    def df_train(self):
        return self.df.iloc[self.train_idxs]

    @property
    def df_test(self):
        return self.df.iloc[self.test_idxs]

    def n_tokens(self):
        '''
        Output the lower bound of number of features
        When randomize searching for the optimal parameter, max_df >= 0.8,
        min_df <= 5
        '''
        vectorizer = CountVectorizer(ngram_range=(1, 1), max_df=1, min_df=1,
                                     tokenizer=self.tokenizer)
        grams = vectorizer.fit_transform(self.df['text'])
        return len(vectorizer.get_feature_names())

    def get_texts(self, set_):
        if set_ == "train":
            return self.df_train['text' ]
        elif set_ == 'test':
            return self.df_test['text']

    def get_labels(self, set_):
        if set_ == 'train':
            return self.df['label'].iloc[self.train_idxs].astype(str)
        elif set_ == 'test':
            return self.df['label'].iloc[self.test_idxs].astype(str)
        else:
            raise ValueError("set_ must be one of ['train', 'test']")

    def read_transform(self, input_):
        '''Read input data and transform to common format

        Selects the right method depending on the data type or
        file type and dispatches apropriate method
        '''
        if isinstance(input_, str):
            inpath = input_
            if inpath.endswith('.csv'):
                df = self.read_from_delim(inpath, delim=',')
            elif inpath.endswith('.tsv'):
                df = self.read_from_delim(inpath, delim='\t')
            elif inpath.endswith('.json'):
                df = self.read_from_json(inpath)
            else:
                raise ValueError('Unsupported file format')
        elif isinstance(input_, pd.core.frame.DataFrame):
            df = self.read_from_df(input_)
        else:
            raise ValueError('input_ has to be str or'
                             'pd.core.frame.DataFrame')
        return df

    def read_from_df(self, input_df):
        out = copy.copy(input_df[[self.field_mapping['label'],
                                  self.field_mapping['text']]])
        out.columns = ['label', 'text']
        out.reset_index(inplace=True, drop=True)
        return out

    def read_from_delim(self, inpath, delim):
        df = pd.read_csv(inpath, delimiter=delim)
        if self.file_subset is not None:
            df = df[df[self.file_subset[0]] == self.file_subset[1]]
        out = df[[self.field_mapping['label'],
                  self.field_mapping['text']]]
        out.columns = ['label', 'text']
        return(out)

    # TODO: Not tested
    def read_from_json(self, inpath):
        with open(inpath) as infile:
            out = {'label': [], 'text': []}
            for line in infile:
                doc = json.loads(line)
                if self.file_subset is not None:
                    if doc[self.file_subset[0]] == self.file_subset[1]:
                        out['label'].append(doc[self.field_mapping['label']])
                        out['text'].append(doc[self.field_mapping['text']])
                else:
                        out['label'].append(doc[self.field_mapping['label']])
                        out['text'].append(doc[self.field_mapping['text']])

        return pd.DataFrame(out)

class SpacyTokenizer:
    def __init__(self):
        self.nlp = spacy.load('en', disable=['ner', 'parser', 'tagger'])

    @staticmethod
    def rescue_hashtags(token_list):
        tokens = iter(token_list)
        return([t + next(tokens, '') if t == '#' else t for t in tokens])

    def tokenize(self, text):
        return self.rescue_hashtags([x.orth_ for x in self.nlp(text)])
