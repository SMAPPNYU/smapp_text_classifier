'''Generate pre-trained word embeddings for Twitter data'''
import os
from data import SpacyTokenizer
from gensim.models import FastText

class TrainDocs:
    '''Iterator over training documents for Fast Text Embeddings'''
    def __init__(self, infile, tokenizer=None):
        self.infile = infile
        if tokenizer is None:
            self.tokenizer = SpacyTokenizer()
    def __iter__(self):
        with open(self.infile) as infile:
            for line in infile:
                line = line.strip('\n')
                yield self.tokenizer.tokenize(line)

if __name__ == '__main__':

    INPUT = '../data/twitter_language_data/all_collections_sample.txt'
    OUTFILE = os.path.join('../data/embedding_models/',
                           os.path.split(INPUT)[-1].strip('.txt'))

    docs = TrainDocs(INPUT)

    # Train fasttext on the complete corpus
    model = FastText(docs, size=100, window=3, min_count=5, iter=20,
                     workers=3)

    model.save(OUTFILE)
