import spacy

class SpacyTokenizer:
    '''Tokenizer that relies on the SpaCy NLP package.

    Arguments:
    ----------
    language: language of the text to be tokenized

    Methods:
    ----------
    tokenize: 
    '''
    def __init__(self, language='en'):
        self.nlp = spacy.load(language, disable=['ner', 'parser', 'tagger'])

    def tokenize(self, doc, rescue_hashtags=False):

        token_list = [x.orth_ for x in self.nlp(doc)]
        if rescue_hashtags:
            tokens = iter(token_list)
            return([t + next(tokens, '') if t == '#' else t for t in tokens])
        else:
            return token_list
