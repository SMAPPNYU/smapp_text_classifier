# SMaPP Text Classification

This package contains some helpful abstractions to allow for easy training of state of the art supervised text classification models. It removes some repetitive and time intensive tasks from like setting up cross validation over various ML classifiers, transformation of text into various feature spaces (word n-grams, character n-grams, and word embeddings), and setting of reasonable defaults for tuning parameters. 

## Installation

```
pip install git+https://github.com/smappnyu/smapp_text_classifier.git
```

## Example Use

This is a bare bones example. For a more extensive executable notebook see `/pipeline_demo`.

```python
from smapp_text_clf.data import DataSet
from smapp_text_clf.models import TextClassifier

from sklearn.model_selection import RandomizedSearchCV

my_tokenizer = lambda x: x.split()
dataset = DataSet(name='my_dataset', input_='my_dataset.csv', 
                  tokenizer=my_tokenizer,
                  field_mapping={'label': 'my_label_column', 
                                 'text': 'my_textcolumn'})
clf = TextClassifier(dataset=dataset, algorithm='svm', 
                     feature_set='embeddings',
                     embedding_model=('my_model', 'path/to/my/model'))

CV = RandomizedSearchCV(clf.pipeline, param_distributions=clf.params,
                        n_iter=5, cv=3, n_jobs=8, scoring='accuracy', 
                        iid=True, return_train_score=True)

X = dataset.get_texts('train')
y = dataset.get_labels('train')
CV = CV.fit(X, y)
print(CV.best_score_)
```

