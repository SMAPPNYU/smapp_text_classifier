# SMaPP Text Classification

This package contains some helpful abstractions to allow for easy training of state of the art supervised text classification models. It removes some repetitive and time intensive tasks from like setting up cross validation over various ML classifiers, transformation of text into various feature spaces (word n-grams, character n-grams, and word embeddings), and setting of reasonable defaults for tuning parameters. 

## Installation

```
pip install git+https://github.com/smappnyu/smapp_text_classifier.git
```

## Example Use

This is a bare bones example. For a more extensive executable notebook see [`/pipeline_demo`](https://github.com/SMAPPNYU/smapp_text_classifier/blob/master/pipeline_demo/pipeline_demo.ipynb).

```python
from smapp_text_classifier.data import DataSet
from smapp_text_classifier.models import TextClassifier

from sklearn.model_selection import RandomizedSearchCV

dataset = DataSet(name='my_dataset', input_='my_dataset.csv')

clf = TextClassifier(
    dataset=dataset, algorithm='svm', 
    feature_set='embeddings',
    embedding_model_name='glove-wiki-gigaword-50'
)

CV = RandomizedSearchCV(clf.pipeline, param_distributions=clf.params,
                        n_iter=5, cv=3, n_jobs=2, scoring='accuracy')

X = dataset.get_texts('train')
y = dataset.get_labels('train')
CV = CV.fit(X, y)
print(CV.best_score_)
```

