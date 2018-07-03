import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from hw_03.project.preprocessing import Preprocess


class CustomModel(BaseEstimator, ClassifierMixin):
    def __init__(self, models, features=None):
        self.features = features
        self.models = models

    def fit(self, X, y):
        if self.features is not None:
            X =X[self.features]

        for model in self.models[:-1]:
            model.fit(X, y)
            X[type(model).__name__] = model.predict_proba(X)[:, 1]
        self.models[-1].fit(X, y)

    def predict_proba(self, X):
        if self.features is not None:
            X =X[self.features]
        for model in self.models[:-1]:
            X[type(model).__name__] = model.predict_proba(X)[:, 1]

        return self.models[-1].predict_proba(X)




def do_predict(preprocess, train):
    test = pd.read_csv('test.csv')
    preprocessed_train = preprocess.preprocess_data(train)
    y = preprocessed_train.target
    x = preprocessed_train.drop(["target"], axis=1)

    lrc = CustomModel(models=[LogisticRegression(C=0.01, class_weight="balanced"),
                                LogisticRegression(C=1, class_weight='balanced', penalty="l2")])

    lrc.fit(x, y)



    preprocessed_test = preprocess.preprocess_data(test)
    my_submission = pd.DataFrame({'_id': test._id, 'target': lrc.predict_proba(preprocessed_test)[:, 1]})
    my_submission.to_csv('submission.csv', index=False)





def log_transform(data):
    train["previous"] = train["previous"].map(lambda x: np.log(x+1))
    train["campaign"] = train["campaign"].map(lambda x: np.log(x+1))
    train["duration"] = train["duration"].map(lambda x: np.log(x+1))
    return data



#--------------------------------------------------------

train = pd.read_csv('train.csv')

preprocess = Preprocess(
    #features_to_drop=["_id", "month", "day_of_week"],
    features_to_drop="_id",
    before_process=log_transform,
    polynomical_power=3
    ,polynomical_features=["duration","age", "euribor3m", "nr.employed", "campaign", "cons.conf.idx", 'cons.price.idx']



)
preprocessed_train = preprocess.preprocess_data(train)
do_predict(preprocess, train)

