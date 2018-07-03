
import pandas as pd

from sklearn.pipeline import Pipeline
from scipy import sparse
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import TruncatedSVD




def submit1():
    train = pd.read_csv('saved_train_lemmatize.csv', index_col='id', sep="\t")
    test = pd.read_csv('saved_test_lemmatize.csv',  sep="\t")


    print("Обучение")

    text_clf = Pipeline([
                        ('vect', CountVectorizer(ngram_range=(1,2))),
                        ('clf', LogisticRegression(C=5))

                         ])

    text_clf.fit(train.name, train.target)

    test.fillna(value="Специалист", inplace=True)

    print("Предсказание")
    predict = text_clf.predict_proba(test.name)



    my_submission = pd.DataFrame({'id': test.id, 'target': predict[:,1]})

    my_submission.to_csv('submission.csv', index=False)
    print("Записано")



def submit2():
    train = pd.read_csv('saved_train_lemmatize.csv', index_col='id', sep="\t")
    test = pd.read_csv('saved_test_lemmatize.csv', sep="\t")

    test.fillna(value="Специалист", inplace=True)

    print("Transform")
    pipe1 = Pipeline([
        ("cv", CountVectorizer(ngram_range=(1, 2))),
        # ("tfidf", TfidfTransformer())
    ])

    pipe2 = Pipeline([
        ("cv", CountVectorizer(min_df=0.1, max_df=0.85)),
        ("tfidf", TfidfTransformer())
    ])

    x1 = pipe1.fit_transform(train.name)
    x1_test = pipe1.transform(test.name)

    x2 = pipe2.fit_transform(train["description"])#ошибся написал pipe1 и пролучил 0.991
    x2_test = pipe2.transform(test["description"])

    xx = sparse.hstack((x1, x2))
    xx_test = sparse.hstack((x1_test, x2_test))

    print("Обучение")
    text_clf = LogisticRegression(C=5)

    text_clf.fit(xx, train.target)



    print("Предсказание")
    predict = text_clf.predict_proba(xx_test)

    my_submission = pd.DataFrame({'id': test.id, 'target': predict[:, 1]})

    my_submission.to_csv('submission.csv', index=False)
    print("Записано")






def submit3():
    train = pd.read_csv('saved_train_lemmatize.csv', index_col='id', sep="\t")
    test = pd.read_csv('saved_test_lemmatize.csv', sep="\t")

    test.fillna(value="Специалист", inplace=True)

    print("Transform")

    with open("stop_words", "rt") as fp:
        stop_words = set(fp.readlines())


    pipe1 = Pipeline([
        ("cv", CountVectorizer(ngram_range=(1, 3))),
        ("tfidf", TfidfTransformer())
    ])

    pipe2 = Pipeline([
        ("cv", CountVectorizer(min_df=0.1, max_df=0.55, stop_words=stop_words, ngram_range=(1,3))),
        ("tfidf", TfidfTransformer()),
        #('pca', TruncatedSVD(n_components=150))
    ])

    x1 = pipe1.fit_transform(train.name)
    x1_test = pipe1.transform(test.name)

    x2 = pipe2.fit_transform(train["description"])#ошибся написал pipe1 и пролучил 0.991
    x2_test = pipe2.transform(test["description"])

    xx = sparse.hstack((x1, x2))
    xx_test = sparse.hstack((x1_test, x2_test))

    print("Обучение")
    text_clf = LogisticRegression(C=5)

    text_clf.fit(xx, train.target)



    print("Предсказание")
    predict = text_clf.predict_proba(xx_test)

    my_submission = pd.DataFrame({'id': test.id, 'target': predict[:, 1]})

    my_submission.to_csv('submission.csv', index=False)
    print("Записано")

submit3()