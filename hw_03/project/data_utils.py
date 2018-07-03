import sklearn.model_selection as ms
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve

def plot_validation_curve(model, X, y, param_name, param_range, cv=10):
    """
    Строит кривые обучения по изменению параметра. Кривая проверки на обучающей и на тестовой выборке от значения параметра.
    :param model:  модель
    :param X:
    :param y:
    :param param_name: имя параметра модели, которое надо менять, например, polynomialfeatures__degree
    :param param_range: массив значений параметра
    :param cv: количество фолдов или объект Fold
    """
    train_score, val_score = validation_curve(model, X, y,  param_name, param_range, cv)
    plt.plot(param_range, np.median(train_score, 1), color='blue', label='training score')
    plt.plot(param_range, np.median(val_score, 1), color='red', label='validation score')
    plt.legend(loc='best')
    plt.ylim(0, 1)
    plt.xlabel('degree')
    plt.draw()
    plt.show()

def plot_roc_curve(y_test, predict):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predict)
    rc_score = metrics.roc_auc_score(y_test, predict)
    plt.title('ROC (AUC=%0.2f)' % (rc_score))
    plt.fill_between(fpr, tpr, alpha=0.2)
    plt.grid(True, linestyle='-', color='0.75')
    plt.plot(fpr, tpr, linewidth=2, label="roc")
    plt.plot(fpr, thresholds, linewidth=1, label="thresholds")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.axis([0, 1, 0, 1])
    plt.draw()
    plt.show()


def plot_pr(y_test, predict):
    p, r, thresholds = metrics.precision_recall_curve(y_test, predict)

    plt.title('PR')
    plt.fill_between(r, p, alpha=0.2)
    plt.grid(True, linestyle='-', color='0.75')
    plt.plot(r, p, linewidth=2, label="roc")
    plt.plot(r[:-1], thresholds, linewidth=1, label="thresholds")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axis([0, 1, 0, 1])
    plt.draw()
    plt.show()




def selection_models(preprocess, train_data, n_splits=10, scoring='roc_auc', models=None):
    """
    Производит кросс-валидацию указанных моделей на обучающем наборе
    Набор должен содержать столбец с именем целевой переменной target
    :param preprocess: подготовленный объект препроцессинга
    :param train_data: датасет
    :param n_splits: количество разбиений датасета
    :param scoring: тип метрики для валидации
    :param models: список моделей(настроенных инстансов)
    :return:
    """
    if models is None:
        raise Exception("Необходимо указать список моделей")
    preprocessed = preprocess.preprocess_data(train_data)
    return selection_models2(preprocessed, n_splits, scoring, models)


def selection_models2(preprocessed, n_splits=10, scoring='roc_auc', models=None):
    """
    Производит кросс-валидацию указанных моделей на обучающем наборе
    :param preprocessed:
    :param n_splits:
    :param scoring:
    :param models: инстанс модели
    :return:
    """
    if models is None:
        raise Exception("Необходимо указать список моделей")

    y = preprocessed.target
    x = preprocessed.drop(["target"], axis=1)

    cross_val_results = dict()

    for model in models:
        print("     Обработка модели:", type(model))
        scores = ms.cross_val_score(
            model,
            x,
            y,
            scoring=scoring,
            cv=ms.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123))

        cross_val_results[model] = {
            "scores": scores,
            "variance": np.var(scores),
            "std": np.std(scores),
            "mean": np.mean(scores)
        }

    return cross_val_results


def print_scores(scores):
    for m, sc in scores.items():
        print("=====", m, "=====", "\n")
        for k, v in sc.items():
            print(k, v, "\n")


def random_forest_one_to_one_cols(pd_dataframe, target_name, scoring='roc_auc', n_splits=10, random_state=123):
    y = pd_dataframe.target
    x = pd_dataframe.drop(["target"], axis=1)

    cross_val_results = []

    for col in x.columns:
        print("     Обработка столбца:", col)
        scores = ms.cross_val_score(
            RandomForestClassifier(),
            x[[col]],
            y,
            scoring=scoring,
            cv=ms.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state))

        cross_val_results.append({
            "column": col,
            "std": np.std(scores),
            "mean": np.mean(scores)
        })

    return sorted(cross_val_results, key=lambda t: t["mean"])


def classification_exp(pd_dataframe, column_names, target_name, scoring='roc_auc', n_splits=10, random_state=123, model=RandomForestClassifier):
    y = pd_dataframe.target
    x = pd_dataframe.drop([target_name], axis=1)

    scores = ms.cross_val_score(
        model(),
        x[column_names],
        y,
        scoring=scoring,
        cv=ms.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state))

    print({
            "std": np.std(scores),
            "mean": np.mean(scores)
        })


def find_key_attrs(data, target_name):
    forest = RandomForestClassifier(n_estimators=400, oob_score=True)
    forest.fit(data.drop([target_name], axis=1), data[target_name])
    feature_importance = forest.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    fi_threshold = 5
    important_idx = np.where(feature_importance > fi_threshold)[0]
    important_features = data.columns[important_idx]
    print("\n", important_features.shape[0], "Important features(>", fi_threshold, "% of max importance)...\n")
    # important_features
    sorted_idx = np.argsort(feature_importance[important_idx])[::-1]
    # get the figure about important features
    pos = np.arange(sorted_idx.shape[0]) + .5
    # plt.subplot(1, 2, 2)
    plt.title('Feature Importance')
    plt.barh(pos, feature_importance[important_idx][sorted_idx[::-1]],
             color='r', align='center')
    plt.yticks(pos, important_features[sorted_idx[::-1]])
    plt.xlabel('Relative Importance')
    plt.draw()
    plt.show()
    print(important_features[sorted_idx[::-1]])



# Не секрет, что зачастую самым важным при решении задачи является умение правильно отобрать и даже создать признаки. В англоязычной литературе это называется Feature Selection и Feature Engineering. В то время как Future Engineering довольно творческий процесс и полагается больше на интуицию и экспертные знания, для Feature Selection есть уже большое количество готовых алгоритмов. «Древесные» алгоритмы допускают расчета информативности признаков:
#
# from sklearn import metrics
# from sklearn.ensemble import ExtraTreesClassifier
# model = ExtraTreesClassifier()
# model.fit(X, y)
# # display the relative importance of each attribute
# print(model.feature_importances_)
#
#
# Все остальные методы так или иначе основаны на эффективном переборе подмножеств признаков с целью найти наилучшее подмножество, на которых построенная модель дает наилучшее качество. Одним из таких алгоритмов перебора является Recursive Feature Elimination алгоритм, который также доступен в библиотеке Scikit-Learn:
#
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# # create the RFE model and select 3 attributes
# rfe = RFE(model, 3)
# rfe = rfe.fit(X, y)
# # summarize the selection of the attributes
# print(rfe.support_)
# print(rfe.ranking_)
