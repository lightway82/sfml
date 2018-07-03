import sklearn.model_selection as ms
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler

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


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

def feature_selection(X_train, y_train):
    """
    Ищит наиболее интересные признаки
    :param X_train:
    :param y_train:
    :return: вернет модель отбора, можно преобразовать датасет через нее  для удаления лишних признаков select.transform(X_test)
    """
    select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold="median")
    select.fit(X_train, y_train)
    X_train_l1 = select.transform(X_train)
    print("форма обуч набора X: {}".format(X_train.shape))
    print("форма обуч набора X c l1: {}".format(X_train_l1.shape))
    mask = select.get_support()
    # визуализируем булевы значения -- черный – True, белый – False
    plt.matshow(mask.reshape(1, -1), cmap='gray_r')
    plt.xlabel("Индекс примера")
    return select



def anova_select_features(X_train, y_train, percentile=50):
    """
    Ищит наиболее интересные признаки. по одномерному тесту по одной фиче с целевой переменной используя anova.
    :param X_train:
    :param y_train:
    :param percentile: граничный процент
    :return: вернет модель отбора, можно преобразовать датасет через нее  для удаления лишних признаков select.transform(X_test)
    """
    from sklearn.feature_selection import SelectPercentile
    select = SelectPercentile(percentile=percentile)
    select.fit(X_train, y_train)
    return select


def RFE_selection_features(X_train, y_train):
    """
        Ищит наиболее интересные признаки. Использует итеративный алгоритм
        :param X_train:
        :param y_train:
        :param percentile: граничный процент
        :return: вернет модель отбора, можно преобразовать датасет через нее  для удаления лишних признаков select.transform(X_test)
        """
    from sklearn.feature_selection import RFE
    select = RFE(RandomForestClassifier(n_estimators=100, random_state=42),  n_features_to_select=40)
    select.fit(X_train, y_train)
    # визуализируем отобранные признаки:
    mask = select.get_support()
    plt.matshow(mask.reshape(1, -1), cmap='gray_r')
    plt.xlabel("Индекс примера")
    return select





def cross_correlation(dataframe):
    f, ax = plt.subplots(figsize=(23, 23))
    corr = dataframe.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        sns.heatmap(corr, ax=ax, annot=True, fmt=".1f", linewidths=.5, mask=mask, square=True)





def PCA_components(pca, feature_names):
    """
    Проценты объясняемой дисперсии по компонентам PCA
    :param pca: модель обученная
    :param feature_names: список имен фич, перед применением PCA
    :return:
    """
    for i, component in enumerate(pca.components_):
        print("{} component: {}% of initial variance".format(i + 1,
                                                             round(100 * pca.explained_variance_ratio_[i], 2)))
        print(" + ".join("%.3f x %s" % (value, name)
                         for value, name in zip(component, feature_names)))



    plt.figure(figsize=(10,7))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)
    plt.xlabel('Number of components')
    plt.ylabel('Total explained variance')
    plt.xlim(0, 63)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.axvline(21, c='b')
    plt.axhline(0.9, c='r')
    plt.show()



def dendrogramm(x, normalize=True, method='single', **dendrogramm_params):
    """
    Строит дендрограмму, кластеризуя x
    :param dendrogramm_params: параметры метода dendrogramm
    :param x: numpy массив строки и стоблцы данных
    :param normalize: нормализовать данные?
    :param method: метод определения дистанции 'single', 'complete', 'average', 'centroid','weighted'  См scipy linkage
    :return:
    """
    if normalize:
        s = StandardScaler()
        x = s.fit_transform(x)

    distance_mat = pdist(x)  # pdist посчитает нам верхний треугольник матрицы попарных расстояний
    Z = hierarchy.linkage(distance_mat, method='single')  # linkage — реализация агломеративного алгоритма
    plt.figure(figsize=(10, 5))
    dn = hierarchy.dendrogram(Z, color_threshold=0.5, **dendrogramm_params)
    return dn



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


def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df


from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib
import matplotlib.patches as mpatches


def plot_LSA(test_data, test_labels, savepath="PCA_demo.csv", plot=True):
    """
    plot_LSA(X_train_counts, y_train)
    :param test_data:
    :param test_labels:
    :param savepath:
    :param plot:
    :return:
    """
    fig = plt.figure(figsize=(16, 16))
    lsa = TruncatedSVD(n_components=2)
    lsa.fit(test_data)
    lsa_scores = lsa.transform(test_data)
    color_mapper = {label:idx for idx,label in enumerate(set(test_labels))}
    color_column = [color_mapper[label] for label in test_labels]
    colors = ['orange','blue','blue']
    if plot:
        plt.scatter(lsa_scores[:,0], lsa_scores[:,1], s=8, alpha=.8, c=test_labels, cmap=matplotlib.colors.ListedColormap(colors))
        red_patch = mpatches.Patch(color='orange', label='Irrelevant')
        green_patch = mpatches.Patch(color='blue', label='Disaster')
        plt.legend(handles=[red_patch, green_patch], prop={'size': 30})

    plt.show()




from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report


def get_metrics(y_test, y_predicted):

    """
    accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
    :param y_test:
    :param y_predicted:
    :return:
    """
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                average='weighted')
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                          average='weighted')

    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')

    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1




import numpy as np
import itertools
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.winter):
    """
     cm = confusion_matrix(y_test, y_predicted_counts)
     fig = plt.figure(figsize=(10, 10))
     plot = plot_confusion_matrix(cm, classes=['Irrelevant','Disaster','Unsure'], normalize=False, title='Confusion matrix')
     plt.show()
     print(cm)

    :param cm:
    :param classes:
    :param normalize:
    :param title:
    :param cmap:
    :return:
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] < thresh else "black", fontsize=40)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)

    return plt



