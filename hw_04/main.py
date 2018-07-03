import seaborn as sns
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_log_error




def prepare_features(data):
    df=data.copy()
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M')
    df['year'] = df.loc[:, 'datetime'].dt.year
    df['month'] = df.loc[:, 'datetime'].dt.month
    df['day'] = df.loc[:, 'datetime'].dt.day
    df['hour'] = df.loc[:, 'datetime'].dt.hour
    df['weekday'] = df.loc[:, 'datetime'].dt.weekday
    year_season = df.year.astype(str) + '-' + df.month.astype(str)
    ## Your code here
    df["year"] = df["year"]
    #заменим 0 скорость ветра, на медиану тк в исхдных данных слишком много 0, что маловероятно
    df['windspeed']=df[['year','month','hour','windspeed']].groupby(['year','month','hour']).transform(lambda x:x.replace(0,np.median([i for i in x if i>0])))
    #категоризация скорости ветра
    df['windspeed']=pd.cut(df['windspeed'],bins=[0,15,35,60],labels=['0','1',"2"])
    df['ftemp'] = df['temp'] + df['atemp']
    df.drop(["datetime","day", "weekday","temp","atemp"], axis=1, inplace=True)#datetime не нужен,day не релевантен, weekday коррелирует с workingday
    return df, year_season

def searchDTRParams(x, y):
    parameters = {'max_depth': range(1, 10), 'min_samples_leaf': range(1, 15)}
    clf = GridSearchCV(DecisionTreeRegressor(), parameters, cv=5)
    clf.fit(x, y)
    print(clf.best_params_)
    return clf



def get_bests_model(x, y):
    _casual = searchDTRParams(X_train, y_train['casual'])
    _registered = searchDTRParams(X_train, y_train['registered'])
    _count = searchDTRParams(X_train, y_train['count'])
    return _casual, _registered, _count

def cross_validate_(estimator, x, y, scoring='neg_mean_squared_log_error'):
    scorres = cross_validate(estimator, x, y, scoring=scoring, cv=5)
    print("scores", scorres['test_score'])


df = pd.read_csv('train.csv')


df_clean, year_season = prepare_features(df)



X_train = df_clean.drop(['casual', 'registered', 'count'], axis=1)
y_train = df_clean[['casual', 'registered', 'count']]



estimator_casual, estimator_registered, estimator_count = get_bests_model(X_train, y_train['casual'])

cross_validate_(DecisionTreeRegressor(max_depth=estimator_casual.best_params_["max_depth"], min_samples_leaf=estimator_casual.best_params_["min_samples_leaf"]), X_train, y_train['casual'])
cross_validate_(DecisionTreeRegressor(max_depth=estimator_registered.best_params_["max_depth"], min_samples_leaf=estimator_registered.best_params_["min_samples_leaf"]), X_train, y_train['registered'])
cross_validate_(DecisionTreeRegressor(max_depth=estimator_count.best_params_["max_depth"], min_samples_leaf=estimator_count.best_params_["min_samples_leaf"]), X_train, y_train['count'])


cv_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=123)


y_labels = ['casual', 'registered', 'count']
y_labels_mask = df_clean.columns.isin(y_labels)
X_labels = df_clean.columns[~y_labels_mask]



for idx_train, idx_test in cv_split.split(df_clean, year_season):
    X_train, y_train = df_clean.loc[idx_train, X_labels], df_clean.loc[idx_train, y_labels]
    X_test, y_test = df_clean.loc[idx_test, X_labels].values, df_clean.loc[idx_test, y_labels]




casual = DecisionTreeRegressor(max_depth=estimator_casual.best_params_["max_depth"], min_samples_leaf=estimator_casual.best_params_["min_samples_leaf"])
registered = DecisionTreeRegressor(max_depth=estimator_registered.best_params_["max_depth"], min_samples_leaf=estimator_registered.best_params_["min_samples_leaf"])

casual.fit(X_train, y_train["casual"])
registered.fit(X_train, y_train["registered"])

predict = casual.predict(X_test)+registered.predict(X_test)
print(mean_squared_log_error(y_test["count"], predict)**0.5)







