import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score

from hw_03.project.preprocessing import Preprocess
import hw_03.project.data_utils as us
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.linear_model import RANSACRegressor


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



def log_transform(data):
    train["previous"] = train["previous"].map(lambda x: np.log(x+1))
    train["campaign"] = train["campaign"].map(lambda x: np.log(x+1))
    train["duration"] = train["duration"].map(lambda x: np.log(x+1))
    return data


train = pd.read_csv('train.csv')


preprocess = Preprocess(
    features_to_drop=["_id", "month", "day_of_week"],
    #features_to_drop="_id",
    before_process=log_transform,
    polynomical_power=3
    ,polynomical_features=["duration", "euribor3m", "nr.employed", "campaign", "cons.conf.idx", 'cons.price.idx'],
    factorizeAge=True,
    factorizeAgeBins=10,
    age_to_cat=True



)
preprocessed_train = preprocess.preprocess_data(train)


#models = {LogisticRegression(C=1, class_weight='balanced', penalty="l2", fit_intercept=False), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier()}
models = {CustomModel(models=[LogisticRegression(C=0.01, class_weight="balanced", fit_intercept=False), LogisticRegression(C=1, class_weight='balanced', penalty="l2", fit_intercept=False)])}
#models = {RANSACRegressor(LogisticRegression( class_weight='balanced'))}
print("Обработка моделей")
scores = us.selection_models(preprocess, train, models=models, n_splits=10)
print("Обработка моделей завершена")
us.print_scores(scores)

#us.find_key_attrs(preprocessed_train, "target")



# ['emp.var.rate' -3.5468861383224652]
# ['education_illiterate' -3.011071020582907]
# ['duration' 1.8099734787848105]
# ['month_mar' 1.7616020707467939]
# ['cons.price.idx' 1.336919503664157]
# ['euribor3m' 1.273671905096912]
# ['month_may' -1.110826114032223]
# ['month_jun' -0.9832311647042554]
# ['month_nov' -0.8675667696097898]
# ['month_aug' 0.8508221841604768]
# ['marital_unknown' -0.8438716422176328]
# ['education_university.degree' 0.6255706852006577]
# ['poutcome_failure' -0.5613110574892684]
# ['contact_telephone' -0.5344440974739847]
# ['education_professional.course' 0.46159835396104326]
# ['education_basic.4y' 0.4134753399019976]
# ['default_unknown' -0.38610039289135534]
# ['education_basic.6y' 0.35988794874468316]
# ['education_basic.9y' 0.32033674308961396]
# ['job_self-employed' -0.3071678426843508]
# ['education_high.school' 0.29187008744773035]
# ['job_unemployed' 0.2859885800298406]
# ['job_student' 0.2855321834405194]
# ['job_unknown' -0.25523574187330517]
# ['month_apr' -0.23559244137218538]
# ['job_retired' 0.23454422327672922]
# ['pdays' -0.22408193353712705]
# ['job_blue-collar' -0.20780746136308959]
# ['loan_no' -0.20511706578526553]
# ['month_jul' -0.1969546087066541]
# ['housing_yes' -0.19426136066845426]
# ['housing_no' -0.18567218226749738]
# ['month_oct' 0.18425927764750594]
# ['job_services' -0.1788410521712328]
# ['loan_yes' -0.17481647713637388]
# ['job_entrepreneur' -0.17348775440328923]
# ['day_of_week_mon' -0.16355416787490826]
# ['nr.employed' 0.16178325256892379]
# ['housing_unknown' -0.15839831928623005]
# ['loan_unknown' -0.15839831928623005]
# ['marital_single' 0.14154374785130786]
# ['poutcome_success' 0.1413801594212502]
# ['month_sep' 0.12686834842697722]
# ['day_of_week_thu' -0.1261061241096322]
# ['poutcome_nonexistent' -0.11840096411504969]
# ['default_yes' -0.11250220918316886]
# ['day_of_week_fri' -0.11106822089825645]
# ['job_management' -0.10479273594864214]
# ['campaign' -0.09833453888449457]
# ['marital_divorced' 0.08402029699812424]
# ['marital_married' 0.07997573514107349]
# ['day_of_week_tue' -0.07978694535692347]
# ['job_technician' -0.07075690159983128]
# ['month_dec' -0.06771264478751707]
# ['day_of_week_wed' -0.05781640398540785]
# ['job_admin.' -0.04505998055123467]
# ['default_no' -0.039729260127859184]
# ['cons.conf.idx' -0.03254400352063246]
# ['previous' -0.024764020116030086]
# ['contact_cellular' -0.0038877647269930284]
# ['age' -0.002693789812501685]
# ['job_housemaid' -0.0012473783869529327]
