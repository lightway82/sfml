from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import category_encoders as ce
import numpy as np

class Preprocess:
    def __init__(self, *, after_process=None, before_process=None, features_to_drop=None, encode_binary_attrs=False, numeric_education_grade=False, numeric_education_grade_scale=False, scalingNumeric=True, factorizeAge=False, factorizeAgeBins=10, age_to_cat=False, scalingFactorizedAge=False, polynomical_power=0,  polynomical_features=None):
        """
        Препроцессинг данных. Перед применением, необходимо первый раз прогнать на выборке( произойдет препроцессинг, после чего можно использовать дальше)
        :param after_process: функция доп. препроцессинга, выполнится после всего препроцесса, принимает датафрейм обработанный, возвращает датафрейм измененный
        :param features_to_drop: имя столбца  или список имен столбцов для удаления
        :param encode_binary_attrs: кодировать ли бинарные атрибуты в 0 1 и предсказывать unknown или интерпретировать как категоральный
        :param numeric_education_grade: переводить ли образование в категории по уровням от 1 итп, иначе интерпретируется как категоральный
        :param numeric_education_grade_scale: производить ли числовое масштабирование градуированного образования
        :param scalingNumeric: масштабировать цифровые значения
        :param factorizeAge: факторизовать возраст
        :param factorizeAgeBins: количество бинов для факторизации возраста
        :param age_to_cat: факторизованное значение возраста перевести в котегории
        :param scalingFactorizedAge:
        :param polynomical_power степень полинома для создания фич. 0 или 1 не будет задавать преобразований
        :param polynomical_features список фич(имен столбцов) для полиномиального преобразования
        """

        self.__polynomical_features = polynomical_features
        self.__polynomical_power = polynomical_power
        self.__before_process = before_process
        self.__after_process = after_process
        self.__age_to_cat = age_to_cat
        self.__encode_binary_attrs = encode_binary_attrs
        self.__features_to_drop = features_to_drop
        self.__factorizeAgeBins = factorizeAgeBins
        self.__scalingFactorizedAge = scalingFactorizedAge
        self.__factorizeAge = factorizeAge
        self.__scalingNumeric = scalingNumeric
        self.__numeric_education_grade_scale = numeric_education_grade_scale
        self.__numeric_education_grade = numeric_education_grade

        self.__numeric_attrs = ['age', 'duration', 'campaign', 'pdays', 'previous',
                              'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
        self.__cat_attrs = ['poutcome', 'education', 'job', 'marital', 'contact', 'month', 'day_of_week']
        self.__binary_attrs = ['default', 'housing', 'loan']
        self.__target_attr = "target"

        if self.__numeric_education_grade:
            self.__cat_attrs.remove("education")

        if self.__factorizeAge:
            self.__numeric_attrs.remove("age")

        if not self.__encode_binary_attrs:
            self.__cat_attrs += self.__binary_attrs
            self.__binary_attrs = []

        self.__binary_predict_models = dict()

        if self.__age_to_cat:
            if not self.__factorizeAge:
                raise Exception("age_to_cat включать вместе с factorizeAge!")
            self.__cat_attrs.append("age")

        if self.__features_to_drop is not None:
            if isinstance(self.__features_to_drop, str):
                if self.__features_to_drop in self.__numeric_attrs:
                    self.__numeric_attrs.remove(self.__features_to_drop)
                if self.__features_to_drop in self.__binary_attrs:
                    self.__binary_attrs.remove(self.__features_to_drop)
                if self.__features_to_drop in self.__cat_attrs:
                    self.__cat_attrs.remove(self.__features_to_drop)

            else:
                for i in self.__features_to_drop:
                    if i in self.__numeric_attrs:
                        self.__numeric_attrs.remove(i)
                    if i in self.__binary_attrs:
                        self.__binary_attrs.remove(i)
                    if i in self.__cat_attrs:
                        self.__cat_attrs.remove(i)

        self.__cat_attrs.append("pdays_none")#доп атрибут по pdays - по факту значения 999

    def preprocess_data(self, data):
        output = data.copy()
        if self.__before_process is not None:
            output = self.__before_process(output)

        if self.__features_to_drop is not None:
            if isinstance(self.__features_to_drop, str):
                output.drop([self.__features_to_drop], axis=1, inplace=True)
            else:
                output.drop(self.__features_to_drop, axis=1, inplace=True)

        self.__process_pdays(output)

        output = self.__preprocess_education(output)
        output = self.__preprocess_age(output)
        output = self.__process_cat(output)
        output = self.__process_numeric(output)
        if self.__encode_binary_attrs:
            output = self.__process_binary(output)

        if self.__polynomical_power > 1:
            output = self.__polynomical_process(output)

        if self.__after_process is not None:
            output = self.__after_process(output)
        return output

    def __process_pdays(self, output):
        median = output["pdays"][output["pdays"]!=999].median()
        output["pdays_none"] = np.where(output["pdays"] == 999, "yes", "no")

        output["pdays"].where(output["pdays"] == 999, median, inplace=True)

        return output

    def __preprocess_age(self, output):
        if self.__factorizeAge:
            output = Preprocess.factorize_numeric_attribute(output, "age", self.__factorizeAgeBins)
            if self.__scalingFactorizedAge:
                output = Preprocess.feature_scaling(output, "age")
        return output


    def __preprocess_education(self, output):
        if self.__numeric_education_grade:
            output = Preprocess.__encode_edu_attrs(output)
            if self.__numeric_education_grade_scale:
                output = Preprocess.feature_scaling(output, "education")
        else:
            output = Preprocess.__fill_unknown_edu_attrs(output)
        return output

    def __fill_unknown_binary(self, data):
        """Заполнит unknown поля бинарных фич на основании предскзания
        по остальным данным, используя RandomForestClassifier
        """

        for i in self.__binary_attrs:
            test_data = data[data[i] == 'unknown']
            if 'target' in test_data.columns:
                testX = test_data.drop(self.__binary_attrs + [self.__target_attr], axis=1)
            else:
                testX = test_data.drop(self.__binary_attrs, axis=1)
            train_data = data[data[i] != 'unknown']
            trainY = train_data[i].astype('int32')
            if 'target' in train_data.columns:
                trainX = train_data.drop(self.__binary_attrs + [self.__target_attr], axis=1)
            else:
                trainX = train_data.drop(self.__binary_attrs, axis=1)
            # при повторном вызове препроцессинга, этот кусок не будет выполняться, а будет использоваться уже готовые модели
            if i not in self.__binary_predict_models:
                self.__binary_predict_models[i] = RandomForestClassifier(n_estimators=100)
                self.__train_unknown_binary(trainX, trainY, self.__binary_predict_models[i])

            test_data[i] = self.__predict_unknown_binary(testX, self.__binary_predict_models[i])
            data = pd.concat([train_data, test_data])
        return data

    def __train_unknown_binary(self, trainX, trainY, model):
        model.fit(trainX, trainY)


    def __predict_unknown_binary(self, testX, model):
        test_predict_y = model.predict(testX).astype(int)
        return pd.DataFrame(test_predict_y, index=testX.index)

    def __process_binary(self, data):
        """Заменяет бинарные атрибуты yes no на 1 и 0.
          Значения unknown заменит предсказанными
          """
        for i in self.__binary_attrs:
            data.loc[data[i] == 'no', i] = 0
            data.loc[data[i] == 'yes', i] = 1

        data = self.__fill_unknown_binary(data)

        for i in self.__binary_attrs:
            data[i] = data[i].astype("int32")

        return data


    def __process_cat(self, data):
        """Преобразует категоральные атрибуты"""
        return pd.get_dummies(data, columns=self.__cat_attrs)

        # encoder = ce.OneHotEncoder(cols=self.__cat_attrs)
        # encoder.fit(data, verbose=1)
        # data = encoder.transform(data)
        # #хак для грязной особенности category_encoders
        # if "col_" in data.columns[0]:
        #     data.rename(columns=lambda x: x[len("col_"):], inplace=True)
        return data

    def __process_numeric(self, data):
        if self.__scalingNumeric:
            return Preprocess.features_scaling(data, self.__numeric_attrs)
        else:
            return data


    @staticmethod
    def features_scaling(data, numeric_attrs):
        """
        Стандартное масштабирование числовых фич, по mean и std
        """
        for i in numeric_attrs:
            std = data[i].std()
            if std != 0:
                data[i] = (data[i] - data[i].mean()) / std

        return data

    @staticmethod
    def feature_scaling(data, feature_name):
        """
        Стандартное масштабирование одной числовой фичи, по mean и std
        """
        std = data[feature_name].std()
        if std != 0:
            data[feature_name] = (data[feature_name] - data[feature_name].mean()) / std

        return data

    @staticmethod
    def factorize_numeric_attribute(data, attr, bins=10):
        """
        Факторизует столбец на диапазоны bins, от 1 и далее
        :param data: DataFrame
        :param attr: строкка название столбца
        :param bins: на сколько частей разбивать
        :return: преобразованный DataFrame
        """
        data[attr] = pd.qcut(data[attr], bins)
        data[attr] = pd.factorize(data[attr])[0] + 1
        return data


    @staticmethod
    def __encode_edu_attrs(data):
        """
        Заменяет строку с образованием на его урове от 1 и до макс уровня
        :param data: DataFrame
        :return:  вернет преобразованный датафрейм
        """
        values = ["illiterate", "basic.4y", "basic.6y", "basic.9y",
                          "high.school", "professional.course", "university.degree"]
        levels = range(1, len(values) + 1)
        dict_levels = dict(zip(values, levels))
        for v in values:
            data.loc[data['education'] == v, 'education'] = dict_levels[v]

        data['education'] = data.education.map(lambda v: v if v != 'unknown' else 4)
        return data


    @staticmethod
    def __fill_unknown_edu_attrs(data):
        """
        Заменяет в графе образование только unknown, заменяет на basic.9y
        :param data: DataFrame
        :return: вернет преобразованный датафрейм
        """
        data['education'] = data.education.map(lambda v: v if v != 'unknown' else "basic.9y")
        return data

    def __polynomical_process(self, output):
        """Добавляет фичи со степенями полинома и кросс фичи"""
        features = self.__polynomical_features if self.__polynomical_features is not None else self.__numeric_attrs
        pf = PolynomialFeatures(self.__polynomical_power, interaction_only=False)
        res = pf.fit_transform(output[features])
        start_pos = len(features) + 1
        df = pd.DataFrame(res[:, start_pos:], columns=pf.get_feature_names(input_features=features)[start_pos:])
        return pd.concat([output, df], axis=1)
