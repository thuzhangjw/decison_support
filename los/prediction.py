import pickle as pk
from sklearn.linear_model import LogisticRegression
from los.common import features_all, ContinuousVariables, CategoryVariables
import numpy as np
import pandas as pd
import math
from fuzzywuzzy import process, fuzz
import os

feature_names = [f.name for f in features_all]
icd_list = pk.load(open('./los/icd_list.pickle', 'rb'))


def demo_bin(item, col_name):
    feature = features_all[feature_names.index(col_name)]
    if isinstance(feature, CategoryVariables):
        return feature.table.index(item)
    elif isinstance(feature, ContinuousVariables):
        return item


def lab_bin(item, col_name):
    feature = features_all[feature_names.index(col_name)]
    assert isinstance(feature, ContinuousVariables)
    (lower, upper) = feature.normal_value_list[0][1:3]
    if pd.isnull(item):
        cat = 1.0
    else:
        cat = math.floor(((item - lower) / (upper - lower))) + 1.0
        if cat < 0:
            cat = 0
        if cat > 10.0:
            cat = 10.0
    return cat


def match_icd(x):
    if x in icd_list:
        return x
    else:
        icd = process.extractOne(x, icd_list)
        return icd


def get_cube(js):
    df = pd.DataFrame(js)
    df = df[feature_names]
    demo_col = list(range(3))
    lab_col = list(range(3, 80))
    diag_col = 80

    demo_raw = df.iloc[:, demo_col]
    for d in demo_raw:
        demo_raw[d] = demo_raw[d].apply(demo_bin, args=(d,))

    lab_raw = df.iloc[:, lab_col]
    for l in lab_raw:
        lab_raw[l] = lab_raw[l].apply(lab_bin, args=(l,))

    s_icd = df.iloc[:, diag_col]
    s_icd = s_icd.apply(match_icd)
    diag = pd.Categorical(s_icd, categories=icd_list)
    diag_dummy = pd.get_dummies(diag).values

    X_in = np.c_[demo_raw.values, lab_raw, diag_dummy]
    return X_in


def get_model(path):
    model = pk.load(open(os.path.join(path, 'length_of_stay.model'), 'rb'))
    assert isinstance(model, LogisticRegression)
    return model


def predict(model, data):
    assert isinstance(model, LogisticRegression)
    data_cube = get_cube(data)
    prob = model.predict_proba(data_cube)
    return prob

