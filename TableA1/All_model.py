import random as rn
import os
import numpy as np
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(123)
import pandas as pd
import pickle
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score

Path = 'D:\\APViaML\\MCS_code\\MCScode'

def get_mc_x_data():
    file = open(Path + '\\data\\mcs_demo_x_100.pkl','rb')
    raw_data = pickle.load(file)
    file.close()
    return raw_data

def get_mc_y_data():
    file = open(Path + '\\data\\mcs_demo_y_100.pkl','rb')
    raw_data = pickle.load(file)
    file.close()
    return raw_data


from sklearn.linear_model import LinearRegression

def rolling_model_Oracle(
        df_X,
        df_Y):

    split_num = 200*60
    X_traindata = df_X[:split_num]
    Y_traindata = df_Y[:split_num]
    X_testdata = df_X[split_num*2:split_num*3]
    Y_testdata = df_Y[split_num*2:split_num*3]

    # specify parameters and distributions to sample from

    clf_OLS = LinearRegression(n_jobs=2)
    clf_OLS.fit(X_traindata, Y_traindata)

    v_pred = clf_OLS.predict(X_traindata)
    v_performance_score = r2_score(Y_traindata, v_pred)

    test_pre_y_array = clf_OLS.predict(X_testdata)
    test_performance_score = r2_score(Y_testdata, test_pre_y_array)

    return v_performance_score, test_performance_score


X = get_mc_x_data()
Y = get_mc_y_data()

y1 = np.array(Y['ret1']).copy()
x1_oracle = np.array(Y['gz1']).copy()
y1_array = np.array(y1).reshape(-1,1)
x1_array_oracle = np.array(x1).reshape(-1,1)

y2 = np.array(Y['ret2']).copy()
x2_oracle = np.array(Y['gz2']).copy()
y2_array = np.array(y2).reshape(-1,1)
x2_array_oracle = np.array(x2).reshape(-1,1)

IS_score, OOS_score = rolling_model_Oracle(df_X=x1_array_oracle, df_Y=y1_array)
print('Oracle Model a IS:', IS_score * 100)
print('Oracle Model a OOS:', OOS_score * 100 )

IS_score_b, OOS_score_b = rolling_model_Oracle(df_X=x2_array_oracle,df_Y=y2_array)
print('Oracle Model a IS:', IS_score_b * 100)
print('Oracle Model a OOS:', OOS_score_b * 100 )


def rolling_model_OLS(
        df_X,
        df_Y):

    split_num = 200*60
    X_traindata = df_X[:split_num]
    Y_traindata = df_Y[:split_num]
    X_testdata = df_X[split_num*2:split_num*3]
    Y_testdata = df_Y[split_num*2:split_num*3]

    # specify parameters and distributions to sample from

    clf_OLS = LinearRegression(n_jobs=2)
    clf_OLS.fit(X_traindata, Y_traindata)

    v_pred = clf_OLS.predict(X_traindata)
    v_performance_score = r2_score(Y_traindata, v_pred)

    test_pre_y_array = clf_OLS.predict(X_testdata)
    test_performance_score = r2_score(Y_testdata, test_pre_y_array)

    return v_performance_score, test_performance_score


x_array = np.array(X)

