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
Path = 'D:\\APViaML'


def get_mc_x_data():
    file = open(Path + '\\data\\mcs_demo_x.pkl', 'rb')
    raw_data = pickle.load(file)
    file.close()
    return raw_data


def get_mc_y_data():
    file = open(Path + '\\data\\mcs_demo_y.pkl', 'rb')
    raw_data = pickle.load(file)
    file.close()
    return raw_data


def Evaluation_fun(predict_array, real_array):
    List1 = []
    List2 = []
    if len(predict_array) != len(real_array):
        print('Something is worng!')
    else:
        for i in range(len(predict_array)):
            List1.append(np.square(predict_array[i] - real_array[i]))
            List2.append(np.square(real_array[i]))
        result = round(100 * (1 - sum(List1) / sum(List2)), 3)
    return result


def my_custom_score_func(ground_truth, predictions):
    num1 = sum((ground_truth - predictions) ** 2)
    num2 = sum(ground_truth ** 2)
    return 1 - num1 / num2


self_score = make_scorer(my_custom_score_func, greater_is_better=True)

from sklearn.linear_model import LinearRegression


def rolling_model_OLS(
        df_X,
        df_Y):
    y_predict_list = []
    v_performance_score_list = []
    test_performance_score_list = []

    id_num = 200
    for i in range(60):
#        print(i)
        ## define data index
        train_num = id_num * 60 + id_num * (i)
        v_num_s = id_num * 60 + id_num * (i)
        v_num_e = id_num * 120 + id_num * (i)
        test_num_s = id_num * 120 + id_num * (i)
        test_num_e = id_num * 120 + id_num * (i + 1)

        X_traindata = df_X[:train_num]
        Y_traindata = df_Y[:train_num]
        X_vdata = df_X[v_num_s:v_num_e]
        Y_vdata = df_Y[v_num_s:v_num_e]
        X_testdata = df_X[test_num_s:test_num_e]
        Y_testdata = df_Y[test_num_s:test_num_e]

        # specify parameters and distributions to sample from

        clf_OLS = LinearRegression(n_jobs=2)
        clf_OLS.fit(X_traindata, Y_traindata)

        v_pred = clf_OLS.predict(X_vdata)
        v_performance_score = my_custom_score_func(Y_vdata, v_pred)
# TODO  v_performance_score = r2_score(Y_vdata, v_pred)
        v_performance_score_list.append(v_performance_score)

        test_pre_y_array = clf_OLS.predict(X_testdata)

        for x in test_pre_y_array:
            y_predict_list.append(x)

        test_performance_score = my_custom_score_func(Y_testdata, test_pre_y_array)
        test_performance_score_list.append(test_performance_score)

    return y_predict_list, v_performance_score_list, test_performance_score_list


X = get_mc_x_data()
Y = get_mc_y_data()
y1 = np.array(Y['ret'])
y2 = np.array(Y['ret1'])
x = np.array(X)

y1_array = np.array(y1).reshape(-1, 1)
y2_array = np.array(y2).reshape(-1, 1)
x_array = np.array(x)


y_predict_list, v_performance_score_list, test_performance_score_list = rolling_model_OLS(df_X=x_array,
                                                                                          df_Y=y1_array)

print('OLS Model a IS:', np.mean(v_performance_score_list) * 100)
y_test_all = y1_array[120 * 200:].copy()
print('OLS Model a OOS:', my_custom_score_func(y_test_all, y_predict_list) * 100)
#print('OLS Model a OOS:', r2_score(y_test_all, y_predict_list) * 100)

y_predict_list, v_performance_score_list, test_performance_score_list = rolling_model_OLS(df_X=x_array,
                                                                                          df_Y=y2_array)

print('OLS Model b IS:', np.mean(v_performance_score_list) * 100)
y_test_all = y2_array[120 * 200:].copy()
print('OLS Model b OOS:', my_custom_score_func(y_test_all, y_predict_list) * 100)
#print('OLS Model b OOS:', r2_score(y_test_all, y_predict_list) * 100)

# y_predict_list1 = []
# for x in y_predict_list:
#     y_predict_list1.append(x[0])
#
# y_test_all1 = []
# for x in y_test_all:
#     y_test_all1.append(x[0])
#
#
# Evaluation_fun(y_predict_list1,y_test_all1)


