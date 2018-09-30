import random as rn
import os
import numpy as np
from pandas import DataFrame
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from scipy.stats import randint as sp_randint
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(123)
import pandas as pd
import pickle

from sklearn.metrics import r2_score


Path = 'D:\\APViaML\\MCS_code\\MCScode'


def get_mc_x_data(m):
    file = open(Path + '\\data\\mcs_demo_x_100' + str(m) + '.pkl', 'rb')
    raw_data = pickle.load(file)
    file.close()
    return raw_data


def get_mc_y_data(m):
    file = open(Path + '\\data\\mcs_demo_y_100' + str(m) + '.pkl', 'rb')
    raw_data = pickle.load(file)
    file.close()
    return raw_data


from sklearn.linear_model import LinearRegression


################################define table
columns_list = ['IS_a_50', 'OOS_a_50', 'IS_b_50', 'OOS_b_50']
index_list = ['Oracle', 'OLS', 'OLS_H', 'PCR', 'PLS', 'Lasso', 'Lasso_H',
              'Ridge', 'Ridge_H', 'ENet', 'ENet_H']
tableA1 = pd.DataFrame(columns=columns_list, index=index_list)


## save the result
def save_result(i):
    tableA1.iloc[i, 0] = round(IS_score * 100, 3)
    tableA1.iloc[i, 1] = round(OOS_score * 100, 3)
    tableA1.iloc[i, 2] = round(IS_score_b * 100, 3)
    tableA1.iloc[i, 3] = round(OOS_score_b * 100, 3)




################################Oracle Model
def rolling_model_Oracle(df_X, df_Y):
    split_num = 200 * 60
    X_traindata = df_X[:split_num]
    Y_traindata = df_Y[:split_num]
    X_testdata = df_X[split_num * 2:split_num * 3]
    Y_testdata = df_Y[split_num * 2:split_num * 3]

    # specify parameters and distributions to sample from

    clf_OLS = LinearRegression(n_jobs=2)
    clf_OLS.fit(X_traindata, Y_traindata)

    v_pred = clf_OLS.predict(X_traindata)
    v_performance_score = r2_score(Y_traindata, v_pred)

    test_pre_y_array = clf_OLS.predict(X_testdata)
    test_performance_score = r2_score(Y_testdata, test_pre_y_array)

    return v_performance_score, test_performance_score

################################OLS Model
def rolling_model_OLS(df_X, df_Y):
    split_num = 200 * 60
    X_traindata = df_X[:split_num]
    Y_traindata = df_Y[:split_num]
    X_testdata = df_X[split_num * 2:split_num * 3]
    Y_testdata = df_Y[split_num * 2:split_num * 3]

    # specify parameters and distributions to sample from

    clf_OLS = LinearRegression(n_jobs=2)
    clf_OLS.fit(X_traindata, Y_traindata)

    v_pred = clf_OLS.predict(X_traindata)
    v_performance_score = r2_score(Y_traindata, v_pred)

    test_pre_y_array = clf_OLS.predict(X_testdata)
    test_performance_score = r2_score(Y_testdata, test_pre_y_array)

    return v_performance_score, test_performance_score

def rolling_model_OLSH(df_X, df_Y):
    split_num = 200 * 60
    X_traindata = df_X[:split_num * 2]
    Y_traindata = df_Y[:split_num * 2]
    X_vdata = df_X[split_num:split_num * 2]
    X_testdata = df_X[split_num * 2:split_num * 3]
    Y_testdata = df_Y[split_num * 2:split_num * 3]

    # specify parameters and distributions to sample from

    num_valid_size = len(X_traindata) - len(X_vdata)
    test_fold = -1 * np.ones(len(X_traindata))
    test_fold[num_valid_size:] = 0
    ps = PredefinedSplit(test_fold)

    # specify parameters and distributions to sample from
    param_dist = {'alpha': uniform(0.00001, 0.1),
                  'power_t': uniform(0.1, 0.9),
                  'l1_ratio': uniform(0.1, 0.9),
                  'eta0': uniform(0.00001, 0.1),
                  'epsilon': uniform(0.01, 0.9),
                  'max_iter': sp_randint(5, 10000),
                  'tol': [0.01, 0.001, 0.0001, 0.00001],
                  'fit_intercept': [True, False]}

    clf = SGDRegressor(shuffle=False, loss='huber', penalty='none', random_state=100)

    # run randomized search
    n_iter_search = 50
    estim = RandomizedSearchCV(clf, param_distributions=param_dist,
                               n_iter=n_iter_search, scoring='r2',
                               cv=ps.split(), iid=False,
                               random_state=100, n_jobs=1)

    estim.fit(X_traindata, Y_traindata)
    best_estimator = estim.best_estimator_
    v_pred = best_estimator.predict(df_X[:split_num])
    v_performance_score = r2_score(df_Y[:split_num], v_pred)
    test_pre_y_array = best_estimator.predict(X_testdata)
    test_performance_score = r2_score(Y_testdata, test_pre_y_array)

    return v_performance_score, test_performance_score



def rolling_model_PCR(df_X, df_Y):
    split_num = 200 * 60
    X_traindata = df_X[:split_num * 2]
    Y_traindata = df_Y[:split_num * 2]
    X_testdata = df_X[split_num * 2:split_num * 3]
    Y_testdata = df_Y[split_num * 2:split_num * 3]

    # specify parameters and distributions to sample from

    v_score = pd.DataFrame(index=range(1, 31, 1), columns=['Is_score', 'v_score', 'test_score'])

    for j in range(1, 30, 1):
        pca = PCA(n_components=j)
        X_reduced_train = pca.fit_transform(X_traindata)
        X_reduced_test = pca.transform(X_testdata)

        X_reduced_train = X_reduced_train[:, :j]
        X_reduced_test = X_reduced_test[:, :j]

        clf_OLS = LinearRegression(n_jobs=2)
        clf_OLS.fit(X_reduced_train, Y_traindata)
        # refit best model & fit

        train_pred = clf_OLS.predict(X_reduced_train[:split_num])
        Is_performance_score = r2_score(df_Y[:split_num], train_pred)
        v_score.iloc[j, 0] = Is_performance_score

        v_pred = clf_OLS.predict(X_reduced_train[split_num:split_num * 2])
        v_performance_score = r2_score(df_Y[split_num:split_num * 2], v_pred)
        v_score.iloc[j, 1] = v_performance_score

        test_pre_y_array = clf_OLS.predict(X_reduced_test)
        test_performance_score = r2_score(Y_testdata, test_pre_y_array)
        v_score.iloc[j, 2] = test_performance_score
    v_score.dropna(inplace=True)
    v_score.sort_values(by=['v_score'], inplace=True, ascending=False)
    v_performance_score = v_score.iloc[0,0]
    test_performance_score = v_score.iloc[0,2]
    return v_performance_score, test_performance_score

def rolling_model_PLS(df_X, df_Y):
    split_num = 200 * 60
    X_traindata = df_X[:split_num * 2]
    Y_traindata = df_Y[:split_num * 2]
    X_vdata = df_X[split_num:split_num * 2]
    X_testdata = df_X[split_num * 2:split_num * 3]
    Y_testdata = df_Y[split_num * 2:split_num * 3]

    # specify parameters and distributions to sample from

    num_valid_size = len(X_traindata) - len(X_vdata)
    test_fold = -1 * np.ones(len(X_traindata))
    test_fold[num_valid_size:] = 0
    ps = PredefinedSplit(test_fold)

    # specify parameters and distributions to sample from
    param_dist = {'n_components': sp_randint(1, 100), 'max_iter': sp_randint(50, len(X_traindata)),
                  'tol': [0.0001, 0.00001, 0.000001, 0.0000001]}
    # param_dist = {'n_components':[3,4]}
    PLS_model = PLSRegression(scale=False)

    # run gridsearchcv make_scorer(r2_score)
    n_iter_search = 50
    estim = RandomizedSearchCV(PLS_model, param_distributions=param_dist, scoring='r2',
                               cv=ps.split(), iid=False, n_jobs=1, n_iter=n_iter_search)

    estim.fit(X_traindata, Y_traindata)
    best_estimator = estim.best_estimator_
    v_pred = best_estimator.predict(df_X[:split_num])
    v_performance_score = r2_score(df_Y[:split_num], v_pred)
    test_pre_y_array = best_estimator.predict(X_testdata)
    test_performance_score = r2_score(Y_testdata, test_pre_y_array)

    return v_performance_score, test_performance_score


def rolling_model_Lasso(df_X, df_Y):
    split_num = 200 * 60
    X_traindata = df_X[:split_num * 2]
    Y_traindata = df_Y[:split_num * 2]
    X_vdata = df_X[split_num:split_num * 2]
    X_testdata = df_X[split_num * 2:split_num * 3]
    Y_testdata = df_Y[split_num * 2:split_num * 3]

    # specify parameters and distributions to sample from

    num_valid_size = len(X_traindata) - len(X_vdata)
    test_fold = -1 * np.ones(len(X_traindata))
    test_fold[num_valid_size:] = 0
    ps = PredefinedSplit(test_fold)

    # specify parameters and distributions to sample from
    param_dist = {'alpha': uniform(0.00001, 0.1),
                  'power_t': uniform(0.1, 0.9),
                  'l1_ratio': uniform(0.1, 0.9),
                  'eta0': uniform(0.00001, 0.1),
                  'epsilon': uniform(0.01, 0.9),
                  'max_iter': sp_randint(5, 10000),
                  'tol': [0.01, 0.001, 0.0001, 0.00001],
                  'fit_intercept': [True, False]}

    clf = SGDRegressor(shuffle=False, loss='squared_loss', penalty='l1', random_state=100)

    # run randomized search
    n_iter_search = 50
    estim = RandomizedSearchCV(clf, param_distributions=param_dist,
                               n_iter=n_iter_search, scoring='r2',
                               cv=ps.split(), iid=False,
                               random_state=100, n_jobs=1)

    estim.fit(X_traindata, Y_traindata)
    best_estimator = estim.best_estimator_
    best_coef = best_estimator.coef_

    v_pred = best_estimator.predict(df_X[:split_num])
    v_performance_score = r2_score(df_Y[:split_num], v_pred)
    test_pre_y_array = best_estimator.predict(X_testdata)
    test_performance_score = r2_score(Y_testdata, test_pre_y_array)

    return v_performance_score, test_performance_score, best_coef


def rolling_model_LassoH(df_X, df_Y):
    split_num = 200 * 60
    X_traindata = df_X[:split_num * 2]
    Y_traindata = df_Y[:split_num * 2]
    X_vdata = df_X[split_num:split_num * 2]
    X_testdata = df_X[split_num * 2:split_num * 3]
    Y_testdata = df_Y[split_num * 2:split_num * 3]

    # specify parameters and distributions to sample from

    num_valid_size = len(X_traindata) - len(X_vdata)
    test_fold = -1 * np.ones(len(X_traindata))
    test_fold[num_valid_size:] = 0
    ps = PredefinedSplit(test_fold)

    # specify parameters and distributions to sample from
    param_dist = {'alpha': uniform(0.00001, 0.1),
                  'power_t': uniform(0.1, 0.9),
                  'l1_ratio': uniform(0.1, 0.9),
                  'eta0': uniform(0.00001, 0.1),
                  'epsilon': uniform(0.01, 0.9),
                  'max_iter': sp_randint(5, 10000),
                  'tol': [0.01, 0.001, 0.0001, 0.00001],
                  'fit_intercept': [True, False]}

    clf = SGDRegressor(shuffle=False, loss='huber', penalty='l1', random_state=100)

    # run randomized search
    n_iter_search = 50
    estim = RandomizedSearchCV(clf, param_distributions=param_dist,
                               n_iter=n_iter_search, scoring='r2',
                               cv=ps.split(), iid=False,
                               random_state=100, n_jobs=1)

    estim.fit(X_traindata, Y_traindata)
    best_estimator = estim.best_estimator_
    best_coef = best_estimator.coef_
    v_pred = best_estimator.predict(df_X[:split_num])
    v_performance_score = r2_score(df_Y[:split_num], v_pred)
    test_pre_y_array = best_estimator.predict(X_testdata)
    test_performance_score = r2_score(Y_testdata, test_pre_y_array)

    return v_performance_score, test_performance_score, best_coef

def rolling_model_Ridge(df_X, df_Y):
    split_num = 200 * 60
    X_traindata = df_X[:split_num * 2]
    Y_traindata = df_Y[:split_num * 2]
    X_vdata = df_X[split_num:split_num * 2]
    X_testdata = df_X[split_num * 2:split_num * 3]
    Y_testdata = df_Y[split_num * 2:split_num * 3]

    # specify parameters and distributions to sample from

    num_valid_size = len(X_traindata) - len(X_vdata)
    test_fold = -1 * np.ones(len(X_traindata))
    test_fold[num_valid_size:] = 0
    ps = PredefinedSplit(test_fold)

    # specify parameters and distributions to sample from
    param_dist = {'alpha': uniform(0.00001, 0.1),
                  'power_t': uniform(0.1, 0.9),
                  'l1_ratio': uniform(0.1, 0.9),
                  'eta0': uniform(0.00001, 0.1),
                  'epsilon': uniform(0.01, 0.9),
                  'max_iter': sp_randint(5, 10000),
                  'tol': [0.01, 0.001, 0.0001, 0.00001],
                  'fit_intercept': [True, False]}

    clf = SGDRegressor(shuffle=False, loss='squared_loss', penalty='l2', random_state=100)

    # run randomized search
    n_iter_search = 50
    estim = RandomizedSearchCV(clf, param_distributions=param_dist,
                               n_iter=n_iter_search, scoring='r2',
                               cv=ps.split(), iid=False,
                               random_state=100, n_jobs=1)

    estim.fit(X_traindata, Y_traindata)
    best_estimator = estim.best_estimator_
    v_pred = best_estimator.predict(df_X[:split_num])
    v_performance_score = r2_score(df_Y[:split_num], v_pred)
    test_pre_y_array = best_estimator.predict(X_testdata)
    test_performance_score = r2_score(Y_testdata, test_pre_y_array)

    return v_performance_score, test_performance_score


def rolling_model_RidgeH(df_X, df_Y):
    split_num = 200 * 60
    X_traindata = df_X[:split_num * 2]
    Y_traindata = df_Y[:split_num * 2]
    X_vdata = df_X[split_num:split_num * 2]
    X_testdata = df_X[split_num * 2:split_num * 3]
    Y_testdata = df_Y[split_num * 2:split_num * 3]

    # specify parameters and distributions to sample from

    num_valid_size = len(X_traindata) - len(X_vdata)
    test_fold = -1 * np.ones(len(X_traindata))
    test_fold[num_valid_size:] = 0
    ps = PredefinedSplit(test_fold)

    # specify parameters and distributions to sample from
    param_dist = {'alpha': uniform(0.00001, 0.1),
                  'power_t': uniform(0.1, 0.9),
                  'l1_ratio': uniform(0.1, 0.9),
                  'eta0': uniform(0.00001, 0.1),
                  'epsilon': uniform(0.01, 0.9),
                  'max_iter': sp_randint(5, 10000),
                  'tol': [0.01, 0.001, 0.0001, 0.00001],
                  'fit_intercept': [True, False]}

    clf = SGDRegressor(shuffle=False, loss='huber', penalty='l2', random_state=100)

    # run randomized search
    n_iter_search = 50
    estim = RandomizedSearchCV(clf, param_distributions=param_dist,
                               n_iter=n_iter_search, scoring='r2',
                               cv=ps.split(), iid=False,
                               random_state=100, n_jobs=1)

    estim.fit(X_traindata, Y_traindata)
    best_estimator = estim.best_estimator_
    v_pred = best_estimator.predict(df_X[:split_num])
    v_performance_score = r2_score(df_Y[:split_num], v_pred)
    test_pre_y_array = best_estimator.predict(X_testdata)
    test_performance_score = r2_score(Y_testdata, test_pre_y_array)

    return v_performance_score, test_performance_score


def rolling_model_ENet(df_X, df_Y):
    split_num = 200 * 60
    X_traindata = df_X[:split_num * 2]
    Y_traindata = df_Y[:split_num * 2]
    X_vdata = df_X[split_num:split_num * 2]
    X_testdata = df_X[split_num * 2:split_num * 3]
    Y_testdata = df_Y[split_num * 2:split_num * 3]

    # specify parameters and distributions to sample from

    num_valid_size = len(X_traindata) - len(X_vdata)
    test_fold = -1 * np.ones(len(X_traindata))
    test_fold[num_valid_size:] = 0
    ps = PredefinedSplit(test_fold)

    # specify parameters and distributions to sample from
    param_dist = {'alpha': uniform(0.00001, 0.1),
                  'power_t': uniform(0.1, 0.9),
                  'l1_ratio': uniform(0.1, 0.9),
                  'eta0': uniform(0.00001, 0.1),
                  'epsilon': uniform(0.01, 0.9),
                  'max_iter': sp_randint(5, 10000),
                  'tol': [0.01, 0.001, 0.0001, 0.00001],
                  'fit_intercept': [True, False]}

    clf = SGDRegressor(shuffle=False, loss='squared_loss', penalty='elasticnet', random_state=100)

    # run randomized search
    n_iter_search = 50
    estim = RandomizedSearchCV(clf, param_distributions=param_dist,
                               n_iter=n_iter_search, scoring='r2',
                               cv=ps.split(), iid=False,
                               random_state=100, n_jobs=1)

    estim.fit(X_traindata, Y_traindata)
    best_estimator = estim.best_estimator_
    best_coef = best_estimator.coef_
    v_pred = best_estimator.predict(df_X[:split_num])
    v_performance_score = r2_score(df_Y[:split_num], v_pred)
    test_pre_y_array = best_estimator.predict(X_testdata)
    test_performance_score = r2_score(Y_testdata, test_pre_y_array)

    return v_performance_score, test_performance_score, best_coef


def rolling_model_ENetH(df_X, df_Y):
    split_num = 200 * 60
    X_traindata = df_X[:split_num * 2]
    Y_traindata = df_Y[:split_num * 2]
    X_vdata = df_X[split_num:split_num * 2]
    X_testdata = df_X[split_num * 2:split_num * 3]
    Y_testdata = df_Y[split_num * 2:split_num * 3]

    # specify parameters and distributions to sample from

    num_valid_size = len(X_traindata) - len(X_vdata)
    test_fold = -1 * np.ones(len(X_traindata))
    test_fold[num_valid_size:] = 0
    ps = PredefinedSplit(test_fold)

    # specify parameters and distributions to sample from
    param_dist = {'alpha': uniform(0.00001, 0.1),
                  'power_t': uniform(0.1, 0.9),
                  'l1_ratio': uniform(0.1, 0.9),
                  'eta0': uniform(0.00001, 0.1),
                  'epsilon': uniform(0.01, 0.9),
                  'max_iter': sp_randint(5, 10000),
                  'tol': [0.01, 0.001, 0.0001, 0.00001],
                  'fit_intercept': [True, False]}

    clf = SGDRegressor(shuffle=False, loss='huber', penalty='elasticnet', random_state=100)

    # run randomized search
    n_iter_search = 50
    estim = RandomizedSearchCV(clf, param_distributions=param_dist,
                               n_iter=n_iter_search, scoring='r2',
                               cv=ps.split(), iid=False,
                               random_state=100, n_jobs=1)

    estim.fit(X_traindata, Y_traindata)
    best_estimator = estim.best_estimator_
    best_coef = best_estimator.coef_
    v_pred = best_estimator.predict(df_X[:split_num])
    v_performance_score = r2_score(df_Y[:split_num], v_pred)
    test_pre_y_array = best_estimator.predict(X_testdata)
    test_performance_score = r2_score(Y_testdata, test_pre_y_array)

    return v_performance_score, test_performance_score, best_coef


def save_to_array(df,round_time):
    df[round_time,0] = IS_score
    df[round_time,1] = OOS_score
    df[round_time,2] = IS_score_b
    df[round_time,3] = OOS_score_b


tableA1_100 = np.zeros(shape=(11,4))
coef_100 = np.zeros(shape=(100,8))
import datetime
for m in range(1):
    starttime = datetime.datetime.now()
    tableA1_temp = np.zeros(shape=(11,4))
    coef_temp = np.zeros(shape=(100, 8))
    print(m)

    ################################define data
    X = get_mc_x_data(m=m)
    Y = get_mc_y_data(m=m)
    x_array = np.array(X)
    y1 = np.array(Y['ret1']).copy()
    x1_oracle = np.array(Y['gz1']).copy()
    y1_array = np.array(y1).reshape(-1, 1)
    x1_array_oracle = np.array(x1_oracle).reshape(-1, 1)

    y2 = np.array(Y['ret2']).copy()
    x2_oracle = np.array(Y['gz2']).copy()
    y2_array = np.array(y2).reshape(-1, 1)
    x2_array_oracle = np.array(x2_oracle).reshape(-1, 1)

    ################################Oracle

    IS_score, OOS_score = rolling_model_Oracle(df_X=x1_array_oracle, df_Y=y1_array)
    print('Oracle Model a IS:', IS_score * 100)
    print('Oracle Model a OOS:', OOS_score * 100)

    IS_score_b, OOS_score_b = rolling_model_Oracle(df_X=x2_array_oracle, df_Y=y2_array)
    print('Oracle Model a IS:', IS_score_b * 100)
    print('Oracle Model a OOS:', OOS_score_b * 100)

    save_to_array(df=tableA1_temp,round_time=0)
    save_result(i=0)

    IS_score, OOS_score = rolling_model_OLS(df_X=x_array, df_Y=y1_array)
    print('OLS Model a IS:', IS_score * 100)
    print('OLS Model a OOS:', OOS_score * 100)

    IS_score_b, OOS_score_b = rolling_model_OLS(df_X=x_array, df_Y=y2_array)
    print('OLS Model b IS:', IS_score_b * 100)
    print('OLS Model b OOS:', OOS_score_b * 100)

    save_to_array(df=tableA1_temp, round_time=1)
    save_result(i=1)


    ################################OLS+H Model

    IS_score, OOS_score = rolling_model_OLSH(df_X=x_array, df_Y=y1)
    print('OLS+H Model a IS:', IS_score * 100)
    print('OLS+H Model a OOS:', OOS_score * 100)

    IS_score_b, OOS_score_b = rolling_model_OLSH(df_X=x_array, df_Y=y2)
    print('OLS+H Model b IS:', IS_score_b * 100)
    print('OLS+H Model b OOS:', OOS_score_b * 100)

    save_to_array(df=tableA1_temp, round_time=2)
    save_result(i=2)

    ################################PCR Model




    IS_score, OOS_score = rolling_model_PCR(df_X=x_array, df_Y=y1)
    print('PCR Model a IS:', IS_score * 100)
    print('PCR Model a OOS:', OOS_score * 100)

    IS_score_b, OOS_score_b = rolling_model_PCR(df_X=x_array, df_Y=y2)
    print('PCR Model b IS:', IS_score_b * 100)
    print('PCR Model b OOS:', OOS_score_b * 100)

    save_to_array(df=tableA1_temp, round_time=3)
    save_result(i=3)


    ################################PLS Model




    IS_score, OOS_score = rolling_model_PLS(df_X=x_array, df_Y=y1)
    print('PLS Model a IS:', IS_score * 100)
    print('PLS Model a OOS:', OOS_score * 100)

    IS_score_b, OOS_score_b = rolling_model_PLS(df_X=x_array, df_Y=y2)
    print('PLS Model b IS:', IS_score_b * 100)
    print('PLS Model b OOS:', OOS_score_b * 100)
    save_to_array(df=tableA1_temp, round_time=4)
    save_result(i=4)


    ################################Lasso Model


    IS_score, OOS_score, best_coef = rolling_model_Lasso(df_X=x_array, df_Y=y1)
    print('Lasso Model a IS:', IS_score * 100)
    print('Lasso Model a OOS:', OOS_score * 100)

    IS_score_b, OOS_score_b, best_coef_b = rolling_model_Lasso(df_X=x_array, df_Y=y2)
    print('Lasso Model b IS:', IS_score_b * 100)
    print('Lasso Model b OOS:', OOS_score_b * 100)
    save_to_array(df=tableA1_temp, round_time=5)
    save_result(i=5)
    coef_temp[:,0] = best_coef
    coef_temp[:,1] = best_coef_b
    ################################Lasso+H Model


    IS_score, OOS_score, best_coef = rolling_model_LassoH(df_X=x_array, df_Y=y1)
    print('LassoH Model a IS:', IS_score * 100)
    print('LassoH Model a OOS:', OOS_score * 100)

    IS_score_b, OOS_score_b, best_coef_b = rolling_model_LassoH(df_X=x_array, df_Y=y2)
    print('LassoH Model b IS:', IS_score_b * 100)
    print('LassoH Model b OOS:', OOS_score_b * 100)
    save_to_array(df=tableA1_temp, round_time=6)
    save_result(i=6)
    coef_temp[:,2] = best_coef
    coef_temp[:,3] = best_coef_b
    ################################Ridge Model


    IS_score, OOS_score = rolling_model_Ridge(df_X=x_array, df_Y=y1)
    print('OLS+H Model a IS:', IS_score * 100)
    print('OLS+H Model a OOS:', OOS_score * 100)

    IS_score_b, OOS_score_b = rolling_model_Ridge(df_X=x_array, df_Y=y2)
    print('OLS+H Model b IS:', IS_score_b * 100)
    print('OLS+H Model b OOS:', OOS_score_b * 100)
    save_to_array(df=tableA1_temp, round_time=7)
    save_result(i=7)


    ################################Ridge+H Model



    IS_score, OOS_score = rolling_model_RidgeH(df_X=x_array, df_Y=y1)
    print('Ridge Model a IS:', IS_score * 100)
    print('Ridge Model a OOS:', OOS_score * 100)

    IS_score_b, OOS_score_b = rolling_model_RidgeH(df_X=x_array, df_Y=y2)
    print('Ridge Model b IS:', IS_score_b * 100)
    print('Ridge Model b OOS:', OOS_score_b * 100)
    save_to_array(df=tableA1_temp, round_time=8)
    save_result(i=8)


    ################################ENet Model



    IS_score, OOS_score, best_coef = rolling_model_ENet(df_X=x_array, df_Y=y1)
    print('ENet Model a IS:', IS_score * 100)
    print('ENet Model a OOS:', OOS_score * 100)

    IS_score_b, OOS_score_b, best_coef_b = rolling_model_ENet(df_X=x_array, df_Y=y2)
    print('ENet Model b IS:', IS_score_b * 100)
    print('ENet Model b OOS:', OOS_score_b * 100)
    save_to_array(df=tableA1_temp, round_time=9)
    save_result(i=9)
    coef_temp[:,4] = best_coef
    coef_temp[:,5] = best_coef_b

    ################################ENet+H Model



    IS_score, OOS_score, best_coef = rolling_model_ENetH(df_X=x_array, df_Y=y1)
    print('ENetH Model a IS:', IS_score * 100)
    print('ENetH Model a OOS:', OOS_score * 100)

    IS_score_b, OOS_score_b, best_coef_b = rolling_model_ENetH(df_X=x_array, df_Y=y2)
    print('ENetH Model b IS:', IS_score_b * 100)
    print('ENetH Model b OOS:', OOS_score_b * 100)
    save_to_array(df=tableA1_temp, round_time=10)
    save_result(i=10)
    coef_temp[:,6] = best_coef
    coef_temp[:,7] = best_coef_b
    ################################save result

    tableA1.to_csv(Path + '//output//tableA1' + str(m) + '.csv')
    tableA1_100 = tableA1_100 + tableA1_temp


    coef_100 = coef_100+coef_temp
    endtime = datetime.datetime.now()
    print ('time_cost',(endtime - starttime).seconds)


for h in range(len(coef_temp)):
    for w in range(len(coef_temp[0])):
        if coef_temp[h,w] != 0:
            coef_temp[h, w] = 1
