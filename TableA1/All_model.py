import random as rn
import os
import numpy as np

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(123)
import pandas as pd
import pickle

from sklearn.metrics import r2_score

Path = 'D:\\APViaML\\MCS_code\\MCScode'


def get_mc_x_data():
    file = open(Path + '\\data\\mcs_demo_x_100.pkl', 'rb')
    raw_data = pickle.load(file)
    file.close()
    return raw_data


def get_mc_y_data():
    file = open(Path + '\\data\\mcs_demo_y_100.pkl', 'rb')
    raw_data = pickle.load(file)
    file.close()
    return raw_data


from sklearn.linear_model import LinearRegression

################################define table
columns_list = ['IS_a_50', 'OOS_a_50', 'IS_b_50', 'OOS_b_50']
index_list = ['Oracle', 'OLS', 'OLS_H', 'PCR', 'PLS', 'Lasso', 'Lasso_H',
              'Ridge', 'Ridge_H', 'ENet', 'ENet_H', 'RF', 'GBRT', 'GBRT_H',
              'NN1', 'NN2', 'NN3', 'NN4', 'NN5']
tableA1 = pd.DataFrame(columns=columns_list, index=index_list)

################################define data
X = get_mc_x_data()
Y = get_mc_y_data()

x_array = np.array(X)
y1 = np.array(Y['ret1']).copy()
x1_oracle = np.array(Y['gz1']).copy()
y1_array = np.array(y1).reshape(-1, 1)
x1_array_oracle = np.array(x1_oracle).reshape(-1, 1)

y2 = np.array(Y['ret2']).copy()
x2_oracle = np.array(Y['gz2']).copy()
y2_array = np.array(y2).reshape(-1, 1)
x2_array_oracle = np.array(x2_oracle).reshape(-1, 1)


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


IS_score, OOS_score = rolling_model_Oracle(df_X=x1_array_oracle, df_Y=y1_array)
print('Oracle Model a IS:', IS_score * 100)
print('Oracle Model a OOS:', OOS_score * 100)

IS_score_b, OOS_score_b = rolling_model_Oracle(df_X=x2_array_oracle, df_Y=y2_array)
print('Oracle Model a IS:', IS_score_b * 100)
print('Oracle Model a OOS:', OOS_score_b * 100)


## save the result
def save_result(i):
    tableA1.iloc[i, 0] = round(IS_score * 100, 3)
    tableA1.iloc[i, 1] = round(OOS_score * 100, 3)
    tableA1.iloc[i, 2] = round(IS_score_b * 100, 3)
    tableA1.iloc[i, 3] = round(OOS_score_b * 100, 3)
save_result(i=0)


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


IS_score, OOS_score = rolling_model_OLS(df_X=x_array, df_Y=y1_array)
print('OLS Model a IS:', IS_score * 100)
print('OLS Model a OOS:', OOS_score * 100)

IS_score_b, OOS_score_b = rolling_model_OLS(df_X=x_array, df_Y=y2_array)
print('OLS Model b IS:', IS_score_b * 100)
print('OLS Model b OOS:', OOS_score_b * 100)

save_result(i=1)

################################OLS+H Model


from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from scipy.stats import randint as sp_randint


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


IS_score, OOS_score = rolling_model_OLSH(df_X=x_array, df_Y=y1)
print('OLS+H Model a IS:', IS_score * 100)
print('OLS+H Model a OOS:', OOS_score * 100)

IS_score_b, OOS_score_b = rolling_model_OLSH(df_X=x_array, df_Y=y2)
print('OLS+H Model b IS:', IS_score_b * 100)
print('OLS+H Model b OOS:', OOS_score_b * 100)

save_result(i=2)

################################PCR Model
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression




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


IS_score, OOS_score = rolling_model_PCR(df_X=x_array, df_Y=y1)
print('PCR Model a IS:', IS_score * 100)
print('PCR Model a OOS:', OOS_score * 100)

IS_score_b, OOS_score_b = rolling_model_PCR(df_X=x_array, df_Y=y2)
print('PCR Model b IS:', IS_score_b * 100)
print('PCR Model b OOS:', OOS_score_b * 100)

save_result(i=3)


################################PLS Model


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
    n_iter_search = 100
    estim = RandomizedSearchCV(PLS_model, param_distributions=param_dist, scoring='r2',
                               cv=ps.split(), iid=False, n_jobs=1, n_iter=n_iter_search)

    estim.fit(X_traindata, Y_traindata)
    best_estimator = estim.best_estimator_
    v_pred = best_estimator.predict(df_X[:split_num])
    v_performance_score = r2_score(df_Y[:split_num], v_pred)
    test_pre_y_array = best_estimator.predict(X_testdata)
    test_performance_score = r2_score(Y_testdata, test_pre_y_array)

    return v_performance_score, test_performance_score


IS_score, OOS_score = rolling_model_PLS(df_X=x_array, df_Y=y1)
print('PLS Model a IS:', IS_score * 100)
print('PLS Model a OOS:', OOS_score * 100)

IS_score_b, OOS_score_b = rolling_model_PLS(df_X=x_array, df_Y=y2)
print('PLS Model b IS:', IS_score_b * 100)
print('PLS Model b OOS:', OOS_score_b * 100)

save_result(i=4)


################################Lasso Model

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
    n_iter_search = 100
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


IS_score, OOS_score = rolling_model_Lasso(df_X=x_array, df_Y=y1)
print('Lasso Model a IS:', IS_score * 100)
print('Lasso Model a OOS:', OOS_score * 100)

IS_score_b, OOS_score_b = rolling_model_Lasso(df_X=x_array, df_Y=y2)
print('Lasso Model b IS:', IS_score_b * 100)
print('Lasso Model b OOS:', OOS_score_b * 100)

save_result(i=5)
################################Lasso+H Model

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
    n_iter_search = 100
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

IS_score, OOS_score = rolling_model_LassoH(df_X=x_array, df_Y=y1)
print('LassoH Model a IS:', IS_score * 100)
print('LassoH Model a OOS:', OOS_score * 100)

IS_score_b, OOS_score_b = rolling_model_LassoH(df_X=x_array, df_Y=y2)
print('LassoH Model b IS:', IS_score_b * 100)
print('LassoH Model b OOS:', OOS_score_b * 100)

save_result(i=6)

################################Ridge Model

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
    n_iter_search = 100
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


IS_score, OOS_score = rolling_model_Ridge(df_X=x_array, df_Y=y1)
print('OLS+H Model a IS:', IS_score * 100)
print('OLS+H Model a OOS:', OOS_score * 100)

IS_score_b, OOS_score_b = rolling_model_Ridge(df_X=x_array, df_Y=y2)
print('OLS+H Model b IS:', IS_score_b * 100)
print('OLS+H Model b OOS:', OOS_score_b * 100)

save_result(i=7)


################################Ridge+H Model

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
    n_iter_search = 100
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


IS_score, OOS_score = rolling_model_RidgeH(df_X=x_array, df_Y=y1)
print('Ridge Model a IS:', IS_score * 100)
print('Ridge Model a OOS:', OOS_score * 100)

IS_score_b, OOS_score_b = rolling_model_RidgeH(df_X=x_array, df_Y=y2)
print('Ridge Model b IS:', IS_score_b * 100)
print('Ridge Model b OOS:', OOS_score_b * 100)

save_result(i=8)


################################ENet Model

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
    n_iter_search = 100
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


IS_score, OOS_score = rolling_model_ENet(df_X=x_array, df_Y=y1)
print('ENet Model a IS:', IS_score * 100)
print('ENet Model a OOS:', OOS_score * 100)

IS_score_b, OOS_score_b = rolling_model_ENet(df_X=x_array, df_Y=y2)
print('ENet Model b IS:', IS_score_b * 100)
print('ENet Model b OOS:', OOS_score_b * 100)

save_result(i=9)


################################ENet+H Model

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
    n_iter_search = 100
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


IS_score, OOS_score = rolling_model_ENetH(df_X=x_array, df_Y=y1)
print('ENetH Model a IS:', IS_score * 100)
print('ENetH Model a OOS:', OOS_score * 100)

IS_score_b, OOS_score_b = rolling_model_ENetH(df_X=x_array, df_Y=y2)
print('ENetH Model b IS:', IS_score_b * 100)
print('ENetH Model b OOS:', OOS_score_b * 100)

save_result(i=10)

################################RF Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


def rolling_model_RF(df_X, df_Y):
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
    param_dist = {"max_features": sp_randint(10, 100),
                  "max_depth": sp_randint(3, 10),
                  "min_samples_split": sp_randint(100, 1000),
                  "min_samples_leaf": sp_randint(100, 1000),
                  "n_estimators": sp_randint(10, 100),
                  "oob_score": [True, False]
                  }

    clf_RF = RandomForestRegressor(random_state=100)

    # run randomized search
    n_iter_search = 100
    estim = RandomizedSearchCV(clf_RF, param_distributions=param_dist,
                               n_iter=n_iter_search, scoring='r2',
                               cv=ps.split(), iid=False, random_state=100)

    estim.fit(X_traindata, Y_traindata)
    best_estimator = estim.best_estimator_
    v_pred = best_estimator.predict(df_X[:split_num])
    v_performance_score = r2_score(df_Y[:split_num], v_pred)
    test_pre_y_array = best_estimator.predict(X_testdata)
    test_performance_score = r2_score(Y_testdata, test_pre_y_array)

    return v_performance_score, test_performance_score


IS_score, OOS_score = rolling_model_RF(df_X=x_array, df_Y=y1)
print('RF Model a IS:', IS_score * 100)
print('RF Model a OOS:', OOS_score * 100)

IS_score_b, OOS_score_b = rolling_model_RF(df_X=x_array, df_Y=y2)
print('RF Model b IS:', IS_score_b * 100)
print('RF Model b OOS:', OOS_score_b * 100)

save_result(i=11)

################################GBRT Model

def rolling_model_GBRT(df_X, df_Y):
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
    param_dist = {"max_features": sp_randint(5, 100),
                  "max_depth": sp_randint(3, 12),
                  "min_samples_split": sp_randint(10, 1000),
                  "min_samples_leaf": sp_randint(10, 1000),
                  "n_estimators": sp_randint(3, 100),
                  "learning_rate": uniform(0.001, 0.1),
                  "subsample": uniform(0.6, 0.4)
                  }

    clf_GBRT = GradientBoostingRegressor(loss='ls', random_state=100)

    # run randomized search
    n_iter_search = 100
    estim = RandomizedSearchCV(clf_GBRT, param_distributions=param_dist,
                               n_iter=n_iter_search, scoring='r2',
                               cv=ps.split(), iid=False, random_state=100)

    estim.fit(X_traindata, Y_traindata)
    best_estimator = estim.best_estimator_
    v_pred = best_estimator.predict(df_X[:split_num])
    v_performance_score = r2_score(df_Y[:split_num], v_pred)
    test_pre_y_array = best_estimator.predict(X_testdata)
    test_performance_score = r2_score(Y_testdata, test_pre_y_array)

    return v_performance_score, test_performance_score


IS_score, OOS_score = rolling_model_GBRT(df_X=x_array, df_Y=y1)
print('GBRT Model a IS:', IS_score * 100)
print('GBRT Model a OOS:', OOS_score * 100)

IS_score_b, OOS_score_b = rolling_model_GBRT(df_X=x_array, df_Y=y2)
print('GBRT Model b IS:', IS_score_b * 100)
print('GBRT Model b OOS:', OOS_score_b * 100)

save_result(i=12)


################################GBRTH Model


def rolling_model_GBRTH(df_X, df_Y):
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
    param_dist = {"max_features": sp_randint(5, 100),
                  "max_depth": sp_randint(3, 12),
                  "min_samples_split": sp_randint(100, 1000),
                  "min_samples_leaf": sp_randint(100, 1000),
                  "n_estimators": sp_randint(5, 100),
                  "learning_rate": uniform(0.001, 0.1),
                  "subsample": uniform(0.6, 0.4)
                  }

    clf_GBRT = GradientBoostingRegressor(loss='huber', random_state=100)

    # run randomized search
    n_iter_search = 100
    estim = RandomizedSearchCV(clf_GBRT, param_distributions=param_dist,
                               n_iter=n_iter_search, scoring='r2',
                               cv=ps.split(), iid=False, random_state=100)

    estim.fit(X_traindata, Y_traindata)
    best_estimator = estim.best_estimator_
    v_pred = best_estimator.predict(df_X[:split_num])
    v_performance_score = r2_score(df_Y[:split_num], v_pred)
    test_pre_y_array = best_estimator.predict(X_testdata)
    test_performance_score = r2_score(Y_testdata, test_pre_y_array)

    return v_performance_score, test_performance_score


IS_score, OOS_score = rolling_model_GBRTH(df_X=x_array, df_Y=y1)
print('GBRT+H Model a IS:', IS_score * 100)
print('GBRT+H Model a OOS:', OOS_score * 100)

IS_score_b, OOS_score_b = rolling_model_GBRTH(df_X=x_array, df_Y=y2)
print('GBRT+H Model b IS:', IS_score_b * 100)
print('GBRT+H Model b OOS:', OOS_score_b * 100)

save_result(i=13)


################################NN1 Model

import tensorflow as tf
from keras import backend as K

K.clear_session()
tf.reset_default_graph()
tf.set_random_seed(1234)
import gc
import pandas as pd
import datetime
import pickle
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras import initializers

from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


def creat_data(df_X, df_Y):
    '''
    Data providing function:

    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    '''
    split_num = 200 * 60
    X_traindata = df_X[:split_num * 2]
    Y_traindata = df_Y[:split_num * 2]
    X_vdata = df_X[split_num:split_num * 2]
    Y_vdata = df_Y[split_num:split_num * 2]
    X_testdata = df_X[split_num * 2:split_num * 3]
    Y_testdata = df_Y[split_num * 2:split_num * 3]

    return X_traindata, Y_traindata, X_vdata, Y_vdata, X_testdata, Y_testdata


space = {'ll_float': hp.uniform('ll_float', 0.01, 0.2),
         'lr': hp.loguniform('lr', np.log(0.005), np.log(0.2)),
         'beta_1_float': hp.uniform('beta_1_float', 0.8, 0.95),
         'beta_2_float': hp.uniform('beta_2_float', 0.98, 0.9999),
         'epsilon_float': hp.uniform('epsilon_float', 1e-09, 1e-07),  ##note
         'batch_size': hp.quniform('batch_size', 5, 500, 1),
         'epochs': hp.quniform('epochs', 20, 50, 1)
         }

## set params random search time,when set 50,something will be wrong
try_num1 = int(50)


X_traindata, Y_traindata, X_vdata, Y_vdata, X_testdata, Y_testdata = creat_data(df_X=x_array, df_Y=y1)

# define NN1
def f_NN1(params):
    split_num = 200 * 60
    ## define params
    ll_float = params["ll_float"]  # 0.1
    learn_rate_float = params["lr"]  # 0.01
    beta_1_float = params["beta_1_float"]  # 0.9
    beta_2_float = params["beta_2_float"]  # 0.999
    epsilon_float = params["epsilon_float"]  # 1e-08
    batch_size_num = params['batch_size']  #
    epochs_num = params['epochs']  # 50
    ## model structure
    model_NN1 = Sequential()
    init = initializers.he_normal(seed=100)
    model_NN1.add(Dense(32, input_dim=len(X_traindata[0]),
                        activation='relu',
                        kernel_initializer=init,
                        kernel_regularizer=regularizers.l1(ll_float)))
    model_NN1.add(Dense(1))

    ## comile model
    adam = Adam(lr=learn_rate_float, beta_1=beta_1_float, beta_2=beta_2_float, epsilon=epsilon_float)
    model_NN1.compile(loss='mse', optimizer=adam)

    ## callback fun
    early_stopping = EarlyStopping(monitor='loss', min_delta=0.00001,
                                   patience=3, verbose=0, mode='auto')
    model_filepath = Path + '\\model\\best_weights.h5'
    checkpoint = ModelCheckpoint(filepath=model_filepath, save_weights_only=False,
                                 monitor='loss', mode='min', save_best_only='True')
    callback_lists = [early_stopping, checkpoint]

    ## fit model
    model_NN1.fit(X_traindata, Y_traindata,
                  batch_size=int(batch_size_num),
                  epochs=int(epochs_num),
                  verbose=0,
                  validation_data=(X_vdata, Y_vdata),
                  callbacks=callback_lists,
                  shuffle=False)

    ##get the best model
    best_model = load_model(model_filepath)

    Y_pre_t = best_model.predict(X_traindata[:split_num], verbose=0)
    Y_pre_tlist = []
    for x in Y_pre_t[:, 0]:
        Y_pre_tlist.append(x)

    train_score = r2_score(Y_traindata[:split_num],Y_pre_tlist)


    Y_pre_v = best_model.predict(X_vdata, verbose=0)
    Y_pre_vlist = []
    for x in Y_pre_v[:, 0]:
        Y_pre_vlist.append(x)

    v_score = r2_score(Y_vdata, Y_pre_vlist)

    ## prediction & save
    Y_pre = best_model.predict(X_testdata, verbose=0)
    Y_pre_list = []
    for x in Y_pre[:, 0]:
        Y_pre_list.append(x)
    test_score = r2_score(Y_testdata, Y_pre_list)
    print('Validate score:', v_score)
    print('OOS score:',test_score)
    K.clear_session()

    return {'loss': -v_score, 'status': STATUS_OK,
            'train_score': train_score, 'test_score': test_score}


trials = Trials()
best = fmin(f_NN1, space, algo=tpe.suggest, max_evals=try_num1, trials=trials)


loss_list = trials.losses()
min_loss = min(loss_list)
for k in range(try_num1):
    if min_loss == loss_list[k]:
        key = k
best_results = trials.results[key]

# save model


IS_score = best_results['train_score']
OOS_score = best_results['test_score']

print('NN1 Model a IS:', IS_score * 100)
print('NN1 Model a OOS:', OOS_score * 100)

K.clear_session()
tf.reset_default_graph()


X_traindata, Y_traindata, X_vdata, Y_vdata, X_testdata, Y_testdata = creat_data(df_X=x_array, df_Y=y2)



trials = Trials()
best = fmin(f_NN1, space, algo=tpe.suggest, max_evals=try_num1, trials=trials)


loss_list = trials.losses()
min_loss = min(loss_list)
for k in range(try_num1):
    if min_loss == loss_list[k]:
        key = k
best_results = trials.results[key]

# save model


IS_score_b = best_results['train_score']
OOS_score_b = best_results['test_score']

print('NN1 Model b IS:', IS_score_b * 100)
print('NN1 Model b OOS:', OOS_score_b * 100)

K.clear_session()
tf.reset_default_graph()


tableA1.to_csv(Path + '//output//tableA1.csv')
