import random as rn
import os
import numpy as np

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(123)
import pandas as pd
import pickle

from sklearn.metrics import r2_score
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

x_array = np.array(X)*100
y1 = np.array(Y['ret1']).copy()*100
x1_oracle = np.array(Y['gz1']).copy()
y1_array = np.array(y1).reshape(-1, 1)
x1_array_oracle = np.array(x1_oracle).reshape(-1, 1)

y2 = np.array(Y['ret2']).copy()*100
x2_oracle = np.array(Y['gz2']).copy()
y2_array = np.array(y2).reshape(-1, 1)
x2_array_oracle = np.array(x2_oracle).reshape(-1, 1)




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

    IS_score, OOS_score = rolling_model_RF(df_X=x_array, df_Y=y1)
    print('RF Model a IS:', IS_score * 100)
    print('RF Model a OOS:', OOS_score * 100)

    IS_score_b, OOS_score_b = rolling_model_RF(df_X=x_array, df_Y=y2)
    print('RF Model b IS:', IS_score_b * 100)
    print('RF Model b OOS:', OOS_score_b * 100)
    save_to_array(df=tableA1_temp, round_time=11)
    save_result(i=11)

    ################################GBRT Model

    IS_score, OOS_score = rolling_model_GBRT(df_X=x_array, df_Y=y1)
    print('GBRT Model a IS:', IS_score * 100)
    print('GBRT Model a OOS:', OOS_score * 100)

    IS_score_b, OOS_score_b = rolling_model_GBRT(df_X=x_array, df_Y=y2)
    print('GBRT Model b IS:', IS_score_b * 100)
    print('GBRT Model b OOS:', OOS_score_b * 100)
    save_to_array(df=tableA1_temp, round_time=12)
    save_result(i=12)
    ################################GBRTH Model

    IS_score, OOS_score = rolling_model_GBRTH(df_X=x_array, df_Y=y1)
    print('GBRT+H Model a IS:', IS_score * 100)
    print('GBRT+H Model a OOS:', OOS_score * 100)

    IS_score_b, OOS_score_b = rolling_model_GBRTH(df_X=x_array, df_Y=y2)
    print('GBRT+H Model b IS:', IS_score_b * 100)
    print('GBRT+H Model b OOS:', OOS_score_b * 100)
    save_to_array(df=tableA1_temp, round_time=13)
    save_result(i=13)




























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

















# ################################NN1 Model
space = {'ll_float': hp.uniform('ll_float', 0.01, 0.3),
         'lr': hp.loguniform('lr', np.log(0.008), np.log(0.05)),
         'beta_1_float': hp.uniform('beta_1_float', 0.8, 0.95),
         'beta_2_float': hp.uniform('beta_2_float', 0.98, 0.9999),
         'epsilon_float': hp.uniform('epsilon_float', 1e-09, 1e-07),  ##note
         'batch_size': hp.quniform('batch_size', 300, 500, 1),
         'epochs': hp.quniform('epochs',500, 600, 1)
         }

## set params random search time,when set 50,something will be wrong
try_num1 = int(30)

X_traindata, Y_traindata, X_vdata, Y_vdata, X_testdata, Y_testdata = creat_data(df_X=x_array, df_Y=y1)

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
    model_NN1.add(BatchNormalization())
    model_NN1.add(Dense(1))

    ## comile model
    adam = Adam(lr=learn_rate_float, beta_1=beta_1_float, beta_2=beta_2_float, epsilon=epsilon_float)
    model_NN1.compile(loss='mse', optimizer=adam)

    ## callback fun
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001,
                                   patience=3, verbose=0, mode='auto')
    model_filepath = Path + '\\model\\best_weights.h5'
    checkpoint = ModelCheckpoint(filepath=model_filepath, save_weights_only=False,
                                 monitor='val_loss', mode='min', save_best_only='True')
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

    K.clear_session()

    return {'loss': -v_score, 'status': STATUS_OK,
            'train_score': train_score, 'test_score': test_score}

# ################################NN2 Model

def f_NN2(params):
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
    model_NN2 = Sequential()
    init = initializers.he_normal(seed=100)
    model_NN2.add(Dense(32, input_dim=len(X_traindata[0]),
                        activation='relu',
                        kernel_initializer=init,
                        kernel_regularizer=regularizers.l1(ll_float)))
    model_NN2.add(BatchNormalization())
    model_NN2.add(Dense(16, 
                        activation='relu',
                        kernel_initializer=init,
                        kernel_regularizer=regularizers.l1(ll_float)))
    model_NN2.add(BatchNormalization())
    
    model_NN2.add(Dense(1))

    ## comile model
    adam = Adam(lr=learn_rate_float, beta_1=beta_1_float, beta_2=beta_2_float, epsilon=epsilon_float)
    model_NN2.compile(loss='mse', optimizer=adam)

    ## callback fun
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001,
                                   patience=3, verbose=0, mode='auto')
    model_filepath = Path + '\\model\\best_weights.h5'
    checkpoint = ModelCheckpoint(filepath=model_filepath, save_weights_only=False,
                                 monitor='val_loss', mode='min', save_best_only='True')
    callback_lists = [early_stopping, checkpoint]

    ## fit model
    model_NN2.fit(X_traindata, Y_traindata,
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

    K.clear_session()

    return {'loss': -v_score, 'status': STATUS_OK,
            'train_score': train_score, 'test_score': test_score}


def f_NN3(params):
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
    model_NN3 = Sequential()
    init = initializers.he_normal(seed=100)
    model_NN3.add(Dense(32, input_dim=len(X_traindata[0]),
                        activation='relu',
                        kernel_initializer=init,
                        kernel_regularizer=regularizers.l1(ll_float)))
    model_NN3.add(BatchNormalization())
    model_NN3.add(Dense(16,
                        activation='relu',
                        kernel_initializer=init,
                        kernel_regularizer=regularizers.l1(ll_float)))

    model_NN3.add(Dense(8,
                        activation='relu',
                        kernel_initializer=init,
                        kernel_regularizer=regularizers.l1(ll_float)))
    
    model_NN3.add(Dense(1))

    ## comile model
    adam = Adam(lr=learn_rate_float, beta_1=beta_1_float, beta_2=beta_2_float, epsilon=epsilon_float)
    model_NN3.compile(loss='mse', optimizer=adam)

    ## callback fun
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001,
                                   patience=3, verbose=0, mode='auto')
    model_filepath = Path + '\\model\\best_weights.h5'
    checkpoint = ModelCheckpoint(filepath=model_filepath, save_weights_only=False,
                                 monitor='val_loss', mode='min', save_best_only='True')
    callback_lists = [early_stopping, checkpoint]

    ## fit model
    model_NN3.fit(X_traindata, Y_traindata,
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

    train_score = r2_score(Y_traindata[:split_num], Y_pre_tlist)

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

    K.clear_session()

    return {'loss': -v_score, 'status': STATUS_OK,
            'train_score': train_score, 'test_score': test_score}


def f_NN4(params):
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
    model_NN4 = Sequential()
    init = initializers.he_normal(seed=100)
    model_NN4.add(Dense(32, input_dim=len(X_traindata[0]),
                        activation='relu',
                        kernel_initializer=init,
                        kernel_regularizer=regularizers.l1(ll_float)))
    model_NN4.add(BatchNormalization())
    model_NN4.add(Dense(16,
                        activation='relu',
                        kernel_initializer=init,
                        kernel_regularizer=regularizers.l1(ll_float)))
    model_NN4.add(BatchNormalization())
    model_NN4.add(Dense(8,
                        activation='relu',
                        kernel_initializer=init,
                        kernel_regularizer=regularizers.l1(ll_float)))
    model_NN4.add(BatchNormalization())
    model_NN4.add(Dense(4,
                        activation='relu',
                        kernel_initializer=init,
                        kernel_regularizer=regularizers.l1(ll_float)))
    model_NN4.add(BatchNormalization())
    
    model_NN4.add(Dense(1))

    ## comile model
    adam = Adam(lr=learn_rate_float, beta_1=beta_1_float, beta_2=beta_2_float, epsilon=epsilon_float)
    model_NN4.compile(loss='mse', optimizer=adam)

    ## callback fun
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001,
                                   patience=3, verbose=0, mode='auto')
    model_filepath = Path + '\\model\\best_weights.h5'
    checkpoint = ModelCheckpoint(filepath=model_filepath, save_weights_only=False,
                                 monitor='val_loss', mode='min', save_best_only='True')
    callback_lists = [early_stopping, checkpoint]

    ## fit model
    model_NN4.fit(X_traindata, Y_traindata,
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

    train_score = r2_score(Y_traindata[:split_num], Y_pre_tlist)

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

    K.clear_session()

    return {'loss': -v_score, 'status': STATUS_OK,
            'train_score': train_score, 'test_score': test_score}


def f_NN5(params):
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
    model_NN5 = Sequential()
    init = initializers.he_normal(seed=100)
    model_NN5.add(Dense(32, input_dim=len(X_traindata[0]),
                        activation='relu',
                        kernel_initializer=init,
                        kernel_regularizer=regularizers.l1(ll_float)))
    model_NN5.add(BatchNormalization())
    model_NN5.add(Dense(16,
                        activation='relu',
                        kernel_initializer=init,
                        kernel_regularizer=regularizers.l1(ll_float)))
    model_NN5.add(BatchNormalization())
    model_NN5.add(Dense(8,
                        activation='relu',
                        kernel_initializer=init,
                        kernel_regularizer=regularizers.l1(ll_float)))
    model_NN5.add(BatchNormalization())
    model_NN5.add(Dense(4,
                        activation='relu',
                        kernel_initializer=init,
                        kernel_regularizer=regularizers.l1(ll_float)))
    model_NN5.add(BatchNormalization())
    model_NN5.add(Dense(2,
                        activation='relu',
                        kernel_initializer=init,
                        kernel_regularizer=regularizers.l1(ll_float)))
    model_NN5.add(BatchNormalization())
    model_NN5.add(Dense(1))

    ## comile model
    adam = Adam(lr=learn_rate_float, beta_1=beta_1_float, beta_2=beta_2_float, epsilon=epsilon_float)
    model_NN5.compile(loss='mse', optimizer=adam)

    ## callback fun
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001,
                                   patience=3, verbose=0, mode='auto')
    model_filepath = Path + '\\model\\best_weights.h5'
    checkpoint = ModelCheckpoint(filepath=model_filepath, save_weights_only=False,
                                 monitor='val_loss', mode='min', save_best_only='True')
    callback_lists = [early_stopping, checkpoint]

    ## fit model
    model_NN5.fit(X_traindata, Y_traindata,
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

    train_score = r2_score(Y_traindata[:split_num], Y_pre_tlist)

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

    K.clear_session()

    return {'loss': -v_score, 'status': STATUS_OK,
            'train_score': train_score, 'test_score': test_score}

NN_result = np.zeros(shape=(5,4))


##############################################NN1
trials = Trials()
best = fmin(f_NN1, space, algo=tpe.suggest, max_evals=try_num1, trials=trials)
loss_list = trials.losses()
min_loss = min(loss_list)
for k in range(try_num1):
    if min_loss == loss_list[k]:
        key = k
best_results = trials.results[key]

IS_score = best_results['train_score']
OOS_score = best_results['test_score']

print('NN1 Model a IS:', IS_score * 100)
print('NN1 Model a OOS:', OOS_score * 100)
NN_result[0,0] = IS_score * 100
NN_result[0,1] = OOS_score * 100
K.clear_session()
tf.reset_default_graph()



##############################################NN2

trials = Trials()
best = fmin(f_NN2, space, algo=tpe.suggest, max_evals=try_num1, trials=trials)
loss_list = trials.losses()
min_loss = min(loss_list)
for k in range(try_num1):
    if min_loss == loss_list[k]:
        key = k
best_results = trials.results[key]

IS_score = best_results['train_score']
OOS_score = best_results['test_score']

print('NN2 Model a IS:', IS_score * 100)
print('NN2 Model a OOS:', OOS_score * 100)
NN_result[1,0] = IS_score * 100
NN_result[1,1] = OOS_score * 100
K.clear_session()
tf.reset_default_graph()



##############################################NN3

trials = Trials()
best = fmin(f_NN3, space, algo=tpe.suggest, max_evals=try_num1, trials=trials)
loss_list = trials.losses()
min_loss = min(loss_list)
for k in range(try_num1):
    if min_loss == loss_list[k]:
        key = k
best_results = trials.results[key]

IS_score = best_results['train_score']
OOS_score = best_results['test_score']

print('NN3 Model a IS:', IS_score * 100)
print('NN3 Model a OOS:', OOS_score * 100)
NN_result[2,0] = IS_score * 100
NN_result[2,1] = OOS_score * 100
K.clear_session()
tf.reset_default_graph()


##############################################NN4

trials = Trials()
best = fmin(f_NN4, space, algo=tpe.suggest, max_evals=try_num1, trials=trials)
loss_list = trials.losses()
min_loss = min(loss_list)
for k in range(try_num1):
    if min_loss == loss_list[k]:
        key = k
best_results = trials.results[key]

IS_score = best_results['train_score']
OOS_score = best_results['test_score']

print('NN4 Model a IS:', IS_score * 100)
print('NN4 Model a OOS:', OOS_score * 100)
NN_result[3,0] = IS_score * 100
NN_result[3,1] = OOS_score * 100
K.clear_session()
tf.reset_default_graph()

##############################################NN5

trials = Trials()
best = fmin(f_NN5, space, algo=tpe.suggest, max_evals=try_num1, trials=trials)
loss_list = trials.losses()
min_loss = min(loss_list)
for k in range(try_num1):
    if min_loss == loss_list[k]:
        key = k
best_results = trials.results[key]

IS_score = best_results['train_score']
OOS_score = best_results['test_score']

print('NN5 Model a IS:', IS_score * 100)
print('NN5 Model a OOS:', OOS_score * 100)
NN_result[4,0] = IS_score * 100
NN_result[4,1] = OOS_score * 100
K.clear_session()
tf.reset_default_graph()



X_traindata, Y_traindata, X_vdata, Y_vdata, X_testdata, Y_testdata = creat_data(df_X=x_array, df_Y=y2)





##############################################NN1
trials = Trials()
best = fmin(f_NN1, space, algo=tpe.suggest, max_evals=try_num1, trials=trials)
loss_list = trials.losses()
min_loss = min(loss_list)
for k in range(try_num1):
    if min_loss == loss_list[k]:
        key = k
best_results = trials.results[key]

IS_score = best_results['train_score']
OOS_score = best_results['test_score']

print('NN1 model b IS:', IS_score * 100)
print('NN1 model b OOS:', OOS_score * 100)
NN_result[0,2] = IS_score * 100
NN_result[0,3] = OOS_score * 100
K.clear_session()
tf.reset_default_graph()



##############################################NN2

trials = Trials()
best = fmin(f_NN2, space, algo=tpe.suggest, max_evals=try_num1, trials=trials)
loss_list = trials.losses()
min_loss = min(loss_list)
for k in range(try_num1):
    if min_loss == loss_list[k]:
        key = k
best_results = trials.results[key]

IS_score = best_results['train_score']
OOS_score = best_results['test_score']

print('NN2 model b IS:', IS_score * 100)
print('NN2 model b OOS:', OOS_score * 100)
NN_result[1,2] = IS_score * 100
NN_result[1,3] = OOS_score * 100
K.clear_session()
tf.reset_default_graph()



##############################################NN3

trials = Trials()
best = fmin(f_NN3, space, algo=tpe.suggest, max_evals=try_num1, trials=trials)
loss_list = trials.losses()
min_loss = min(loss_list)
for k in range(try_num1):
    if min_loss == loss_list[k]:
        key = k
best_results = trials.results[key]

IS_score = best_results['train_score']
OOS_score = best_results['test_score']

print('NN3 model b IS:', IS_score * 100)
print('NN3 model b OOS:', OOS_score * 100)
NN_result[2,2] = IS_score * 100
NN_result[2,3] = OOS_score * 100
K.clear_session()
tf.reset_default_graph()



##############################################NN4

trials = Trials()
best = fmin(f_NN4, space, algo=tpe.suggest, max_evals=try_num1, trials=trials)
loss_list = trials.losses()
min_loss = min(loss_list)
for k in range(try_num1):
    if min_loss == loss_list[k]:
        key = k
best_results = trials.results[key]

IS_score = best_results['train_score']
OOS_score = best_results['test_score']

print('NN4 model b IS:', IS_score * 100)
print('NN4 model b OOS:', OOS_score * 100)
NN_result[3,2] = IS_score * 100
NN_result[3,3] = OOS_score * 100
K.clear_session()
tf.reset_default_graph()

##############################################NN5

trials = Trials()
best = fmin(f_NN5, space, algo=tpe.suggest, max_evals=try_num1, trials=trials)
loss_list = trials.losses()
min_loss = min(loss_list)
for k in range(try_num1):
    if min_loss == loss_list[k]:
        key = k
best_results = trials.results[key]

IS_score = best_results['train_score']
OOS_score = best_results['test_score']

print('NN5 model b IS:', IS_score * 100)
print('NN5 model b OOS:', OOS_score * 100)
NN_result[4,2] = IS_score * 100
NN_result[4,3] = OOS_score * 100
K.clear_session()
tf.reset_default_graph()








