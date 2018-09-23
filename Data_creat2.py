import pandas as pd
import numpy as np
import pickle

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
Path = 'D:\\APViaML'
from scipy.stats import norm
from scipy.stats import t

# 0 set size & df
T_num = 180
id_num = 200
Xt_list = ['x1']
pc1 = 50
pc2 = 100
temp = list(range(1, pc1 + 1))
ct_list = []
for x in temp:
    ct_list.append('c' + str(x))
other_columns_name = ['gz1', 'gz2', 'e', 'ret']
columns_name = Xt_list + ct_list + other_columns_name

final_data = pd.DataFrame(columns=columns_name)

# 1 simulate x1_t
x1 = np.zeros(shape=T_num * id_num)
x1_0 = np.zeros(shape=id_num)

# 1.1 x1
import math

p = 0.95
data_mean = 0
data_std = math.sqrt(1 - 0.95 * 0.95)
data_size = T_num
random_seed1 = 12

u1_t_array = norm.rvs(loc=data_mean, scale=data_std, size=data_size, random_state=random_seed1)

for i in range(T_num):
    if i == 0:
        temp_x_1 = x1_0 * p + u1_t_array[i]
        x1[200 * i:200 * (i + 1)] = temp_x_1.copy()
    else:
        temp_x_1 = temp_x_1 * p + u1_t_array[i]
        x1[200 * i:200 * (i + 1)] = temp_x_1.copy()

# save x1
final_data_100 = final_data
final_data_100['x1'] = x1

# 2 simulate Cij,t

from scipy.stats import uniform

data_low = 0.9
data_scale = 0.1
data_size = pc1
random_seed = 123

pj = uniform.rvs(loc=data_low, scale=data_scale, size=data_size, random_state=random_seed)

data_mean = 0
data_std = 1
data_size = (id_num)
random_seed2 = 123
epsilon_ij_t = norm.rvs(loc=data_mean, scale=data_std, size=data_size, random_state=random_seed2)

c = np.zeros(shape=(id_num * T_num, pc1))
c_0 = np.zeros(shape=(id_num, pc1))

for i in range(T_num):
    if i == 0:
        temp_x_1 = c_0.copy()
        for j in range(pc1):
            temp_x_1[:, j] = temp_x_1[:, j] * pj[j] + norm.rvs(loc=data_mean, scale=data_std, size=data_size)
        c[200 * i:200 * (i + 1)] = temp_x_1.copy()
    else:
        for j in range(pc1):
            temp_x_1[:, j] = temp_x_1[:, j] * pj[j] + norm.rvs(loc=data_mean, scale=data_std, size=data_size)
        c[200 * i:200 * (i + 1)] = temp_x_1.copy()

c_new = c.copy()
for i in range(T_num):
    for j in range(pc1):
        temp_series = pd.Series(c_new[200 * i:200 * (i + 1), j])
        temp_series = temp_series.rank()
        temp_series = 2 * temp_series / (len(temp_series) + 1) - 1
        c_new[200 * i:200 * (i + 1), j] = temp_series.copy()

final_data_100.loc[:, ct_list] = c_new

# 3 gz

seita1 = np.array([[0.02, 0.02, 0.02]]).T
temp = final_data_100['x1'] * final_data_100['c3']
c13 = c_new[:, :3].copy()
c13[:, 2] = np.array(temp)

gz1 = np.dot(c13, seita1)

temp_c1 = final_data_100['c1'] ** 2
temp_c2 = final_data_100['c1'] * final_data_100['c2']
temp_c3 = np.sign(final_data_100['x1'] * final_data_100['c3'])

c23 = c_new[:, :3].copy()
c23[:, 0] = np.array(temp_c1)
c23[:, 1] = np.array(temp_c2)
c23[:, 2] = np.array(temp_c3)

seita2 = np.array([[0.04, 0.03, 0.012]]).T

gz2 = np.dot(c23, seita2)

final_data_100['gz1'] = gz1
final_data_100['gz2'] = gz2

##4 e


data_mean = 0
data_std = 0.05
data_size = T_num
random_seed1 = 12
random_seed2 = 121
random_seed3 = 123

vt1_array = norm.rvs(loc=data_mean, scale=data_std, size=data_size, random_state=random_seed1)
vt2_array = norm.rvs(loc=data_mean, scale=data_std, size=data_size, random_state=random_seed2)
vt3_array = norm.rvs(loc=data_mean, scale=data_std, size=data_size, random_state=random_seed3)
temp_array = np.zeros(shape=[3, T_num])

temp_array[0, :] = vt1_array
temp_array[1, :] = vt2_array
temp_array[2, :] = vt3_array

beta = c_new[:, :3].copy()
for i in range(T_num):
    for j in range(3):
        beta[200 * i:200 * (i + 1), j] = beta[200 * i:200 * (i + 1), j] * temp_array[j, i]





data_fr = 5
data_mean = 0
data_scale = 0.05
data_size = id_num*T_num
random_seed = 123
data_size2 = id_num

epsilon2_it = np.zeros(shape=(data_size,1))
for i in range(T_num):
    epsilon2_it[200 * i:200 * (i + 1),0] = t.rvs(df=data_fr, loc=data_mean, scale=data_scale, size=data_size2, random_state=random_seed+i)

e = beta[:, 0] + beta[:, 1] + beta[:, 2] + epsilon2_it[:,0]

final_data_100['e'] = e

final_data_100['ret'] = final_data_100['e'] + final_data_100['gz1']
final_data_100['ret1'] = final_data_100['e'] + final_data_100['gz2']

final_data_100_x1 = np.zeros(shape=(id_num*T_num,100))
final_data_100_x1[:,:50] = final_data_100.loc[:,ct_list]

final_data_100.loc[:,'c1'] = final_data_100.loc[:,'c1']*final_data_100.loc[:,'x1']

for i in range(50):
    final_data_100.iloc[:,i+1] = final_data_100.iloc[:,i+1]*final_data_100.loc[:,'x1']

final_data_100_x1[:,50:100] = final_data_100.loc[:,ct_list]

final_data_100_y = final_data_100.loc[:,['ret','ret1','gz1','gz2']]


file = open(Path + '\\data\\mcs_demo_x.pkl', 'wb')
pickle.dump(final_data_100_x1, file)
file.close()

file = open(Path + '\\data\\mcs_demo_y.pkl', 'wb')
pickle.dump(final_data_100_y, file)
file.close()



temp_index = np.array(range(180))*200
temp_c1 = c_new[:,25].copy()
temp_c1 = final_data_100['ret']
temp_c1 = temp_c1[temp_index]
import matplotlib.pyplot as plt
plt.plot(range(180),temp_c1)
plt.show()


