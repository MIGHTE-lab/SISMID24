import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


###### from previous exercise ######
df = pd.read_csv('./data/MX_Dengue_trends.csv')
df['Date'] = pd.to_datetime(df['Date'])


df_train = df.iloc[:36, :]

mod = LinearRegression()
mod.fit(df_train[['dengue']], df_train['Dengue CDC'])
m = mod.coef_[0]
b = mod.intercept_


df_valid = df.iloc[36:, :]
pred_static = df_valid['dengue'] * m + b


########### Part 2 ############

# (a)
pred_dyn = np.zeros(df_valid.shape[0])

for month in range(36, df.shape[0]):
    
    # define training set
    start = month - 36
    end = month
    df_train = df.iloc[start:end]
    df_pred = df.iloc[month:month+1]
    
    # fit model
    lr = LinearRegression()
    lr.fit(df_train[['dengue']], df_train['Dengue CDC'])
    
    # make prediction
    pred = lr.predict(df_pred[['dengue']])
    pred_dyn[month - 36] = pred
    

# (b)
plt.plot(df_valid['Date'], df_valid['Dengue CDC'], c='k', label='Cases')
plt.plot(df_valid['Date'], pred_static, c='r', label='Static')
plt.plot(df_valid['Date'], pred_dyn, c='g', label='Dynamic')
plt.legend()


# (c)
def rmse(truth, pred):
    return np.sqrt(np.mean((truth - pred) ** 2))

print('Static ', rmse(df_valid['Dengue CDC'], pred_static))
print('Dynamic ', rmse(df_valid['Dengue CDC'], pred_dyn))


# (d)
pred_dyn_all = np.zeros(df_valid.shape[0])
features = ['dengue', 'sintomas de dengue', 'mosquito', 'dengue sintomas']

for month in range(36, df.shape[0]):
    
    # define training set
    start = month - 36
    end = month
    df_train = df.iloc[start:end]
    df_pred = df.iloc[month:month+1]
    
    # fit model
    lr = LinearRegression()
    lr.fit(df_train[features], df_train['Dengue CDC'])
    
    # make prediction
    pred = lr.predict(df_pred[features])
    pred_dyn_all[month - 36] = pred

plt.plot(df_valid['Date'], df_valid['Dengue CDC'], c='k', label='Cases')
plt.plot(df_valid['Date'], pred_static, c='r', label='Static')
plt.plot(df_valid['Date'], pred_dyn, c='g', label='Dynamic')
plt.plot(df_valid['Date'], pred_dyn_all, c='b', label='Dynamic all vars')
plt.legend()

print('Static ', rmse(df_valid['Dengue CDC'], pred_static))
print('Dynamic ', rmse(df_valid['Dengue CDC'], pred_dyn))
print('Dynamic all vars', rmse(df_valid['Dengue CDC'], pred_dyn_all))


# (e)

# add AR terms
df['ar1'] = df['Dengue CDC'].shift(1)
df['ar2'] = df['Dengue CDC'].shift(2)

pred_dyn_ar = np.zeros(df_valid.shape[0])
features = ['dengue', 'sintomas de dengue',
            'mosquito', 'dengue sintomas', 'ar1', 'ar2']

for month in range(36, df.shape[0]):
    
    # define training set
    start = max(month - 36, 2)      # notice difference here
    end = month
    df_train = df.iloc[start:end]
    df_pred = df.iloc[month:month+1]
    
    # fit model
    lr = LinearRegression()
    lr.fit(df_train[features], df_train['Dengue CDC'])
    
    # make prediction
    pred = lr.predict(df_pred[features])
    pred_dyn_ar[month - 36] = pred

plt.plot(df_valid['Date'], df_valid['Dengue CDC'], c='k', label='Cases')
plt.plot(df_valid['Date'], pred_static, c='r', label='Static')
plt.plot(df_valid['Date'], pred_dyn, c='g', label='Dynamic')
plt.plot(df_valid['Date'], pred_dyn_all, c='b', label='Dynamic all vars')
plt.plot(df_valid['Date'], pred_dyn_ar, c='m', label='ARGO')
plt.legend()

print('Static ', rmse(df_valid['Dengue CDC'], pred_static))
print('Dynamic ', rmse(df_valid['Dengue CDC'], pred_dyn))
print('Dynamic all vars', rmse(df_valid['Dengue CDC'], pred_dyn_all))
print('ARGO', rmse(df_valid['Dengue CDC'], pred_dyn_ar))
