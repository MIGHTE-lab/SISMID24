import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# (a)

df = pd.read_csv('./data/MX_Dengue_trends.csv')
df['Date'] = pd.to_datetime(df['Date'])
plt.plot(df['Date'], df['Dengue CDC'])


# (b)

df_train = df.iloc[:36, :]

mod = LinearRegression()
mod.fit(df_train[['dengue']], df_train['Dengue CDC'])
m = mod.coef_[0]
b = mod.intercept_
print(b, m)


# (c)

plt.clf()
plt.scatter(df_train['dengue'], df_train['Dengue CDC'])
xg = np.linspace(0, 30, 100)
yg = xg * m + b
plt.plot(xg, yg, c='r')

plt.clf()
pred_insample = df_train['dengue'] * m + b
plt.plot(df_train['Date'], df_train['Dengue CDC'], c='k', label='Cases')
plt.plot(df_train['Date'], pred_insample, c='b', label='Pred')
plt.legend()

# (d)

df_valid = df.iloc[36:, :]
pred_static = df_valid['dengue'] * m + b

plt.clf()
plt.plot(df_valid['Date'], df_valid['Dengue CDC'], c='k', label='Cases')
plt.plot(df_valid['Date'], pred_static, c='b', label='Pred')
plt.legend()
