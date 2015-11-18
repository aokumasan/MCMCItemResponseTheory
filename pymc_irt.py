import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano.tensor as T

iter = 2000
burn = 1000


# データ
# LSATを読み込む
df = pd.read_csv("lsat.csv")
del df['Unnamed: 0']
data = np.array(df)

# 項目数nI, 反応パターン数nS
nS, nI = data.shape

# 項目番号行列を作成
item = np.array([list(range(nI)) for i in range(nS)])

# 反応パターン行列を作成
pattern = np.array([[i]*nI for i in range(nS)])


# 2次元配列から1次元配列に変換
data = data.flatten()
item = item.flatten()
pattern = pattern.flatten()

def invlogit(x):
    return np.exp(x)/(np.exp(x)+1)

def tinvlogit(v):
    return T.exp(v)/(T.exp(v)+1)

with pm.Model() as model:
    # hyper prior
    tau_b = pm.Uniform("tau_b", lower=0, upper=100)
    tau_theta = pm.Uniform("tau_theta", lower=0, upper=100)

    # prior
    b = pm.Normal("b", mu=0, sd=tau_b, shape=nI)
    theta = pm.Normal("theta", mu=0, sd=tau_theta, shape=nS)

    q = tinvlogit(1.7*(theta[pattern]-b[item]))

    y = pm.Bernoulli('y', p=q, observed=data)

    step = pm.NUTS()
    trace = pm.sample(iter, step=step)

with model:
    pm.traceplot(trace)

# 結果グラフの表示
plt.show()
