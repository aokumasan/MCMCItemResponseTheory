import pystan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

irt_model = """
data {
  int<lower=1> J;              // number of students
  int<lower=1> K;              // number of questions
  int<lower=1> N;              // number of observations
  int<lower=1,upper=J> jj[N];  // student for observation n
  int<lower=1,upper=K> kk[N];  // question for observation n
  int<lower=0,upper=1> y[N];   // correctness for observation n
}
parameters {    
  real alpha[J];               // ability of student j - mean ability
  real beta[K];                // difficulty of question k
  real tau_a;
  real tau_b;
}
model {
  tau_a ~ uniform(0, 100);
  tau_b ~ uniform(0, 100);
  alpha ~ normal(0,1/(tau_a*tau_a));         // informative true prior
  beta ~ normal(0,1/(tau_b*tau_b));          // informative true prior
  for (n in 1:N)
    y[n] ~ bernoulli_logit(alpha[jj[n]] - beta[kk[n]]);
}
"""



# RのLSATを読み込む
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
item = item.flatten()+1
pattern = pattern.flatten()+1

# データをディクショナリに変換
irt_data = {"J":nS,
            "K":nI,
            "N":nS*nI,
            "jj":pattern,
            "kk":item,
            "y":data}

# MCMC
fit = pystan.stan(model_code=irt_model, data=irt_data, iter=2000, chains=4)

# 結果表示
print(fit)

# グラフ表示
fit.plot().show()
