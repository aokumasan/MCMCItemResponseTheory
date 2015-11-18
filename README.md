# IRT_by_Python

## 内容
項目反応理論（Item Response Theory）のパラメータを推定します．
1PLモデル（Rasch）のみ対応です．

## 必要なもの
 - Python3系
 - PyStan
 - PyMC3
 
Python3系で動きます．
PyStanかPyMC3が必要です．
以下のページを参考にインストールしてください．

- [PyMC3](https://github.com/pymc-devs/pymc3)
- [PyStan](https://github.com/stan-dev/pystan)

## 使い方
```
python3 pystan_irt.py
```
または，
```
python3 pymc_irt.py
```
で実行できます．途中でエラーとかワーニングを吐きまくりますが，無視して大丈夫です．