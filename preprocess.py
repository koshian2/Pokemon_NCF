import numpy as np
import pandas as pd

# レーティングデータの読み込み
pd_data = pd.read_csv("data.csv",encoding="SHIFT-JIS")
# ポケモン名
pokemon_names = pd_data.values[:,0]
# レーティングデータ
data = pd_data.values[:,1:].astype(float)

# トレーナー別の特徴量係数の読み込み
pd_data = pd.read_csv("trainer_features.csv",encoding="SHIFT-JIS")
# トレーナー名
trainer_names = pd_data.values[:,0]
# 係数行列
trainer_coefs = pd_data.values[:, 1:]

# ポケモン別の特徴量（種族値）の読み込み
pd_data = pd.read_csv("species_normalized.csv",encoding="SHIFT-JIS")
# 種族値
pokemon_spicies = pd_data.values[:,1:]

# あとから読みやすいようにNumpy配列として保存
np.savez("data", data=data, pokemon_names=pokemon_names, pokemon_spicies=pokemon_spicies, 
         trainer_names=trainer_names, trainer_coefs=trainer_coefs)