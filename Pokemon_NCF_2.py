import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Dot
from keras.models import Model
from keras.optimizers import Adam

# データの読み込み
pokemon = np.load("data.npz")
# 定数
n_pokemon, n_trainer = pokemon["data"].shape #(151, 25)

# より実践的な例にするために未評価の項目を入れる
np.random.seed(151)
rand = np.random.uniform(size=(n_pokemon, n_trainer))
# 乱数が0.6以下なら未評価とする
mask = (rand > 0.6).astype(int)
data = pokemon["data"]
data[mask==0] = 0
# ポケモンごとの平均値を計算
avg = np.zeros(n_pokemon)
for i in range(n_pokemon):
    slice = data[i, mask[i, :]==1]
    if(len(slice) > 0):
        avg[i] = np.mean(slice)
# 観測値から平均値を引く
data = (data - np.tile(avg, (n_trainer, 1)).T) * mask
# データありで0になってしまったところに微小量を加える
data[(mask==1) & (data==0)] = 1e-8

# NCFモデル
input_a = Input(shape=(n_pokemon, ))
x_a = Dense(5, activation="relu")(input_a)

input_b = Input(shape=(n_trainer, ))
x_b = Dense(5, activation="relu")(input_b)

y = Dot(axes=-1)([x_a, x_b])
model = Model(inputs=[input_a, input_b], outputs=y)

# fit_generator
def fit_gen(data):
    global n_pokemon, n_trainer
    print(data[:,0].shape, data[0,:].shape)
    while True:
        for i in np.random.permutation(n_pokemon):
            for j in np.random.permutation(n_trainer):
                yield [data[:,j].reshape(1,-1), data[i,:].reshape(1,-1)], data[i,j].reshape(1,-1)

# カスタム損失関数
def loss_function(y_true, y_pred):
    squared = K.square(y_true - y_pred)
    # sign(0)=0, sign(正の数)=1、sign(負の数)=-1なので、sign関数の2乗で未評価の項目をフィルタリングできる
    return squared * K.square(K.sign(y_true))

# コンパイル
model.compile(optimizer=Adam(lr=0.0001), loss=loss_function)

# generatorを使ってフィット
n_epochs = 25
history = model.fit_generator(generator=fit_gen(data), 
                    steps_per_epoch=n_pokemon*n_trainer, epochs=n_epochs).history

plt.plot(np.arange(n_epochs), history["loss"])
plt.show()

# おすすめのポケモンを上位10件表示
def recommendation_view(pred, mask, avg):
    global pokemon
    # 評価済みポケモン
    print("評価済みポケモン")
    print(pokemon["pokemon_names"][mask==1])
    # 予測レート
    score = (pred + avg) * (1-mask)
    # ポケモンのインデックス
    index = np.argsort(score)[::-1][:10]
    cnt = 1
    print("おすすめのポケモン一覧")
    for i in index:
        if mask[i] == 1: break
        print(cnt, "位 : id =", i+1, "(", pokemon["pokemon_names"][i], ")", "予測レート :", score[i])
        cnt += 1
    print()

# 予測
def recommendation(column_id):
    pred = model.predict([np.tile(data[:,column_id], (n_pokemon, 1)), data])
    # おすすめ表示
    recommendation_view(np.ravel(pred), mask[:,column_id], avg)

# 攻撃極振りが好きなトレーナー2の場合
print("-攻撃極振りのトレーナー2の場合")
recommendation(1)
print()
# 防御極振りが好きなトレーナー3の場合
print("-防御極振りのトレーナー3の場合")
recommendation(2)
print()

# ロケット団のポケモンを入れてみる
rocket_score, rocket_mask = np.zeros(n_pokemon), np.zeros(n_pokemon)
# アーボ(22)は4点、アーボック(23)は5点
rocket_score[22], rocket_score[23] = 4, 5
rocket_mask[22], rocket_mask[23] = 1, 1
# ベロリンガ(107)も5点
rocket_score[107], rocket_mask[107] = 5, 1
# シェルダー(89)は4点
rocket_score[89], rocket_mask[89] = 4, 1
# ドガース(108)は4点、マタドガス(109)は5点
rocket_score[108], rocket_score[109] = 4, 5
rocket_mask[108], rocket_mask[109] = 1, 1
# ガーディ(57)は4点
rocket_score[57], rocket_mask[57] = 4, 1
# ウツドン（69）は4点、ウツボット（70）は5点
rocket_score[69], rocket_score[70] = 4, 5
rocket_mask[69], rocket_mask[70] = 1, 1
# 標準化
rocket_score = rocket_score - avg
rocket_score[(rocket_score==0) & (rocket_mask==1)] = 1e-8
# おすすめ表示
print("-ロケット団の場合")
rocket_pred = model.predict([np.tile(rocket_score, (n_pokemon, 1)), data])
recommendation_view(np.ravel(rocket_pred), rocket_mask, avg)
