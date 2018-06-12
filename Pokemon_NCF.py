import numpy as np
from keras.layers import Input, Dense, Dot
from keras.models import Model
from keras.optimizers import Adam

# 先にpreprocess.pyを実行してNumpy配列を作っておく
pokemon = np.load("data.npz")
# 定数
n_pokemon, n_trainer = pokemon["data"].shape #(151, 25)

# トレーナー別のポケモンに対するレーティングを入力するモデル input_shape=(151,)
input_a = Input(shape=(n_pokemon, ))
# 特徴量にマッピング
x_a = Dense(5, activation="relu")(input_a)

# ポケモン別のユーザーごとのレーティングを入力するモデル input_shape=(25,)
input_b = Input(shape=(n_trainer, ))
# 特徴量にマッピング
x_b = Dense(5, activation="relu")(input_b)

# モデルを結合（ここでは内積を取る）
y = Dot(axes=-1)([x_a, x_b])
# コンパイル
model = Model(inputs=[input_a, input_b], outputs=y)
model.compile(optimizer=Adam(lr=0.0001), loss="mse", metrics=["mse"])

# fit_generator
def fit_gen(data):
    global n_pokemon, n_trainer
    print(data[:,0].shape, data[0,:].shape)
    while True:
        for i in np.random.permutation(n_pokemon):
            for j in np.random.permutation(n_trainer):
                # 1ランクのベクトルになるのでreshapeで2ランクにすること
                yield [data[:,j].reshape(1,-1), data[i,:].reshape(1,-1)], data[i,j].reshape(1,-1)

# generatorを使ってフィット
model.fit_generator(generator=fit_gen(pokemon["data"]), 
                    steps_per_epoch=n_pokemon*n_trainer, epochs=10)

# id=24 ピカチュウ
# HPしか興味ない人（トレーナー1）のピカチュウのレーティングは？
pred_24_0 = model.predict([pokemon["data"][:,0].reshape(1,-1), pokemon["data"][24,:].reshape(1,-1)])
# 攻撃しか興味ない人（トレーナー1）のピカチュウのレーティングは？
pred_24_1 = model.predict([pokemon["data"][:,1].reshape(1,-1), pokemon["data"][24,:].reshape(1,-1)])
# 防御、素早さ、特殊以下同様
pred_24_2 = model.predict([pokemon["data"][:,2].reshape(1,-1), pokemon["data"][24,:].reshape(1,-1)])
pred_24_3 = model.predict([pokemon["data"][:,3].reshape(1,-1), pokemon["data"][24,:].reshape(1,-1)])
pred_24_4 = model.predict([pokemon["data"][:,4].reshape(1,-1), pokemon["data"][24,:].reshape(1,-1)])

print("予測結果")
print("トレーナー1（HP）のピカチュウのレートのデータは、", pokemon["data"][24,0])
print("トレーナー1（HP）のピカチュウのレートの予測値は、", pred_24_0)
print("トレーナー2（攻撃）のピカチュウのレートのデータは、", pokemon["data"][24,1])
print("トレーナー2（攻撃）のピカチュウのレートの予測値は、", pred_24_1)
print("トレーナー3（防御）のピカチュウのレートのデータは、", pokemon["data"][24,2])
print("トレーナー3（防御）のピカチュウのレートの予測値は、", pred_24_2)
print("トレーナー4（素早さ）のピカチュウのレートのデータは、", pokemon["data"][24,3])
print("トレーナー4（素早さ）のピカチュウのレートの予測値は、", pred_24_3)
print("トレーナー5（特殊）のピカチュウのレートのデータは、", pokemon["data"][24,4])
print("トレーナー5（特殊）のピカチュウのレートの予測値は、", pred_24_4)

#3775/3775 [==============================] - 5s 1ms/step - loss: 0.5702 - mean_s
#quared_error: 0.5702
#予測結果
#トレーナー1（HP）のピカチュウのレートのデータは、 2.0
#トレーナー1（HP）のピカチュウのレートの予測値は、 [[1.788103]]
#トレーナー2（攻撃）のピカチュウのレートのデータは、 3.0
#トレーナー2（攻撃）のピカチュウのレートの予測値は、 [[3.095326]]
#トレーナー3（防御）のピカチュウのレートのデータは、 2.0
#トレーナー3（防御）のピカチュウのレートの予測値は、 [[2.2915573]]
#トレーナー4（素早さ）のピカチュウのレートのデータは、 4.0
#トレーナー4（素早さ）のピカチュウのレートの予測値は、 [[2.5494237]]
#トレーナー5（特殊）のピカチュウのレートのデータは、 2.0
#トレーナー5（特殊）のピカチュウのレートの予測値は、 [[2.6407263]]