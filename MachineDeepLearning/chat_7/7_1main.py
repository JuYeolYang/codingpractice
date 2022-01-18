from tensorflow import keras
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
print(train_input.shape, train_target.shape)#(60000, 28, 28) (60000,)
print(test_input.shape, test_target.shape)#(10000, 28, 28) (10000,)

#샘플 그림 출력
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 10, figsize=(10, 10))
for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].axis('off')
plt.show()

#10개의 샘플의 타깃값을 리스트로 만든 후 출력
print([train_target[i] for i in range(10)])#[9, 0, 0, 3, 0, 2, 7, 2, 5, 5]

#레이블 당 샘플 개수 확인
import numpy as np
print(np.unique(train_target, return_counts=True))#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000]))

#SGDClassifier은 2차원 입력을 다루지 못하기 때문에 1차원 배열로 바꿔줘야한다.
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)
print(train_scaled.shape)#(60000, 784)

from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier
sc = SGDClassifier(loss='log', max_iter=5, random_state=42)
scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
print(np.mean(scores['test_score']))#0.8192833333333333

#인공 신경망
#텐서플로와 케라스

import tensorflow as tf

#인공 신경망으로 모델 만들기
from sklearn.model_selection import train_test_split
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

print(train_scaled.shape, train_target.shape)#(48000, 784) (48000,)
print(val_scaled.shape, val_target.shape)#(12000, 784) (12000,)

#activation='softmax' : 10개의 뉴런에서 출력되는 값을 확률로 바꾼다
dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))
model = keras.Sequential(dense)                     #모델 -> 사이킷런 모델에선 sc = SGDClassifier()해당

#인공 신경망으로 패션 아이템 분류하기
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(train_target[:10])
model.fit(train_scaled, train_target, epochs=5)     #훈련 -> 사이킷런 모델에선 sc.fit()해당
model.evaluate(val_scaled, val_target)              #평가 -> 사이킷런 모델에선 sc.score()해당