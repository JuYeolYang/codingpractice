from tensorflow import keras
from sklearn.model_selection import train_test_split

#입력 이미지는 항상 깊이(채널)차원이 있어야 한다
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0#깊이를 1로 만든다
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

"""## 합성곱 신경망 만들기"""

#32개의 필터를 사용한 Conv2D를 추가한다. 
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(28,28,1)))
model.add(keras.layers.MaxPooling2D(2))

#64개의 필터를 사용한 Conv2D를 추가한다
model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(2))

#은닉충은 100개의 뉴런을 사용하고 렐루 함수를 사용한다
#드롭아웃 층은 은닉층의 과대적합을 막아 성능을 조금 더 개선한다
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10, activation='softmax'))
model.summary()

#plot_model(): 층의 구성을 그림으로 표현해 준다
keras.utils.plot_model(model)
#show_shapes:True로 설정하면 그림에 입력과 출력의 크기를 표시해 준다
#to_file:파일 이름을 지정하면 출력한 이미지를 파일로 저장한다
#dpi:해상도를 지정할 수 있다
keras.utils.plot_model(model, show_shapes=True, to_file='cnn-architecture.png', dpi=300)

"""## 모델 컴파일과 훈련"""
#Adam 옵티마이저를 사용하고, ModelCheckpoint 콜백과 EarlyStopping 콜백을 함께 사용한다
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.h5', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True)
history = model.fit(train_scaled, train_target, epochs=20,validation_data=(val_scaled, val_target),callbacks=[checkpoint_cb, early_stopping_cb])

#손실 그래프를 그려 조기 종료가 잘 이루어졌는지 확인한다
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

#세트에 대한 성늘 평가
model.evaluate(val_scaled, val_target)
#흑백 이미지에 깊이 차원이 없으므로 (28, 28, 1)을 (28, 28)로 바꾸어 출력한다
plt.imshow(val_scaled[0].reshape(28, 28), cmap='gray_r')
plt.show()

#10개의 클래스에 대한 예측 확률을 출력한다
preds = model.predict(val_scaled[0:1])
print(preds)

plt.bar(range(1, 11), preds[0])
plt.xlabel('class')
plt.ylabel('prob.')
plt.show()

classes = ['티셔츠', '바지', '스웨터', '드레스', '코트',
           '샌달', '셔츠', '스니커즈', '가방', '앵클 부츠']

#preds배열에서 가장 큰 인덱스를 찾아 classes리스트의 인덱스로 사용한다
import numpy as np
print(classes[np.argmax(preds)])#가방

#테스트 세트는 모델을 출시하기 직전에 딱 한번 사용해야 한다. 이유는 모델을 실전에 투입했을 때 성능을 올바르게 예측하지 못한다
test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0
model.evaluate(test_scaled, test_target)