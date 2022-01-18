from tensorflow import keras
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

#모델을 만드는 간단한 함수 정의
def model_fn(a_layer=None):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(100, activation='relu'))
    if a_layer:
        model.add(a_layer)
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

model = model_fn()
model.summary()

model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#history 객체에는 훈련 측정값이 담겨 있는 history 딕셔너리가 들어있다
history = model.fit(train_scaled, train_target, epochs=5, verbose=0)
print(history.history.keys())#dict_keys(['loss', 'accuracy'])->손실과 정확도가 포함되어 있다

import matplotlib.pyplot as plt
#손실 그래프 출력
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

#정확도 그래프 출력
plt.plot(history.history['accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

#에포크 횟수를 20으로 늘려 모델 훈련
model = model_fn()
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_scaled, train_target, epochs=20, verbose=0)

plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

"""## 검증 손실"""
#에포크마다 검증 손실을 계산하기 위해 케라스 모델의 fit()메서드에 검증 데이터를 전달할 수 있다
#validation_data매개변수에 검증에 사용할 입력과 타깃값을 튜플로 만들어 전달한다
model = model_fn()
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target))
print(history.history.keys())#dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

#옵티마이저를 Adam으로 사용
model = model_fn()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

"""## 드롭아웃"""
#30%정도 드롭아웃시킨다
model = model_fn(keras.layers.Dropout(0.3))
model.summary()#일부 뉴런의 출력을 0으로 만들지만 전체 출력배열의 크기를 비꾸진 않는다

#텐서플로와 케라스는 모델을 평가와 예측에 사용할 때는 자동으로 드롭아웃을 적용하지 않는다
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

"""## 모델 저장과 복원"""
#에포크 횟수를 10으로 지정하고 모델을 훈련한다
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_scaled, train_target, epochs=10, verbose=0, validation_data=(val_scaled, val_target))

#케라스 모델은 훈련된 모델의 파라미터를 저장하는 save_weights()메서드를 제공한다
model.save_weights('model-weights.h5')
#모델 구조와 모델 파라미터를 함께 저장하는 save()메서드를 제공한다
model.save('model-whole.h5')

#훈련을 하지 않은 새로운 모델을 만들고 model-weights.h5파일에서 훈련된 모델 파라미터를 읽어서 사용한다
#save_weights()는 load_weigths()메서드를 사용해서 불러온다. load_weigths()사용할 때 save_weights()로 저장했던 모델과 정확히 같은 구조를 가져야 한다.
model = model_fn(keras.layers.Dropout(0.3))
model.load_weights('model-weights.h5')

import numpy as np

#10개 확률 중에 가장 큰 값의 인덱스를 골라 타깃 레이블과 비교하여 정확도를 계산해본다.
#argmax()는 배열에서 가장 큰 값의 인덱스를 반환한다.
val_labels = np.argmax(model.predict(val_scaled), axis=-1)#(12000:샘플, 10:확률)크기의 배열 반환
print(np.mean(val_labels == val_target))#0.87575

#모델 전체를 파일에서 읽은 다음 검증 세트의 정확도를 출력해본다.
model = keras.models.load_model('model-whole.h5')
model.evaluate(val_scaled, val_target)
#같은 모델을 저장하고 다시 불러들였기 때문에 동일한 정확도를 얻는다.

"""## 콜백"""
#ModelCheckpoint 콜백은 기본적으로 최상의 검증 점수를 만드는 모델을 저장한다
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.h5', save_best_only=True)
model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target), callbacks=[checkpoint_cb])

model = keras.models.load_model('best-model.h5')
model.evaluate(val_scaled, val_target)

model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#patience: 지정한 횟수만큼 연속 검증 점수가 향상되지 안하으면 훈련을 중지한다
#restore_best_weights: True로 지정하면 가장 낮은 검증 손실을 낸 모델 파라미터로 되돌린다
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.h5', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True)
history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target), callbacks=[checkpoint_cb, early_stopping_cb])

#훈련을 마치고 나면 몇 번째 에포크에서 훈련이 중지되었는지 early_stopping_cb.stopped_epoch 속성에서 확인할 수 있다
print(early_stopping_cb.stopped_epoch)#12

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

model.evaluate(val_scaled, val_target)