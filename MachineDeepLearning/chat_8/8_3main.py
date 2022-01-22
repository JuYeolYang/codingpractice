from tensorflow import keras
#케라스 모델에 추가한 층은 layers 속성에 저장되어 있다
model = keras.models.load_model('best-cnn-model.h5')
print(model.layers)

#weights의 가중치(첫 번째 원소)와 절편(두 번째 원소)의 크기 출력
conv = model.layers[0]
print(conv.weights[0].shape, conv.weights[1].shape)#(3, 3, 1, 32) (32,)

#가중치 배열의 평균과 표준편차 출력
conv_weights = conv.weights[0].numpy()
print(conv_weights.mean(), conv_weights.std())#-0.018889196 0.23722129

#가중치의 분포도를 히스토그램으로 그리기
import matplotlib.pyplot as plt
plt.hist(conv_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()

#커널 출력
fig, axs = plt.subplots(2, 16, figsize=(15,2))
for i in range(2):
    for j in range(16):
        axs[i, j].imshow(conv_weights[:,:,0,i*16 + j], vmin=-0.5, vmax=0.5)
        axs[i, j].axis('off')

plt.show()

#훈련하지 않은 빈 합성곱 신경망 만들기
no_training_model = keras.Sequential()
no_training_model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(28,28,1)))

#Conv2D층의 가중치를 no_training_conv 변수에 저장
no_training_conv = no_training_model.layers[0]
print(no_training_conv.weights[0].shape)#(3, 3, 1, 32)

#가중치의 평균과 표준편차 확인
no_training_weights = no_training_conv.weights[0].numpy()
print(no_training_weights.mean(), no_training_weights.std())#-0.0032394065 0.079241686

plt.hist(no_training_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()

fig, axs = plt.subplots(2, 16, figsize=(15,2))
for i in range(2):
    for j in range(16):
        axs[i, j].imshow(no_training_weights[:,:,0,i*16 + j], vmin=-0.5, vmax=0.5)
        axs[i, j].axis('off')

plt.show()

"""## 함수형 API"""

print(model.input)#KerasTensor(type_spec=TensorSpec(shape=(None, 28, 28, 1), 
#                   dtype=tf.float32, name='conv2d_input'), name='conv2d_input', description="created by layer 'conv2d_input'")

conv_acti = keras.Model(model.input, model.layers[0].output)

"""## 특성 맵 시각화"""

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

plt.imshow(train_input[0], cmap='gray_r')
plt.show()

#predict()메서드는 항상 입력의 첫 번째 차원이 배치 차원일 것으로 기대하기 때문에 (784,)크기를 (28, 28, 1)로 변경한다
inputs = train_input[0:1].reshape(-1, 28, 28, 1)/255.0
feature_maps = conv_acti.predict(inputs)

print(feature_maps.shape)#(1, 28, 28, 32)

fig, axs = plt.subplots(4, 8, figsize=(15,8))
for i in range(4):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0,:,:,i*8 + j])
        axs[i, j].axis('off')

plt.show()

conv2_acti = keras.Model(model.input, model.layers[2].output)

feature_maps = conv2_acti.predict(train_input[0:1].reshape(-1, 28, 28, 1)/255.0)
print(feature_maps.shape)#(1, 14, 14, 64)

fig, axs = plt.subplots(8, 8, figsize=(12,12))
for i in range(8):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0,:,:,i*8 + j])
        axs[i, j].axis('off')

plt.show()