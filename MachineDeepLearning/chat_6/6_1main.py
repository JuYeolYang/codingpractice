import numpy as np
import matplotlib.pyplot as plt

fruits = np.load('MachineDeepLearning/chat_6/fruits_300.npy')
#첫 번째:샘플의 개수, 두 번째:이미지 높이, 세 번째:이미지 너비
print(fruits.shape)#(300, 100, 100)
print(fruits[0, 0, :])
#gray: 흑백이미지, gray_r:흑백 이미지 반전
plt.imshow(fruits[0], cmap='gray')
plt.show()

plt.imshow(fruits[0], cmap='gray_r')
plt.show()

#subplots():여러 개의 그래프를 배열처럼 쌓는다
#axs:서브 그래프를 담고 있는 배열
fig, axs = plt.subplots(1, 2)
axs[0].imshow(fruits[100], cmap='gray_r')
axs[1].imshow(fruits[200], cmap='gray_r')
plt.show()

#reshape()의 첫 번째 차원을 -1 로 지정하면 자동으로 남은 자원을 할당한다
apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)

#샘플마다 픽셀의 평균값 계산
#axis=0(행, 세로), axis=1(열, 가로)
print(apple.shape)#(100, 10000)
print(apple.mean(axis=1))

#히스토그램:값이 발생한 빈도를 그래프로 표시
#alpha:1보다 작게 하면 투명도를 줄 수 있다
#legend():범례를 만든다
plt.hist(np.mean(apple, axis=1), alpha=0.8)
plt.hist(np.mean(pineapple, axis=1), alpha=0.8)
plt.hist(np.mean(banana, axis=1), alpha=0.8)
plt.legend(['apple', 'pineapple', 'banana'])
plt.show()

#픽셀의 평균 계산
fig, axs = plt.subplots(1, 3, figsize=(20,5))
axs[0].bar(range(10000), np.mean(apple, axis=0))
axs[1].bar(range(10000), np.mean(pineapple, axis=0))
axs[2].bar(range(10000), np.mean(banana, axis=0))
plt.show()

#이미지화
apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
banana_mean = np.mean(banana, axis=0).reshape(100, 100)

fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].imshow(apple_mean, cmap='gray_r')
axs[1].imshow(pineapple_mean, cmap='gray_r')
axs[2].imshow(banana_mean, cmap='gray_r')
plt.show()

#abs():절댓값 계산 함수
abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff,axis=(1, 2))
print(abs_mean.shape)#(300,)

#np.argsort():작은 것에서 큰 순서대로 나열한 abs_mean배열의 인덱스를 반환
apple_index = np.argsort(abs_mean)[:100]
fig, axs = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[apple_index[i*10 + j]], cmap='gray_r')
        axs[i, j].axis('off')    
plt.show()    
