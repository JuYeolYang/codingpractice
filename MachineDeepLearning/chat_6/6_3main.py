import numpy as np
fruits = np.load('MachineDeepLearning/chat_6/fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)

#PCA클래스
#n_components:주성분의 개수를 지정, components_:주성분 속성이 저장되어 있다
#비지도 학습이기 때문에 타깃값이 없다
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
pca.fit(fruits_2d)
print(pca.components_.shape)#(50, 10000)

#이미지 출력
import matplotlib.pyplot as plt
def draw_fruits(arr, ratio=1):
    n = len(arr)#n:샘플 개수
    #한 줄에 10개씩 이미지를 그린다. 샘플 개수를 10으로 나누어 전체 행 개수를 계산한다
    rows = int(np.ceil(n/10))
    #행이 1개이면 열의 개수는 샘플 개수이다. 그렇지 않으면 10개이다
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols, figsize=(cols*ratio, rows*ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:#n개까지만 그린다
                axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
                axs[i, j].axis('off')
    plt.show()        
                
draw_fruits(pca.components_.reshape(-1, 100, 100))

#transform()로 원본 데이터의 차원을 50으로 줄인다
print(fruits_2d.shape)#(300, 10000)
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)#(300, 50)

#원본 데이터 재구성
fruits_inverse = pca.inverse_transform(fruits_pca)
print(fruits_inverse.shape)#(300, 10000)

fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)
for start in [0, 100, 200]:
    draw_fruits(fruits_reconstruct[start:start+100])
    print("\n")
    
#설명된 분산
#explained_variance_ratio_:각 주성분의 설명된 분산 비율을 기록되어 있다
print(np.sum(pca.explained_variance_ratio_))#0.921568126215531

#설명된 분산을 그래프로 출력
#설명된 분산의 비율을 그래프로 그려 보면 적절한 주성분의 개수를 찾는데 도움이 된다
plt.plot(pca.explained_variance_ratio_)
plt.show()

#원본 데이터와 PCA로 축소한 데이터의 차이점
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
#0:사과, 1:파인애플, 2:바나나
target = np.array([0]*100 + [1]*100 + [2]*100)

from sklearn.model_selection import cross_validate
scores = cross_validate(lr, fruits_2d, target)
print(np.mean(scores['test_score']))#0.9966666666666667
print(np.mean(scores['fit_time']))#0.27936840057373047

#fruits_pca를 사용했을 때와 비교
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))#1.0
print(np.mean(scores['fit_time']))#0.014868831634521485

#설명된 분산을 50%에 달하는 PCA모델
pca = PCA(n_components=0.5)
pca.fit(fruits_2d)
print(pca.n_components_)#2

#이 모델로 원본 데이터 변환
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)#(300, 2)

#교차 검증 결과 확인
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))#0.9933333333333334
print(np.mean(scores['fit_time']))#0.025211811065673828

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_pca)
print(np.unique(km.labels_, return_counts=True))

for label in range(0, 3):
    draw_fruits(fruits[km.labels_ == label])
    print("\n")

#훈련데이터 시각화
for label in range(0, 3):
    data = fruits_pca[km.labels_ == label]
    plt.scatter(data[:,0], data[:,1])
plt.legend(['apple', 'banana', 'pineapple'])
plt.show()

