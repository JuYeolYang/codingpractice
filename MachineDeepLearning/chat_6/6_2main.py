'''k-평균 알고리즘
1.무작위로 k개의 클러스터 중심을 정한다
2.각 샘플에서 가장 가까운 클러스터 중심을 찾아 해당 클러스터의 샘플로 지정한다
3.클러스터에 속한 샘플의 평균값으로 클러스터 중심을 변경한다
4.클러스터 중심에 변화가 없을 때까지 2번으로 돌아가 반복한다
'''
import numpy as np
fruits = np.load('MachineDeepLearning/chat_6/fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)

#KMeans클래스
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)
print(km.labels_)
#레이블 0,1,2로 모은 샘플의 개수를 확인한다
print(np.unique(km.labels_, return_counts=True))#(array([0, 1, 2], dtype=int32), array([111,  98,  91]))

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

#km.labels_==0:배열에서 값이 0인 위치는 True, 그 외는 모두 False가 된다
#True인 위치의 원소만 모두 추출한다
draw_fruits(fruits[km.labels_==0])
draw_fruits(fruits[km.labels_==1])
draw_fruits(fruits[km.labels_==2])

#KMeans클래스가 최종적으로 찾은 클러스터 중심은 cluster_centers_속성에 저장되어 있다
draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3)

#transform()메소드도 2차원 배열을 기대하기 때문에 (1,100000)크기의 배열을 전달한다
print(km.transform(fruits_2d[100:101]))
#predict():가장 가까운 클러스터 중심을 예측 클래스로 출력한다
print(km.predict(fruits_2d[100:101]))#[2]
#예측한게 맞는지 확인해 본다
draw_fruits(fruits[100:101])
#n_iter_:알고리즘 반복한 횟수 저장
print(km.n_iter_)

inertia = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(fruits_2d)
    inertia.append(km.inertia_)
plt.plot(range(2, 7), inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()