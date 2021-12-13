import pandas as pd
from scipy.sparse.construct import random
fish = pd.read_csv('https://bit.ly/fish_csv_data')

fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

#표준화 전처리
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
#훈련세트로 학습한 통계값으로 변환해야한다
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

from sklearn.linear_model import SGDClassifier
sc = SGDClassifier(loss='log', max_iter=10, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))#0.773109243697479
print(sc.score(test_scaled, test_target))#0.775

#partial_fit은 fit()메소드와 사용법은 같지만 1 에포크씩 이어서 훈련할 수 있다
sc.partial_fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))#0.8151260504201681 -> 성능이 향상되었다
print(sc.score(test_scaled, test_target))#.85
#얼마나 더 훈련시켜야 할 지 기준이 필요하다

import numpy as np
sc = SGDClassifier(loss='log', random_state=42)
train_score = []
test_score = []
classes = np.unique(train_target)#train_target에 있는 7개 생선의 목록을 만든다

#300번의 에포크 동안 훈련세트와 테스트세트의 점수를 계산해서 저장한다
for _ in range(0, 300):
    sc.partial_fit(train_scaled, train_target, classes=classes)
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))
    
import matplotlib.pyplot as plt
plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()#100번째 에포크가 적절한 반복 횟수로 보인다

#SGDClassifier에서 일정 에포크 동안 성능이 향상되지 않으면 더 훈련하지 않고 자동으로 멈춘다
#tol 매개변수로 자동으로 멈추지 않게 할 수 있다
sc = SGDClassifier(loss='log', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))#0.957983193277311
print(sc.score(test_scaled, test_target))#0.925

#SGDClassifier은 여러 종류의 손실 함수를 loss 매개변수에 지정하면 다양한 머신러닝 알고리즘을 지원한다
#loss매개변수의 기본값은 'hinge'이다
sc = SGDClassifier(loss='hinge', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))#0.9495798319327731
print(sc.score(test_scaled, test_target))#0.925