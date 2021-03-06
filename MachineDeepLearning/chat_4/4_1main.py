import pandas as pd
from scipy.sparse.construct import random
from scipy.sparse.data import _minmax_mixin

fish = pd.read_csv('https://bit.ly/fish_csv_data')
print(fish.head())

#Species 열에 고유한 값을 추출
print(pd.unique(fish['Species']))#['Bream' 'Roach' 'Whitefish' 'Parkki' 'Perch' 'Pike' 'Smelt']

#원하는 열을 리스트로 나열하면 나열 순으로 배열이 만들어진다
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
print(fish_input[:5])

fish_target = fish['Species'].to_numpy()

#훈련세트와 테스트세트 나누기
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

#훈련센트, 테스트 세트 표준화 전처리-StandardScaler : 변환기 클래스
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

#훈련세트, 테스트 세트 점수 확인, k-최근접 이웃 분류기로 확률 예측
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target))#0.8907563025210085
print(kn.score(test_scaled, test_target))#0.85

#정렬된 타깃값은 classes_속성에 저장되어있다
print(kn.classes_)#['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']
#샘플 타깃값 예측
print(kn.predict(test_scaled[:5]))#['Perch' 'Smelt' 'Pike' 'Perch' 'Perch']

import numpy as np
#예측은 어떤 확률로 만들어 졌는지 확인
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))

#이웃 확인
distances, indexes = kn.kneighbors(test_scaled[3:4])
print(train_target[indexes])#[['Roach' 'Perch' 'Perch']]

#시그모이드(로지스틱) 함수
import matplotlib.pyplot as plt
z = np.arange(-5, 5, 0.1)
#np.exp : 지수함수
phi = 1 / (1 + np.exp(-z))
plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()

#불리언 인덱싱 테스트
char_arr = np.array(['A', 'B', 'C', 'D', 'E'])
print(char_arr[[True, False, True, False, False]])#['A' 'C'], True값만 출력함

#도미와 빙어행 골라내기
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)
print(lr.predict(train_bream_smelt[:5]))#['Bream' 'Smelt' 'Bream' 'Bream' 'Bream']
print(lr.predict_proba(train_bream_smelt[:5]))
print(lr.classes_)#['Bream' 'Smelt']
#학습한 계수 확인
#weight:-0.404, length:-0.576, diagonal:-0.663, height:1.013, width:-0.732, 절편:-2.161
print(lr.coef_, lr.intercept_)#[[-0.4037798  -0.57620209 -0.66280298 -1.01290277 -0.73168947]] [-2.16155132]

#z값 출력
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)#[-6.02927744  3.57123907 -5.26568906 -4.24321775 -6.0607117 ], 양성 클래스에 대한 z값을 반환

from scipy.special import expit
print(expit(decisions))#[0.00240145 0.97264817 0.00513928 0.01415798 0.00232731]

#LogisticRegression는 릿지 회귀와 같이 게수의 제곱을 규제한다 - 'L2규제'라고 부른다
#릿지 회귀와는 반대로 C값이 증가 할 수록 규제가 약해진다
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))#0.9327731092436975
print(lr.score(test_scaled, test_target))#0.925

print(lr.predict(test_scaled[:5]))#['Perch' 'Smelt' 'Pike' 'Roach' 'Perch']
#5개 샘플에 대한 예측 확률 출력
proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))

#농어에 대한 확륙 확인
print(lr.classes_)#['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']

#coef_와 intercept_의 크기 출력
print(lr.coef_.shape, lr.intercept_.shape)#(7, 5) (7,)

#5개 샘플에 대한 z1~z7의 값 구하기
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))

from scipy.special import softmax
proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))