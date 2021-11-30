from sklearn.model_selection import train_test_split
import perchData as pd
import numpy as np

#훈련 세트와 테스트 세트로 나눈다.
train_input, test_input, train_target, test_target = train_test_split(pd.perch_length, pd.perch_weight, random_state=42)
#훈련 세트와 테스트 세트를 2차원 배열로 바꿉니다.
train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)


from sklearn.neighbors import KNeighborsClassifier

knr = KNeighborsClassifier(n_neighbors=3)
#k-최근접 이웃 회귀 모델을 훈련합니다.
knr.fit(train_input, train_target)
print(knr.predict([[50]]))#[1000.]

import matplotlib.pyplot as plt

#50cm 농어의 이웃을 구합니다.
distances, indexes = knr.kneighbors([[50]])

#훈련 세트의 산점도를 그립니다.
plt.scatter(train_input, train_target)

#훈련 세트 중에서 이웃 샘플만 다시 그립니다.
plt.scatter(train_input[indexes], train_target[indexes], marker='D')

#50cm 농어 데이터
plt.scatter(50, 1000, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

print(np.mean(train_target[indexes]))#1033.3333333333333
print(knr.predict([[100]]))

#100cm 농어의 이웃을 구합니다.
distances, indexes = knr.kneighbors([[100]])

#훈련 세트의 산점도를 그립니다.
plt.scatter(train_input, train_target)

#훈련 세트 중에서 이웃 샘플만 다시 그린다.
plt.scatter(train_input[indexes], train_target[indexes], marker='D')

#100cm 농어의 데이터
plt.scatter(100, 1033, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

#선형 회귀 모델을 훈련합니다.
lr.fit(train_input, train_target)

#50cm 농어에 대해 예측합니다.
print(lr.predict([[50]]))#[1241.83860323]

print(lr.coef_, lr.intercept_)#[39.01714496] -709.0186449535474

#훈련 세트의 산점도를 그립니다.
plt.scatter(train_input, train_target)

#15에서 50까지 1차 방정식 그래프를 그립니다.
plt.plot([15,50], [15 * lr.coef_ + lr.intercept_, 50 * lr.coef_ + lr.intercept_])

#50cm 농어 데이터
plt.scatter(50, 1241.8, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

print(lr.score(train_input, train_target))#0.9398463339976041
print(lr.score(test_input, test_target))#0.824750312331356

train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))

print(train_poly.shape, test_poly.shape)#(42, 2) (14, 2)

lr = LinearRegression()
lr.fit(train_poly, train_target)

print(lr.predict([[50**2, 50]])) #[1573.98423528]

print(lr.coef_, lr.intercept_) #[  1.01433211 -21.55792498] 116.05021078278259

#구간별 직선을 그리기 위해 15에서 49까지 정수 배열을 만든다.
point = np.arange(15, 50)

#훈련 센트의 산점도를 그린다.
plt.scatter(train_input, train_target)

#15에서 49가지 2차 방정식 그래프를 그린다.
plt.plot(point, 1.01 * point ** 2 - 21.6 * point + 116.05)

#50cm 농어 데이터
plt.scatter(50, 1574, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

print(lr.score(train_poly, train_target))#0.9706807451768623
print(lr.score(test_poly, test_target))#0.9775935108325121