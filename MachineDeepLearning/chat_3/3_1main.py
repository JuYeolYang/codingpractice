import matplotlib.pyplot as plt
import perchdata as pd
import numpy as np

plt.scatter(pd.perch_length,pd.perch_weight)
plt.xlabel('lenght')
plt.ylabel('weight')
plt.show()

from sklearn.model_selection import train_test_split
#1차원 배열로 생성된다.
train_input, test_input, train_target, test_target = train_test_split(pd.perch_length, pd.perch_weight, random_state=42)

#1차원 배열을 2차원 배열로 바꾸는 테스트
test_array = np.array([1,2,3,4]) #(4,)
print(test_array.shape)
test_array = test_array.reshape(2,2)#(2,2)
print(test_array.shape)

train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)
print(train_input.shape, test_input.shape)#(42,1) (14,1)

from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
knr.fit(train_input, train_target)
print(knr.score(test_input, test_target))#0.992809406101064

from sklearn.metrics import mean_absolute_error
#테스트 세트에 대한 예측을 만든다.
test_prediction = knr.predict(test_input)

#테스트 세트에 대한 평균 절댓값 오차를 계산한다.
mae = mean_absolute_error(test_target, test_prediction)
print(mae)#19.157142857142862

print(knr.score(train_input, train_target))#0.9698823289099254

# 이웃의 개수를 3으로 설정
knr.n_neighbors = 3
knr.fit(train_input, train_target)
print(knr.score(train_input, train_target))#0.9804899950518966
print(knr.score(test_input, test_target))#0.9746459963987609