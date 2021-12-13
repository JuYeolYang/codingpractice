import pandas as pd
from scipy.sparse.construct import random
wine = pd.read_csv('https://bit.ly/wine_csv_data')
print(wine.head())
'''
   alcohol  sugar    pH  class
0      9.4    1.9  3.51    0.0
1      9.8    2.6  3.20    0.0
2      9.8    2.3  3.26    0.0
3      9.8    1.9  3.16    0.0
4      9.4    1.9  3.51    0.0
'''
#데이터프레임의 각 열의 데이터 타입과 누락된 데이터가 있는지 확인하는데 유용하다
#데이터프레임의 요약된 정보를 출력한다
print(wine.info())

#열에 대한 간략한 통계를 출력한다
#mean:평균, std:표준편차, min:최소, max:최대, 중간값 1사분위수, 3사분위수도 알려준다
print(wine.describe())

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split
#train_test_split()함수의 테스트세트 지정 디폴트 설정값은 25%이므로 test_size로 테스트세트 지정 값을 바꿀 수 있다
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
print(train_input.shape, test_input.shape)#(5197, 3) (1300, 3), 3개의 특성을 가지고 있다

#훈련,테스트세트 변환
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))#0.7808350971714451
print(lr.score(test_scaled, test_target))#0.7776923076923077
#계수, 절편 확인
print(lr.coef_, lr.intercept_)#[[ 0.51270274  1.6733911  -0.68767781]] [1.81777902]

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))#0.996921300750433
print(dt.score(test_scaled, test_target))#0.8592307692307692

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(10,7))
plot_tree(dt)
plt.show()

#트리의 깊이 제한해서 출력
#max_depth:1로 맞추면 루트 노드를 데외하고 하나의 노드를 더 확장하여 그린다
#filled:클래스에 맞게 노드의 색을 칠할 수 있다
#feature_names:특성의 이름을 전달할 수 있다
plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))#0.8454877814123533
print(dt.score(test_scaled, test_target))#0.8415384615384616

plt.figure(figsize=(20, 15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

#결정 트리 알고리즘은 특성값의 스케일에 영향을 받지 않는다
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)
#전처리 했을 때와 결과는 같다
print(dt.score(train_input, train_target))#0.8454877814123533
print(dt.score(test_input, test_target))#0.8415384615384616

#같은 트리이지만 특성값을 표준점수로 바꾸지 않아 이해하기 쉽다
plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

#특성 중요도는 feature_importances_ 속성에 저장되어 있다
#특성 중요도는 각 노드의 정보 이득과 전체 샘플에 대한 비율을 곱한 후 특성별로 더하여 계산한다
print(dt.feature_importances_)#[0.12345626 0.86862934 0.0079144 ]-['alcohol', 'sugar', 'pH']순
