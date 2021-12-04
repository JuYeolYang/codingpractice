from re import S
import pandas as pd
import perchdata
import numpy as np

df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()
print(perch_full)

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(perch_full, perchdata.perch_weight, random_state=42)

#PolynomiaFeatures는 변환기 클래스이다.
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures()
poly.fit([[2, 3]])
print(poly.transform([[2, 3]]))#[[1. 2. 3. 4. 6. 9.]], 1은 절편

#include_bias -> 절편 추가 여부, false -> 절편 추가하지 않음
poly = PolynomialFeatures(include_bias=False)#사이킷런 모델은 include=False로 지정하지않아도 자동으로 특성에 추가된 절편 항을 무시한다.

#훈련세트 변환
poly.fit(train_input)
train_poly = poly.transform(train_input)
print(train_poly.shape)#(42, 9), 샘플 수 : 42, 특성 수 : 9
#어떻게 특성을 만들었는지 출력해준다.
print(poly.get_feature_names_out())#['x0' 'x1' 'x2' 'x0^2' 'x0 x1' 'x0 x2' 'x1^2' 'x1 x2' 'x2^2']

#테스트세트 변환
test_poly = poly.transform(test_input)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))#0.9903183436982124

#고차항 특성 만들기
poly = PolynomialFeatures(degree=5, include_bias=False)#5제곱까지 특성 만듬
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print(train_poly.shape)#(42, 55), 샘플 수:42, 특성 수:55

lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))#0.9999999999994632
print(lr.score(test_poly, test_target))#-144.40578108808137 -> 과대적합-훈련세트에 너무 맞춰 훈련됬다

#StandardScaler도 변환기 클래스 중 하나이다
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)



#릿지 모델 만들기-계수를 제곱한 값을 기준으로 규제한다
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))#0.9896101671037343
print(ridge.score(test_scaled, test_target))#0.9790693977615388


#alpha값을 찾는 방법 중 하나는 R^2값의 그래프를 그려보는 것이다
import matplotlib.pyplot as plt
train_score = []
test_score = []

#alpha값이 클 수록 규제 강도가 세지므로 과소적합으로 유도하고
#alpha값이 작을 수록 규제 강도가 약해져 과대적합이 될 가능성이 커진다
alpha_list = [0.001, 0.01, 0.1, 1, 10 ,100]
for alpha in alpha_list:
    #릿지 모델을 만든다.
    ridge = Ridge(alpha=alpha)
    #릿지 모델을 훈련합니다.
    ridge.fit(train_scaled, train_target)
    #훈련 점수와 텟트 점수를 저장합니다.
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))
    
#이대로 그래프를 그리면 그래프를 확인하기 어렵기 때문에 밑이 10인 로그함수를 사용한다
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

ridge = Ridge(alpha=0.1)#alpha = 0.1일 때 두 그래프가 가장 가깝고 테스트 세테의 점수가 가장 높았다
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))#0.9903815817570368
print(ridge.score(test_scaled, test_target))#0.9827976465386795


#라쏘 회귀 만들기-계수의 절댓값을 기준으로 규제한다
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(train_scaled, train_target)
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))#0.989789897208096
print(lasso.score(test_scaled, test_target))#0.9800593698421886

train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    #라쏘 모델을 만듭니다.
    lasso = Lasso(alpha=alpha, max_iter=10000)
    #라쏘 모델을 훈련합니다.
    lasso.fit(train_scaled, train_target)
    #훈련 점수와 테스트 점수를 저장합니다.
    train_score.append(lasso.score(train_scaled, train_target))
    test_score.append(lasso.score(test_scaled, test_target))
    
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()    

lasso = Lasso(alpha=10)#alpha = 10일 때 두 그래프가 가깝고 테스트점수가 높다
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))#0.9888067471131866
print(lasso.score(test_scaled, test_target))#0.9824470598706696

#라쏘 모델의 계수는 coef_ 속성에 저장되어 있다
#라쏘 모델을 유용한 특성을 골라내는 용도로 사용할 수 있다
print(np.sum(lasso.coef_ == 0))#계수가 0이된 계수가 40개가 된다 -> 라쏘 모델이 사용한 특성은 15개 밖에 되지 않는다