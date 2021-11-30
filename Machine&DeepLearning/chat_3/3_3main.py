from re import S
import pandas as pd
import perchData as perchdata
import numpy as np

df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()
print(perch_full)

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(perch_full, perchdata.perch_weight, random_state=42)

from sklearn.preprocessing import PolynomialFeatures

#
poly = PolynomialFeatures()
poly.fit([[2, 3]])
print(poly.transform([[2, 3]]))#[[1. 2. 3. 4. 6. 9.]]

poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
print(train_poly.shape)#(42, 9)

print(poly.get_feature_names_out())

test_poly = poly.transform(test_input)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))#0.9903183436982124

poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print(train_poly.shape)#(42, 55)

lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))#0.9999999999994632
print(lr.score(test_poly, test_target))#-144.40578108808137 -> 과대적합

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)



#릿지 모델 만들기
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))#0.9896101671037343
print(ridge.score(test_scaled, test_target))#0.9790693977615388

import matplotlib.pyplot as plt
train_score = []
test_score = []


alpha_list = [0.001, 0.01, 0.1, 1, 10 ,100]
for alpha in alpha_list:
    #릿지 모델을 만든다.
    ridge = Ridge(alpha=alpha)
    #릿지 모델을 훈련합니다.
    ridge.fit(train_scaled, train_target)
    #훈련 점수와 텟트 점수를 저장합니다.
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))
    
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))#0.9903815817570368
print(ridge.score(test_scaled, test_target))#0.9827976465386795


#라쏘 회귀 만들기
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

lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))#0.9888067471131866
print(lasso.score(test_scaled, test_target))#0.9824470598706696


print(np.sum(lasso.coef_ == 0))#40