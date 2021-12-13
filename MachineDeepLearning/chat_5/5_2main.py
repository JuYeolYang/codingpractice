import pandas as pd
from scipy.sparse.construct import rand
wine = pd.read_csv('https://bit.ly/wine_csv_data')
data=wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
#검증 세트
sub_input, val_input, sub_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)
print(sub_input.shape, val_input.shape)#(4157, 3) (1040, 3)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)
print(dt.score(sub_input, sub_target))#0.9908167229431201
print(dt.score(val_input, val_target))#0.864423076923077

#교차 검증
#cross_validate()함수는 기본적으로 5-폴드 교차 검증을 수행한다
#fit_time:훈련하는 시간, score_time:검증하는 시간, test_score:교차 검증의 최종 점수
from sklearn.model_selection import cross_validate
scores = cross_validate(dt, train_input, train_target)
print(scores)
'''
{
'fit_time': array([0.00594783, 0.00567293, 0.005409  , 0.00499105, 0.0044229 ]), 
'score_time': array([0.00070405, 0.00056696, 0.00058222, 0.00046587, 0.00044489]), 
'test_score': array([0.298845  , 0.169733  , 0.30308435, 0.2243649 , 0.08649028])
}
'''

import numpy as np
print(np.mean(scores['test_score']))#0.855300214703487

#위의 교차 검증 코드와 아래 코드는 동일하다
from sklearn.model_selection import StratifiedKFold
scores = cross_validate(dt, train_input, train_target, cv=StratifiedKFold())
print(np.mean(scores['test_score']))

#10-폴드 교차 검증
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean(scores['test_score']))#.8574181117533719

#그리드 서치
#cv 매개변수 기본값은 5이다-5폴드 교차 검증
#n_jobs:cpu코어 수 지정
#params:탐색할 모델의 매개변수와 확률 분포 객체
from sklearn.model_selection import GridSearchCV
params = {'min_impurity_decrease':[0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)

#교차 검증에서 하이퍼파라미터를 찾으면 전체 훈련 세트로 모델을 다시 만들어야한다
#그리드 서치는 훈련이 끝나면 25개의 모델 중에 검증 점수가 가장 높은 모델의 매개변수 조합으로 전체 훈련 세트에서 자동으로 다시 모델을 훈련한다
dt = gs.best_estimator_
print(dt.score(train_input, train_target))#0.9615162593804117

#최적의 매견수는 best_params_ 속성에 저장되어 있다
print(gs.best_params_)#{'min_impurity_decrease': 0.0001}

#각 매개변수에서 수행한 교차 검증의 편균점수는 cv_results_ 속성의 'mean_test_score'키에 저장되어 있다
print(gs.cv_results_['mean_test_score'])#[0.86819297 0.86453617 0.86492226 0.86780891 0.86761605]

#넘파이 argmax()함수를 사용하여 가장 큰 값의 인덱스를 추출 후 이 인덱스를 사용해서 params키에 저장된 매개변수를 출력할 수 있다
#gs.best_params_와 동일하다
best_index = np.argmax(gs.cv_results_['mean_test_score'])
print(gs.cv_results_['params'][best_index])#{'min_impurity_decrease': 0.0001}

#이 매개변수로 수행할 교차 검증 횟수 : 9 x 15 x 10 = 1,350개
#5-폴드 교차검증 수행 시 모델의 수 6,750개
params = {'min_impurity_decrease':np.arange(0.0001, 0.001, 0.0001),
          'max_depth':range(5, 20, 1),
          'min_samples_split':range(2, 100, 10)
          }
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
print(gs.best_params_)#{'max_depth': 14, 'min_impurity_decrease': 0.0004, 'min_samples_split': 12}
print(np.max(gs.cv_results_['mean_test_score']))#0.8683865773302731

#랜덤 서치
from scipy.stats import uniform, randint

#randint사용
#10개의 숫자 샘플링
rgen = randint(0, 10)
print(rgen.rvs(10))#[1 7 3 6 7 5 4 2 0 5]
#1000개의 숫자 샘플링, return_counts:고유의 값 개수 배열 반환 여부
print(np.unique(rgen.rvs(1000), return_counts=True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([101,  85,  87, 105, 109, 106,  94, 105,  95, 113]))

#uniform사용
#0~1사이에서 10개의 실수 추출
ugen = uniform(0, 1)
print(ugen.rvs(10))
#[0.31724145 0.0801095  0.9166998  0.79340702 0.57182555 0.57973792 0.35539882 0.22341284 0.32951723 0.844419  ]

#min_samples_leaf:리프 노드가 되기 위한 최소 샘플의 개수
params = {'min_impurity_decrease':uniform(0.0001,0.001),
          'max_depth':randint(20, 50),
          'min_samples_split':randint(2, 25),
          'min_samples_leaf':randint(1,25),
          }

from sklearn.model_selection import RandomizedSearchCV
gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_iter=100, n_jobs=-1, random_state=42)
gs.fit(train_input, train_target)
print(gs.best_params_)
'''
{
'max_depth': 39,
'min_impurity_decrease': 0.00034102546602601173,
'min_samples_leaf': 7,
'min_samples_split': 13}
'''
print(np.max(gs.cv_results_['mean_test_score']))#0.8695428296438884

#best_estimator_속성에 전체 훈련세트로 훈련된 최적의 모델이 저장되어있다
dt = gs.best_estimator_
print(dt.score(test_input, test_target))#0.86