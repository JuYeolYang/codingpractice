import numpy as np
import pandas as pd
from scipy.sparse.construct import rand, random
from sklearn.model_selection import train_test_split
wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

#
#랜덤 포레스트
#
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)
#과대적합되어있다
print(np.mean(scores['train_score']), np.mean(scores['test_score']))#0.9973541965122431 0.8905151032797809

#중요도 확인
rf.fit(train_input,train_target)
print(rf.feature_importances_)#[0.23167441 0.50039841 0.26792718]-alcohol, sugar, pH

#oob_score 매개변수를 True로 지정하면 남는 샘플을 사용하여 부트스트랩 샘플로 훈련한 결정 트리를 평가할 수 있다
rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)
rf.fit(train_input, train_target)
print(rf.oob_score_)#0.8934000384837406

#
#엑스트라 트리
#
from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(et, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))#0.9974503966084433 0.8887848893166506

#중요도 확인
et.fit(train_input, train_target)
print(et.feature_importances_)#[0.20183568 0.52242907 0.27573525]

#
#그레이디언트 부스팅
#
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state=42)

#교차 검증 점수 확인
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))#0.8881086892152563 0.8720430147331015gb

gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2, random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))#0.9464595437171814 0.8780082549788999

gb.fit(train_input, train_target)
print(gb.feature_importances_)#[0.15872278 0.68010884 0.16116839]

#
#히스토그램 기반 그레이디언트 부스팅
#
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
hgb = HistGradientBoostingClassifier(random_state=42)
scores = cross_validate(hgb, train_input, train_target, return_train_score=True)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))#0.9321723946453317 0.8801241948619236

#permutation_importance:특성 중요도 계산
#n_repeats:랜덤하게 섞을 횟수(기본 5)
from sklearn.inspection import permutation_importance
hgb.fit(train_input, train_target)
result = permutation_importance(hgb, train_input, train_target, n_repeats=10, random_state=42, n_jobs=-1)
print(result.importances_mean)#[0.08876275 0.23438522 0.08027708]

#permutation_importance()함수가 반환하는 객체는 특성 중요도, 평균, 표준편차를 담고 있다
result = permutation_importance(hgb, test_input, test_target, n_repeats=10, random_state=42, n_jobs=-1)
print(result.importances_mean)#[0.05969231 0.20238462 0.049     ]
hgb.score(test_input, test_target)

#히스토그램 기반 그레이디언트 부스팅 알고리즘

#
#XGBoost
#
from xgboost import XGBClassifier
xgb = XGBClassifier(tree_method='hist', random_state=42)
scores = cross_validate(xgb, train_input, train_target, return_train_score=True)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))#0.9555033709953124 0.8799326275264677

#
#LightGBM
#
from lightgbm import LGBMClassifier
lgb = LGBMClassifier(random_state=42)
scores = cross_validate(lgb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))#0.935828414851749 0.8801251203079884

