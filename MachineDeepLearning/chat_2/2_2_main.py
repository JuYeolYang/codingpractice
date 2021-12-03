import numpy as np
import TrainingData as td

print(np.column_stack(([1,2,3], [4,5,6])))

#make tuple into (fish_length, fish_weight)
fish_data = np.column_stack((td.fish_length, td.fish_weight))
print(fish_data[:5])
print(np.ones(5))

#             -----35----- ----14---
#make target (1,1,1,,...,1,0,0,...,0)
fish_target = np.concatenate((np.ones(35), np.zeros(14)))
print(fish_target)

#make training input and target, test input and target
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42)
print(train_input.shape, test_input.shape)
print(train_target.shape, test_target.shape)

print(test_target) # it shows source of bias

train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)
#print(train_input)
print(test_target)#solve the source of bias


from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
print(kn.score(test_input, test_target)) # 1.0
print(kn.predict([[25, 150]])) #[0.]

import matplotlib.pyplot as plt
#train_iniput[:,0] = fish length, train_input[:,1] = fish weigth
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

distances, indexes = kn.kneighbors([[25, 150]])

plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
print(train_input[indexes])
print(train_target[indexes]) #[[1. 0. 0. 0. 0.]]
print(distances)#[[ 92.00086956 130.48375378 130.73859415 138.32150953 138.39320793]]


plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
plt.xlim((0, 1000))
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

#평균 구하기
mean = np.mean(train_input, axis=0)
#표준편차 구하기
std = np.std(train_input, axis=0)
print(mean ,std)#[ 27.29722222 454.09722222] [  9.98244253 323.29893931]
#표준점수 구하기
train_scaled = (train_input - mean) / std #broadcasting

plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(25, 150, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

new = ([25, 150] - mean) / std
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

kn.fit(train_scaled, train_target)

test_scaled = (test_input - mean) / std
print(kn.score(test_scaled, test_target)) #1.0
print(kn.predict([new]))#[1.]



distances, indexes = kn.kneighbors([new])
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_scaled[indexes,0], train_scaled[indexes,1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()