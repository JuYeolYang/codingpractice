import TrainingData as td

fish_data = [[l, w] for l, w in zip(td.fish_length, td.fish_weight)]
fish_target = [1] * 35 + [0] * 14

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
print(fish_data[4])
print(fish_data[0:5])# fish_data[:5]

train_input = fish_data[:35]
train_target = fish_target[:35]
test_input = fish_data[35:]
test_target = fish_target[35:]

kn = kn.fit(train_input, train_target)
print(kn.score(test_input, test_target)) # result = 0.0

import numpy as np

#change the fish_data array into numpy array
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

print(input_arr)
print(input_arr.shape) # print(sample count, feature count) == (49,2)

np.random.seed(42) # make seed
index = np.arange(49)
np.random.shuffle(index)
print(index) # result = [13 45 47 44 17 27 26 25 31 19 12 4 34 8 3 6 40 41 46 15 9 16 24 33 30 0 43 32 5 29 11 36 1 21 2 37 35 23 39 10 22 18 48 20 7 42 14 28 38]

print(input_arr[[1,3]]) # result = [[ 26.3 290. ], [ 29.  363. ]]

train_input = input_arr[index[:35]] #normal array -> numpy array
train_target = target_arr[index[:35]] #normal array -> numpy array
print(input_arr[13], train_input[0]) # result = [ 32. 340.] [ 32. 340.]

test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

import matplotlib.pyplot as plt
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(test_input[:,0], test_input[:,1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

#training
kn = kn.fit(train_input, train_target)
print(kn.score(test_input, test_target))

#predict test_input
print(kn.predict(test_input))
print(test_target)