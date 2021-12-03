import matplotlib.pyplot as plt
import BreamData as bd, SmeltData as sd


#print("hello world")

#show BreamData 
plt.scatter(bd.bream_length, bd.bream_weight)
plt.xlabel('length') #x축은 길이
plt.ylabel('weigth') #y축은 무게
plt.show()


#show SmeltData and BreamData
plt.scatter(bd.bream_length, bd.bream_weight)
plt.scatter(sd.smelt_length, sd.smelt_weight)
plt.xlabel('length') #x축은 길이
plt.ylabel('weigth') #y축은 무게
plt.show()

#chain the bream and smelt data
length = bd.bream_length + sd.smelt_length
weight = bd.bream_weight + sd.smelt_weight

#skikit-learn
fish_data = [[l, w] for l, w in zip(length, weight)]
print(fish_data)

#make fish target : beram is '1', smelt is '0'
fish_target = [1] * 35 + [0] * 14
print(fish_target)

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()

#training
kn.fit(fish_data, fish_target)

#show how it works : 0~1
print(kn.score(fish_data, fish_target)) #will show 1.0

#predict the data[30, 600]
print(kn.predict([[30, 600]])) #will show array[1]

#print fish_data
print(kn._fit_X)
print(kn._y)#[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]


kn49 = KNeighborsClassifier(n_neighbors=49)
kn49.fit(fish_data, fish_target)
print(kn49.score(fish_data, fish_target))#0.7142857142857143

