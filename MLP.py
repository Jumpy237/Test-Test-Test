from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import csv
import random
import numpy as np
import csv


dataset = []
targets = []
with open('golf_dataset_preprocessed.csv', newline='') as csvfile:
    linereader = csv.reader(csvfile, delimiter=',', quotechar='|')

    next(linereader)
    
    for row in linereader:
        dataset.append(row)
        


d = len(dataset) - round(len(dataset) * 0.3) # 70% of dataset
train = dataset[:d] #X is 30% of dataset for training
test = dataset[d:] #Y is 30% of dataset for testing

train = np.asarray(train, dtype=np.int) #convert 2d list into numpy array
test = np.asarray(test, dtype=np.int)

#print(train)

x_train = train[:,:-1]
y_train = train[:,-1]
X_test = test[:,:-1]
Y_test = test[:,-1]
#print(x_train)
#print(y_train)

mlp = MLPClassifier(solver='lbfgs',max_iter=400)
mlp.fit(x_train,y_train)
Y_pred = mlp.predict(X_test)

print('Accuracy: %.2f' % accuracy_score(Y_test, Y_pred))


print(confusion_matrix(Y_test, Y_pred))



