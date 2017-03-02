# Takes as input the dataset name, attribute name which is used as the class label, 
# training fraction and the name of the file containing the learned embeddings (for that dataset) 
# of a specific technique. We assume that the attribute (relationship) has been masked while learing the 
# embedding for that technique.


import sys
import json
import random
import numpy as np
from sklearn import linear_model
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# Run as python classifier.py Linkedin location 0.5 deepwalk.emb.txt

dataset = sys.argv[1]
attributeName = sys.argv[2]
trainFraction = float(sys.argv[3])
embeddingFile = sys.argv[4]


def train(trainData, trainLabels, testData, testLabels):
	print "training ..."
	logistic = linear_model.LogisticRegression()
	model = OneVsRestClassifier(logistic)
	#print trainLabels.shape
	#print trainData.shape
	model.fit(trainData,trainLabels)
	result = model.predict(testData)
	flag = {}
	count = 0
	for i in range(0, len(testData)):
		flag[i] = 0
		for j in range(0, len(testData[i])):
			if testData[i][j] > 0 and testLabels[i][j] == 1:
				flag[i] = 1
				print "test",i, j
		if flag[i] == 1:
			count += 1
	print count, len(testData)
	print "done training..."
	


def splitTrainTest(user_list, trainFraction):
	random.shuffle(user_list)
	trainLen = int(trainFraction*len(user_list))
	train = user_list[0:trainLen]
	test = user_list[trainLen:len(user_list)]
	return (train, test)

user_embeddings = {}
with open('Datasets/'+dataset+'/network.txt') as f:   
	network = json.load(f)

with open('Datasets/'+dataset+'/user_attributes.txt') as f:   
	user_attributes = json.load(f)


with open(embeddingFile) as f:
	lines = f.read().splitlines()
	for l in lines:
		split_line = l.split()
		user = int(split_line[0])
		emb = map(lambda x : float(x) , split_line[1:len(split_line)])
		user_embeddings[user] = emb


# we set window size as 10, walk length as 40, walks per vertex as 40 for deepwalk. - emb = 128.

# First get all users who are labeled with that attribute.

users_labels = {}
user_list = []
labels = set()
labelToIdx = {}
IdxToLabel = []

for u in user_attributes:
	if attributeName in user_attributes[u]:
		users_labels[int(u)] = user_attributes[u][attributeName]
		labels.update(user_attributes[u][attributeName])
		user_list.append(int(u))


print len(users_labels)
#print len(users_labels)

print len(labels)
ctr = 0
for l in labels:
	labelToIdx[l] = ctr
	IdxToLabel.append(l)
	ctr += 1


(train_users, test_users) = splitTrainTest(user_list, trainFraction)

train_data = []
train_labels = []

test_data = []
test_labels = []

for u in train_users:
	train_data.append(user_embeddings[u])


for u in train_users:
	labels_u = users_labels[u]
	label_vector = np.zeros(shape= (len(labelToIdx)))
	for l in labels_u:
		label_vector[labelToIdx[l]] = 1
	train_labels.append(label_vector)


for u in test_users:
	test_data.append(user_embeddings[u])


#print train_labels
for u in test_users:
	labels_u = users_labels[u]
	label_vector = np.zeros(shape= (len(labelToIdx), 1))
	for l in labels_u:
		label_vector[labelToIdx[l]] = 1
	test_labels.append(label_vector)

model = train(np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels))
print "hello"
#print len(set(users_labels))





