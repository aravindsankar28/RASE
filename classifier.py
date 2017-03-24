# Takes as input the dataset name, attribute name which is used as the class label, 
# training fraction and the name of the file containing the learned embeddings (for that dataset) 
# of a specific technique. We assume that the attribute (relationship) has been masked while learing the 
# embedding for that technique.

# Parameter settings for node2vec and deepwalk.
# We set window size as 10, walk length as 40, walks per vertex as 40 for deepwalk. - emb = 128.
# First get all users who are labeled with that attribute.

import sys
import json
import random
import numpy as np
from sklearn import linear_model
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import operator
# Run as python classifier.py Linkedin location 0.5 deepwalk.emb.txt
dataset = sys.argv[1]
attributeName = sys.argv[2]
trainFraction = float(sys.argv[3])
embeddingFile = sys.argv[4]

def readEmbeddings(filename):
	user_embeddings = {}
	with open(filename) as f:
		lines = f.read().splitlines()
		for l in lines:
			split_line = l.split()
			user = int(split_line[0])
			emb = map(lambda x : float(x) , split_line[1:len(split_line)])
			user_embeddings[user] = emb
	return user_embeddings

def buildLabelIndex(attributeName, user_attributes):
	users_labels = {}
	labels = set()
	user_list = []
	labelToIdx = {}
	IdxToLabel = []
	for u in user_attributes:
		if attributeName in user_attributes[u]:
			users_labels[int(u)] = user_attributes[u][attributeName]
			labels.update(user_attributes[u][attributeName])
			user_list.append(int(u))
	ctr =0
	for l in labels:
		labelToIdx[l] = ctr
		IdxToLabel.append(l)
		ctr += 1
	numLabels = len(labelToIdx)
	for u in user_list:
		users_labels[int(u)] = map(lambda x : labelToIdx[x] ,user_attributes[str(u)][attributeName])
	return (user_list,users_labels,labelToIdx,IdxToLabel, numLabels)

# Filter data based on a subset of labels
def filterData(data, labels, labels_subset):
	retain = []
	for i in range(0, len(labels)):
		for j in range(0, len(labels[i])):
			if labels[i][j] == 1 and j in labels_subset:
				retain.append(i)
				break
	data_filtered = []
	labels_filtered = []
	for i in retain:
		data_filtered.append(data[i])
		temp = []
		for j in labels_subset:
			temp.append(labels[i][j])
		labels_filtered.append(temp)
	return (np.array(data_filtered), np.array(labels_filtered))

def createDataLabelVectors(users, user_embeddings, users_labels, numLabels):
	labels = []
	data = []
	for u in users:
		data.append(user_embeddings[u])	
	for u in users:
		labels_u = users_labels[u]
		label_vector = np.zeros(shape= (numLabels))
		for l in labels_u:
			label_vector[l] = 1
		labels.append(label_vector)
	return (data, labels)

def splitTrainTest(user_list, trainFraction, user_embeddings, users_labels, numLabels):
	random.shuffle(user_list)
	trainLen = int(trainFraction*len(user_list))
	train_users = user_list[0:trainLen]
	test_users = user_list[trainLen:len(user_list)]
	(train_data, train_labels) = createDataLabelVectors(train_users, user_embeddings, users_labels, numLabels)
	(test_data, test_labels) = createDataLabelVectors(test_users, user_embeddings, users_labels, numLabels)
	return (train_data, train_labels, test_data, test_labels)

def train(trainData, trainLabels, testData, testLabels):
	print "training ..."
	label_count = {}
	for i in range(0, len(trainLabels[0])):
		label_count[i] =0 

	for i in range(0, len(trainData)):
		for j in range(0, len(trainLabels[i])):
			if trainLabels[i][j] == 1:
				label_count[j] += 1
	print "sorting the label counts"
	label_count_sorted = sorted(label_count.items(), key = operator.itemgetter(1), reverse = True)	
	labels_retain = []
	labels_retain_new = []
	print "removing sparse labels"
	for i in range(0, 20):
		labels_retain.append(label_count_sorted[i][0])

	print "Old train and test", trainData.shape, testData.shape
	(trainData_filtered, trainLabels_filtered) = filterData(trainData, trainLabels, labels_retain)
	(testData_filtered, testLabels_filtered) = filterData(testData, testLabels, labels_retain)

	print "New train data ", trainData_filtered.shape,
	print "label ", trainLabels_filtered.shape
	print "New test data", testData_filtered.shape,
	print "label", testLabels_filtered.shape

	logistic = linear_model.LogisticRegression()
	model = OneVsRestClassifier(logistic)
	model.fit(trainData_filtered,trainLabels_filtered)
	result = model.predict(testData_filtered)
	flag = {}
	count = 0
	for i in range(0, len(testData_filtered)):
		flag[i] = 0
		for j in range(0, len(result[i])):
			if result[i][j]>0:
				flag[i] += 1
			if result[i][j] > 0 and testLabels_filtered[i][j] == 1:
				flag[i] = 1
		if flag[i] > 0:
			count += 1
	print count, len(testData_filtered), count*1.0/len(testData_filtered)
	print "done training..."


user_embeddings = readEmbeddings(embeddingFile)
with open('Datasets/'+dataset+'/network.txt') as f:   
	network = json.load(f)

with open('Datasets/'+dataset+'/user_attributes.txt') as f:   
	user_attributes = json.load(f)

(user_list, users_labels, IdxToLabel, labelToIdx, numLabels) = buildLabelIndex(attributeName, user_attributes)

(train_data, train_labels, test_data, test_labels) =  splitTrainTest(user_list, trainFraction, user_embeddings, users_labels, numLabels)

model = train(np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels))

