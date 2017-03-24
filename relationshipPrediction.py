# Here, we take an relation name as input and predict links on that relation.
# First, we create positive and negative pairs for that relation type. 
# Then, we do an 80-20 train test split to train a logitic regression classifer to 
# predict the existence of relation R. 

import sys
import json
import random
import numpy as np 
from sklearn import linear_model
from sklearn.model_selection import train_test_split

dataset = sys.argv[1]
attributeName = sys.argv[2]
embeddingFile = sys.argv[3]
operatorTypes = ["AVG", "HAD","L1", "L2"]
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

def samplePositiveLinks(attributeName, network):
	# Get all links of type R. Return pairs of nodes.
	positiveExamples = []
	for e in network:
		if attributeName in network[e]:
			positiveExamples.append((int(e.split()[0]),int(e.split()[1])))
	return positiveExamples

def sampleNegativeLinks(attributeName, network, n):
	# Get an equal # links which are not related by R.
	remainingLinks = []
	for e in network:
		if attributeName not in  network[e]:
			remainingLinks.append((int(e.split()[0]),int(e.split()[1])))
	random.shuffle(remainingLinks)
	return remainingLinks[0:n]


def getLinkScore(fu, fv, operator):
	fu = np.array(fu)
	fv = np.array(fv)
	if operator == "AVG":
		return (fv+fv)/2.0
	elif operator == "HAD":
		return np.multiply(fu,fv)
	elif operator == "L1":
		return np.abs(fu-fv)
	elif operator =="L2":
		return (fu-fv)**2
	else:
		print "ERROR"
		return -1

# Operator can be one of 4 types :
# 1) AVG 2) HAD 3) L1 4) L2
def getLinkFeatures(links, user_embeddings, operator):
	features = []
	for l in links:
		a = l[0]
		b = l[1]
		f = getLinkScore(user_embeddings[a], user_embeddings[b], operator)
		features.append(f)
	return features


def splitTrainTest(user_list, trainFraction, user_embeddings):
	random.shuffle(user_list)
	trainLen = int(trainFraction*len(user_list))
	train_users = user_list[0:trainLen]

def evaluate(positiveExamples, negativeExamples, user_embeddings):

	for operator in operatorTypes:
		positiveExamplesFeatures = np.array(getLinkFeatures(positiveExamples, user_embeddings, operator))
		negativeExamplesFeatures = np.array(getLinkFeatures(negativeExamples, user_embeddings, operator))

		positiveLabels = np.array([1]*len(positiveExamplesFeatures))
		negativeLabels = np.array([-1]*len(negativeExamplesFeatures))

		data = np.vstack((positiveExamplesFeatures, negativeExamplesFeatures))
		labels = np.append(positiveLabels, negativeLabels)
		X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2)
		print X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
		logistic = linear_model.LogisticRegression()
		logistic.fit(X_train,Y_train)
		test_predict = logistic.predict(X_test)
		count = 0.0
		for i in range(0, len(test_predict)):
			if test_predict[i] == Y_test[i]:
				count += 1
		print operator, "Accuracy", count, len(Y_test), count/len(Y_test)
		#print result	


user_embeddings = readEmbeddings(embeddingFile)
with open('Datasets/'+dataset+'/network.txt') as f:   
	network = json.load(f)

with open('Datasets/'+dataset+'/user_attributes.txt') as f:   
	user_attributes = json.load(f)

positiveExamples = samplePositiveLinks(attributeName, network)
negativeExamples = sampleNegativeLinks(attributeName, network, len(positiveExamples))


evaluate(positiveExamples, negativeExamples, user_embeddings)
