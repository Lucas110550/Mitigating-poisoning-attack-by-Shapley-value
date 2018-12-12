import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F
import torchvision.models as models

import torchvision.transforms as transforms

import torch.optim as optim
import numpy as np
import scipy
from scipy.stats import mode
import pickle
import gzip
import torch.utils.data as dutils
import argparse

#parser = argparse.ArgumentParser(description=None)
#parser.add_argument('--choose', type = int, required = True)
#args = parser.parse_args()

transform = transforms.Compose(
	[#transforms.Resize(256),
#	 transforms.CenterCrop(224),
	 transforms.ToTensor(),
#	 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
#        self.fc = nn.Linear(100, 100)
#        self.fc2 = nn.Linear(100, 100)
        self.fc = nn.Linear(100, 2)
    def forward(self, x):
#    	return self.fc3(F.relu(self.fc2(F.relu(self.fc(x)))))
    	return F.log_softmax(self.fc(x))

model = LR()
model = model.cuda()
import torchvision.utils as utils
with open("CIFARtrain.pkl", "rb") as tf:
	trainX, trainy = pickle.load(tf)
with open("CIFARval.pkl", "rb") as tf:
	valX, valy = pickle.load(tf)
with open("CIFARtest.pkl", "rb") as tf:
	testX, testy = pickle.load(tf)


#choose = args.choose
#idx = np.random.permutation(35000)[:choose]
idx = [trainy[i] <= 1 for i in range(35000)]
trainX = trainX[idx]
trainy = trainy[idx]

idx = [valy[i] <= 1 for i in range(15000)]
valX = valX[idx]
valy = valy[idx]

idx = [testy[i] <= 1 for i in range(10000)]
testX = testX[idx]
testy = testy[idx]

testset = dutils.TensorDataset(torch.from_numpy(testX), torch.from_numpy(testy))
trainset = dutils.TensorDataset(torch.from_numpy(trainX), torch.from_numpy(trainy))
valset = dutils.TensorDataset(torch.from_numpy(valX), torch.from_numpy(valy))
#trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
#testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)

print(trainX.shape)
print(testX.shape)
print(valX.shape)

N = trainX.shape[0]
TN = testX.shape[0]
VN = valX.shape[0]


############################################## Preprocessing ... ######################################################
'''
P = 10000
dataloader = torch.utils.data.DataLoader(trainset, batch_size = 20, shuffle = True, num_workers = 2)
valloader = torch.utils.data.DataLoader(valset, batch_size = 20, shuffle = False, num_workers = 2)
testloader = torch.utils.data.DataLoader(testset, batch_size = 20, shuffle = False, num_workers = 2)
K = 6
G = 3 * 5 * P // 10

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

bestans = 0.0
historyval = []
historytest = []
for epoch in range(60):
	model.train()
	running_loss = 0.0
	for i, data in enumerate(dataloader, 0):
		input, label = data
#		W = value[label.numpy() // 10]
#		label = torch.from_numpy(label.numpy() % 10)
		optimizer.zero_grad()
		input = input.cuda()
		#label = label.cuda()
		label = label.cuda()
#		W = torch.from_numpy(W).cuda()
		output = model(input)
		#loss = 0
#		W = W.float()
#		bs = input.size()[0]
#		for j in range(bs):
#			loss += criterion(output[j].view(1, 10), label[j].view(1))

		loss = criterion(output, label)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
	#	if (i % 500 == 0 + 499):
	#		print("Epoch [%02d/20]: %d - loss: %03f" %(epoch, i,running_loss / (20500.0-20000)))
	#		running_loss = 0.0

	ans = 0.0
	model.eval()
	for i, data in enumerate(valloader, 0):
		input, label = data
		input = input.cuda()
		label = label.cuda()
 
		output = model(input)
		_, predicted = torch.max(output.data, 1)
		ans += (predicted == label).sum().item()

	print("Epoch [%02d/20]: val Accuracy - %03f" % (epoch, ans * 100.0 / VN))
	historyval.append(ans * 100.0 / P)
	if (ans > bestans):
		bestans = ans
		M = model
	ans = 0.0
	for i, data in enumerate(testloader, 0):
		input, label = data
		input = input.cuda()
		label = label.cuda()
		output = model(input)
		_, predicted = torch.max(output.data, 1)
		ans += (predicted == label).sum().item()

	print("Epoch [%02d/20]: test Accuracy - %03f" % (epoch, ans * 100.0 / TN))
	historytest.append(ans * 100.0 / TN)

#with open("randomshapley" + str(choose) + ".pkl", "wb") as tf:
#	pickle.dump((historyval, historytest), tf)
'''
 
import influence
import sys
sys.path.append("..")
from influence.dataset import DataSet

Train = DataSet(trainX, trainy)
Test = DataSet(testX, testy)
Validation = None
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base

dataset = base.Datasets(train = Train, validation = Validation, test = Test)

from influence.binaryLogisticRegressionWithLBFGS import BinaryLogisticRegressionWithLBFGS

num_classes = 2
input_dim = 100

weight_decay = 0.01
batch_size = 100
initial_learning_rate = 0.001 
keep_probs = None
decay_epochs = [1000, 10000]
max_lbfgs_iter = 1000

tf.reset_default_graph()

tf_model = BinaryLogisticRegressionWithLBFGS(
    input_dim=input_dim,
    weight_decay=weight_decay,
    max_lbfgs_iter=max_lbfgs_iter,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=dataset,
    initial_learning_rate=initial_learning_rate,
    keep_probs=keep_probs,
    decay_epochs=decay_epochs,
    mini_batch=False,
    train_dir='tmp',
    log_dir='tmp',
    model_name='titanic')

tf_model.train()


# Retrieve test predictions and reference labels
preds_p = tf_model.get_preds().tolist()
preds = [1 if el[0] < 0.5 else 0 for el in preds_p]
ref = tf_model.data_sets.test.labels

# True/False - Positives/Negatives    
true_pos = [(i, p) for i, p in enumerate(preds_p) if p[0] < p[1] and ref[i] == 1]
true_neg = [(i, p) for i, p in enumerate(preds_p) if p[0] > p[1] and ref[i] == 0]
false_pos = [(i, p) for i, p in enumerate(preds_p) if p[0] < p[1] and ref[i] == 0]
false_neg = [(i, p) for i, p in enumerate(preds_p) if p[0] > p[1] and ref[i] == 1]

# Confusion matrix data
print("true_positives:", len(true_pos))
print("true_negatives:", len(true_neg))
print("false_positives", len(false_pos))
print("false_negatives", len(false_neg))

# Sort true_positives and true_negatives by how confident the model is
true_pos_top = sorted(true_pos, key=lambda x : x[1][0], reverse=False)
true_neg_top = sorted(true_neg, key=lambda x : x[1][0], reverse=True)

# Sample down (top 10)
true_pos_top = true_pos_top[:N_BEST_PREDS]
true_neg_top = true_neg_top[:N_BEST_PREDS]


def get_top_train_influence(idx, n_train_points=N_INFLUENT_TRAIN):
    """
    Approximate most influential train points for a test point
    idx : index of test point
    """
    num_train = len(tf_model.data_sets.train.labels)
    influences = tf_model.get_influence_on_test_loss(
        [idx], 
        np.arange(len(tf_model.data_sets.train.labels)),
        force_refresh=True) * num_train
    influences_sorted = sorted(enumerate(influences),
                               key=lambda x:x[1],
                               reverse=True)
    influences_sorted = influences_sorted[:n_train_points]
    return influences_sorted

print(get_top_train_influence(0))