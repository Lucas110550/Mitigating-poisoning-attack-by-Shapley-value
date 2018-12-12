import IPython
import numpy as np

from load_animals import load_animals, load_dogfish_with_koda, load_dogfish_with_orig_and_koda
import pickle
import os
from shutil import copyfile

from influence.inceptionModel import BinaryInceptionModel
from influence.binaryLogisticRegressionWithLBFGS import BinaryLogisticRegressionWithLBFGS
import influence.experiments
from influence.dataset import DataSet
from influence.dataset_poisoning import iterative_attack, select_examples_to_attack, get_projection_to_box_around_orig_point, generate_inception_features

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import base
import argparse
from tqdm import tqdm
img_side = 299
num_channels = 3

initial_learning_rate = 0.001 
keep_probs = None
decay_epochs = [1000, 10000]

weight_decay = 0.001

num_classes = 2
max_lbfgs_iter = 1000

parser = argparse.ArgumentParser(description = None)
parser.add_argument('--dataset', type=str, required = True)
parser.add_argument('--eps', type=float, required = True)
parser.add_argument('--num', type=int, required = True)
args = parser.parse_args()

k = args.k
datasetname = args.dataset
poison_num = args.num

### DogFish, jg12, gpu1
# num_train_ex_per_class = 900
# num_test_ex_per_class = 300
# batch_size = 100

# dataset_name = 'dogfish_%s_%s' % (num_train_ex_per_class, num_test_ex_per_class)
# data_sets = load_animals(
#	 num_train_ex_per_class=num_train_ex_per_class, 
#	 num_test_ex_per_class=num_test_ex_per_class,
#	 classes=['dog', 'fish'])

### DogFish with Koda, jg13, gpu0
batch_size = 30
dataset_name = 'dogfish_koda'
data_sets = load_dogfish_with_koda(datasetname)	

### DogFish with orig and Koda, jg12, gpu2
# batch_size = 30
# dataset_name = 'dogfish_orig_and_koda'
# data_sets = load_dogfish_with_orig_and_koda()	


full_graph = tf.Graph()
top_graph = tf.Graph()

print(data_sets.train.labels.shape)
print(data_sets.test.labels.shape)
print('*** Full:')
with full_graph.as_default():
	full_model_name = '%s_inception_wd-%s' % (dataset_name, weight_decay)
	full_model = BinaryInceptionModel(
		img_side=img_side,
		num_channels=num_channels,
		weight_decay=weight_decay,
		num_classes=num_classes, 
		batch_size=batch_size,
		data_sets=data_sets,
		initial_learning_rate=initial_learning_rate,
		keep_probs=keep_probs,
		decay_epochs=decay_epochs,
		mini_batch=True,
		train_dir='output',
		log_dir='log',
		model_name=full_model_name)

	for data_set, label in [
		(data_sets.train, 'train'),
		(data_sets.validation, 'validation'),
		(data_sets.test, 'test')]:

		inception_features_path = 'output/%s_inception_features_new_%s.npz' % (dataset_name, label)
		if not os.path.exists(inception_features_path):

			print('Inception features do not exist. Generating %s...' % label)
			data_set.reset_batch()
			
			num_examples = data_set.num_examples
			assert num_examples % batch_size == 0

			inception_features_val = generate_inception_features(
				full_model, 
				data_set.x, 
				data_set.labels, 
				batch_size=batch_size)
			
			np.savez(
				inception_features_path, 
				inception_features_val=inception_features_val,
				labels=data_set.labels)


train_f = np.load('output/%s_inception_features_new_train.npz' % dataset_name)
train = DataSet(train_f['inception_features_val'], train_f['labels'])
test_f = np.load('output/%s_inception_features_new_test.npz' % dataset_name)
test = DataSet(test_f['inception_features_val'], test_f['labels'])
validation_f = np.load('output/%s_inception_features_new_validation.npz' % dataset_name)
validation = DataSet(validation_f['inception_features_val'], validation_f['labels'])
print(len(train_f['labels']))


#validation = None

inception_data_sets = base.Datasets(train=train, validation=None, test=test)

print('*** Top:')
with top_graph.as_default():
	top_model_name = '%s_inception_onlytop_wd-%s' % (dataset_name, weight_decay)
	input_dim = 2048
	top_model = BinaryLogisticRegressionWithLBFGS(
		input_dim=input_dim,
		weight_decay=weight_decay,
		max_lbfgs_iter=max_lbfgs_iter,
		num_classes=num_classes, 
		batch_size=batch_size,
		data_sets=inception_data_sets,
		initial_learning_rate=initial_learning_rate,
		keep_probs=keep_probs,
		decay_epochs=decay_epochs,
		mini_batch=False,
		train_dir='output',
		log_dir='log',
		model_name=top_model_name)
	top_model.train()
	weights = top_model.sess.run(top_model.weights)
	orig_weight_path = 'output/inception_weights_%s.npy' % top_model_name
	np.save(orig_weight_path, weights)


with full_graph.as_default():
	full_model.load_weights_from_disk(orig_weight_path, do_save=False, do_check=True)
	full_model.reset_datasets()
import shapley
from shapley import get_value 

print('Creating poisoned dataset...')

step_size = 0.02

num_train = len(top_model.data_sets.train.labels)
print("**************" + str(num_train))
num_test = len(top_model.data_sets.test.labels)
print("**************" + str(num_test))
max_num_to_poison = 10
loss_type = 'normal_loss'


### Try attacking each test example individually

orig_X_train = np.copy(data_sets.train.x)
orig_Y_train = np.copy(data_sets.train.labels)
print(data_sets.validation.x.shape)

success = 0
robust = 0
total = 0
count = 0
pp = 0
qq = 0
ww = 0
truth = 0
M = top_model



with open("poison.pkl", "rb") as tf:
	idx = pickle.load(tf)
pbar = tqdm(range(600), unit='steps', ascii=True)
for test_idx in pbar:
	top_model = M
	test_indices = [test_idx]
	#print(top_model.data_sets.test.x.shape)
	test_predX = top_model.sess.run(top_model.preds, feed_dict=top_model.fill_feed_dict_with_some_ex(
		top_model.data_sets.test,
		test_indices))

	predvalue = test_predX[0, int(full_model.data_sets.test.labels[test_indices])]
	if (predvalue >= 0.5):
		count += 1
		with open(datasetname + "Cache%d.%d.pkl" % (poison_num, count), "rb") as tf:
			inceptiontrainX, inceptiontrainy, perm, perm2 = pickle.load(tf)
						
			mark = np.ones(perm.shape[0])
		
			for i in range(int(perm.shape[0] * k)):
				mark[perm[i]] = 0
			XX = top_model.data_sets.train.x
			YY = top_model.data_sets.train.labels
			X = []
			y = []
			for i in range(perm.shape[0]):
				if (mark[i] == 0):
					X.append(inceptiontrainX[i].reshape(1, 2048))
					y.append(inceptiontrainy[i])
			X = np.concatenate(X)
			y = np.array(y)
			top_model.update_train_x_y(X, y)
			top_model.train()
			test_predX = top_model.sess.run(top_model.preds, feed_dict=top_model.fill_feed_dict_with_some_ex(
				top_model.data_sets.test,
				test_indices))

			predvalue = test_predX[0, int(full_model.data_sets.test.labels[test_indices])]
			if (predvalue >= 0.5):
				pp += 1
			top_model.update_train_x_y(XX, YY)
			top_model.train()
			
			perm = perm2
			mark = np.ones(perm.shape[0])
		
			for i in range(int(perm.shape[0] * k)):
				mark[perm[i]] = 0
			XX = top_model.data_sets.train.x
			YY = top_model.data_sets.train.labels
			X = []
			y = []
			for i in range(perm.shape[0]):
				if (mark[i] == 0):
					X.append(inceptiontrainX[i].reshape(1, 2048))
					y.append(inceptiontrainy[i])
			X = np.concatenate(X)
			y = np.array(y)
			top_model.update_train_x_y(X, y)
			top_model.train()
			test_predX = top_model.sess.run(top_model.preds, feed_dict=top_model.fill_feed_dict_with_some_ex(
				top_model.data_sets.test,
				test_indices))

			predvalue = test_predX[0, int(full_model.data_sets.test.labels[test_indices])]
			if (predvalue >= 0.5):
				qq += 1
			top_model.update_train_x_y(XX, YY)
			top_model.train()
			
			perm = perm2
			mark = np.zeros(perm.shape[0])
		
			for i in range(poison_num):
				mark[int(idx[count - 1][i])] = 1
			XX = top_model.data_sets.train.x
			YY = top_model.data_sets.train.labels
			X = []
			y = []
			for i in range(perm.shape[0]):
				if (mark[i] == 0):
					X.append(inceptiontrainX[i].reshape(1, 2048))
					y.append(inceptiontrainy[i])
			X = np.concatenate(X)
			y = np.array(y)
			top_model.update_train_x_y(X, y)
			top_model.train()
			test_predX = top_model.sess.run(top_model.preds, feed_dict=top_model.fill_feed_dict_with_some_ex(
				top_model.data_sets.test,
				test_indices))

			predvalue = test_predX[0, int(full_model.data_sets.test.labels[test_indices])]
			if (predvalue >= 0.5):
				truth += 1
			top_model.update_train_x_y(XX, YY)
			top_model.train()
			
			mark = np.ones(perm.shape[0])
			perm = np.random.permutation(perm.shape[0])
			for i in range(int(perm.shape[0] * k)):
				mark[perm[i]] = 0
			XX = top_model.data_sets.train.x
			YY = top_model.data_sets.train.labels
			X = []
			y = []
			for i in range(perm.shape[0]):
				if (mark[i] == 0):
					X.append(inceptiontrainX[i].reshape(1, 2048))
					y.append(inceptiontrainy[i])
			X = np.concatenate(X)
			y = np.array(y)
			top_model.update_train_x_y(X, y)
			top_model.train()
			test_predX = top_model.sess.run(top_model.preds, feed_dict=top_model.fill_feed_dict_with_some_ex(
				top_model.data_sets.test,
				test_indices))

			predvalue = test_predX[0, int(full_model.data_sets.test.labels[test_indices])]
			if (predvalue >= 0.5):
				ww += 1
			top_model.update_train_x_y(XX, YY)
			top_model.train()
			
			pbar.set_description('Current => Strong = %d, shapley = %d, inf = %d, random = %d' % (truth, pp, qq, ww))
print(count)
print(truth)
print(pp)
print(qq)
print(ww)
