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
from PIL import Image

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

batch_size = 30
dataset_name = 'dogfish_koda'
data_sets = load_dogfish_with_koda(datasetname)	



full_graph = tf.Graph()
top_graph = tf.Graph()
ref_graph = tf.Graph()
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
other = base.Datasets(train=train, validation=validation, test=test)
ref_data_sets = base.Datasets(train = train, validation = None, test = validation)
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
print('*** Ref:')
with ref_graph.as_default():
	ref_model_name = '%s_inception_onlyref_wd-%s' % (dataset_name, weight_decay)
	input_dim = 2048
	ref_model = BinaryLogisticRegressionWithLBFGS(
		input_dim=input_dim,
		weight_decay=weight_decay,
		max_lbfgs_iter=max_lbfgs_iter,
		num_classes=num_classes, 
		batch_size=batch_size,
		data_sets=ref_data_sets,
		initial_learning_rate=initial_learning_rate,
		keep_probs=keep_probs,
		decay_epochs=decay_epochs,
		mini_batch=False,
		train_dir='output',
		log_dir='log',
		model_name=ref_model_name)
	ref_model.train()

with full_graph.as_default():
	full_model.load_weights_from_disk(orig_weight_path, do_save=False, do_check=True)
	full_model.reset_datasets()
import shapley
from shapley import get_value 

### Create poisoned dataset
print('Creating poisoned dataset...')

step_size = 0.02
num_train = len(top_model.data_sets.train.labels)
print("**************" + str(num_train))
num_test = len(top_model.data_sets.test.labels)
print("**************" + str(num_test))
max_num_to_poison = poison_num
loss_type = 'normal_loss'



orig_X_train = np.copy(data_sets.train.x)
orig_Y_train = np.copy(data_sets.train.labels)
print(data_sets.validation.x.shape)

success = 0
robust = 0
total = 0
for test_idx in range(0, 600):

	print('****** Attacking test_idx %s ******' % test_idx)
	test_description = test_idx

	filenames = [filename for filename in os.listdir('./output') if (
		(('%s_attack_%s_testidx-%s_trainidx-' % (full_model.model_name, loss_type, test_description)) in filename) and		
		(filename.endswith('stepsize-%s_proj_final.npz' % step_size)))]
	with top_graph.as_default():
		top_model.get_influence_on_test_loss(
			[test_idx], 
			[0],		
			test_description=test_description,
			force_refresh=True)
	copyfile(
		'output/%s-cg-normal_loss-test-%s.npz' % (top_model_name, test_description),
		'output/%s-cg-normal_loss-test-%s.npz' % (full_model_name, test_description))

	with full_graph.as_default():
		grad_influence_wrt_input_val = full_model.get_grad_of_influence_wrt_input(
			np.arange(num_train), 
			[test_idx], 
			force_refresh=False,
			test_description=test_description,
			loss_type=loss_type)	
		all_indices_to_poison = select_examples_to_attack(
			full_model, 
			max_num_to_poison, 
			grad_influence_wrt_input_val,
			step_size=step_size)
	M = top_model
	for num_to_poison in [poison_num]:
		indices_to_poison = all_indices_to_poison[:num_to_poison]
		orig_X_train_subset = np.copy(full_model.data_sets.train.x[indices_to_poison, :])
		orig_X_train_inception_features_subset = np.copy(top_model.data_sets.train.x[indices_to_poison, :])
		project_fn = get_projection_to_box_around_orig_point(orig_X_train_subset, box_radius_in_pixels=0.5)
		test_indices = [test_idx]
		print(top_model.data_sets.test.x.shape)
		top_model = M
		test_predX = top_model.sess.run(top_model.preds, feed_dict=top_model.fill_feed_dict_with_some_ex(
			top_model.data_sets.test,
			test_indices))
		mask, poisonX, idx, inceptionrecover, inceptiontrainX, inceptiontrainy, inceptiontestX, inceptiontesty = iterative_attack(top_model, full_model, top_graph, full_graph, project_fn, [test_idx], test_description=test_description, 
			indices_to_poison=indices_to_poison,
			num_iter=100,
			step_size=step_size,
			save_iter=50,
			loss_type=loss_type,
			early_stop=0.5)

		cur = False
		if (test_predX[0, int(full_model.data_sets.test.labels[test_indices])] >= 0.5):
			cur = True
		if (cur == True):
			total += 1
		if (mask == True and cur == True):
			success += 1
		valX = other.validation.x
		valy = other.validation.labels
		value = get_value(inceptiontrainX, inceptiontrainy, valX, valy, 6)
		with ref_graph.as_default():
			ref_model.update_train_x(inceptiontrainX)
			ref_model.train()
			value2 = ref_model.get_influence_on_test_loss(
				np.arange(len(ref_model.data_sets.test.labels)), 
				np.arange(len(ref_model.data_sets.train.labels)),		
				test_description = "all",
				force_refresh=True)
		print("Shapley value has been calculated")
		perm = np.argsort(-value2)
		perm = np.random.permutation(value.shape[0])
		ordperm = np.zeros(perm.shape[0], dtype=np.int64)
		mark = np.ones(perm.shape[0])
		for i in range(perm.shape[0]):
			ordperm[perm[i]] = i
	
		for i in range(int(perm.shape[0] * k)):
			mark[perm[i]] = 0
		for i in range(idx.shape[0]):
			print(int(ordperm[idx[i]]))
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

		print(X.shape)
		print(y.shape)



				#vutil.save_image(torch.from_numpy(data_sets.test.x[test_idx]), "Picture%d.png" % (test_idx))
		with top_graph.as_default():
			test_predx = top_model.sess.run(top_model.preds, feed_dict=top_model.fill_feed_dict_with_some_ex(
				top_model.data_sets.test,
				test_indices))
			top_model = M
			top_model.update_train_x_y(X, y)
			top_model.train()
			weights = top_model.sess.run(top_model.weights)
			test_pred = top_model.sess.run(top_model.preds, feed_dict=top_model.fill_feed_dict_with_some_ex(
				top_model.data_sets.test,
				test_indices))
		print("Origin: %03f" % (test_predX[0, int(full_model.data_sets.test.labels[test_indices])]))
		print("Before: %03f" % (test_predx[0, int(full_model.data_sets.test.labels[test_indices])]))
		print("After: %03f" % (test_pred[0, int(full_model.data_sets.test.labels[test_indices])]))

		mm = False
		if (test_pred[0, int(full_model.data_sets.test.labels[test_indices])] < 0.5 and cur == True):
			robust += 1

		if (test_pred[0, int(full_model.data_sets.test.labels[test_indices])] >= 0.5 and cur == True):
			mm = True
		if (cur == True):
				print("Good data, Saving... ")
				with open(datasetname + "Cache" + "%d.%d" % (num_to_poison, total) + ".pkl", "wb") as tf:
					pickle.dump((inceptiontrainX, inceptiontrainy, np.argsort(-value), np.argsort(-value2)), tf)
				#G = data_sets.test.x[test_idx]
				#G = G / 2.0 + 0.5
				#G = (G * 255.0).astype(np.uint8)
				#print(G.shape)
				#G = G.reshape(3, 299, 299)[0]
				#im = Image.fromarray(G)
				#im.save("G%d.png" % (test_idx))
				#T = poisonX
				#T = T / 2.0 + 0.5
				#T = (T * 255.0).astype(np.uint8)
				#print(T.shape)
				#T = T.reshape(3, 299, 299)[0]
				#im = Image.fromarray(T)
				#im.save("T%d.png" % (test_idx))

				#T = orig_X_train_subset
				#T = T / 2.0 + 0.5
				#T = (T * 255.0).astype(np.uint8)
				#print(T.shape)
				#T = T.reshape(3, 299, 299)[0]
				#im = Image.fromarray(T)
				#im.save("TT%d.png" % (test_idx))
		print("Current Success %d, Robust %d in %d (%d)" % (success, robust, total, test_idx + 1))
		with full_graph.as_default():
			# X_train = full_model.data_sets.train.x
			# X_train[idx_to_poison, :] = orig_X_train_subset
			# full_model.update_train_x(X_train)
			full_model.update_train_x_y(orig_X_train, orig_Y_train)
			full_model.load_weights_from_disk(orig_weight_path, do_save=False, do_check=False)
		with top_graph.as_default():
			X_train = XX
			X_train[indices_to_poison, :] = orig_X_train_inception_features_subset
			top_model = M
			top_model.update_train_x_y(X_train, YY)
			top_model.train()
			ref_model.update_train_x_y(X_train, YY)
			ref_model.train()
	
print("Measure Attack success rate (%d in %d)" % (success, total))
print("Measure Attack success rate after defense (%d in %d)" % (robust, total))