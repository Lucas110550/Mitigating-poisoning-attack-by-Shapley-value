import os
import pickle
from tensorflow.contrib.learn.python.learn.datasets import base
import numpy as np
import IPython

from subprocess import call

from keras.preprocessing import image

from influence.dataset import DataSet
from influence.inception_v3 import preprocess_input

BASE_DIR = 'data' # TODO: change

def fill(X, Y, idx, label, img_path, img_side):
    img = image.load_img(img_path, target_size=(img_side, img_side))
    x = image.img_to_array(img)
    X[idx, ...] = x
    Y[idx] = label

     
def extract_and_rename_animals():
    class_maps = [
        ('dog', 'n02084071'),
        ('cat', 'n02121808'),
        ('bird', 'n01503061'),
        ('fish', 'n02512053'),
        ('horse', 'n02374451'),
        ('monkey', 'n02484322'),
        ('zebra', 'n02391049'),
        ('panda', 'n02510455'),
        ('lemur', 'n02496913'),
        ('wombat', 'n01883070'),
        ]


    for class_string, class_id in class_maps:
        
        class_dir = os.path.join(BASE_DIR, class_string)
        print(class_dir)
        call('mkdir %s' % class_dir, shell=True)
        call('tar -xf %s.tar -C %s' % (os.path.join(BASE_DIR, class_id), class_dir), shell=True)
        
        for filename in os.listdir(class_dir):

            file_idx = filename.split('_')[1].split('.')[0]
            src_filename = os.path.join(class_dir, filename)
            dst_filename = os.path.join(class_dir, '%s_%s.JPEG' % (class_string, file_idx))
            os.rename(src_filename, dst_filename)



def load_animals(num_train_ex_per_class=300, 
                 num_test_ex_per_class=100,
                 num_valid_ex_per_class=0,
                 classes=None,name="ImageNet"
                 ):    

    num_channels = 3
    img_side = 299

    num_classes = len(classes)
    num_train_examples = num_train_ex_per_class * num_classes
    num_test_examples = num_test_ex_per_class * num_classes
    num_valid_examples = num_valid_ex_per_class * num_classes

    print('Loading animals from disk...')
    
    if (name != "ImageNet"):
        if (name == 'MNIST'):
            with open("MTrain.pkl", "rb") as tf:
                tX, ty = pickle.load(tf)
            with open("MTest.pkl", "rb") as tf:
                X_test, Y_test = pickle.load(tf)
        else:
            with open("CTrain.pkl", "rb") as tf:
                tX, ty = pickle.load(tf)
            with open("CTest.pkl", "rb") as tf:
                X_test, Y_test = pickle.load(tf)

        X_train = tX[:1200]
        X_valid = tX[1200:]
        Y_valid = ty[1200:]
        Y_train = ty[:1200]
    else:
        f = np.load("ImageNet.pkl")
        X_train = f['X_train'][:1200]
        Y_train = f['Y_train'][:1200]

        X_test = f['X_test']
        Y_test = f['Y_test']

        X_valid = f['X_train'][1200:]
        Y_valid = f['Y_train'][1200:]


    train = DataSet(X_train, Y_train)
    if (X_valid is not None) and (Y_valid is not None):
        validation = DataSet(X_valid, Y_valid)
    else:
        validation = None

    test = DataSet(X_test, Y_test)

    return base.Datasets(train=train, validation=validation, test=test)


def load_koda():
    num_channels = 3
    img_side = 299

    data_filename = os.path.join(BASE_DIR, 'dataset_koda.npz')

    if os.path.exists(data_filename):
        print('Loading Koda from disk...')
        f = np.load(data_filename)
        X = f['X']
        Y = f['Y']
    else:
        # Returns all class 0
        print('Reading Koda from raw images...')

        image_files = [image_file for image_file in os.listdir(os.path.join(BASE_DIR, 'koda')) if (image_file.endswith('.jpg'))]
        # Hack to get the image files in the right order
        # image_files = [image_file for image_file in os.listdir(os.path.join(BASE_DIR, 'koda')) if (image_file.endswith('.jpg') and not image_file.startswith('124'))]
        # image_files += [image_file for image_file in os.listdir(os.path.join(BASE_DIR, 'koda')) if (image_file.endswith('.jpg') and image_file.startswith('124'))]


        num_examples = len(image_files)
        X = np.zeros([num_examples, img_side, img_side, num_channels])
        Y = np.zeros([num_examples])

        class_idx = 0
        for counter, image_file in enumerate(image_files):
            img_path = os.path.join(BASE_DIR, 'koda', image_file)
            fill(X, Y, counter, class_idx, img_path, img_side)

        X = preprocess_input(X)

        np.savez(data_filename, X=X, Y=Y)

    return X, Y
    

def load_dogfish_with_koda(datasetname):        
    classes = ['dog', 'fish']
    #X_test, Y_test = load_koda()

    data_sets = load_animals(num_train_ex_per_class=900, 
                 num_test_ex_per_class=300,
                 num_valid_ex_per_class=400,
                 classes=classes, name=datasetname)
    train = data_sets.train
    validation = data_sets.validation
    print(train.x.shape)
    print(validation.x.shape)
    test = data_sets.test
    #test = DataSet(X_test, Y_test)
    print(test.x.shape)
    return base.Datasets(train=train, validation=validation, test=test)


def load_dogfish_with_orig_and_koda():
    classes = ['dog', 'fish']
    X_test, Y_test = load_koda()
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    data_sets = load_animals(num_train_ex_per_class=900, 
                 num_test_ex_per_class=300,
                 num_valid_ex_per_class=0,
                 classes=classes)
    train = data_sets.train
    validation = data_sets.validation

    test = DataSet(
        np.concatenate((data_sets.test.x, X_test), axis=0), 
        np.concatenate((data_sets.test.labels, Y_test), axis=0))

    return base.Datasets(train=train, validation=validation, test=test)

