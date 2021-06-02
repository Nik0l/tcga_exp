# Using example test set -- works
import random
import numpy as np
from mil.data.datasets import musk1
from mil.metrics import AUC, BinaryAccuracy
from mil.validators import KFold
from mil.trainer.trainer import Trainer
from mil.models import LogisticRegression, SVC
from mil.bag_representation.mapping import DiscriminativeMapping, MILESMapping
from mil.preprocessing import StandarizerBagsList
from sklearn.model_selection import train_test_split
import pickle


def mil_example():
    (bags_train, y_train), (bags_test, y_test) = musk1.load()
    trainer = Trainer()
    metrics = [AUC, BinaryAccuracy]
    model = LogisticRegression(solver='liblinear', C=1, class_weight='balanced')
    pipeline = [('scale', StandarizerBagsList()), ('disc_mapping', DiscriminativeMapping(m=30))]
    trainer.prepare(model, preprocess_pipeline=pipeline, metrics=metrics)
    valid = KFold(n_splits=2, shuffle=True)
    history = trainer.fit(bags_train, y_train, sample_weights='balanced', validation_strategy=valid, verbose=1)


# Using WSI dataset -- too slow
def pickleOpen(file_name):
    open_file = open(file_name, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()
    return(loaded_list)


def an_example():
    bags_subset = pickleOpen('data/sophie_ML/bags_subset.pkl')
    labels_subset = pickleOpen('data/sophie_ML/labels_subset.pkl')
    X_train, X_test, y_train, y_test = train_test_split(bags_subset, labels_subset, stratify=labels_subset)
    trainer = Trainer()
    metrics = [AUC, BinaryAccuracy]
    model = LogisticRegression(solver='liblinear', C=1, class_weight='balanced')
    pipeline = [('scale', StandarizerBagsList()), ('disc_mapping', DiscriminativeMapping(m=30))]
    trainer.prepare(model, preprocess_pipeline=pipeline, metrics=metrics)
    valid = KFold(n_splits=2, shuffle=True)
    history = trainer.fit(X_train, y_train, sample_weights='balanced', validation_strategy=valid, verbose=1)
    #history = trainer.fit(X_train, y_train, sample_weights='balanced', validation_strategy=None, verbose=1)


def mil_test(path_to_data, use_smaller_bags=True, n_patches_max=100):

    bags_subset = pickleOpen(path_to_data + 'bags_subset.pkl')
    labels_subset = pickleOpen(path_to_data + 'labels_subset.pkl')

    X_train, X_test, y_train, y_test = train_test_split(bags_subset, labels_subset, stratify=labels_subset)
    X_train = [l.tolist() for l in X_train]

    for train_bag in X_train:
        print('%d samples in a train bag' % len(train_bag))
        print('Embeddings are of %d size' % len(train_bag[0]))

    # to speed up training, one can limit max number of patches per WSI:
    if use_smaller_bags:
        X_train_small = []
        for train_bag in X_train:
            print('%d samples in a bag' % len(train_bag))
            new_bag = random.sample(train_bag, n_patches_max)
            X_train_small.append(new_bag)
        X_train = X_train_small

    trainer = Trainer()
    metrics = [AUC, BinaryAccuracy]
    model = LogisticRegression(solver='liblinear', C=1, class_weight='balanced')
    pipeline = [('scale', StandarizerBagsList()), ('disc_mapping', DiscriminativeMapping(m=30))]
    trainer.prepare(model, preprocess_pipeline=pipeline, metrics=metrics)
    # there are no many positive examples (only one?), you can see it if printing y_train vector:
    print(y_train)
    # this means that with Kfold where n=2 you end up with a training without positive examples
    # you can fix it by a. validation_strategy=None b. include enough positive samples; this can be done by increasing
    # the training dataset or by oversampling the underrepresenting dataset, for example, duplicate positive samples
    # y_train = np.array([0, 0, 0, 0, 1, 1, 1])
    valid = KFold(n_splits=2, shuffle=True)
    history = trainer.fit(X_train, y_train, sample_weights='balanced', validation_strategy=valid, verbose=1)


# an axample what must work
#mil_example()
# wrapped the function for training on the embeddings
mil_test(path_to_data='/Users/kmwx127/Downloads/data_for_ML/', use_smaller_bags=True, n_patches_max=100)
