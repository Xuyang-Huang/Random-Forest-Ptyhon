#-- coding: utf-8 --
#@Time : 2021/4/10 19:53
#@Author : HUANG XUYANG
#@Email : xhuang032@e.ntu.edu.sg
#@File : Random_Forest.py
#@Software: PyCharm

import sklearn.datasets as sk_dataset
import numpy as np
from Decision_Tree import DecisionTree
import multiprocessing as mp
import os


class RandomForest:
    """
    Attributes:
        __n_tree: An integer of tree number.
        __min_leaf: An integer of minimum leaf.
        __n_class: An integer of class number.
        tree: A list, stored all decision trees.
        __split_method: String, choose 'gini' or 'entropy'.
        __pruning_prop: A floating number, proportion of data number to prune DT.
        __n_try: n_try: AAn integer or None or string, if 'sqrt' use sqrt(n_feature), if 'log2' use log2(n_feature),
            if None use all n_feature, if integer number of feature pick randomly.

    """
    def __init__(self, n_tree, min_leaf, n_try, split_method='gini', pruning_prop=0.2):
        self.__n_tree = n_tree
        self.__min_leaf = min_leaf
        self.__n_class = None
        self.tree = []
        self.__split_method = split_method
        self.__pruning_prop = pruning_prop
        self.__n_try = n_try

    def train(self, data, label, n_class, multi_processing=True):
        """Train random forest.

        :param data: A 2-D Numpy array.
        :param label: A 1-D Numpy array.
        :param n_class: An integer of class number.
        :param multi_processing: A bool, if True using multi-processing.
        :return:
        """
        self.__n_class = n_class
        if multi_processing:
            n_cores = mp.cpu_count() - 1
            p = mp.pool.Pool(n_cores)
            results = [p.apply_async(self.train_mp, args=(data, label, self.__n_tree // n_cores)) for _ in range(n_cores)]
            if self.__n_tree % n_cores != 0:
                results.append(p.apply_async(self.train_mp, args=(data, label, self.__n_tree % n_cores)))
            results = [p.get() for p in results]

            [self.tree.extend(item) for item in results]

        else:
            while len(self.tree) < self.__n_tree:
                train, val = self.bagging(data, label)
                dt = DecisionTree(self.__min_leaf, self.__split_method, self.__pruning_prop, self.__n_try)
                dt.train(train[0], train[1], self.__n_class, val[0], val[1])
                if dt.root is not None:
                    self.tree.append(dt)

    def train_mp(self, data, label, n_tree):
        """Training on multi-processing.

        :param data: A 2-D Numpy array.
        :param label: A 1-D Numpy array.
        :param n_tree: An integer of tree number.
        :return: decision trees.
        """
        sub_tree = []
        while len(sub_tree) < n_tree:
            train, val = self.bagging(data, label)
            dt = DecisionTree(self.__min_leaf, self.__split_method, self.__pruning_prop, self.__n_try)
            dt.train(train[0], train[1], self.__n_class, val[0], val[1])
            if dt.root is not None:
                sub_tree.append(dt)
        print('finish pid:', os.getpid())
        return sub_tree

    def predict(self, data):
        """Traverse RF then get a result.

        :param data: A 2-D Numpy array.
        :return: Prediction.
        """
        raw_result = [[] for _ in range(self.__n_class)]
        result_bin = []
        result = np.zeros(len(data))

        for i in range(self.__n_tree):
            _result = self.tree[i].predic(data)
            for j in range(self.__n_class):
                raw_result[j].extend(_result[j])

        for i in range(self.__n_class):
            result_bin.append(np.bincount(raw_result[i], minlength=self.__n_class))

        for i in range(len(data)):
            result[i] = np.argmax([result_bin[j][i] for j in range(self.__n_class)])

        return result

    def eval(self, data, label):
        """

        :param data: A 2-D Numpy array.
        :param label: A 1-D Numpy array.
        :return: Prediction, Accuracy.
        """
        raw_result = [[] for _ in range(self.__n_class)]
        result_bin = []
        result = np.zeros(len(data))

        for i in range(self.__n_tree):
            _result, _, _ = self.tree[i].eval(data, label)
            for j in range(self.__n_class):
                raw_result[j].extend(_result[j])

        for i in range(self.__n_class):
            result_bin.append(np.bincount(raw_result[i], minlength=len(data)))

        for i in range(len(data)):
            result[i] = np.argmax([result_bin[j][i] for j in range(self.__n_class)])

        acc = np.mean(result == label)

        return result, acc

    def bagging(self, data, label):
        """Bagging method to prepare input data.

        :param data: A 2-D Numpy array.
        :param label: A 1-D Numpy array.
        :return: (train data and label), (val data and label).
        """
        train_rand_index = np.arange(len(data))
        np.random.shuffle(train_rand_index)
        val_rand_index = train_rand_index[int(0.65 * len(data)):]
        train_rand_index = train_rand_index[:int(0.65 * len(data))]
        train_rand_index = np.random.choice(train_rand_index, len(data))
        return (data[train_rand_index], label[train_rand_index]), (data[val_rand_index], label[val_rand_index])


def prepare_data(proportion):
    dataset = sk_dataset.load_wine()
    label = dataset['target']
    data = dataset['data']
    n_class = len(dataset['target_names'])

    shuffle_index = np.arange(len(label))
    np.random.shuffle(shuffle_index)

    train_number = int(proportion * len(label))
    train_index = shuffle_index[:train_number]
    val_index = shuffle_index[train_number:]
    data_train = data[train_index]
    label_train = label[train_index]
    data_val = data[val_index]
    label_val = label[val_index]
    return (data_train, label_train), (data_val, label_val), n_class


if __name__ == '__main__':
    minimum_leaf = 1
    num_tree = 50

    train, val, num_class = prepare_data(0.8)
    num_try = int(np.sqrt(train[0].shape[1]))
    rf = RandomForest(n_tree=num_tree, min_leaf=minimum_leaf, n_try=num_try, split_method='entropy', pruning_prop=0.2)
    rf.train(train[0], train[1], num_class, multi_processing=True)
    _, train_acc = rf.eval(train[0], train[1])
    pred, val_acc = rf.eval(val[0], val[1])
    print('train_acc', train_acc)
    print('val_acc', val_acc)
