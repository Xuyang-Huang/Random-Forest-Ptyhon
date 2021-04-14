#-- coding: utf-8 --
#@Time : 2021/4/9 14:20
#@Author : HUANG XUYANG
#@Email : xhuang032@e.ntu.edu.sg
#@File : Cart_Decision_Tree_on_Continuous_Value_by_Recursion.py
#@Software: PyCharm

import numpy as np
import sklearn.datasets as sk_dataset

"""
This decision tree can be used on continuous or discrete data.

"""


class TreeNode:
    """A decision tree node

    Attributes:
        feature_index: An integer of feature index, specify the decision feature.
        children: A list of children nodes.
        children_class: A list of children nodes class belonging.
        children_acc: A list of floating number,  children accuracy for pruning.
        acc: A floating number, this node's accuracy while pruning.
        is_discrete: A bool, if this node's feature is discrete.
        children_values: A list of unique value in this feature index.
        thr: A floating number of threshold to split the data.

    """
    def __init__(self, feature_index, thr, is_discrete):
        self.feature_index = feature_index
        self.children = []
        self.children_class = []
        self.children_acc = []
        self.acc = None
        self.is_discrete = is_discrete
        if self.is_discrete:
            self.children_values = thr
        else:
            self.thr = thr

    def split(self, data, label=None):
        """Split the input data.

        :param data: A 2-D Numpy array.
        :param label: A 1-D Numpy array.
        :return: [left split data, right split data], [left split data ground truth, right split data ground truth]
        """
        if self.is_discrete:
            children_data = []
            children_label = []
            for i in range(len(self.children_values)):
                mask = data[:, self.feature_index] == self.children_values[i]
                children_data.append(data[mask])
                if label is not None:
                    children_label.append(label[mask])

            if label is not None:
                return children_data, children_label
            return children_data

        else:
            left_mask = data[:, self.feature_index] <= self.thr
            if left_mask.all():
                left_mask = data[:, self.feature_index] < self.thr
            if np.unique(data[:, self.feature_index]).__len__() == 1:
                left_mask[:len(data)//2] = True
            right_mask = ~left_mask

            left_leaf_data = data[left_mask]
            right_leaf_data = data[right_mask]

            if label is not None:
                left_leaf_label = label[left_mask]
                right_leaf_label = label[right_mask]
                return [left_leaf_data, right_leaf_data], [left_leaf_label, right_leaf_label]
            else:
                return [left_leaf_data, right_leaf_data]

    def split_predict(self, data, index, label=None):
        """

        :param data: A 2-D numpy array.
        :return: [left split data, right split data],
                 [left split data index, right split data index]
                 [left split data ground truth, right split data ground truth]
        """
        if self.is_discrete:
            children_data = []
            children_index = []
            children_label = []
            for i in range(len(self.children_values)):
                mask = data[:, self.feature_index] == self.children_values[i]
                children_data.append(data[mask])
                children_index.append(index[mask])
                if label is not None:
                    children_label.append(label[mask])

            if label is not None:
                return children_data, children_index, children_label
            return children_data, children_index

        else:
            left_mask = data[:, self.feature_index] <= self.thr
            if left_mask.all():
                left_mask = data[:, self.feature_index] < self.thr
            if np.unique(data[:, self.feature_index]).__len__() == 1:
                left_mask[:len(data)//2] = True
            right_mask = ~left_mask

            left_leaf_data = data[left_mask]
            right_leaf_data = data[right_mask]

            left_leaf_data_index = index[left_mask]
            right_leaf_data_index = index[right_mask]
            if label is not None:
                left_leaf_label = label[left_mask]
                right_leaf_label = label[right_mask]
                return [left_leaf_data, right_leaf_data], [left_leaf_data_index, right_leaf_data_index], [left_leaf_label, right_leaf_label]
            else:
                return [left_leaf_data, right_leaf_data], [left_leaf_data_index, right_leaf_data_index]


class DecisionTree:
    """
    Attributes:
        root: A decision tree node class.
        __min_leaf: An integer of minimum leaf number.
        __n_class: An integer of class number.
        split_method: String, choose 'gini' or 'entropy'.
        __pruning_prop: A floating number, proportion of data number to prune DT.
        __pruning: Bool, if True prune the DT, or not.
        __n_try: An integer or None or string, if 'sqrt' use sqrt(n_feature), if 'log2' use log2(n_feature),
            if None use all n_feature, if integer number of feature pick randomly.

    """
    def __init__(self, min_leaf, split_method='gini', pruning_prop=None, n_try=None):
        self.root = None
        self.__min_leaf = min_leaf
        self.__n_class = None
        c = Criterion()
        self.__train_node = getattr(c, split_method)
        self.__pruning_prop = pruning_prop
        self.__pruning = pruning_prop is not None
        self.__n_try = n_try

    def train(self, data, label, n_class, pruning_data=None, pruning_label=None):
        """Train a decision tree.

        Using recursion to train a DT.

        :param data: A 2-D Numpy array.
        :param label: A 1-D Numpy array.
        :param n_class: An integer of class number.
        :return: No return.
        """
        self.__n_class = n_class
        assert (pruning_label is None) == (pruning_data is None), 'Please provide data and label both or not!'

        if self.__pruning:
            if pruning_data is None:
                pruning_data = data[:int(len(data) * self.__pruning_prop)]
                pruning_label = label[:int(len(label) * self.__pruning_prop)]
                data = data[int(len(data) * self.__pruning_prop):]
                label = label[int(len(label) * self.__pruning_prop):]

        is_discrete_feature = [isinstance(item, str) for item in range(data.shape[1])]

        def grow(_data, _label):
            # Train single node.
            _feature_index, _thr, _is_discrete = self.__train_node(_data, _label, is_discrete_feature, self.__n_try)
            _node = TreeNode(_feature_index, _thr, _is_discrete)
            _split_data, _split_label = _node.split(_data, _label)

            if np.sum([len(_split_label[j]) == 0 for j in range(len(_split_data))]) > 0:
                return None
            for i in range(len(_split_data)):
                _node.children_class.append(np.argmax(np.bincount(_split_label[i])))
                _node.children_acc.append(None)
                if not ((len(_split_label[i]) <= self.__min_leaf) | (_split_label[i] == _split_label[i][0]).all()):
                    _node.children.append(grow(_split_data[i], _split_label[i]))
                else:
                    _node.children.append(None)

            return _node

        self.root = grow(data, label)
        if self.__pruning:
            self.__post_pruning(pruning_data, pruning_label)

    def __post_pruning(self, data, label):
        """Prune DT.

        :param data: A 2-D Numpy array.
        :param label: A 1-D Numpy array.
        :return:
        """

        if self.root is None:
            return None
        index = np.arange(len(data))

        def __inference(root, _data, _index, _label):
            _split_data, _split_data_index, _split_label = root.split_predict(_data, _index, _label)
            if len(_label) == 0:
                root.acc = 0
            else:
                root.acc = np.mean(_label == np.argmax(np.bincount(_label)))
            for i in range(len(root.children_class)):
                if root.children[i] is None:
                    if len(_split_data[i]) != 0:
                        tmp_acc = np.mean(_split_label[i] == root.children_class[i])
                    else:
                        tmp_acc = 0
                    root.children_acc[i] = tmp_acc
                else:
                    __inference(root.children[i], _split_data[i], _split_data_index[i], _split_label[i])

            if np.sum([root.children_acc[j] is None for j in range(len(root.children_class))]) == 0:
                child_acc = np.mean([_acc for _acc in root.children_acc])
                if root.acc > child_acc:
                    del root.children
                    root.children = [None] * len(root.children_class)
                    root.children_acc = [None] * len(root.children_class)
        __inference(self.root, data, index, label)

    def predict(self, data):
        """Traverse DT get a result.

        :param data: A 2-D Numpy array.
        :return: Prediction.
        """
        result = [[] for _ in range(self.__n_class)]
        index = np.arange(len(data))

        def __inference(root, _data, _index):
            if len(_data) == 0:
                return None
            _split_data, _split_data_index, _split_label = root.split_predict(_data, _index)
            for i in range(len(root.children_class)):
                if len(_split_data[i]) == 0:
                    break
                if root.children[i] is None:
                    if len(_split_data[i]) != 0:
                        result[root.children_class[i]].extend(_split_data_index[i])
                else:
                    __inference(root.children[i], _split_data[i], _split_data_index[i])
            return None

        __inference(self.root, data, index)
        return result

    def eval(self, data, label):
        """

        :param data: A 2-D Numpy array.
        :param label: A 1-D Numpy array.
        :return: Prediction, Prediction with label, Accuracy.
        """
        result = [[] for _ in range(self.__n_class)]
        result_label = [[] for _ in range(self.__n_class)]
        index = np.arange(len(data))

        def __inference(root, _data, _index, _label):
            if len(_data) == 0:
                return None
            _split_data, _split_data_index, _split_label = root.split_predict(_data, _index, _label)
            for i in range(len(root.children_class)):
                if root.children[i] is None:
                    if len(_split_data[i]) != 0:
                        result[root.children_class[i]].extend(_split_data_index[i])
                        result_label[root.children_class[i]].extend(_split_label[i])
                else:
                    __inference(root.children[i], _split_data[i], _split_data_index[i], _split_label[i])
            return None

        __inference(self.root, data, index, label)
        acc = []
        for i in range(self.__n_class):
            if len(result_label[i]) == 0:
                acc.append(0)
            else:
                acc.append(np.mean(np.array(result_label[i]) == i))
        acc = np.mean(acc)
        return result, result_label, acc


class Criterion:
    def gini(self, data, label, is_discrete_feature, n_try=None):
        """Traverse all features and value, find the best split feature and threshold.

        :param data: A 2-D Numpy array.
        :param label: A 1-D Numpy array.
        :param is_discrete_feature: A list of bool for all features if they are discrete or not.
        :param n_try: AAn integer or None or string, if 'sqrt' use sqrt(n_feature), if 'log2' use log2(n_feature),
            if None use all n_feature, if integer number of feature pick randomly.
        :return: Best feature to split, best threshold to split (unique value for discrete), is_discrete.        """

        if n_try == 'log2':
            n_try = int(np.log2(data.shape[1]))
        elif n_try == 'sqrt':
            n_try = int(np.sqrt(data.shape[1]))
        if n_try is not None:
            rand_feature_index = np.arange(data.shape[1])
            np.random.shuffle(rand_feature_index)
            rand_feature_index = rand_feature_index[:n_try]
            data = data[:, rand_feature_index]
        else:
            rand_feature_index = np.arange(data.shape[1])


        best_gini = np.inf

        for i in range(data.shape[1]):
            _is_discrete = is_discrete_feature[rand_feature_index[i]]
            if _is_discrete:
                unique_values = np.unique(data[:, i])
                tmp_gini_value = np.sum([np.sum(data[:, i] == _value) / len(data[:, i]) * self.__gini(label[data[:, i] == _value])
                                        for _value in unique_values])
                if tmp_gini_value < best_gini:
                    best_gini = tmp_gini_value
                    best_feature = rand_feature_index[i]
                    is_discrete = True
                    best_unique_values = unique_values
            else:
                sort_index = np.argsort(data[:, i])
                sub_data = data[sort_index, i]
                sub_label = label[sort_index]
                if data.shape[0] > 100:
                    sub_data = np.percentile(data[:, i], np.arange(100))
                    sub_label = sub_label[np.percentile(np.arange(len(sub_label)), np.arange(100)).astype(np.int32)]

                for j in range(1, len(sub_data)):
                    tmp_gini_value = j / len(sub_data) * self.__gini(sub_label[:j]) + \
                                         (len(sub_data) - j) / len(sub_data) * self.__gini(sub_label[j:])
                    if tmp_gini_value < best_gini:
                        best_gini = tmp_gini_value
                        best_thr = np.mean([sub_data[j-1], sub_data[j]])
                        best_feature = rand_feature_index[i]
                        is_discrete = False
        if is_discrete:
            return best_feature, best_unique_values, is_discrete
        else:
            return best_feature, best_thr, is_discrete

    def entropy(self, data, label, is_discrete_feature, n_try=None):
        """Traverse all features and value, find the best split feature and threshold.

        Find the gain higher than average, pick the highest gain ratio one.

        :param data: A 2-D Numpy array.
        :param label: A 1-D Numpy array.
        :param is_discrete_feature: A list of bool for all features if they are discrete or not.
        :param n_try: An integer or None or string, if 'sqrt' use sqrt(n_feature), if 'log2' use log2(n_feature),
            if None use all n_feature, if integer number of feature pick randomly.
        :return: Best feature to split, best threshold to split (unique value for discrete), is_discrete.
        """
        gain = []
        gain_ratio = []
        ent_before = self.__ent(label)
        if n_try == 'log2':
            n_try = int(np.log2(data.shape[1]))
        elif n_try == 'sqrt':
            n_try = int(np.sqrt(data.shape[1]))
        if n_try is not None:
            rand_feature_index = np.arange(data.shape[1])
            np.random.shuffle(rand_feature_index)
            rand_feature_index = rand_feature_index[:n_try]
            data = data[:, rand_feature_index]
        else:
            rand_feature_index = np.arange(data.shape[1])
        for i in range(data.shape[1]):
            _is_discrete = is_discrete_feature[rand_feature_index[i]]
            if _is_discrete:
                unique_values = np.unique(data[:, i])
                tmp_gain = ent_before - np.sum([np.sum(data[:, i] == _value) / len(data) * self.__ent(label[data[:, i] == _value]) for _value in unique_values])
                tmp_gain_ratio = tmp_gain / (-np.sum([np.sum(data[:, i] == _value) / len(data[:, i]) * np.log2(np.sum(data[:, i] == _value) / len(data[:, i])) for _value in unique_values]) - 10e-5)
                [gain.append(tmp_gain) for _ in range(np.minimum(data.shape[0] - 1, 100 - 1))]
                [gain_ratio.append(tmp_gain_ratio) for _ in range(np.minimum(data.shape[0] - 1, 100 - 1))]
            else:
                sort_index = np.argsort(data[:, i])
                sub_label = label[sort_index]
                sub_data = data[sort_index, i]
                if data.shape[0] > 100:
                    sub_data = np.percentile(data[:, i], np.arange(100))
                    sub_label = sub_label[np.percentile(np.arange(len(sub_label)), np.arange(100)).astype(np.int32)]
                for j in range(1, len(sub_data)):
                    tmp_gain = ent_before - \
                               (j / len(sub_data) * self.__ent(sub_label[:j]) + (len(sub_data) - j) / len(sub_data) * self.__ent(sub_label[j:]))
                    tmp_gain_ratio = tmp_gain / (- j / len(sub_data) * np.log2(j / len(sub_data)) -
                                                 (len(sub_data) - j) / len(sub_data) * np.log2((len(sub_data) - j) / len(sub_data)))
                    gain.append(tmp_gain)
                    gain_ratio.append(tmp_gain_ratio)
        gain = np.array(gain)
        gain_ratio = np.array(gain_ratio)
        gain_ratio[gain < np.mean(gain)] = -np.inf
        best_index = np.argmax(gain_ratio)
        mat_index = np.unravel_index(best_index, [data.shape[1], np.minimum(data.shape[0] - 1, 100 - 1)])
        best_feature = rand_feature_index[mat_index[0]]
        if is_discrete_feature[best_feature]:
            best_unique_values = np.unique(data[:, mat_index[0]])
            return best_feature, best_unique_values, is_discrete_feature[best_feature]
        else:
            sub_data = np.sort(data[:, mat_index[0]])
            best_thr = np.mean([sub_data[mat_index[1]], sub_data[mat_index[1] + 1]])
            return best_feature, best_thr, is_discrete_feature[best_feature]

    @staticmethod
    def __gini(label):
        _label_class = np.unique(label)
        gini_value = 1 - np.sum([(np.sum(label == i) / len(label)) ** 2 for i in _label_class])
        return gini_value

    @staticmethod
    def __ent(label):
        _label_class = np.unique(label)
        ent_value = - np.sum([np.sum(label == i) / len(label) * np.log2(np.sum(label == i) / len(label)) for i in _label_class])
        return ent_value


def prepare_data(proportion):
    dataset = sk_dataset.load_iris()
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
    train, val, num_class = prepare_data(0.8)
    num_try = int(np.sqrt(train[0].shape[1]))
    dt = DecisionTree(minimum_leaf, 'gini', pruning_prop=0.3, n_try=num_try)
    dt.train(train[0], train[1], num_class)
    _, _, train_acc = dt.eval(train[0], train[1])
    pred, pred_gt, val_acc = dt.eval(val[0], val[1])
    print('train_acc', train_acc)
    print('val_acc', val_acc)
