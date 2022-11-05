############################################################################
# Name: Connor Deide
# Class: CPSC 322, Fall 2022
# Programming Assignment 5
# 11/2/2022
# Did not attempt the bonus
# 
# Description: This program implements 4 different evaluation approaches for
# evaluating data before forming predictions of unseen instances.
############################################################################

import numpy as np # use numpy's random number generation
import operator

############################################################################
# Utility Functions for myevaluation.py
############################################################################

def shuffle_data(random_state, X, y=None):
    """
    This function shuffles two parallel lists randomly while keeping them parellel.
    If y is none, only shuffle X

    Args:
        random_state (int): an integer used for seeding np.random
        X (list of list): A 2D list of values to be sorted parallel to y
        y (list, optional): A list of values to be sorted parallel to X. Defaults to None.

    Returns:
        X and y: returns the now shuffled parellel lists
    """

    if random_state is not None:
        np.random.seed(random_state)

    # shuffle indexes
    for i in range(len(X)):
        rand_idx = np.random.randint(len(X))
        X[i], X[rand_idx] = X[rand_idx], X[i]
        if y is not None:
            y[i], y[rand_idx] = y[rand_idx], y[i]

    return X, y

def fold_data(n_splits, splits):
    """
    Function 'folds' the data into k tuples. Each tuple has a test set and a training set

    Args:
        n_splits (int): the number of splits in the lists splits
        splits (list of list): A 2D list, each row is a split from the dataset that needs to be folded

    Returns:
        list of list: a list of tuples, each with a test set and train set
    """

    folds = []
    for test_idx in range(n_splits):
        train_set = []
        for idx in range(n_splits):
            if idx == test_idx:
                test_set = splits[idx]
            elif idx != test_idx:
                train_set += splits[idx]
        folds.append((train_set, test_set))
        del train_set
        del test_set
    
    return folds

def split_data(n_splits, X):
    """
    Function splits the dataset X as evenly as possible into n partitions

    Args:
        n_splits (int): number of partitions to split data into
        X (list of list): dataset that needs to be split up

    Returns:
        list of list: A 2D list with lists of split data from X
    """

    splits = []
    for _ in range(n_splits):
        splits.append([])
    idx = 0
    for i in range(len(X)):
        splits[idx].append(i)
        # Check for looping back around
        if idx == n_splits - 1:
            idx = 0
        else:
            idx += 1

    return splits

def split_stratified_data(n_splits, group_indices):
    """
    Function splits the stratified dataset as evenly as possible into n partitions
    Differs from split_data as the original dataset has already been split into
    groups based on classification

    Args:
        n_splits (int): number of partitions to split data into
        group_indices (list of list): each row in this 2D list is a list of all of the
        indexes with corresponding attributes

    Returns:
        list of list: A 2D list with lists of stratified split data from X
    """

    splits = []
    for _ in range(n_splits):
        splits.append([])
    idx = 0
    for group in group_indices:
        for i in range(len(group)):
            splits[idx].append(group[i])
            # Check for looping back around
            if idx == n_splits - 1:
                idx = 0
            else:
                idx += 1

    return splits

def group_by(y):
    """
    Function creates a list for each classification present in y.
    Then groups the indexes of the elements into their corresponding
    classification

    Args:
        y (list): listof classification values

    Returns:
        list of list: A 2D list where each list is a group of indexes with the
        same classification value
    """

    groups = []
    classes = []
    for i in range(len(y)):
        if y[i] not in classes:
            classes.append(y[i])
            groups.append([])

    for i in range(len(y)):
        for j, classification in enumerate(classes):
            if y[i] == classification:
                groups[j].append(i)
                break

    return groups

############################################################################
# Utility Functions for pa5.ipynb
############################################################################

def random_subsample(X, y, evaluation, classifier):

    avg_accuracy = 0
    avg_error = 0
    k = 10
    for i in range(k):
        X_train, X_test, y_train, y_true = evaluation.train_test_split(X, y, 0.50, i, True)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        # Compare y_test to y_pred for accuracy
        accuracy = evaluation.accuracy_score(y_true, y_pred)

        avg_accuracy += accuracy

    # Get averages
    avg_accuracy = round((avg_accuracy / 10), 2)
    avg_error = 1.0 - avg_accuracy

    return avg_accuracy, avg_error

def cross_val_predict(X, y, evaluation, classifier, stratify=False):

    if stratify:
        folds = evaluation.stratified_kfold_split(X, y, 10)
    else:
        folds = evaluation.kfold_split(X, 10)

    X_train = []
    y_train = []
    X_test = []
    y_true = []

    avg_accuracy = 0
    for fold in folds:
        for train in fold[0]:
            X_train.append(X[train])
            y_train.append(y[train])
        for test in fold[1]:
            X_test.append(X[test])
            y_true.append(y[test])
    
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        # Compare y_test to y_pred for accuracy
        accuracy = evaluation.accuracy_score(y_true, y_pred)

        avg_accuracy += accuracy

    # Get averages
    avg_accuracy = round((avg_accuracy / 10), 2)
    avg_error = 1.0 - avg_accuracy

    if stratify:
        return avg_accuracy, avg_error, y_true, y_pred
    else:
        return avg_accuracy, avg_error

def bootstrap_method(X, y, evaluation, classifier):

    avg_accuracy = 0
    k = 10
    for _ in range(k):
        X_train, X_test, y_train, y_true = evaluation.bootstrap_sample(X, y)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        # Compare y_test to y_pred for accuracy
        accuracy = evaluation.accuracy_score(y_true, y_pred)

        avg_accuracy += accuracy

    # Get averages
    avg_accuracy = round((avg_accuracy / 10), 2)
    avg_error = 1.0 - avg_accuracy

    return avg_accuracy, avg_error

def normalize(raw):
    """
    Function normalizes the values of a list

    Args:
        raw (list): list of raw values that need to be normalized

    Returns:
        list: list of the normalized values
    """
    raw_max = max(raw)
    raw_min = min(raw)
    range = raw_max - raw_min

    normal = []
    for value in raw:
        normal.append((value - raw_min) / range)
    
    return normal

def get_labels(y):
    """
    Function builds a list of labels. One label for every classification
    in the given list. (Gets rid of dups)

    Args:
        y (list): list of classifications

    Returns:
        list: list of one of each unique classification within y
    """

    labels = []
    for i in range(len(y)):
        if y[i] not in labels:
            labels.append(y[i])
    return labels

def mpg_data_discretizer(value):
    """
    Function uses a python switch statement to create a new list of categorical data out of
    a list of continuous data. 

    Args:
        mpg_values_list (list): list of instances of the 'mpg' attribute from the auto-mpg.txt
        file

    Returns:
        list: list of ratings based off of the mpg value
    """
    if (value == 'NA'):
        return None
    elif (value <= 13):
        return 1
    elif (value > 13 and value < 15):
        return 2
    elif (value >= 15 and value < 17):
        return 3
    elif (value >= 17 and value < 20):
        return 4
    elif (value >= 20 and value < 24):
        return 5
    elif (value >= 24 and value < 27):
        return 6
    elif (value >= 27 and value < 31):
        return 7
    elif (value >= 31 and value < 37):
        return 8
    elif (value >= 37 and value < 45):
        return 9
    elif (value >= 45):
        return 10
    else:
        return None

############################################################################
# Utility Functions for classifiers
############################################################################

def get_indices_and_dists(MyKNeighborsClassifier, X_test):
    """
    Function creates a 2D list of tuples (index, distance) based off of the
    distances computed between and instance of X_train and X_test

    Args:
        X_test (list of list): 2D list of point values to be used in computing
        distances

    Returns:
        list of list: 2D list of tuples (index, distance)
    """
    row_indexes_dists = []
    for test_point in X_test:
        index_distances = []
        for i, train_point in enumerate(MyKNeighborsClassifier.X_train):
            distance = round(compute_euclidean_distance(\
                test_point, train_point), 3)
            index_distances.append((i, distance))
        row_indexes_dists.append(index_distances)
    return row_indexes_dists

def get_kclosest_neighbors(knn_clf, row_indexes_dists):
    """
    Function finds the k closest neighbors for each list in the 2D list
    row_indexes_dists by finding the items with the shortest distances

    Args:
        knn_clf (MyKNeighborsClassifier): MyKNeighborsClassifier object used to
        obtain the n_neighbors attribute
        row_indexes_dists (list of list): 2D list of tuples (index, distance) used
        to find the k closest neighbors

    Returns:
        list of list: 2D list of tuples
    """
    total_neighbors = []
    for index_distances in row_indexes_dists:
        index_distances.sort(key=operator.itemgetter(-1))
        total_neighbors.append(index_distances[:knn_clf.n_neighbors])
    return total_neighbors

def compute_euclidean_distance(v1, v2):
    """
    Function computes the distance between two values using the euclidean
    distance algorithm

    Args:
        v1 (float): point one
        v2 (float): point two

    Returns:
        float: the distance value between the two points
    """
    return np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))

def find_majority_2D(classifications):
    """
    Function returns a list of the most common classification values from each row
    in a 2D list

    Args:
        classifications (list of list): 2D list of classification values from y_train

    Returns:
        list: the most common classification from each row of classifications in 
        the list
    """
    y_predicted = []
    for row in classifications:
        class_frequency = {}
        for value in row:
            if value in class_frequency:
                class_frequency[value] += 1
            else:
                class_frequency[value] = 1
        y_predicted.append(max(class_frequency, key=class_frequency.get))

    return y_predicted

def find_majority_1D(classifications):
    """
    Function returns the most common classification value from a 1D list

    Args:
        classifications (list): list of classification values from y_train

    Returns:
        obj: the most common classification from the list
    """
    y_predicted = ""
    class_frequency = {}
    for value in classifications:
        if value in class_frequency:
            class_frequency[value] += 1
        else:
            class_frequency[value] = 1
    y_predicted = max(class_frequency, key=class_frequency.get)

    return y_predicted