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

from mysklearn import myutils
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor

class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        X_train = [x[0] for x in X_train] # convert 2D list with 1 col to 1D list
        self.slope, self.intercept = self.regressor.compute_slope_intercept(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_values = []
        if self.regressor.slope is not None and self.regressor.intercept is not None:
            for test_instance in X_test:
                y_values.append(self.regressor.slope * test_instance[0] + self.regressor.intercept)
        # Discretize the y values
        predictions = []
        for value in y_values:
            predictions.append(self.discretizer(value))
        return predictions

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closest neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        # Get indices and distances
        row_indexes_dists = myutils.get_indices_and_dists(self, X_test)
        # Get the k_closest neighbors indexes and values for each
        # row within total distances
        total_neighbors = myutils.get_kclosest_neighbors(self, row_indexes_dists)
        # Create neighbor_indices and distances from total_neighbors
        neighbor_indices = []
        distances = []
        for row in total_neighbors:
            row_of_neighbor_indices = []
            row_of_distances = []
            for index_distance in row:
                row_of_neighbor_indices.append(index_distance[0])
                row_of_distances.append(index_distance[1])
            neighbor_indices.append(row_of_neighbor_indices)
            distances.append(row_of_distances)

        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # Like the kneighbors function, need the neighbor indices to gather the corresponding
        # values in y_train, then use majority rule to predict a classification for X_test
        # Get indices and distances
        row_indexes_dists = myutils.get_indices_and_dists(self, X_test)
        # Get the k_closest neighbors indexes and values for each
        # row within total distances
        total_neighbors = myutils.get_kclosest_neighbors(self, row_indexes_dists)
        # Get the indexes of the k_closest_neighbors
        neighbor_indices = []
        for row in total_neighbors:
            row_of_neighbor_indices = []
            for index_distance in row:
                row_of_neighbor_indices.append(index_distance[0])
            neighbor_indices.append(row_of_neighbor_indices)
        # Find the corresponding classification in y_train
        k_neighbors_classifications = []
        for row in neighbor_indices:
            k_neighbors_classification = []
            for index in row:
                k_neighbors_classification.append(self.y_train[index])
            k_neighbors_classifications.append(k_neighbors_classification)
        # Use majority rule to predict the class
        y_predicted = myutils.find_majority_2D(k_neighbors_classifications)

        return y_predicted

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        # Find the most frequent class label
        self.most_common_label = myutils.find_majority_1D(y_train)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        if self.most_common_label is None:
            return None
        else:
            y_predicted = []
            for _ in range(len(X_test)):
                y_predicted.append(self.most_common_label)
        return y_predicted
