############################################################################
# Name: Connor Deide
# 11/4/2022
# Version 1.0.0
# Developed in Docker Container: continuumio/anaconda3
#
# Description: This file contains unit tests for the three different machine
# learning classfiers found in mysklearn/myclassifiers.py. While the other
# two test files were written by one of my professors, I wrote the unit tests
# contained in this file
############################################################################

import numpy as np
from scipy import stats

from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
from mysklearn.myclassifiers import MySimpleLinearRegressionClassifier,\
    MyKNeighborsClassifier,\
    MyDummyClassifier

def high_low_discretizer(value):
    if value >= 100:
        return "high"
    else:
        return "low"

def test_simple_linear_regression_classifier_fit():
    # Create random data for testing
    np.random.seed(0)
    X_train = [[value] for value in range(100)]
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train]
    lin_clf = MySimpleLinearRegressionClassifier(high_low_discretizer, MySimpleLinearRegressor())
    lin_clf.fit(X_train, y_train)

    # Test
    # Desk Check
    slope_solution = 1.924917458430444
    intercept_solution = 5.211786196055144
    assert np.isclose(lin_clf.slope, slope_solution)
    assert np.isclose(lin_clf.intercept, intercept_solution)

def test_simple_linear_regression_classifier_predict():
    # Create random data for testing
    lin_clf_one = MySimpleLinearRegressionClassifier(high_low_discretizer, \
        MySimpleLinearRegressor(2, 10)) # y = 2x + 10
    X_test_one = [[78], [12], [7]]
    y_predicted_solution_one = ["high", "low", "low"] # Hard coded

    lin_clf_two = MySimpleLinearRegressionClassifier(high_low_discretizer, \
        MySimpleLinearRegressor(4, 6)) # y = 4x + 6
    X_test_two = [[25], [76], [4]]
    y_predicted_solution_two = ["high", "high", "low"] # Hard coded

    # Test
    # Desk Check
    y_predicted_one = lin_clf_one.predict(X_test_one)
    y_predicted_two = lin_clf_two.predict(X_test_two)
    assert y_predicted_one == y_predicted_solution_one
    assert y_predicted_two == y_predicted_solution_two

def test_kneighbors_classifier_kneighbors():
    # In-class training set 1 (4 instances)
    X_train_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_example1 = ["bad", "bad", "good", "good"]
    X_test_example1 = [[0.33, 1]]
    knn_clf_example1 = MyKNeighborsClassifier(3)
    knn_clf_example1.fit(X_train_example1, y_train_example1)
    # Create expected returns
    example1_distances_expected = [[0.670, 1.00, 1.053]]
    example1_neighbor_indices_expected = [[0, 2, 3]]
    # Get actual returns
    example1_distances_returned, example1_neighbor_indices_returned = \
        knn_clf_example1.kneighbors(X_test_example1)
    # Assert
    assert example1_neighbor_indices_returned == example1_neighbor_indices_expected
    assert np.allclose(example1_distances_returned, example1_distances_expected, 0.01)

    # In-class training set 2 (8 instances)
    X_train_example2 = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]]
    y_train_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    X_test_example2 = [[2, 3]]
    knn_clf_example2 = MyKNeighborsClassifier(3)
    knn_clf_example2.fit(X_train_example2, y_train_example2)
    # Create Expected returns
    example2_distances_expected = [[1.414, 1.414, 2.0]]
    example2_neighbor_indices_expected = [[0, 4, 6]]
    # Get actual returns
    example2_distances_returned, example2_neighbor_indices_returned = \
        knn_clf_example2.kneighbors(X_test_example2)
    # Assert
    assert example2_neighbor_indices_returned == example2_neighbor_indices_expected
    assert np.allclose(example2_distances_returned, example2_distances_expected, 0.01)

    # Bramer training set
    header_bramer_example = ["Attribute 1", "Attribute 2"]
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
            "-", "-", "+", "+", "+", "-", "+"]
    X_test_bramer_example = [[9.1, 11.0]]
    knn_clf_bramer_example = MyKNeighborsClassifier(5)
    knn_clf_bramer_example.fit(X_train_bramer_example, y_train_bramer_example)
    # Create expected returns
    bramer_example_distances_expected = [[0.608, 1.237, 2.202, 2.802, 2.915]]
    bramer_example_neighbor_indices_expected = [[6, 5, 7, 4, 8]]
    # Get actual returns
    bramer_example_distances_returned, bramer_example_neighbor_indices_returned = \
        knn_clf_bramer_example.kneighbors(X_test_bramer_example)
    # Assert
    assert bramer_example_neighbor_indices_returned == bramer_example_neighbor_indices_expected
    assert np.allclose(bramer_example_distances_returned, bramer_example_distances_expected, 0.01)

def test_kneighbors_classifier_predict():
    # In-class training set 1 (4 instances)
    X_train_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_example1 = ["bad", "bad", "good", "good"]
    X_test_example1 = [0.33, 1]
    knn_clf_example1 = MyKNeighborsClassifier(3)
    knn_clf_example1.fit(X_train_example1, y_train_example1)
    # Create Expected returns
    example1_y_predicted_solution = ["good"]
    # Get actual returns
    example1_y_predicted = knn_clf_example1.predict([X_test_example1])
    # Assert
    assert example1_y_predicted == example1_y_predicted_solution

    # In-class training set 2 (8 instances)
    X_train_example2 = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]]
    y_train_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    X_test_example2 = [2, 3]
    knn_clf_example2 = MyKNeighborsClassifier(3)
    knn_clf_example2.fit(X_train_example2, y_train_example2)
    # Create Expected returns
    example2_y_predicted_solution = ["yes"]
    # Get actual returns
    example2_y_predicted = knn_clf_example2.predict([X_test_example2])
    # Assert
    assert example2_y_predicted == example2_y_predicted_solution

    # Bramer training set
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
            "-", "-", "+", "+", "+", "-", "+"]
    X_test_bramer_example = [9.1, 11.0]
    knn_clf_bramer_example = MyKNeighborsClassifier(5)
    knn_clf_bramer_example.fit(X_train_bramer_example, y_train_bramer_example)
    # Create expected returns
    bramer_example_y_predicted_solution = ["+"]
    # Get actual returns
    bramer_example_y_predicted = knn_clf_bramer_example.predict([X_test_bramer_example])
    # Assert
    assert bramer_example_y_predicted == bramer_example_y_predicted_solution

def test_dummy_classifier_fit():
    # Create data for test case A
    np.random.seed(0)
    X_train_A = [[value] for value in range(100)]
    y_train_A = list(np.random.choice(["yes", "no"], 100, replace=True, p = [0.7, 0.3]))
    lin_clf_A = MyDummyClassifier()
    lin_clf_A.fit(X_train_A, y_train_A)

    # Test
    # Desk Check
    most_common_solution_A = "yes"
    assert lin_clf_A.most_common_label == most_common_solution_A

    # Create data for test case B
    X_train_B = [[value] for value in range(100)]
    y_train_B = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, \
        p = [0.2, 0.6, 0.2]))
    lin_clf_B = MyDummyClassifier()
    lin_clf_B.fit(X_train_B, y_train_B)

    # Test
    # Desk Check
    most_common_solution_B = "no"
    assert lin_clf_B.most_common_label == most_common_solution_B

    # Create data for test case C
    X_train_C = [[value] for value in range(100)]
    y_train_C = list(np.random.choice(["dog", "cat", "parrot", "whale"], 100, replace=True, \
        p = [0.2, 0.1, 0.3, 0.4]))
    lin_clf_C = MyDummyClassifier()
    lin_clf_C.fit(X_train_C, y_train_C)

    # Test
    # Desk Check
    most_common_solution_C = "whale"
    assert lin_clf_C.most_common_label == most_common_solution_C

def test_dummy_classifier_predict():
    # Create data for test case A
    np.random.seed(0)
    X_train_A = [[value] for value in range(100)]
    y_train_A = list(np.random.choice(["yes", "no"], 100, replace=True, p = [0.7, 0.3]))
    X_test_A = []
    for _ in range(3):
        X_test_A.append(np.random.randint(100))
    dum_clf_A = MyDummyClassifier()
    dum_clf_A.fit(X_train_A, y_train_A)

    # Test
    # Desk Check
    most_common_solutions_A = ["yes", "yes", "yes"]
    assert dum_clf_A.predict(X_test_A) == most_common_solutions_A

    # Create data for test case B
    X_train_B = [[value] for value in range(100)]
    y_train_B = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, \
        p = [0.2, 0.6, 0.2]))
    X_test_B = []
    for _ in range(2):
        X_test_B.append(np.random.randint(100))
    dum_clf_B = MyDummyClassifier()
    dum_clf_B.fit(X_train_B, y_train_B)

    # Test
    # Desk Check
    most_common_solutions_B = ["no", "no"]
    assert dum_clf_B.predict(X_test_B) == most_common_solutions_B 

    # Create data for test case C
    X_train_C = [[value] for value in range(100)]
    y_train_C = list(np.random.choice(["dog", "cat", "parrot", "whale"], 100, replace=True, \
        p = [0.2, 0.1, 0.3, 0.4]))
    X_test_C = []
    for _ in range(3):
        X_test_C.append(np.random.randint(100))
    dum_clf_C = MyDummyClassifier()
    dum_clf_C.fit(X_train_C, y_train_C)

    # Test
    # Desk Check
    most_common_solutions_C = ["whale", "whale", "whale"]
    assert dum_clf_C.predict(X_test_C) == most_common_solutions_C
