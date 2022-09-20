import math as m

import numpy as np
from scipy import sparse as sp
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC


# Cosine normalization for sparse feature vectors, i.e., \ell_2 normalization.
def normalize_feature_vector(feature_vectors):
    n = feature_vectors.shape[0]

    for i in range(0, n):
        norm = sp.linalg.norm(feature_vectors[i])

        if norm != 0:
            feature_vectors[i] = feature_vectors[i] / norm

    return feature_vectors


def normalize_feature_vector_dense(feature_vectors):
    n = feature_vectors.shape[0]

    for i in range(0, n):
        norm = np.linalg.norm(feature_vectors[i])

        feature_vectors[i] = feature_vectors[i] / norm

    return feature_vectors


# Cosine normalization for a gram matrix.
def normalize_gram_matrix(gram_matrix):
    n = gram_matrix.shape[0]
    gram_matrix_norm = np.zeros([n, n], dtype=np.float64)

    for i in range(0, n):
        for j in range(i, n):
            if not (gram_matrix[i][i] == 0.0 or gram_matrix[j][j] == 0.0):
                g = gram_matrix[i][j] / m.sqrt(gram_matrix[i][i] * gram_matrix[j][j])
                gram_matrix_norm[i][j] = g
                gram_matrix_norm[j][i] = g

    return gram_matrix_norm


# 10-CV for linear svm with sparse feature vectors and hyperparameter selection.
def linear_svm_evaluation(all_feature_matrices, classes, num_repetitions=10,
                          C=[10 ** 3, 10 ** 2, 10 ** 1, 10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3], all_std=False):
    # Acc. over all repetitions.
    test_accuracies_all = []
    # All acc. over all folds and repetitions.
    test_accuracies_complete = []

    margins = []

    for i in range(num_repetitions):
        # Test acc. over all folds.
        test_accuracies = []
        kf = KFold(n_splits=10, shuffle=True)

        for train_index, test_index in kf.split(list(range(len(classes)))):
            # Sample 10% split from training split for validation.
            train_index, val_index = train_test_split(train_index, test_size=0.1)
            best_val_acc = 0.0
            best_gram_matrix = all_feature_matrices[0]
            best_c = C[0]

            for gram_matrix in all_feature_matrices:
                train = gram_matrix[train_index]
                val = gram_matrix[val_index]

                c_train = classes[train_index]
                c_val = classes[val_index]

                for c in C:
                    clf = LinearSVC(C=c, tol=0.01, dual=False)
                    clf.fit(train, c_train)
                    val_acc = accuracy_score(c_val, clf.predict(val)) * 100.0

                    if val_acc > best_val_acc:
                        # Get test acc.
                        best_val_acc = val_acc
                        best_c = c
                        best_gram_matrix = gram_matrix

            # Determine test accuracy.
            train = best_gram_matrix[train_index]
            test = best_gram_matrix[test_index]

            c_train = classes[train_index]
            c_test = classes[test_index]
            clf = LinearSVC(C=best_c, tol=0.01, dual=False)
            clf.fit(train, c_train)

            margins.append(1.0 / np.linalg.norm(clf.coef_))

            best_test = accuracy_score(c_test, clf.predict(test)) * 100.0

            test_accuracies.append(best_test)
            if all_std:
                test_accuracies_complete.append(best_test)
        test_accuracies_all.append(float(np.array(test_accuracies).mean()))

    if all_std:
        return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std(),
                np.array(test_accuracies_complete).std())
    else:
        return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std(), np.array(margins).mean())


# 10-CV for kernel svm and hyperparameter selection.
def kernel_svm_evaluation(all_matrices, classes, num_repetitions=10,
                          C=[10 ** 3, 10 ** 2, 10 ** 1, 10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3], all_std=False):
    # Acc. over all repetitions.
    test_accuracies_all = []
    train_accuracies_all = []

    for i in range(num_repetitions):
        # Test acc. over all folds.
        test_accuracies = []
        train_accuracies = []
        kf = KFold(n_splits=10, shuffle=True)

        for train_index, test_index in kf.split(list(range(len(classes)))):
            # Determine hyperparameters
            train_index, val_index = train_test_split(train_index, test_size=0.1)
            best_val_acc = 0.0
            best_gram_matrix = all_matrices[0]
            best_c = C[0]

            for gram_matrix in all_matrices:
                train = gram_matrix[train_index, :]
                train = train[:, train_index]
                val = gram_matrix[val_index, :]
                val = val[:, train_index]

                c_train = classes[train_index]
                c_val = classes[val_index]

                for c in C:
                    clf = SVC(C=c, kernel="precomputed", tol=0.001)
                    clf.fit(train, c_train)
                    val_acc = accuracy_score(c_val, clf.predict(val)) * 100.0

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_c = c
                        best_gram_matrix = gram_matrix

            # Determine test accuracy.
            train = best_gram_matrix[train_index, :]
            train = train[:, train_index]
            test = best_gram_matrix[test_index, :]
            test = test[:, train_index]

            c_train = classes[train_index]
            c_test = classes[test_index]
            clf = SVC(C=best_c, kernel="precomputed", tol=0.001)
            clf.fit(train, c_train)

            best_train = accuracy_score(c_train, clf.predict(train)) * 100.0
            best_test = accuracy_score(c_test, clf.predict(test)) * 100.0

            test_accuracies.append(best_test)
            train_accuracies.append(best_train)

        test_accuracies_all.append(float(np.array(test_accuracies).mean()))
        train_accuracies_all.append(float(np.array(train_accuracies).mean()))

        return (np.array(train_accuracies_all).mean(), np.array(test_accuracies_all).mean())
