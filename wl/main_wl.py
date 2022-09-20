import numpy as np

from auxiliarymethods.auxiliary_methods import read_txt
from auxiliarymethods.svm import kernel_svm_evaluation, normalize_gram_matrix
from wl import wl_simple

datasets = [["PTC_FM", True]]

for dataset, labels in datasets:
    accs = []
    accs_train = []

    graph_db, classes = read_txt(dataset)


    for r in range(1, 10):
        gram_matrices = []
        gram_matrix = wl_simple(graph_db, h=r, degree=False, uniform=not labels, gram_matrix=True)
        gram_matrix = normalize_gram_matrix(gram_matrix)
        gram_matrices.append(gram_matrix)

        train, test = kernel_svm_evaluation(gram_matrices, classes, num_repetitions=10)

        print(train, test, train-test)
    #accs.append(acc)
    #accs_train.append(at)

#print("WL", dataset, np.array(accs).mean(), np.array(accs).std())

