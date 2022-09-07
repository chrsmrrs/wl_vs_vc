import numpy as np

from auxiliarymethods.auxiliary_methods import read_txt
from auxiliarymethods.svm import kernel_svm_evaluation, normalize_gram_matrix
from wl import wl_simple

datasets = [["ENZYMES", True]]

for dataset, labels in datasets:
    accs = []
    accs_train = []

    graph_db, classes = read_txt(dataset)

    gram_matrices = []
    for r in range(1, 5):
        feature_vector = wl_simple(graph_db, h=r, degree=False, uniform=not labels, gram_matrix=False)
        print(feature_vector.shape)
        #gram_matrix = normalize_gram_matrix(gram_matrix)
        #gram_matrices.append(gram_matrix)

    #acc, at = kernel_svm_evaluation(gram_matrices, classes, num_repetitions=10)
    #accs.append(acc)
    #accs_train.append(at)

#print("WL", dataset, np.array(accs).mean(), np.array(accs).std())

