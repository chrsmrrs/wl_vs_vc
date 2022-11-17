import numpy as np
import os.path as osp
import csv
from auxiliarymethods.auxiliary_methods import read_txt
#from auxiliarymethods.svm import kernel_svm_evaluation, linear_svm_evaluation, normalize_gram_matrix, normalize_feature_vector_dense
from wl import wl_simple, wl_simple_color_count

datasets = [ ["AIDS", True]]

color_count = []
for dataset, labels in datasets:
    with open(dataset + '.csv', 'w') as file:
        writer = csv.writer(file, delimiter=' ', lineterminator='\n')

        accs = []
        accs_train = []

        graph_db, _ = read_txt(dataset)
        print("ss")

        #color_count = wl_simple_color_count(graph_db, h=9)

        print(dataset)
        for i in range(0, 6):
            print(i)

            graph_db, _ = read_txt(dataset)
            feature_matrix = wl_simple(graph_db, h=i, degree=False, uniform=not labels, gram_matrix=False)

            u = np.unique(feature_matrix, axis=0).shape[0]
            print(u, len(graph_db))

            # graph_db, classes = read_txt(dataset)
            # gram_matrices = []
            # gram_matrix = wl_simple(graph_db, h=i, degree=False, uniform=not labels, gram_matrix=True)
            #
            # gram_matrix = normalize_gram_matrix(gram_matrix)
            #
            # gram_matrices.append(gram_matrix)
            #
            # train, train_std, test, test_std = kernel_svm_evaluation(gram_matrices, classes, num_repetitions=10)
            #
            # print(dataset, str(i), train, test, train - test, )

            #print(dataset, str(i), train, train_std, test, test_std, train - test, color_count[i])
            #writer.writerow([dataset, str(i), train, train_std, test, test_std, train - test, color_count[i]])
