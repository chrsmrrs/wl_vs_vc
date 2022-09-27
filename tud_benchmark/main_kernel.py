import auxiliarymethods.auxiliary_methods as aux
import auxiliarymethods.datasets as dp
import kernel_baselines as kb
from auxiliarymethods.kernel_evaluation import kernel_svm_evaluation
from auxiliarymethods.kernel_evaluation import linear_svm_evaluation


def main():
    results = []

    # Number of repetitions of 10-CV.
    num_reps = 3

    ### Larger datasets using LIBLINEAR with edge labels.
    dataset = [["ENZYMES", True, False]]

    print("wwww")

    for d, use_labels, use_edge_labels in dataset:
        dataset = d
        classes = dp.get_dataset(dataset)

        # 1-WL kernel, number of iterations in [1:6].
        all_matrices = []
        for i in range(0, 12):
            gm = kb.compute_wl_1_sparse_count(dataset, 1, use_labels, False)
            print(gm)
            exit()

            print(i)

            gm_n = aux.normalize_feature_vector(gm)
            all_matrices.append(gm_n)

            acc, s_1, s_2 = linear_svm_evaluation(all_matrices, classes, num_repetitions=num_reps, all_std=True)
            print(d + " " + "WL1SP " + str(acc) + " " + str(s_1) + " " + str(s_2))
            results.append(d + " " + "WL1SP " + str(acc) + " " + str(s_1) + " " + str(s_2))


    for r in results:
        print(r)


if __name__ == "__main__":
    main()
