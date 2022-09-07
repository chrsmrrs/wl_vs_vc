import auxiliarymethods.datasets as dp
from auxiliarymethods.gnn_evaluation import gnn_evaluation, gnn_evaluation_no_val
from gnn_baselines.gnn_architectures import Conv

def main():
    # TODO
    num_reps = 1

    ### Smaller datasets.
    dataset = [["ENZYMES", True]]

    results = []
    for d, use_labels in dataset:
        # Download dataset.
        dp.get_dataset(d)

        # GIN, dataset d, layers in [1:6], hidden dimension in {32,64,128}.
        # TODO currently evaluating without validation set.
        train_acc, train_std, test_acc, test_std,  = gnn_evaluation_no_val(Conv, d, [5], [8], max_num_epochs=1000, batch_size=128,
                                       start_lr=0.01, factor=99.99, patience=100, num_repetitions=num_reps, all_std=True)
        print(d + " " + "CONV " + str(train_acc) + " " + str(train_std) + " " + str(test_acc) + " " + str(test_std))
        results.append(d + " " + "CONV " + str(train_acc) + " " + str(train_std) + " " + str(test_acc) + " " + str(test_std))

        # train_acc, train_std, test_acc, test_std,  = gnn_evaluation_no_val(Conv, d, [5], [32], max_num_epochs=1000, batch_size=64,
        #                                start_lr=0.01, factor=99.99, patience=100, num_repetitions=num_reps, all_std=True)
        # print(d + " " + "CONV " + str(train_acc) + " " + str(train_std) + " " + str(test_acc) + " " + str(test_std))
        # results.append(d + " " + "CONV " + str(train_acc) + " " + str(train_std) + " " + str(test_acc) + " " + str(test_std))

        train_acc, train_std, test_acc, test_std,  = gnn_evaluation_no_val(Conv, d, [5], [1024], max_num_epochs=1000, batch_size=128,
                                       start_lr=0.01, factor=99.99, patience=100, num_repetitions=num_reps, all_std=True)
        print(d + " " + "CONV " + str(train_acc) + " " + str(train_std) + " " + str(test_acc) + " " + str(test_std))
        results.append(d + " " + "CONV " + str(train_acc) + " " + str(train_std) + " " + str(test_acc) + " " + str(test_std))


    for r in results:
        print(r)



if __name__ == "__main__":
    main()
