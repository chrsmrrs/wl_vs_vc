import auxiliarymethods.datasets as dp
from auxiliarymethods.gnn_evaluation import gnn_evaluation, gnn_evaluation_no_val
from gnn_baselines.gnn_architectures import Conv

def main():
    num_reps = 3

    ### Smaller datasets.
    dataset = [["MUTAG", True]]

    results = []
    for d, use_labels in dataset:
        # Download dataset.
        dp.get_dataset(d)

        # GIN, dataset d, layers in [1:6], hidden dimension in {32,64,128}.
        # TODO val
        train_acc, train_std, test_acc, test_std,  = gnn_evaluation_no_val(Conv, d, [3], [32], max_num_epochs=100, batch_size=64,
                                       start_lr=0.01, num_repetitions=num_reps, all_std=True)
        print(d + " " + "CONV " + str(train_acc) + " " + str(train_std) + " " + str(test_acc) + " " + str(test_std))
        results.append(d + " " + "CONV " + str(train_acc) + " " + str(train_std) + " " + str(test_acc) + " " + str(test_std))


    for r in results:
        print(r)



if __name__ == "__main__":
    main()
