import matplotlib.pyplot as plt
import numpy as np

""" Utility functions for logging and plotting training metrics. """

# Function to generate plots for AUPR, F1, Top-k Precision, and Top-k F1 over epochs
def gen_AUPR_plot(plot_path, start_epoch,
        train_aupr_list, val_aupr_list, test_aupr_list, fold):
    epochs = range(start_epoch, start_epoch + len(train_aupr_list))

    # Plot AUPR
    fig_aupr, ax_aupr = plt.subplots(figsize=(10, 6))
    ax_aupr.plot(epochs, train_aupr_list, label='Train AUPR', color='blue')
    ax_aupr.plot(epochs, val_aupr_list, label='Val AUPR', color='orange')
    ax_aupr.plot(epochs, test_aupr_list, label='Test AUPR', color='green')
    ax_aupr.set_xlabel('Epoch')
    ax_aupr.set_ylabel('AUPR')
    ax_aupr.legend(loc='upper left')
    plt.title(f"AUPR over Epochs{' (Fold ' + str(fold+1) + ')' if fold is not None else ''}")
    plt.savefig(plot_path.replace('.png', '_aupr.png'))
    plt.close(fig_aupr)


def gen_F1_plot(plot_path, start_epoch, train_f1_list, val_f1_list, test_f1_list, fold):
    
    epochs = range(start_epoch, start_epoch + len(train_f1_list))
    
    # Plot F1

    fig_f1, ax_f1 = plt.subplots(figsize=(10, 6))
    ax_f1.plot(epochs, train_f1_list, label='Train F1', color='blue')
    ax_f1.plot(epochs, val_f1_list, label='Val F1', color='orange')
    ax_f1.plot(epochs, test_f1_list, label='Test F1', color='green')
    ax_f1.set_xlabel('Epoch')
    ax_f1.set_ylabel('F1 Score')
    ax_f1.legend(loc='upper left')
    plt.title(f"F1 Score over Epochs{' (Fold ' + str(fold+1) + ')' if fold is not None else ''}")
    plt.savefig(plot_path.replace('.png', '_f1.png'))
    plt.close(fig_f1)

def gen_topkprecf1_plots(plot_path, start_epoch, train_topk_prec_list, val_topk_prec_list, test_topk_prec_list,
        train_topk_f1_list, val_topk_f1_list, test_topk_f1_list, topk_k, fold):
    
    epochs = range(start_epoch, start_epoch + len(train_topk_prec_list))

    # Plot Top-k Precision and F1
    fig_topk, ax_topk = plt.subplots(figsize=(10, 6))
    ax_topk.plot(epochs, train_topk_prec_list, label=f'Train Top{topk_k} Prec', color='blue', linestyle='dotted')
    ax_topk.plot(epochs, val_topk_prec_list, label=f'Val Top{topk_k} Prec', color='orange', linestyle='dotted')
    ax_topk.plot(epochs, test_topk_prec_list, label=f'Test Top{topk_k} Prec', color='green', linestyle='dotted')
    ax_topk.plot(epochs, train_topk_f1_list, label=f'Train Top{topk_k} F1', color='blue', linestyle='dashdot')
    ax_topk.plot(epochs, val_topk_f1_list, label=f'Val Top{topk_k} F1', color='orange', linestyle='dashdot')
    ax_topk.plot(epochs, test_topk_f1_list, label=f'Test Top{topk_k} F1', color='green', linestyle='dashdot')
    ax_topk.set_xlabel('Epoch')
    ax_topk.set_ylabel(f'Top{topk_k} Precision / F1')
    ax_topk.legend(loc='upper left')
    plt.title(f"Top{topk_k} Precision & F1 over Epochs{' (Fold ' + str(fold+1) + ')' if fold is not None else ''}")
    plt.savefig(plot_path.replace('.png', f'_top{topk_k}.png'))
    plt.close(fig_topk)

# Function to write epoch log and train results to the log file
def output_epoch_results(log_file_path, epoch_log, train_results):
    """
    Write epoch log and train results to the log file.
    """
    with open(log_file_path, "a") as f:
        f.write(str(epoch_log) + "\n")
        f.write("train_results:" + str(train_results) + "\n")

# Function to write final k-fold results to the log file
def output_final_kfold_results(final_output_path, avg_valid, avg_train, avg_test, fold_results):
    """
    Write final k-fold results to the log file.
    """
    with open(final_output_path, "w") as f:
        f.write(f"===== Stratified 5-Fold CV Results =====\nAvg highest_valid: {avg_valid:.3f}, Avg final_train: {avg_train:.3f}, Avg final_test: {avg_test:.3f}\n")
        f.write(f"\nDetails for each fold:\n")
        for fold, r in enumerate(fold_results):
            f.write(f"Fold {fold+1} results: highest_valid: {r['highest_valid']:.3f}, final_train: {r['final_train']:.3f}, final_test: {r['final_test']:.3f}\n")
            f.write(f"Fold {fold+1} details:\nhighest_valid_desc: {r['highest_valid_desc']}\nfinal_train_desc: {r['final_train_desc']}\nfinal_test_desc: {r['final_test_desc']}\n\n")

# Helper function to analyze the makeup of train, val, test splits in terms of positives and negatives
def analyze_split_makeup(train_idx, val_idx, test_idx, y, outpath=None):
    """
    Analyze the makeup of train, val, test splits in terms of positives and negatives.

    :param train_idx: Indices for the training set.
    :param val_idx: Indices for the validation set.
    :param test_idx: Indices for the test set.
    :param y: Array-like of labels (0/1 or similar).
    :return: Prints counts for each split.
    """
    def count_pos_neg(indices):
        labels = np.array(y)[indices]
        n_pos = np.sum(labels == 1)
        n_neg = np.sum(labels == 0)
        return n_pos, n_neg

    train_pos, train_neg = count_pos_neg(train_idx)
    val_pos, val_neg = count_pos_neg(val_idx)
    test_pos, test_neg = count_pos_neg(test_idx)
    
    # output the counts for each split to a file with outpath + "_split_makeup.txt"
    if outpath:
        with open(outpath + "_split_makeup.txt", "w") as f:
            f.write(f"Train: {len(train_idx)} samples | Positives: {train_pos} | Negatives: {train_neg}\n")
            f.write(f"Val: {len(val_idx)} samples | Positives: {val_pos} | Negatives: {val_neg}\n")
            f.write(f"Test: {len(test_idx)} samples | Positives: {test_pos} | Negatives: {test_neg}\n")
    else:
        print(f"Train: {len(train_idx)} samples | Positives: {train_pos} | Negatives: {train_neg}")
        print(f"Val: {len(val_idx)} samples | Positives: {val_pos} | Negatives: {val_neg}")
        print(f"Test: {len(test_idx)} samples | Positives: {test_pos} | Negatives: {test_neg}")
