import matplotlib.pyplot as plt

""" Utility functions for logging and plotting training metrics. """

# Function to generate plots for AUPR, F1, Top-k Precision, and Top-k F1 over epochs
def gen_epoch_metric_plot(plot_path, start_epoch,
        train_aupr_list, val_aupr_list, test_aupr_list,
        train_f1_list, val_f1_list, test_f1_list,
        train_topk_prec_list, val_topk_prec_list, test_topk_prec_list,
        train_topk_f1_list, val_topk_f1_list, test_topk_f1_list,
        topk_k, fold):
    """
    Generate and save the epoch metric plot for AUPR, F1, Top-k Precision, Top-k F1.
    """
    fig, ax1 = plt.subplots(figsize=(12, 7))
    epochs = range(start_epoch, start_epoch + len(train_aupr_list))
    ax1.plot(epochs, train_aupr_list, label='Train AUPR', color='blue')
    ax1.plot(epochs, val_aupr_list, label='Val AUPR', color='orange')
    ax1.plot(epochs, test_aupr_list, label='Test AUPR', color='green')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('AUPR')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(epochs, train_f1_list, label='Train F1', color='blue', linestyle='dashed')
    ax2.plot(epochs, val_f1_list, label='Val F1', color='orange', linestyle='dashed')
    ax2.plot(epochs, test_f1_list, label='Test F1', color='green', linestyle='dashed')
    ax2.plot(epochs, train_topk_prec_list, label=f'Train Top{topk_k} Prec', color='blue', linestyle='dotted')
    ax2.plot(epochs, val_topk_prec_list, label=f'Val Top{topk_k} Prec', color='orange', linestyle='dotted')
    ax2.plot(epochs, test_topk_prec_list, label=f'Test Top{topk_k} Prec', color='green', linestyle='dotted')
    ax2.plot(epochs, train_topk_f1_list, label=f'Train Top{topk_k} F1', color='blue', linestyle='dashdot')
    ax2.plot(epochs, val_topk_f1_list, label=f'Val Top{topk_k} F1', color='orange', linestyle='dashdot')
    ax2.plot(epochs, test_topk_f1_list, label=f'Test Top{topk_k} F1', color='green', linestyle='dashdot')
    ax2.set_ylabel('F1 / Top-k Metrics')
    ax2.legend(loc='upper right')

    plt.title(f"AUPR, F1, Top{topk_k} Precision & F1 over Epochs{' (Fold ' + str(fold+1) + ')' if fold is not None else ''}")
    plt.savefig(plot_path)
    plt.close(fig)

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
