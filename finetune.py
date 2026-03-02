import argparse
import os
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from dataloader.image_dataloader import ImageDataset, load_filenames_and_labels_multitask, get_datasets
from dataloader.balanced_sampler import BalancedBatchSampler
from model.cnn_model_utils import load_model, train_one_epoch_multitask, evaluate_on_multitask, save_finetune_ckpt
from model.train_utils import fix_train_random_seed, load_smiles
from utils.public_utils import cal_torch_model_params, setup_device, is_left_better_right
from utils.splitter import split_train_val_test_idx, split_train_val_test_idx_stratified, scaffold_split_train_val_test, \
    random_scaffold_split_train_val_test, scaffold_split_balanced_train_val_test, stratified_k_fold_split_train_val_test


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of ImageMol')

    # basic
    parser.add_argument('--dataset', type=str, default="bbbp", help='dataset name, e.g. bbbp, tox21, to pull from bucket')
    parser.add_argument('--dataroot', type=str, default="./data_process/data/", help='data root in the bucket')
    parser.add_argument('--bucket_name', type=str, default=None, help='GCP bucket name')
    parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use (default: 1)')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')

    # optimizer
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', default=-5, type=float, help='weight decay pow (default: -5)')
    parser.add_argument('--momentum', default=0.9, type=float, help='moment um (default: 0.9)')
    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam', 'adamw'], help='optimizer type (sgd, adam, adamw)')

    # train
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42) to split dataset')
    parser.add_argument('--runseed', type=int, default=2021, help='random seed to run model (default: 2021)')
    parser.add_argument('--split', default="random", type=str,
                        choices=['random', 'stratified', 'scaffold', 'random_scaffold', 'scaffold_balanced', 'stratified-k-fold'],
                        help='regularization of classification loss')
    parser.add_argument('--epochs', type=int, default=100, help='number of total epochs to run (default: 100)')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=128, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--batch-sampler-ratio', default=0.5, type=float, help='portion of negatives in each batch when using balanced sampler (default: 0.5)')
    parser.add_argument('--resume', default='None', type=str, metavar='PATH', help='path to checkpoint (default: None)')
    parser.add_argument('--resume_key', default="state_dict", type=str, help='key of checkpoint')
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
    parser.add_argument('--image_model', type=str, default="ResNet18", help='e.g. ResNet18, ResNet34')
    parser.add_argument('--image_aug', action='store_true', default=False, help='whether to use data augmentation')
    parser.add_argument('--weighted_CE', action='store_true', default=False, help='whether to use global imbalanced weight')
    parser.add_argument('--task_type', type=str, default="classification", choices=["classification", "regression"], help='task type')
    parser.add_argument('--save_finetune_ckpt', type=int, default=1, choices=[0, 1], help='1 represents saving best ckpt, 0 represents no saving best ckpt')

    # log
    parser.add_argument('--log_dir', default='./logs/finetune/', help='path to log')

    return parser.parse_args()

def run_training_fold(args, device, device_ids, num_tasks, eval_metric, valid_select, min_value,
                      name_train, labels_train, name_val, labels_val, name_test, labels_test,
                      img_transformer_train, img_transformer_test, fold=None):
    
    # make the datasets and dataloaders according to the fold indices
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = ImageDataset(name_train, labels_train, img_transformer=transforms.Compose(img_transformer_train), normalize=normalize, args=args)
    val_dataset = ImageDataset(name_val, labels_val, img_transformer=transforms.Compose(img_transformer_test), normalize=normalize, args=args)
    test_dataset = ImageDataset(name_test, labels_test, img_transformer=transforms.Compose(img_transformer_test), normalize=normalize, args=args)

    if args.task_type == "classification":
        unique_labels = np.unique(labels_train[labels_train != -1])
        if len(unique_labels) == 2:
            sampler = BalancedBatchSampler(labels_train, args.batch, args.batch_sampler_ratio)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=sampler, num_workers=args.workers, pin_memory=True)
        else:
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    else:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    # train the model
    model = load_model(args.image_model, imageSize=args.imageSize, num_classes=num_tasks)
    if args.resume and fold is None:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            ckp_keys = list(checkpoint[args.resume_key])
            cur_keys = list(model.state_dict())
            model_sd = model.state_dict()
            if args.image_model == "ResNet18":
                ckp_keys = ckp_keys[:120]
                cur_keys = cur_keys[:120]
            for ckp_key, cur_key in zip(ckp_keys, cur_keys):
                model_sd[cur_key] = checkpoint[args.resume_key][ckp_key]
            model.load_state_dict(model_sd)
            arch = checkpoint['arch']
            print(f"resume model info: arch: {arch}")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")
    print(model)
    print(f"params: {cal_torch_model_params(model)}")
    if torch.cuda.is_available():
        model = model.cuda()
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # optimizer 
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            filter(lambda x: x.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=10 ** args.weight_decay
        )
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=10 ** args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            filter(lambda x: x.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=10 ** args.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {args.optimizer}")

    weights = None
    if args.task_type == "classification":
        if args.weighted_CE:
            labels_train_list = labels_train[labels_train != -1].flatten().tolist()
            count_labels_train = Counter(labels_train_list)
            imbalance_weight = {key: 1 - count_labels_train[key] / len(labels_train_list) for key in count_labels_train.keys()}
            weights = np.array(sorted(imbalance_weight.items(), key=lambda x: x[0]), dtype="float")[:, 1]
        criterion = nn.BCEWithLogitsLoss(reduction="none")
    elif args.task_type == "regression":
        criterion = nn.MSELoss()
    else:
        raise Exception(f"param {args.task_type} is not supported.")

    results = {'highest_valid': min_value, 'final_train': min_value, 'final_test': min_value, 'highest_train': min_value, 'highest_valid_desc': None, "final_train_desc": None, "final_test_desc": None}
    early_stop = 0
    patience = 30
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch_multitask(model=model, optimizer=optimizer, data_loader=train_dataloader, criterion=criterion, weights=weights, device=device, epoch=epoch, task_type=args.task_type)
        train_loss, train_results, train_data_dict = evaluate_on_multitask(model=model, data_loader=train_dataloader, criterion=criterion, device=device, epoch=epoch, task_type=args.task_type, return_data_dict=True)
        val_loss, val_results, val_data_dict = evaluate_on_multitask(model=model, data_loader=val_dataloader, criterion=criterion, device=device, epoch=epoch, task_type=args.task_type, return_data_dict=True)
        test_loss, test_results, test_data_dict = evaluate_on_multitask(model=model, data_loader=test_dataloader, criterion=criterion, device=device, epoch=epoch, task_type=args.task_type, return_data_dict=True)

        train_result = train_results[eval_metric.upper()]
        valid_result = val_results[eval_metric.upper()]
        test_result = test_results[eval_metric.upper()]

        epoch_log = {"fold": fold+1 if fold is not None else None, "epoch": epoch, "patience": early_stop, "Loss": train_loss, 'Train AUPR': train_result, 'Validation AUPR': valid_result, 'Test AUPR': test_result}
        print(epoch_log)
        log_file_path = os.path.join(args.log_dir, f"training_log_fold{fold+1}.txt" if fold is not None else "training_log.txt")
        with open(log_file_path, "a") as f:
            f.write(str(epoch_log) + "\n")
            f.write("train_results:" + str(train_results) + "\n")

        if is_left_better_right(train_result, results['highest_train'], standard=valid_select):
            results['highest_train'] = train_result
        if is_left_better_right(valid_result, results['highest_valid'], standard=valid_select):
            results['highest_valid'] = valid_result
            results['final_train'] = train_result
            results['final_test'] = test_result
            results['highest_valid_desc'] = val_results
            results['final_train_desc'] = train_results
            results['final_test_desc'] = test_results
            early_stop = 0
        else:
            early_stop += 1
            if early_stop > patience:
                break
        for k, v in results.items():
            if isinstance(v, np.generic):
                results[k] = float(v)
        if args.save_finetune_ckpt == 1:
            save_finetune_ckpt(model, optimizer, round(train_loss, 4), epoch, args.log_dir, f"fold{fold+1}_epoch_{epoch}" if fold is not None else f"epoch_{epoch}", lr_scheduler=None, result_dict=results)

    print(f"Fold {fold+1} results: highest_valid: {results['highest_valid']:.3f}, final_train: {results['final_train']:.3f}, final_test: {results['final_test']:.3f}" if fold is not None else "final results: highest_valid: {:.3f}, final_train: {:.3f}, final_test: {:.3f}".format(results["highest_valid"], results["final_train"], results["final_test"]))
    
    return results

def main(args):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.image_folder, args.txt_file = get_datasets(args.dataset, args.dataroot, data_type="processed", bucket_name=args.bucket_name)

    if torch.cuda.is_available():
        device, device_ids = setup_device(args.ngpu)
    else:
        device = torch.device('cpu')
        device_ids = []

    # fix random seeds
    fix_train_random_seed(args.runseed)

    # architecture name
    print('Architecture: {}'.format(args.image_model))

    ##################################### initialize some parameters #####################################
    if args.task_type == "classification":
        eval_metric = "aupr"
        valid_select = "max"
        min_value = -np.inf
    elif args.task_type == "regression":
        if args.dataset == "qm7" or args.dataset == "qm8" or args.dataset == "qm9":
            eval_metric = "mae"
        else:
            eval_metric = "rmse"
        valid_select = "min"
        min_value = np.inf
    else:
        raise Exception("{} is not supported".format(args.task_type))

    print("eval_metric: {}".format(eval_metric))

    ##################################### load data #####################################
    if args.image_aug:
        img_transformer_train = [transforms.CenterCrop(args.imageSize), transforms.RandomGrayscale(p=0.2), transforms.RandomRotation(degrees=360), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    else:
        img_transformer_train = [transforms.CenterCrop(args.imageSize), transforms.ToTensor()]
    img_transformer_test = [transforms.CenterCrop(args.imageSize), transforms.ToTensor()]
    names, labels = load_filenames_and_labels_multitask(args.image_folder, args.txt_file, task_type=args.task_type)
    names, labels = np.array(names), np.array(labels)
    num_tasks = labels.shape[1]

    if args.split != "stratified-k-fold":
        # Single split
        if args.split == "random":
            train_idx, val_idx, test_idx = split_train_val_test_idx(list(range(0, len(names))), frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed)
        elif args.split == "stratified":
            train_idx, val_idx, test_idx = split_train_val_test_idx_stratified(list(range(0, len(names))), labels, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed)
        elif args.split == "scaffold":
            smiles = load_smiles(args.txt_file)
            train_idx, val_idx, test_idx = scaffold_split_train_val_test(list(range(0, len(names))), smiles, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        elif args.split == "random_scaffold":
            smiles = load_smiles(args.txt_file)
            train_idx, val_idx, test_idx = random_scaffold_split_train_val_test(list(range(0, len(names))), smiles, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed)
        elif args.split == "scaffold_balanced":
            smiles = load_smiles(args.txt_file)
            train_idx, val_idx, test_idx = scaffold_split_balanced_train_val_test(list(range(0, len(names))), smiles, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed, balanced=True)

        name_train, name_val, name_test, labels_train, labels_val, labels_test = names[train_idx], names[val_idx], names[test_idx], labels[train_idx], labels[val_idx], labels[test_idx]
        run_training_fold(args, device, device_ids, num_tasks, eval_metric, valid_select, min_value,
                         name_train, labels_train, name_val, labels_val, name_test, labels_test,
                         img_transformer_train, img_transformer_test)
    else:
        # Stratified k-fold branch
        splits = stratified_k_fold_split_train_val_test(list(range(0, len(names))), labels, n_splits=5, seed=args.seed)
        fold_results = []
        for fold, (train_idx, val_idx, test_idx) in enumerate(splits):
            print(f"\n===== Fold {fold+1}/5 =====")
            name_train, name_val, name_test = names[train_idx], names[val_idx], names[test_idx]
            labels_train, labels_val, labels_test = labels[train_idx], labels[val_idx], labels[test_idx]
            result = run_training_fold(args, device, device_ids, num_tasks, eval_metric, valid_select, min_value,
                                      name_train, labels_train, name_val, labels_val, name_test, labels_test,
                                      img_transformer_train, img_transformer_test, fold=fold)
            fold_results.append(result)
        # Aggregate results across folds
        avg_valid = np.mean([r['highest_valid'] for r in fold_results])
        avg_train = np.mean([r['final_train'] for r in fold_results])
        avg_test = np.mean([r['final_test'] for r in fold_results])
        print(f"\n===== Stratified 5-Fold CV Results =====\nAvg highest_valid: {avg_valid:.3f}, Avg final_train: {avg_train:.3f}, Avg final_test: {avg_test:.3f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
