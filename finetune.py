import argparse
import copy
import os
import pickle
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.ops import sigmoid_focal_loss
from dataloader.image_dataloader import ImageDataset, load_filenames_and_labels_multitask, get_datasets
from dataloader.fingerprint_dataloader import get_ecfp4_fingerprints
from dataloader.balanced_sampler import BalancedBatchSampler
from model.cnn_model_utils import load_model, train_one_epoch_multitask, evaluate_on_multitask, save_finetune_ckpt
from model.train_utils import fix_train_random_seed, load_smiles
from utils.public_utils import cal_torch_model_params, setup_device, is_left_better_right
from utils.splitter import split_train_val_test_idx, split_train_val_test_idx_stratified, scaffold_split_train_val_test, \
    random_scaffold_split_train_val_test, scaffold_split_balanced_train_val_test, stratified_k_fold_split_train_val_test
from utils.splitter import load_existing_k_fold_split
from model.evaluate import compute_topk_precision_f1, compute_topk_hit_rate, metric_multitask
import yaml

from utils.logger import output_epoch_results, gen_AUPR_plot, gen_F1_plot, gen_topkprecf1_plots, gen_topk_hitrate_plot, output_final_kfold_results, \
    output_final_kfold_results, analyze_split_makeup, gen_topk_hitrate_plot, gen_fold_validation_bars, gen_fold_model_comparison_bars


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of ImageMol')

    # config file
    parser.add_argument('--config', default=None, type=str, help='path to config file (default: None)')

    # basic
    parser.add_argument('--dataset', type=str, default="bbbp", help='dataset name, e.g. bbbp, tox21, to pull from bucket')
    parser.add_argument('--dataroot', type=str, default="./data_process/data/", help='data root in the bucket')
    parser.add_argument('--bucket_name', type=str, default=None, help='GCP bucket name')
    parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use (default: 1)')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')

    # optimizer
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate (default: 0.01)')
    parser.add_argument('--backbone_lr_ratio', default=0.1, type=float,
                        help='backbone lr = lr * backbone_lr_ratio (default: 0.1)')
    parser.add_argument('--weight_decay', default=-5, type=float, help='weight decay pow (default: -5)')
    parser.add_argument('--momentum', default=0.9, type=float, help='moment um (default: 0.9)')
    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam', 'adamw'], help='optimizer type (sgd, adam, adamw)')
    parser.add_argument('--lr_scheduler', default='cosine', type=str, choices=['cosine', 'step', 'none'], help='learning rate scheduler type (cosine, step, none)')
    parser.add_argument('--lr_step_size', default=1, type=int, help='step size for StepLR in epochs (default: 1)')
    parser.add_argument('--lr_gamma', default=0.1, type=float, help='gamma for StepLR (default: 0.1)')

    # train
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42) to split dataset')
    parser.add_argument('--runseed', type=int, default=2021, help='random seed to run model (default: 2021)')
    parser.add_argument('--split', default="random", type=str,
                        choices=['random', 'stratified', 'scaffold', 'random_scaffold', 'scaffold_balanced', 
                                 'stratified-k-fold', 'scaffold-k-fold', 'butina-k-fold', 'agglo-k-fold',
                                  'random-k-fold', 'random200-k-fold'],
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
    parser.add_argument('--focal_loss', action='store_true', default=False, help='whether to use sigmoid focal loss instead of BCE')
    parser.add_argument('--task_type', type=str, default="classification", choices=["classification", "regression"], help='task type')
    parser.add_argument('--save_finetune_ckpt', type=int, default=1, choices=[0, 1], help='1 represents saving best ckpt, 0 represents no saving best ckpt')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='dropout rate before the classifier layer (default: 0.5)')
    parser.add_argument('--freeze_layers', type=int, choices=[0, 1, 2, 3, 4], default=0, help='how many embedding layers to freeze')

    # lgbm baseline
    parser.add_argument('--lgbm', type=int, default=0, choices=[0, 1], help='whether to run LGBM baseline using ECFP4 fingerprints')
    parser.add_argument('--data_type', type=str, default='processed', help='data type path level used for fingerprint parquet in bucket path')
    parser.add_argument('--save_lgbm_ckpt', type=int, default=1, choices=[0, 1], help='1 saves trained LGBM model artifacts, 0 disables saving')
    parser.add_argument('--gmu', type=int, default=1, choices=[0, 1], help='whether to train GMU fusion over ImageMol embeddings and LGBM scores')
    parser.add_argument('--gmu_hidden_dim', type=int, default=256, help='hidden dimension for GMU fusion model')
    parser.add_argument('--gmu_epochs', type=int, default=50, help='number of epochs for GMU fusion training')
    parser.add_argument('--gmu_lr', type=float, default=1e-3, help='learning rate for GMU fusion training')
    parser.add_argument('--save_gmu_ckpt', type=int, default=1, choices=[0, 1], help='1 saves trained GMU fusion checkpoints, 0 disables saving')

    # log
    parser.add_argument('--log_dir', default='./logs/finetune/', help='path to log')
    parser.add_argument('--run_num', type=int, default=0, help='unique run number for output folder, if not specified, will use 0')

    args = parser.parse_args()
    # Update log_dir to include run_num as a subdirectory
    args.log_dir = os.path.join(args.log_dir, f"run_{args.run_num}")
    return args

# Run LGBM baseline for one fold and return results and artifacts
# These scores will be used as input features for the GMU fusion stage, 
# and the per-task models will be saved as artifacts if save_lgbm_ckpt is enabled
def run_lgbm_fold(args, labels_train, labels_val, labels_test,
                  fp_train, fp_val, fp_test, has_test, fold=None):
    if args.task_type != "classification":
        print("Skipping LGBM baseline because it currently supports classification only.")
        return None

    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError("LightGBM is not installed. Install 'lightgbm' to use --lgbm 1.") from exc

    fp_train = np.asarray(fp_train, dtype=np.float32)
    fp_val = np.asarray(fp_val, dtype=np.float32)
    fp_test = np.asarray(fp_test, dtype=np.float32) if has_test else None

    num_tasks = labels_train.shape[1]
    y_pro_train = np.full((labels_train.shape[0], num_tasks), 0.5, dtype=np.float32)
    y_pro_val = np.full((labels_val.shape[0], num_tasks), 0.5, dtype=np.float32)
    y_pred_train = np.zeros((labels_train.shape[0], num_tasks), dtype=np.int32)
    y_pred_val = np.zeros((labels_val.shape[0], num_tasks), dtype=np.int32)

    if has_test:
        y_pro_test = np.full((labels_test.shape[0], num_tasks), 0.5, dtype=np.float32)
        y_pred_test = np.zeros((labels_test.shape[0], num_tasks), dtype=np.int32)

    task_models = [None] * num_tasks

    for task_idx in range(num_tasks):
        y_task = labels_train[:, task_idx]
        valid = y_task != -1
        valid_classes = np.unique(y_task[valid]) if np.any(valid) else np.array([])
        if len(valid_classes) < 2:
            continue
        
        # best seen parameters for now, can be exposed as args if desired
        lgbm = LGBMClassifier(
            n_estimators=3000,
            learning_rate=0.02,
            num_leaves=31,
            min_child_samples=100,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_alpha=0.5,
            reg_lambda=5.0
        )
        lgbm.fit(fp_train[valid], y_task[valid])
        task_models[task_idx] = lgbm

        train_proba = lgbm.predict_proba(fp_train)[:, 1]
        val_proba = lgbm.predict_proba(fp_val)[:, 1]
        y_pro_train[:, task_idx] = train_proba
        y_pro_val[:, task_idx] = val_proba
        y_pred_train[:, task_idx] = (train_proba > 0.5).astype(np.int32)
        y_pred_val[:, task_idx] = (val_proba > 0.5).astype(np.int32)

        if has_test:
            test_proba = lgbm.predict_proba(fp_test)[:, 1]
            y_pro_test[:, task_idx] = test_proba
            y_pred_test[:, task_idx] = (test_proba > 0.5).astype(np.int32)

    train_results = metric_multitask(labels_train, y_pred_train, y_pro_train, num_tasks=num_tasks, empty=-1)
    val_results = metric_multitask(labels_val, y_pred_val, y_pro_val, num_tasks=num_tasks, empty=-1)
    topk_k = 200

    train_topk_hitrate = compute_topk_hit_rate(y_pro_train.flatten(), labels_train.flatten(), k=topk_k)
    val_topk_hitrate = compute_topk_hit_rate(y_pro_val.flatten(), labels_val.flatten(), k=topk_k)

    if has_test:
        test_results = metric_multitask(labels_test, y_pred_test, y_pro_test, num_tasks=num_tasks, empty=-1)
        test_topk_hitrate = compute_topk_hit_rate(y_pro_test.flatten(), labels_test.flatten(), k=topk_k)

    lgbm_log = {
        "fold": fold + 1 if fold is not None else None,
        "model": "lgbm",
        "Train AUPR": train_results.get("AUPR"),
        "Validation AUPR": val_results.get("AUPR"),
        "topk_hitrate": {
            f"Train Top{topk_k} Hit Rate": train_topk_hitrate,
            f"Val Top{topk_k} Hit Rate": val_topk_hitrate,
        },
    }
    if has_test:
        lgbm_log["Test AUPR"] = test_results.get("AUPR")
        lgbm_log["topk_hitrate"][f"Test Top{topk_k} Hit Rate"] = test_topk_hitrate

    print(lgbm_log)
    lgbm_log_file = os.path.join(args.log_dir, f"training_log_lgbm_fold{fold+1}.txt" if fold is not None else "training_log_lgbm.txt")
    output_epoch_results(lgbm_log_file, lgbm_log, train_results)

    # save the trained LGBM models as artifacts if enabled, along with metadata about the fold, model, and metrics
    if args.save_lgbm_ckpt == 1:
        ckpt_dir = os.path.join(args.log_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        lgbm_ckpt_path = os.path.join(
            ckpt_dir,
            f"lgbm_fold{fold+1}.pkl" if fold is not None else "lgbm.pkl"
        )
        artifact = {
            "fold": fold + 1 if fold is not None else None,
            "model": "lgbm",
            "num_tasks": num_tasks,
            "task_models": task_models,
            "valid_task_mask": [m is not None for m in task_models],
            "decision_threshold": 0.5,
            "feature_layout": {
                "concat_order": ["ecfp4"],
            },
        }
        with open(lgbm_ckpt_path, "wb") as f:
            pickle.dump(artifact, f)
        print(f"Saved LGBM artifact to: {lgbm_ckpt_path}")

    # generate and save AUPR and top-k hit rate plots for this fold
    lgbm_plot_path = os.path.join(args.log_dir, f"lgbm_plot_fold{fold+1}.png" if fold is not None else "lgbm_plot.png")
    if has_test:
        gen_AUPR_plot(lgbm_plot_path, 0, [train_results.get("AUPR")], [val_results.get("AUPR")], [test_results.get("AUPR")], fold)
        gen_topk_hitrate_plot(lgbm_plot_path, 0, [train_topk_hitrate], [val_topk_hitrate], [test_topk_hitrate], topk_k, fold)
    else:
        gen_AUPR_plot(lgbm_plot_path, 0, [train_results.get("AUPR")], [val_results.get("AUPR")], [], fold)
        gen_topk_hitrate_plot(lgbm_plot_path, 0, [train_topk_hitrate], [val_topk_hitrate], [], topk_k, fold)

    lgbm_results = {
        "highest_valid": val_results.get("AUPR"),
        "val_top200_hitrate": val_topk_hitrate,
        "final_train": train_results.get("AUPR"),
        "final_test": test_results.get("AUPR") if has_test else None,
        "highest_valid_desc": val_results,
        "final_train_desc": train_results,
        "final_test_desc": test_results if has_test else None,
        "task_models": task_models,
        "train_scores": y_pro_train,
        "val_scores": y_pro_val,
        "test_scores": y_pro_test if has_test else None,
    }
    return lgbm_results

# Helper function to extract fingerprint features for a given set of indices from the full ECFP4 array
def _extract_fp_features(ecfp4_pairs, indices):
    """Extract fingerprint vectors from object rows shaped as [fingerprint, label]."""
    selected = ecfp4_pairs[indices]
    if selected.ndim != 2 or selected.shape[1] < 1:
        raise ValueError("Expected ecfp4 pairs shaped (n, 2) as [[fingerprint, label], ...].")
    return np.vstack(selected[:, 0]).astype(np.float32)

# Helper function to extract image embeddings for a given set of indices from the full dataset using the ResNet backbone.
@torch.no_grad()
def _extract_image_embeddings(model, dataset, batch_size, workers, device):
    """Extract ResNet backbone embeddings for each sample in dataset order."""
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    backbone = model.module if isinstance(model, torch.nn.DataParallel) else model
    embeddings = []

    for images, _ in loader:
        images = images.to(device)
        x = backbone.conv1(images)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)
        x = backbone.layer1(x)
        x = backbone.layer2(x)
        x = backbone.layer3(x)
        x = backbone.layer4(x)
        x = backbone.avgpool(x)
        x = torch.flatten(x, 1)
        embeddings.append(x.detach().cpu().numpy())

    return np.vstack(embeddings).astype(np.float32)

def _predict_lgbm_multitask_scores(task_models, features):
    """Predict per-task probabilities from fitted per-task LGBM models."""
    x = np.asarray(features, dtype=np.float32)
    num_tasks = len(task_models)
    y_pro = np.full((x.shape[0], num_tasks), 0.5, dtype=np.float32)
    for task_idx, clf in enumerate(task_models):
        if clf is None:
            continue
        y_pro[:, task_idx] = clf.predict_proba(x)[:, 1]
    return y_pro

# Class definition for the GMU fusion model, which takes image embeddings and LGBM scores as input
# and learns to fuse them for final predictions.
class GMUFusionModel(nn.Module):
    """GMU fusion block adapted for image embeddings + fingerprint-model scores."""

    # Equations as implemented in GMU paper:

    def __init__(self, image_dim, fp_dim, hidden_dim, out_dim):
        super().__init__()
        self.img_proj = nn.Linear(image_dim, hidden_dim)
        self.fp_proj = nn.Linear(fp_dim, hidden_dim)
        self.gate = nn.Linear(image_dim + fp_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, image_x, fp_x):
        h_img = torch.tanh(self.img_proj(image_x))
        h_fp = torch.tanh(self.fp_proj(fp_x))
        z = torch.sigmoid(self.gate(torch.cat([image_x, fp_x], dim=1)))
        h = z * h_img + (1.0 - z) * h_fp
        return self.out(h)


@torch.no_grad()
def _evaluate_gmu_split(model, x_img, x_fp, y_true, topk_k):
    """Run inference + metrics for one split during GMU training/evaluation."""
    model.eval()
    logits = model(x_img, x_fp)
    y_pro = torch.sigmoid(logits).cpu().numpy()
    y_pred = (y_pro > 0.5).astype(np.int32)
    y_np = y_true.detach().cpu().numpy()
    results = metric_multitask(y_np, y_pred, y_pro, num_tasks=y_np.shape[1], empty=-1)
    topk_hitrate = compute_topk_hit_rate(y_pro.flatten(), y_np.flatten(), k=topk_k)
    return {
        "logits": logits,
        "y_pro": y_pro,
        "y_pred": y_pred,
        "y_np": y_np,
        "results": results,
        "topk_hitrate": topk_hitrate,
    }

# Train and evaluate a GMU fusion model on one fold, returning results and artifacts. 
# This will be called for each fold in k-fold CV, as well as for the single split scenario.
def _run_gmu_fusion_fold(args, device, fold,
                         labels_train, labels_val, labels_test,
                         img_train, img_val, img_test,
                         fp_train_scores, fp_val_scores, fp_test_scores,
                         has_test):
    """Train and evaluate a GMU fusion model on one fold."""
    if args.task_type != "classification":
        print("Skipping GMU stage because it currently supports classification only.")
        return None

    x_img_train = torch.from_numpy(np.asarray(img_train, dtype=np.float32)).to(device)
    x_img_val = torch.from_numpy(np.asarray(img_val, dtype=np.float32)).to(device)
    x_fp_train = torch.from_numpy(np.asarray(fp_train_scores, dtype=np.float32)).to(device)
    x_fp_val = torch.from_numpy(np.asarray(fp_val_scores, dtype=np.float32)).to(device)

    y_train = torch.from_numpy(np.asarray(labels_train, dtype=np.float32)).to(device)
    y_val = torch.from_numpy(np.asarray(labels_val, dtype=np.float32)).to(device)

    if has_test:
        x_img_test = torch.from_numpy(np.asarray(img_test, dtype=np.float32)).to(device)
        x_fp_test = torch.from_numpy(np.asarray(fp_test_scores, dtype=np.float32)).to(device)
        y_test = torch.from_numpy(np.asarray(labels_test, dtype=np.float32)).to(device)

    model = GMUFusionModel(
        image_dim=x_img_train.shape[1],
        fp_dim=x_fp_train.shape[1],
        hidden_dim=args.gmu_hidden_dim,
        out_dim=y_train.shape[1],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.gmu_lr, weight_decay=10**args.weight_decay)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    best_topk = -np.inf
    best_state = None
    patience = 10
    early_stop = 0
    topk_k = 200

    for epoch in range(args.gmu_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x_img_train, x_fp_train)
        mask = y_train != -1
        y_train_safe = torch.where(mask, y_train, torch.zeros_like(y_train))
        loss_mat = criterion(logits, y_train_safe)
        denom = torch.clamp(mask.sum().float(), min=1.0)
        loss = (loss_mat * mask.float()).sum() / denom
        loss.backward()
        optimizer.step()

        train_eval = _evaluate_gmu_split(model, x_img_train, x_fp_train, y_train, topk_k)
        val_eval = _evaluate_gmu_split(model, x_img_val, x_fp_val, y_val, topk_k)
        train_results = train_eval["results"]
        val_results = val_eval["results"]

        val_topk = float(val_eval["topk_hitrate"])
        if np.isfinite(val_topk) and val_topk > best_topk:
            best_topk = val_topk
            best_state = copy.deepcopy(model.state_dict())
            early_stop = 0
        else:
            early_stop += 1
            if early_stop > patience:
                break

        gmu_epoch_log = {
            "fold": fold + 1 if fold is not None else None,
            "epoch": epoch,
            "Loss": float(loss.item()),
            "Train AUPR": train_results.get("AUPR"),
            "Validation AUPR": val_results.get("AUPR"),
            "topk_hitrate": {
                f"Train Top{topk_k} Hit Rate": train_eval["topk_hitrate"],
                f"Val Top{topk_k} Hit Rate": val_eval["topk_hitrate"],
            },
        }
        gmu_log_file = os.path.join(args.log_dir, f"training_log_gmu_fold{fold+1}.txt" if fold is not None else "training_log_gmu.txt")
        output_epoch_results(gmu_log_file, gmu_epoch_log, train_results)

    if best_state is not None:
        model.load_state_dict(best_state)

    train_eval = _evaluate_gmu_split(model, x_img_train, x_fp_train, y_train, topk_k)
    val_eval = _evaluate_gmu_split(model, x_img_val, x_fp_val, y_val, topk_k)
    train_results = train_eval["results"]
    val_results = val_eval["results"]
    train_topk_hitrate = train_eval["topk_hitrate"]
    val_topk_hitrate = val_eval["topk_hitrate"]

    gmu_results = {
        "highest_valid": val_results.get("AUPR"),
        "val_top200_hitrate": val_topk_hitrate,
        "final_train": train_results.get("AUPR"),
        "final_test": None,
        "highest_valid_desc": val_results,
        "final_train_desc": train_results,
        "final_test_desc": None,
    }

    if has_test:
        test_eval = _evaluate_gmu_split(model, x_img_test, x_fp_test, y_test, topk_k)
        test_results = test_eval["results"]
        test_topk_hitrate = test_eval["topk_hitrate"]
        gmu_results["final_test"] = test_results.get("AUPR")
        gmu_results["final_test_desc"] = test_results
    else:
        test_results = None
        test_topk_hitrate = None

    if args.save_gmu_ckpt == 1:
        ckpt_dir = os.path.join(args.log_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        gmu_ckpt_path = os.path.join(ckpt_dir, f"gmu_fold{fold+1}.pth" if fold is not None else "gmu.pth")
        state = {
            "fold": fold + 1 if fold is not None else None,
            "model_state_dict": model.state_dict(),
            "model_config": {
                "image_dim": x_img_train.shape[1],
                "fp_dim": x_fp_train.shape[1],
                "hidden_dim": args.gmu_hidden_dim,
                "out_dim": y_train.shape[1],
            },
            "decision_threshold": 0.5,
            "feature_layout": {
                "modalities": ["image_embedding", "lgbm_task_scores"],
            },
            "metrics": gmu_results,
        }
        torch.save(state, gmu_ckpt_path)
        print(f"Saved GMU artifact to: {gmu_ckpt_path}")

    gmu_plot_path = os.path.join(args.log_dir, f"gmu_plot_fold{fold+1}.png" if fold is not None else "gmu_plot.png")
    if has_test:
        gen_AUPR_plot(gmu_plot_path, 0, [train_results.get("AUPR")], [val_results.get("AUPR")], [test_results.get("AUPR")], fold)
        gen_topk_hitrate_plot(gmu_plot_path, 0, [train_topk_hitrate], [val_topk_hitrate], [test_topk_hitrate], topk_k, fold)
    else:
        gen_AUPR_plot(gmu_plot_path, 0, [train_results.get("AUPR")], [val_results.get("AUPR")], [], fold)
        gen_topk_hitrate_plot(gmu_plot_path, 0, [train_topk_hitrate], [val_topk_hitrate], [], topk_k, fold)

    return gmu_results

# ImageMol + LGBM training fold runner that handles dataset and dataloader creation, 
# LGBM baseline training and evaluation, image model training and evaluation, 
# and GMU fusion training and evaluation for one fold of k-fold CV or the single split scenario.
def run_training_fold(args, device, device_ids, num_tasks, eval_metric, valid_select, min_value,
                      name_train, labels_train, name_val, labels_val, name_test, labels_test,
                      img_transformer_train, img_transformer_test, fold=None,
                      fp_train=None, fp_val=None, fp_test=None):
    
    # var if we are doing k fold with predefined splits, to distinguish from single split scenario in the logs and outputs
    has_test = args.split not in {'scaffold-k-fold', 'butina-k-fold', 'agglo-k-fold', 'random-k-fold', 'random200-k-fold'}
    
    # make the datasets and dataloaders according to the fold indices
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = ImageDataset(name_train, labels_train, img_transformer=transforms.Compose(img_transformer_train), normalize=normalize, args=args)
    val_dataset = ImageDataset(name_val, labels_val, img_transformer=transforms.Compose(img_transformer_test), normalize=normalize, args=args)
    
    if has_test:
        test_dataset = ImageDataset(name_test, labels_test, img_transformer=transforms.Compose(img_transformer_test), normalize=normalize, args=args)

    # Dataloaders

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
    if has_test:
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)


    ### LGBM ###

    lgbm_results = None
    if args.lgbm:
        if fp_train is None or fp_val is None or (has_test and fp_test is None):
            raise ValueError("LGBM is enabled but fingerprint arrays are missing for this fold.")
        lgbm_results = run_lgbm_fold(
            args=args,
            labels_train=labels_train,
            labels_val=labels_val,
            labels_test=labels_test,
            fp_train=fp_train,
            fp_val=fp_val,
            fp_test=fp_test,
            has_test=has_test,
            fold=fold,
        )

    ### Image Model ###
    model = load_model(args.image_model, imageSize=args.imageSize, num_classes=num_tasks, dropout_rate=args.dropout_rate)
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

    # freeze the embedding layers according to the freeze_layers parameter
    if args.freeze_layers:
        # Freeze stem + first N residual layers
        freeze_stages = args.freeze_layers
        # stem
        for p in model.conv1.parameters():
            p.requires_grad = False
        for p in model.bn1.parameters():
            p.requires_grad = False

        if freeze_stages >= 1:
            for p in model.layer1.parameters():
                p.requires_grad = False
        if freeze_stages >= 2:
            for p in model.layer2.parameters():
                p.requires_grad = False
        if freeze_stages >= 3:
            for p in model.layer3.parameters():
                p.requires_grad = False
        if freeze_stages >= 4:
            for p in model.layer4.parameters():
                p.requires_grad = False

    if torch.cuda.is_available():
        model = model.cuda()
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # Initialize optimizer with differential LR:
    # - head (fc/classifier): args.lr
    # - backbone: args.lr * args.backbone_lr_ratio
    # Frozen params are excluded by checking requires_grad.
    head_prefixes = (
        "fc.", "module.fc.",
        "classifier.", "module.classifier.",
        "head.", "module.head.",
    )
    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(head_prefixes):
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr * args.backbone_lr_ratio})
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr})

    if not param_groups:
        raise ValueError("No trainable parameters found. Check freeze settings.")
    
    ### Image Model Optimizer and Scheduler ###

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=args.momentum,
            weight_decay=10 ** args.weight_decay
        )
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            weight_decay=10 ** args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=10 ** args.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {args.optimizer}")

    # initialize the image model scheduler

    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    else:
        scheduler = None

    # Initialize the loss function
    weights = None
    if args.task_type == "classification":
        if args.weighted_CE:
            labels_train_list = labels_train[labels_train != -1].flatten().tolist()
            count_labels_train = Counter(labels_train_list)
            imbalance_weight = {key: 1 - count_labels_train[key] / len(labels_train_list) for key in count_labels_train.keys()}
            weights = np.array(sorted(imbalance_weight.items(), key=lambda x: x[0]), dtype="float")[:, 1]
        if args.focal_loss:
            # Set up focal loss parameters (can be exposed as args if desired)
            focal_alpha =  0.9
            focal_gamma = 2.0
            focal_reduction = "none"
            def criterion(inputs, targets):
                # sigmoid_focal_loss expects float targets in [0,1]
                return sigmoid_focal_loss(inputs, targets, alpha=focal_alpha, gamma=focal_gamma, reduction=focal_reduction)
        else:
            criterion = nn.BCEWithLogitsLoss(reduction="none")
    elif args.task_type == "regression":
        criterion = nn.MSELoss()
    else:
        raise Exception(f"param {args.task_type} is not supported.")

    results = {'highest_valid': min_value, 'final_train': min_value, 'final_test': min_value, 'highest_train': min_value, 'highest_valid_desc': None, "final_train_desc": None, "final_test_desc": None}
    early_stop = 0
    patience = 30

    # Lists to store metrics for plotting
    train_aupr_list, val_aupr_list, test_aupr_list = [], [], []
    train_f1_list, val_f1_list, test_f1_list = [], [], []
    train_topk_hitrate_list, val_topk_hitrate_list, test_topk_hitrate_list = [], [], []
    topk_k = 200

    ########### Train the image model for the required epochs ################

    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch_multitask(
            model=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            criterion=criterion,
            weights=weights,
            device=device,
            epoch=epoch,
            task_type=args.task_type,
            freeze_layers=args.freeze_layers,
        )
        if scheduler is not None:
            scheduler.step()

        train_loss, train_results, train_data_dict = evaluate_on_multitask(model=model, data_loader=train_dataloader, criterion=criterion, device=device, epoch=epoch, task_type=args.task_type, return_data_dict=True)
        val_loss, val_results, val_data_dict = evaluate_on_multitask(model=model, data_loader=val_dataloader, criterion=criterion, device=device, epoch=epoch, task_type=args.task_type, return_data_dict=True)
        
        if has_test:
            test_loss, test_results, test_data_dict = evaluate_on_multitask(model=model, data_loader=test_dataloader, criterion=criterion, device=device, epoch=epoch, task_type=args.task_type, return_data_dict=True)

        # Store all of the metrics for plotting
        train_aupr = train_results.get('AUPR', None)
        val_aupr = val_results.get('AUPR', None)
        train_f1 = train_results.get('F1', None)
        val_f1 = val_results.get('F1', None)
        train_aupr_list.append(train_aupr)
        val_aupr_list.append(val_aupr)
        train_f1_list.append(train_f1)
        val_f1_list.append(val_f1)

        if has_test:
            test_aupr = test_results.get('AUPR', None)
            test_f1 = test_results.get('F1', None)
            test_aupr_list.append(test_aupr)
            test_f1_list.append(test_f1)


        # compute the top-k hit rate for train/val/test
        train_topk_hitrate_list.append(compute_topk_hit_rate(train_data_dict['y_pro'].flatten(), 
                                                   train_data_dict['y_true'].flatten(), 
                                                   k=topk_k) if 'y_pro' in train_data_dict \
                                                   and 'y_true' in train_data_dict else None)
        val_topk_hitrate_list.append(compute_topk_hit_rate(val_data_dict['y_pro'].flatten(), 
                                                            val_data_dict['y_true'].flatten(), 
                                                            k=topk_k) if 'y_pro' in val_data_dict \
                                                            and 'y_true' in val_data_dict else None)
        
        if has_test:
            test_topk_hitrate_list.append(compute_topk_hit_rate(test_data_dict['y_pro'].flatten(), 
                                                                 test_data_dict['y_true'].flatten(), 
                                                                 k=topk_k) if 'y_pro' in test_data_dict \
                                                                 and 'y_true' in test_data_dict else None)

        if eval_metric == "topk_hitrate":
            train_result = train_topk_hitrate_list[-1]
            valid_result = val_topk_hitrate_list[-1]
            if has_test:
                test_result = test_topk_hitrate_list[-1]
        else:
            train_result = train_results[eval_metric.upper()]
            valid_result = val_results[eval_metric.upper()]
            if has_test:
                test_result = test_results[eval_metric.upper()]

        if has_test:
            epoch_log = {"fold": fold+1 if fold is not None else None, "epoch": epoch, "patience": early_stop, "Loss": train_loss, 'Train AUPR': train_result, 'Validation AUPR': valid_result, 'Test AUPR': test_result, \
                     "topk_hitrate": {f'Train Top{topk_k} Hit Rate': train_topk_hitrate_list[-1], f'Val Top{topk_k} Hit Rate': val_topk_hitrate_list[-1], f'Test Top{topk_k} Hit Rate': test_topk_hitrate_list[-1]}}
        else:
            epoch_log = {"fold": fold+1 if fold is not None else None, "epoch": epoch, "patience": early_stop, "Loss": train_loss, 'Train AUPR': train_result, 'Validation AUPR': valid_result, \
                     "topk_hitrate": {f'Train Top{topk_k} Hit Rate': train_topk_hitrate_list[-1], f'Val Top{topk_k} Hit Rate': val_topk_hitrate_list[-1]}}
        
        print(epoch_log)
        
        log_file_path = os.path.join(args.log_dir, f"training_log_fold{fold+1}.txt" if fold is not None else "training_log.txt")
        output_epoch_results(log_file_path, epoch_log, train_results)

        # save the best model based on validation metric, and implement early stopping based on validation metric with patience
        if is_left_better_right(train_result, results['highest_train'], standard=valid_select):
            results['highest_train'] = train_result
        if is_left_better_right(valid_result, results['highest_valid'], standard=valid_select):
            results['highest_valid'] = valid_result
            results['final_train'] = train_result
            if has_test:
                results['final_test'] = test_result
            results['highest_valid_desc'] = val_results
            results['final_train_desc'] = train_results
            if has_test:
                results['final_test_desc'] = test_results
            early_stop = 0
        else:
            early_stop += 1
            if early_stop > patience:
                break
        for k, v in results.items():
            if isinstance(v, np.generic):
                results[k] = float(v)

        # Save checkpoint only in the final 2 epochs
        if args.save_finetune_ckpt == 1 and epoch >= args.epochs - 2:
            checkpoint_dir = os.path.join(args.log_dir, "checkpoints")
            save_finetune_ckpt(
                model, optimizer, round(train_loss, 4), epoch, checkpoint_dir,
                f"fold{fold+1}_epoch_{epoch}" if fold is not None else f"epoch_{epoch}",
                lr_scheduler=None, result_dict=results
            )

    plot_path = os.path.join(args.log_dir, f"aupr_f1_topk_plot_fold{fold+1}.png" if fold is not None else "aupr_f1_topk_plot.png")
    
    # Plot the image model metrics over epochs
    if has_test:
        gen_AUPR_plot(plot_path, args.start_epoch, train_aupr_list, val_aupr_list, test_aupr_list,
                    fold)
    else:
        gen_AUPR_plot(plot_path, args.start_epoch, train_aupr_list, val_aupr_list, [], fold)
    
    if has_test:
        gen_topk_hitrate_plot(plot_path, args.start_epoch, train_topk_hitrate_list, val_topk_hitrate_list, test_topk_hitrate_list, topk_k, fold)
    else:
        gen_topk_hitrate_plot(plot_path, args.start_epoch, train_topk_hitrate_list, val_topk_hitrate_list, [], topk_k, fold)

    if has_test:
        print(f"Fold {fold+1} results: highest_valid: {results['highest_valid']:.3f}, final_train: {results['final_train']:.3f}, final_test: {results['final_test']:.3f}" if fold is not None else "final results: highest_valid: {:.3f}, final_train: {:.3f}, final_test: {:.3f}".format(results["highest_valid"], results["final_train"], results["final_test"]))
    else:
        print(f"Fold {fold+1} results: highest_valid: {results['highest_valid']:.3f}, final_train: {results['final_train']:.3f}" if fold is not None else "final results: highest_valid: {:.3f}, final_train: {:.3f}".format(results["highest_valid"], results["final_train"]))

    ### GMU Fusion ###
    # Occurs after both image model and LGBM training for the fold

    # currently this code just takes in the last image model from the fold training loop, 
    # not necessarily the best one. This can be refactored, but due to prior results
    # showing that imagemol plateaus, I have not implemented this yet.

    gmu_results = None
    if args.gmu and args.lgbm and lgbm_results is not None:
        task_models = lgbm_results.get("task_models")
        if task_models is None:
            raise ValueError("GMU requires fitted LGBM task models, but none were found.")

        img_train = _extract_image_embeddings(model, train_dataset, args.batch, args.workers, device)
        img_val = _extract_image_embeddings(model, val_dataset, args.batch, args.workers, device)
        img_test = _extract_image_embeddings(model, test_dataset, args.batch, args.workers, device) if has_test else None

        fp_train_scores = _predict_lgbm_multitask_scores(task_models, fp_train)
        fp_val_scores = _predict_lgbm_multitask_scores(task_models, fp_val)
        fp_test_scores = _predict_lgbm_multitask_scores(task_models, fp_test) if has_test else None

        gmu_results = _run_gmu_fusion_fold(
            args=args,
            device=device,
            fold=fold,
            labels_train=labels_train,
            labels_val=labels_val,
            labels_test=labels_test,
            img_train=img_train,
            img_val=img_val,
            img_test=img_test,
            fp_train_scores=fp_train_scores,
            fp_val_scores=fp_val_scores,
            fp_test_scores=fp_test_scores,
            has_test=has_test,
        )

    if lgbm_results is not None:
        results["lgbm_results"] = lgbm_results
    if gmu_results is not None:
        results["gmu_results"] = gmu_results
    return results

### MAIN FUNCTION ###

def main(args):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.image_folder, args.txt_file = get_datasets(args.dataset, args.dataroot, data_type="processed", bucket_name=args.bucket_name)

    if args.lgbm:
        args.ecfp4 = get_ecfp4_fingerprints(args.dataset, args.dataroot, args.data_type, args.bucket_name)

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
        eval_metric = "topk_hitrate"
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
        img_transformer_train = [transforms.CenterCrop(args.imageSize), 
                                 transforms.RandomGrayscale(p=0.2), 
                                 transforms.RandomRotation(degrees=360), 
                                 transforms.RandomHorizontalFlip(), 
                                 transforms.RandomVerticalFlip(), # new
                                 transforms.RandomResizedCrop(args.imageSize, scale=(0.8, 1.0)), # new
                                 transforms.ToTensor(),
                                 transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))] # new
    else:
        img_transformer_train = [transforms.CenterCrop(args.imageSize), transforms.ToTensor()]
    img_transformer_test = [transforms.CenterCrop(args.imageSize), transforms.ToTensor()]
    names, labels = load_filenames_and_labels_multitask(args.image_folder, args.txt_file, task_type=args.task_type)
    names, labels = np.array(names), np.array(labels)
    num_tasks = labels.shape[1]
    
    kfold_splits = {'stratified-k-fold', 'scaffold-k-fold', 'butina-k-fold', 'agglo-k-fold', 'random-k-fold', 'random200-k-fold'}

    if args.split not in kfold_splits:
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

        # log the makeup of the splits in terms of positives and negatives
        analyze_split_makeup(train_idx, val_idx, test_idx, labels, outpath=os.path.join(args.log_dir, "split_makeup"))

        name_train, name_val, name_test, labels_train, labels_val, labels_test = names[train_idx], names[val_idx], names[test_idx], labels[train_idx], labels[val_idx], labels[test_idx]
        fp_train = _extract_fp_features(args.ecfp4, train_idx) if args.lgbm else None
        fp_val = _extract_fp_features(args.ecfp4, val_idx) if args.lgbm else None
        fp_test = _extract_fp_features(args.ecfp4, test_idx) if (args.lgbm and len(test_idx) > 0) else None
        run_training_fold(args, device, device_ids, num_tasks, eval_metric, valid_select, min_value,
                         name_train, labels_train, name_val, labels_val, name_test, labels_test,
                 img_transformer_train, img_transformer_test,
                 fp_train=fp_train, fp_val=fp_val, fp_test=fp_test)
    else:
        # k-fold branch
        if args.split == "stratified-k-fold":
            splits = stratified_k_fold_split_train_val_test(list(range(0, len(names))), labels, n_splits=5, seed=args.seed)
        elif args.split == "scaffold-k-fold":
            splits = load_existing_k_fold_split('scaffold')
        elif args.split == "butina-k-fold":
            splits = load_existing_k_fold_split('butina')
        elif args.split == "agglo-k-fold":
            splits = load_existing_k_fold_split('agglo')
        elif args.split == "random-k-fold":
            splits = load_existing_k_fold_split('random')
        elif args.split == "random200-k-fold":
            splits = load_existing_k_fold_split('random200')
        fold_results = []
        for fold, (train_idx, val_idx, test_idx) in enumerate(splits):
            print(f"\n===== Fold {fold+1}/5 =====")
            
            # convert indices to numpy arrays for easier indexing
            train_idx = np.array(train_idx, dtype=int)
            val_idx = np.array(val_idx, dtype=int)
            test_idx = np.array(test_idx, dtype=int)

            name_train, name_val, name_test = names[train_idx], names[val_idx], names[test_idx]
            labels_train, labels_val, labels_test = labels[train_idx], labels[val_idx], labels[test_idx]
            fp_train = _extract_fp_features(args.ecfp4, train_idx) if args.lgbm else None
            fp_val = _extract_fp_features(args.ecfp4, val_idx) if args.lgbm else None
            fp_test = _extract_fp_features(args.ecfp4, test_idx) if (args.lgbm and len(test_idx) > 0) else None
            result = run_training_fold(args, device, device_ids, num_tasks, eval_metric, valid_select, min_value,
                                      name_train, labels_train, name_val, labels_val, name_test, labels_test,
                                      img_transformer_train, img_transformer_test, fold=fold,
                                      fp_train=fp_train, fp_val=fp_val, fp_test=fp_test)
            fold_results.append(result)
        # Aggregate results across folds
        avg_valid = np.mean([r['highest_valid'] for r in fold_results])
        avg_train = np.mean([r['final_train'] for r in fold_results])
        avg_test = np.mean([r['final_test'] for r in fold_results])

        final_output_path = os.path.join(args.log_dir, "final_results.txt")
        output_final_kfold_results(final_output_path, avg_valid, avg_train, avg_test, fold_results)
        if args.lgbm:
            fold_val_aupr = []
            fold_val_topk = []
            for r in fold_results:
                lgbm_r = r.get("lgbm_results") if isinstance(r, dict) else None
                if lgbm_r is None:
                    continue
                fold_val_aupr.append(float(lgbm_r.get("highest_valid", np.nan)))
                fold_val_topk.append(float(lgbm_r.get("val_top200_hitrate", np.nan)))

            if len(fold_val_aupr) > 0 and len(fold_val_topk) == len(fold_val_aupr):
                fold_bar_path = os.path.join(args.log_dir, "lgbm_validation_bars.png")
                gen_fold_validation_bars(fold_bar_path, fold_val_aupr, fold_val_topk, topk_k=200)

        if args.lgbm and args.gmu:
            lgbm_fold_val_aupr = []
            lgbm_fold_val_topk = []
            gmu_fold_val_aupr = []
            gmu_fold_val_topk = []

            for r in fold_results:
                if not isinstance(r, dict):
                    continue
                lgbm_r = r.get("lgbm_results")
                gmu_r = r.get("gmu_results")
                if lgbm_r is None or gmu_r is None:
                    continue

                lgbm_fold_val_aupr.append(float(lgbm_r.get("highest_valid", np.nan)))
                lgbm_fold_val_topk.append(float(lgbm_r.get("val_top200_hitrate", np.nan)))
                gmu_fold_val_aupr.append(float(gmu_r.get("highest_valid", np.nan)))
                gmu_fold_val_topk.append(float(gmu_r.get("val_top200_hitrate", np.nan)))

            if len(lgbm_fold_val_aupr) > 0 and len(lgbm_fold_val_aupr) == len(gmu_fold_val_aupr):
                compare_plot_path = os.path.join(args.log_dir, "lgbm_vs_gmu_validation_bars.png")
                gen_fold_model_comparison_bars(
                    compare_plot_path,
                    lgbm_fold_val_aupr,
                    lgbm_fold_val_topk,
                    gmu_fold_val_aupr,
                    gmu_fold_val_topk,
                    topk_k=200,
                )

if __name__ == "__main__":
    args = parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            config_args = yaml.safe_load(f)
        for key, value in config_args.items():
            if hasattr(args, key):
                setattr(args, key, value)

    main(args)
