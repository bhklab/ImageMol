# ImageMol

The code is an official PyTorch-based implementation in the paper [Accurate prediction of molecular properties and drug targets using a self-supervised image representation learning framework](https://www.nature.com/articles/s42256-022-00557-6) (accepted in *Nature Machine Intelligence*, 2022).

This fork contains augmentations to ImageMol by Alexander Leonardos for testing on AIRCHECK DEL/ASMS Datasets.

This README will only focus on the augmentations made for finetuning ImageMol.

## Install Environment

A conda_env yaml file is included to import these requirements to a conda environment. 

 `conda env create -f conda_env.yml -n new_env_name`

## Finetuning

#### 1. Download pre-trained ImageMol

You can download [pre-trained model](https://drive.google.com/file/d/1wQfby8JIhgo3DxPvFeHXPc14wS-b4KB5/view?usp=sharing) and push it into the folder ``ckpts/``



#### 2. Finetune with pre-trained ImageMol

**a)** You can download [molecular property prediction datasets](https://drive.google.com/file/d/1IdW6J6tX4j5JU0bFcQcuOBVwGNdX7pZp/view?usp=sharing), [CYP450 datasets](https://drive.google.com/file/d/1mBsgGWXYqej5McsLwy1_fs_-VGGQnCro/view?usp=sharing), [multi-label CYP450 dataset](https://drive.google.com/file/d/1VVV5HjIUwlm1yYjksz-e37LUPh-iuB4y/view?usp=sharing), [SARS-CoV-2 datasets](https://drive.google.com/file/d/1UfROoqR_aU6f5xWwxpLoiJnkwyUzDsui/view?usp=sharing), [kinases datasets](https://drive.google.com/file/d/1HVHrxJfW16-5uxQ-7DxgQTxroXxeFDcQ/view?usp=sharing) and [KinomeScan datasets](https://drive.google.com/file/d/1V4xVgjbzBWcIu0CRcHoYVZH9jdnZXpPj/view?usp=sharing) and put it into ``datasets/finetuning/``

**b)** The usage is as follows:

```bash
options:
  -h, --help            show this help message and exit
  --config CONFIG       path to config file (default: None)
  --dataset DATASET     dataset name, e.g. bbbp, tox21, to pull from bucket
  --dataroot DATAROOT   data root in the bucket
  --bucket_name BUCKET_NAME
                        GCP bucket name
  --gpu GPU             index of GPU to use
  --ngpu NGPU           number of GPUs to use (default: 1)
  --workers WORKERS     number of data loading workers (default: 4)
  --lr LR               learning rate (default: 0.01)
  --backbone_lr_ratio BACKBONE_LR_RATIO
                        backbone lr = lr * backbone_lr_ratio (default: 0.1)
  --weight_decay WEIGHT_DECAY
                        weight decay pow (default: -5)
  --momentum MOMENTUM   moment um (default: 0.9)
  --optimizer {sgd,adam,adamw}
                        optimizer type (sgd, adam, adamw)
  --lr_scheduler {cosine,step,none}
                        learning rate scheduler type (cosine, step, none)
  --lr_step_size LR_STEP_SIZE
                        step size for StepLR in epochs (default: 1)
  --lr_gamma LR_GAMMA   gamma for StepLR (default: 0.1)
  --seed SEED           random seed (default: 42) to split dataset
  --runseed RUNSEED     random seed to run model (default: 2021)
  --split {random,stratified,scaffold,random_scaffold,scaffold_balanced,stratified-k-fold,scaffold-k-fold,butina-k-fold,agglo-k-fold,random-k-fold,random200-k-fold}
                        regularization of classification loss
  --epochs EPOCHS       number of total epochs to run (default: 100)
  --start_epoch START_EPOCH
                        manual epoch number (useful on restarts) (default: 0)
  --batch BATCH         mini-batch size (default: 128)
  --batch-sampler-ratio BATCH_SAMPLER_RATIO
                        portion of negatives in each batch when using balanced sampler
                        (default: 0.5)
  --resume PATH         path to checkpoint (default: None)
  --resume_key RESUME_KEY
                        key of checkpoint
  --imageSize IMAGESIZE
                        the height / width of the input image to network
  --image_model IMAGE_MODEL
                        e.g. ResNet18, ResNet34
  --image_aug           whether to use data augmentation
  --weighted_CE         whether to use global imbalanced weight
  --focal_loss          whether to use sigmoid focal loss instead of BCE
  --task_type {classification,regression}
                        task type
  --save_finetune_ckpt {0,1}
                        1 represents saving best ckpt, 0 represents no saving best ckpt
  --dropout_rate DROPOUT_RATE
                        dropout rate before the classifier layer (default: 0.5)
  --freeze_layers {0,1,2,3,4}
                        how many embedding layers to freeze
  --lgbm {0,1}          whether to run LGBM baseline using ECFP4 fingerprints
  --data_type DATA_TYPE
                        data type path level used for fingerprint parquet in bucket path
  --save_lgbm_ckpt {0,1}
                        1 saves trained LGBM model artifacts, 0 disables saving
  --gmu {0,1}           whether to train GMU fusion over ImageMol embeddings and LGBM scores
  --gmu_hidden_dim GMU_HIDDEN_DIM
                        hidden dimension for GMU fusion model
  --gmu_epochs GMU_EPOCHS
                        number of epochs for GMU fusion training
  --gmu_lr GMU_LR       learning rate for GMU fusion training
  --save_gmu_ckpt {0,1}
                        1 saves trained GMU fusion checkpoints, 0 disables saving
  --log_dir LOG_DIR     path to log
  --run_num RUN_NUM     unique run number for output folder, if not specified, will use 0
```

**c)** You can run ImageMol by simply using the following code:

```bash
python finetune.py --gpu ${gpu_no} \
                   --save_finetune_ckpt ${save_finetune_ckpt} \
                   --log_dir ${log_dir} \
                   --dataroot ${dataroot} \
                   --dataset ${dataset} \
                   --task_type ${task_type} \
                   --resume ${resume} \
                   --image_aug \
                   --lr ${lr} \
                   --batch ${batch} \
                   --epochs ${epoch}
```

For example:

```bash
python finetune.py --gpu 0 \
                   --save_finetune_ckpt 1 \
                   --log_dir ./logs/toxcast \
                   --dataroot ./datasets/finetuning/benchmarks \
                   --dataset toxcast \
                   --task_type classification \
                   --resume ./ckpts/ImageMol.pth.tar \
                   --image_aug \
                   --lr 0.5 \
                   --batch 64 \
                   --epochs 20
```

## ImageMol Augmentations

The following are a list of added CLI arguments, as well as how they interact with the `finetune.py` script.

1. `--bucket_name`: This argument allows for pulling the dataset from a specified GCP bucket, provided the script is run either on Vertex AI or through an authenticated terminal.
2. `--backbone_lr_ratio`: Specifies the ratio between the learning rate for params in the ResNet backbone vs ones in the final FC layer. 
3. `--optimizer`: Support added for Adam and AdamW optimizers.
4. `--lr_scheduler`: Support added for cosine and step learning rate schedulers. 

**Note:** if StepLR is chosen, `--lr_step_size` and `--lr_gamma` should also be configured. 

5. `--split`: Added support for many splitting strategies, including scaffold splitting.

**Note:** Current functionality for the scaffold k fold splits is to read from fold csvs, as specified by Shay Reza.

6. `--batch_sampler_ratio`: Splits the batches into a specified percentage of positives to negatives. E.g. 0.25 => 1:3 ratio per batch.

7. `--focal_loss`: Support for sigmoid focal loss.

8. `--dropout_rate`: Adds dropout before the final FC layer in the ResNet architecture.

9. `--freeze_layers`: Prevents the training code from updating parameters in layers up to the specified layer. Useful for maintaining useful training embeddings.

10. `--run_num`: Outputs the logs and model checkpoints to a subdirectory of the `--log_dir`, allowing for maintenance of many run outputs.

# Supported Pipelines

Different architectures are stored in different branches in this repository.

Other branches have been kept as documentation of development steps, and they have been renamed to contain `archive/` before their branch names.

These branches of interest contain the final pipelines for the architectures of interest:

1. **ImageMol Branch:** `master`

This branch contains the ImageMol code augmented with the features above.

2. **Fingerprint Concatenation Branch:** `concat-fingerprints`

This branch contains the training loop for concatenating ImageMol embeddings to an ECFP4 fingerprint, and passing this concatenated vector into a LightGBM model.

3. **GMU Branch:** `gated-multimodal-unit`

This branch contains the training loop for training a GMU layer on ImageMol embeddings and LightGBM outputs. The architecture for the GMU is implemented as described by [Arevalo et. al](https://doi.org/10.1145/3136755.3136814).

4. **General Fusion Branch:** `LGBM-fusion`

This branch contains a training loop which trains the LGBM model alongside the ImageMol model. No fusion or experiments were performed in this branch, but it is not archived as future fusion implementations should be rooted at this branch.

# Dockerization

Each of these branch environments can be dockerized as specified by `Dockerfile`. Instructions for dockerization and running on Google Cloud Platform are found in `pipeline_instructions.md`.
