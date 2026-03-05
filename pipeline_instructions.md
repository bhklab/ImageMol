# GCP Imagemol Finetuning Instructions:

Alexander Leonardos


## Set up dataset on GCP:

0. Make sure that the terminal is authenticated with GCP. Instructions are on the [BHKLab handbook website.](https://bhklab.github.io/handbook/latest/software_development/Remote_Development/Google_Cloud_Platform/introduction/?h=#accessing-gcp-via-terminal)

1.  First set up a conda environment for the following modules:

    `conda env create -f imagemol_data_utils.yml`

2. Use `parquet_explore.py` to check column names for SMILES and Labels.

    `usage: parquet_explore.py [-h] [--dataset_path DATASET_PATH] [--head HEAD]`

3. Use `image_generator.py` to generate the images and directly upload them to the desired bucket on GCP.

    `usage: image_generator.py [-h] --datapath DATAPATH --outpath OUTPATH [--gcp_bucket] [--small]`

4. Use `finetuned_csv_generator.py` to generate the csv that ImageMol will use for the finetuning process.

    `usage: finetuned_csv_generator.py [-h] --datapath DATAPATH --outpath OUTPATH [--gcp_bucket] --cols COLS [COLS ...] [--small] [--name NAME]`

## Docker Image:
1. Clone my [ImageMol Fork](https://github.com/AlexLeonardos/ImageMol).
2. Build the docker image:

    `docker build -t finetuningimage:tag . `

3. Docker tag the image so that it can be uploaded to GCP.

    `docker tag [IMAGE_NAME] [REGION]-docker.pkg.dev/[PROJECT_ID]/[REPOSITORY_NAME]/[IMAGE_NAME]:[TAG]`

4. Docker push

    `docker push [REGION]-docker.pkg.dev/[PROJECT_ID]/[REPOSITORY_NAME]/[IMAGE_NAME]:[TAG]`

## Running Vertex AI Jobs:
1. Select Vertex AI: Training on GCP.

2. Parameters to choose and region to run:

    a. Train New Model (custom training)

    b. Use a custom imagemol container held in the imagemol-finetuning bucket in the Artifact Registry.

    c. Under arguments, paste in args. They must be in this form for Vertex to parse, with line breaks between each command.

    `--gpu`
    
    `0`

    `--save_finetune_ckpt`
    
    `1`

    `--log_dir`

    `/gcs/aircheck-huwe1baylor/huwe1full/logs`

    `--dataroot`

    `datasets/finetuning`

    `--dataset`

    `huwe1full`
    
    `--bucket_name`
    
    `aircheck-huwe1baylor`
    
    `--task_type`
    
    `classification`
    
    `--resume`
    
    `ckpts/ImageMol.pth.tar`
    
    `--image_aug`
    
    `--lr`
    
    `0.5`
    
    `--batch`
    
    `64`
    
    `--epochs`
    
    `20`

    `--batch-sampler-ratio`

    `0.5`
    
    d. Can do hyperparameter tuning.

    e. Compute and Pricing: 
    
        Region: northamerica-northeast2
        Deploy to new worker pool
        Machine Type: g2-standard-8/16
        Accelerator Type: NVIDIA_L4
        Accelerator Count: 1

    f. Can do inference training on GCP with anotehr container from Artifact Registry, but it's likely easier to do locally.

## Run Inference:

1. Download the ckpt file(s) from the proper bucket as specified by your training params.

2. Run `inference.py` to test the performance of the finetuned model(s) on a different test set (parquet file).

    `usage: inference.py [-h] --datapath DATAPATH --outpath OUTPATH --ckpt CKPT [--small] [--threshold THRESHOLD] [--topkprecision TOPKPRECISION] [--batch_size BATCH_SIZE]`
    