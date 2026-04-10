# GCP Imagemol Finetuning Instructions:

Alexander Leonardos


## Set up dataset on GCP: This should be done using the following Repo: [aircheck_image_utils](https://github.com/bhklab/aircheck_image_utils).

**0.** Make sure that the terminal is authenticated with GCP. Instructions are on the [BHKLab handbook website.](https://bhklab.github.io/handbook/latest/software_development/Remote_Development/Google_Cloud_Platform/introduction/?h=#accessing-gcp-via-terminal)

**1.**  First set up a conda environment for the following modules:

    conda env create -f conda_env.yml -n new_env_name

**2.** Use `parquet_explore.py` to check column names for SMILES and Labels.

    usage: parquet_explore.py [-h] [--dataset_path DATASET_PATH] [--head HEAD] [--ecfp4_info]
                          [--export_ecfp4_subset] [--export_path EXPORT_PATH] [--print_index_head]

**3.** Use `image_generator.py` to generate the images and directly upload them to the desired bucket on GCP.

    usage: image_generator.py [-h] --datapath DATAPATH --outpath OUTPATH [--gcp_bucket] [--small]

**4.** Use `finetuned_csv_generator.py` to generate the csv that ImageMol will use for the finetuning process.

    usage: finetuned_csv_generator.py [-h] --datapath DATAPATH --outpath OUTPATH [--gcp_bucket] --cols COLS [COLS ...] [--small] [--name NAME]

## Docker Image:
**1.** Clone my [ImageMol Fork](https://github.com/AlexLeonardos/ImageMol).
**2.** Build the docker image with a specified tag relevant to the current update or the pipeline currently being tested (ImageMol, Fingerprint Concatenation, GMU, etc.):

    docker build -t finetuningimage:tag . 

**3.** Docker tag the image so that it can be uploaded to GCP.

    docker tag [IMAGE_NAME] [REGION]-docker.pkg.dev/[PROJECT_ID]/[REPOSITORY_NAME]/[IMAGE_NAME]:[TAG]

**4.** Docker push the image to the GCP artifact registry.

    docker push [REGION]-docker.pkg.dev/[PROJECT_ID]/[REPOSITORY_NAME]/[IMAGE_NAME]:[TAG]

## Running Vertex AI Jobs:
**1.** Select Vertex AI: Training on GCP.

**2.** Parameters to choose and region to run:

**a.** Train New Model (custom training)

**b.** Use a custom imagemol container held in the imagemol-finetuning bucket in the Artifact Registry.

**c.** Under arguments, paste in args. They must be in this form for Vertex to parse, with **line breaks between each command**.

    --gpu
    0
    --save_finetune_ckpt
    1
    --log_dir
    /gcs/aircheck-huwe1baylor/huwe1full/logs
    --dataroot
    datasets/finetuning
    --dataset
    huwe1full
    --bucket_name
    aircheck-huwe1baylor
    --task_type
    classification
    --resume
    ckpts/ImageMol.pth.tar
    --image_aug
    --lr
    0.001
    --batch
    32
    --epochs
    20
    --batch-sampler-ratio
    0.5

Full commandline arguments are described in the ImageMol Fork's README, as well as in the `finetune.py` script. If running a specific model fusion, add the flags for running and saving the model checkpoints as well.
    
**d.** Can do hyperparameter tuning.

**e.** Compute and Pricing: 
    
        Region: northamerica-northeast2
        Deploy to new worker pool
        Machine Type: g2-standard-8/16
        Accelerator Type: NVIDIA_L4
        Accelerator Count: 1

**Note:** The current quota for these NVIDIA L4 GPUs is 1, meaning that any job will have to wait for currently running jobs to terminate before starting.

With the balanced batch sampling method implemented, these runs take significantly less time. A typical 5-fold CV run on PGK2 DEL data takes ~1-2 hours.

**f.** Can do inference training on GCP with another container from Artifact Registry, but it's likely easier to do locally. All of my tests were done locally using inference scripts found in the 330_project GitHub.

## Run Inference:

**1.** Download the ckpt/pkl file(s) from the proper bucket as specified by your training params.

**2.** Run the appropriate inference script to test the performance of the finetuned model(s) on a different test set (parquet/csv file). These inference scripts are written to process batches of molecules, hence the `--batch_size` argument.


    ImageMol (ResNet) Inference:
    usage: inference.py [-h] --datapath DATAPATH --outpath OUTPATH --ckpt CKPT [--small] [--threshold THRESHOLD] [--topkprecision TOPKPRECISION] [--batch_size BATCH_SIZE]

    Fingerprint Concatenation Fusion Inference:
    usage: concat_inference.py [-h] --datapath DATAPATH --outpath OUTPATH --imagemol_ckpt IMAGEMOL_CKPT
                           --lgbm_model LGBM_MODEL [--smiles_col SMILES_COL]
                           [--label_col LABEL_COL] [--small] [--threshold THRESHOLD]
                           [--batch_size BATCH_SIZE] [--ecfp_radius ECFP_RADIUS]
                           [--ecfp_bits ECFP_BITS] [--positive_label POSITIVE_LABEL]
    
    GMU Fusion Inference:
    usage: gmu_inference.py [-h] --datapath DATAPATH --outpath OUTPATH --imagemol_ckpt IMAGEMOL_CKPT
                        --lgbm_model LGBM_MODEL --gmu_ckpt GMU_CKPT [--smiles_col SMILES_COL]        
                        [--label_col LABEL_COL] [--small] [--start_index START_INDEX]
                        [--max_rows MAX_ROWS] [--task_index TASK_INDEX] [--threshold THRESHOLD]      
                        [--batch_size BATCH_SIZE] [--ecfp_radius ECFP_RADIUS]
                        [--ecfp_bits ECFP_BITS] [--positive_label POSITIVE_LABEL]
                        
**Note:** Inferences were run locally for my experiments, but could be similarly Dockerized and ran on GCP for future implementations.
    
