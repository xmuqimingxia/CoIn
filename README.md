# CoIn: Contrastive Instance Feature Mining for Outdoor 3D Object Detection with Very Limited Annotations(ICCV2023)

This is a official code release of CoIn (Contrastive Instance Feature Mining for Outdoor 3D Object Detection with Very Limited Annotations). This code is mainly based on OpenPCDet.

## Detection Framework

![image](https://github.com/xmuqimingxia/CoIn/assets/108978798/787379ac-4a0f-41ab-9e04-e7c8c0fd61b2)

## Getting Started
###Prepare dataset
coming soon

## Training & Testing

### Train a model

You could optionally add extra command line parameters `--batch_size ${BATCH_SIZE}` and `--epochs ${EPOCHS}` to specify your preferred parameters. 
  

* Train with multiple GPUs or multiple machines
```shell script
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}

# or 

sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
```

* Train with a single GPU:
```shell script
python train.py --cfg_file ${CONFIG_FILE}
```
