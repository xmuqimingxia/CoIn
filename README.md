# CoIn: Contrastive Instance Feature Mining for Outdoor 3D Object Detection with Very Limited Annotations(ICCV2023)

This is a official code release of CoIn (Contrastive Instance Feature Mining for Outdoor 3D Object Detection with Very Limited Annotations). This code is mainly based on OpenPCDet.

## Detection Framework

![image](https://github.com/xmuqimingxia/CoIn/assets/108978798/787379ac-4a0f-41ab-9e04-e7c8c0fd61b2)

## Getting Started
### Prepare dataset
coming soon

## Training & Testing

### Train a model

You could optionally add extra command line parameters `--batch_size ${BATCH_SIZE}` and `--epochs ${EPOCHS}` to specify your preferred parameters. 
  

* Train with multiple GPUs or multiple machines
```shell script
sh scripts/dist_train.sh ${NUM_GPUS} tools/cfgs/kitti_models/CoIn.yaml
```

* Train with a single GPU:
```shell script
python train.py tools/cfgs/kitti_models/CoIn.yaml
```

### Test a model

* To test with multiple GPUs:
```shell script
sh scripts/dist_test.sh ${NUM_GPUS} \
    tools/cfgs/kitti_models/CoIn.yaml --batch_size ${BATCH_SIZE}
```

* To test all the saved checkpoints of a specific training setting and draw the performance curve on the Tensorboard, add the `--eval_all` argument: 
```shell script
python test.py tools/cfgs/kitti_models/CoIn.yaml --batch_size ${BATCH_SIZE} --eval_all
```
## Acknowledgement
[OpenPCDET](https://github.com/open-mmlab/OpenPCDet)


## Citation 
If you find this project useful in your research, please consider cite:


```
@inproceedings{CoIn2023,
    title={CoIn: Contrastive Instance Feature Mining for Outdoor 3D Object Detection with Very Limited Annotations},
    author={Xia, Qiming and Deng, Jinhao and Wen, Chenglu and Wu, Hai and Shi, Shaoshuai and Li, Xin and Wang, Cheng},
    booktitle = {ICCV},
    year={2023}
}
```
