# Image Classification Part (Section 4.1)

## Image Classification
### 1. Requirements

torch>=1.7.0; torchvision>=0.8.0; pyyaml; [apex-amp](https://github.com/NVIDIA/apex) (if you want to use fp16); [timm](https://github.com/rwightman/pytorch-image-models) (`pip install git+https://github.com/rwightman/pytorch-image-models.git@9d6aad44f8fd32e89e5cca503efe3ada5071cc2a`)

data prepare: ImageNet with the following folder structure, you can extract ImageNet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```



### 2. PoolFormer Models

| Model    |  #Params | Image resolution | #MACs* | Top1 Acc| Download | Log |
| :---     |   :---:    |  :---: |  :---: |  :---:  |  :---:  | :---:  |
| poolformer_s12  |    12M     |   224  |  1.8G |  79.0  | [here]() |[log]() |
| poolformer_s24 |   21M     |   224 | 3.4G | 80.3  | [here]() |  |[log]() |




### 3. Validation

To evaluate our PoolFormer models, run:

```bash
MODEL=poolformer_s12 # poolformer_{s12, s24, s36, m36, m48}
python3 validate.py /path/to/imagenet  --model $MODEL -b 128 \
  --pretrained # or --checkpoint /path/to/checkpoint 
```



### 4. Train
We show how to train PoolFormers on 8 GPUs. The relation between learning rate and batch size is lr=bs/1024*1e-3.
For convenience, assuming the batch size is 1024, then the learning rate is set as 1e-3 (for batch size of 1024, setting the learning rate as 2e-3 sometimes sees better performance). 


```bash
MODEL=poolformer_s12 # poolformer_{s12, s24, s36, m36, m48}
DROP_PATH=0.1 # drop path rates [0.1, 0.1, 0.2, 0.3, 0.4] responding to model [s12, s24, s36, m36, m48]
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 /path/to/imagenet \
  --model $MODEL -b 128 --lr 1e-3 --drop-path $DROP_PATH --apex-amp
```


## Acknowledgment
Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

[pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [mmdetection](https://github.com/open-mmlab/mmdetection), [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).


