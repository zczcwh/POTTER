# Human Mesh Recovery (HMR) part.




<div align="center">
<img src="assets/potter.gif" height="160"> 
</div>


## News :triangular_flag_on_post:

[2023/03/29] The demo code is released!


## TODO
- Provide pretrained model 
- Provide training and testing code

## Installation instructions

Please follow the installation instructions from [HybrIK](https://github.com/Jeff-sjtu/HybrIK) since our code is build based on HybrIK. 


## Download models

Please follow download instructions from [HybrIK](https://github.com/Jeff-sjtu/HybrIK) to get the required files since our code is build based on HybrIK. 

## Demo
First make sure you download the pretrained model and place it in the `${ROOT}/eval` directory, i.e., `./eval/potter_demo.pth` or `./eval/feater_demo.pth`.

* Visualize POTTER on **images**:

``` bash
python scripts/demo_image_potter.py --img-dir examples/coco --out-dir examples/res_coco
```

* Visualize POTTER on **videos** (frame by frame reconstruction) and save results:

``` bash
python scripts/demo_video_potter.py --video-name examples/d1.mp4 --out-dir examples/res_d1
```

Similarly 

* Visualize FeatER on **images**:

``` bash
python scripts/demo_image_feater.py --img-dir examples/coco --out-dir examples/res_coco
```

* Visualize FeatER on **videos** (frame by frame reconstruction) and save results:

``` bash
python scripts/demo_video_feater.py --video-name examples/d1.mp4 --out-dir examples/res_d1
```



## Fetch data
Please follow download instructions from [HybrIK](https://github.com/Jeff-sjtu/HybrIK) to get the dataset since our code is build based on HybrIK. 



## Train from scratch
Code will be released soon.




## Citing
If our code helps your research, please consider citing the following paper:

    @inproceedings{zheng2023potter,
        title={POTTER: Pooling Attention Transformer for Efficient Human Mesh Recovery},
        author={Zheng, Ce and Liu, Xianpeng and Qi, Guo-Jun and Chen, Chen},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        year={2023}
    }
    
        @inproceedings{zheng2023feater,
        title={FeatER: An Efficient Network for Human Reconstruction via Feature Map-Based TransformER},
        author={Zheng, Ce and Mendieta, Matias and Yang, Taojiannan and Qi, Guo-Jun and Chen, Chen},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        year={2023}
    }
