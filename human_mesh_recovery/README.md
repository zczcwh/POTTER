# Human Mesh Recovery (HMR) part for our two papers.

"FeatER: An Efficient Network for Human Reconstruction via Feature Map-Based TransformER", CVPR 2023

"POTTER: Pooling Attention Transformer for Efficient Human Mesh Recovery", CVPR 2023

Please check the instruction for running FeatER or POTTER.

<div align="center">
<img src="assets/potter.gif" height="160"> 
</div>


## News :triangular_flag_on_post:

[2023/03/29] The demo code (both FeatER and POTTER) are released!


## Installation instructions

Please follow the installation instructions from [HybrIK](https://github.com/Jeff-sjtu/HybrIK) since our code is built based on HybrIK. 


## Download models

Please follow the download instructions from [HybrIK](https://github.com/Jeff-sjtu/HybrIK) to get the required files since our code is built based on HybrIK.

* Download the SMPL model `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from [here](https://smpl.is.tue.mpg.de/) at `common/utils/smplpytorch/smplpytorch/native/models`.
* Download our POTTER demo model from [ [Google Drive](https://drive.google.com/file/d/1tLpMCbC6-M3Yxxsn5OoHbo8JuJLO5opZ/view?usp=sharing)].
* Download our FeatER demo model from [ [Google Drive](https://drive.google.com/file/d/1uAyla25E15BLezs1wpHk2GprszF5q-5e/view?usp=sharing)].

## Demo
First make sure you download the pretrained model and place it in the `${ROOT}/eval` directory, i.e., `./eval/potter_demo.pth` or `./eval/feater_demo.pth`.

Also, make sure you have run this command in the installation instruction.

``` bash
python setup.py develop 
```

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

* Visualize FeatER on **videos** (frame-by-frame reconstruction) and save results:

``` bash
python scripts/demo_video_feater.py --video-name examples/d1.mp4 --out-dir examples/res_d1
```



## Fetch data
Please follow the download instructions from [HybrIK](https://github.com/Jeff-sjtu/HybrIK) to get the dataset since our code is built based on HybrIK. 



## Training
Please download the pretrained weights first and place it in the `${ROOT}/model_files` directory, :
* Download our POTTER pretrained weights from [ [Google Drive](https://drive.google.com/file/d/1Nr4uFGryG7v6Tl9sqz3v9u7_w1p6lwW1/view?usp=sharing)].
* Download our FeatER pretrained weights from [ [Google Drive](https://drive.google.com/file/d/1ULMN1U0GHjcQ5nUQpQ1SAhLxc5P1EULB/view?usp=sharing)].

To train POTTER:
``` bash
./scripts/train_smpl_cam.sh train_potter ./configs/potter_cam_w_pw3d.yaml
```
To train FeatER:
``` bash
./scripts/train_smpl_cam.sh train_feater ./configs/feater_cam_w_pw3d.yaml
```
To evaluate:
``` bash
./scripts/validate_smpl_cam.sh ./configs/potter_cam_w_pw3d.yaml [ckp_path]
```
or
``` bash
./scripts/validate_smpl_cam.sh ./configs/feater_cam_w_pw3d.yaml [ckp_path]
```


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
