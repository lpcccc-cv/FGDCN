# FGDCN: Flow Guidance Deformable Compensation Network for Video Frame Interpolation (TMM2023)

## Highlights
Motion-based video frame interpolation (VFI) methods have made remarkable progress with the development of deep convolutional networks over the past years. While their performance is often jeopardized by the inaccuracy of flow map estimation, especially in the case of large motion and occlusion. In this paper, we propose a flow guidance deformable compensation network (FGDCN) to overcome the drawbacks of existing motion-based methods. FGDCN decomposes the frame sampling process into two steps: a flow step and a deformation step. Specifically, the flow step utilizes a coarse-to-fine flow estimation network to directly estimate the intermediate flows and synthesizes an anchor frame simultaneously. To ensure the accuracy of the estimated flow, a distillation loss and a task-oriented loss are jointly employed in this step. Under the guidance of the flow priors learned in step one, the deformation step designs a pyramid deformable compensation network to compensate for the missing details of the flow step. In addition, a pyramid loss is proposed to supervise the model in both the image and frequency domains. Experimental results show that the proposed algorithm achieves excellent performance on various datasets with fewer parameters.

## DCNv2
```
cd $ROOT/models/modules/DCNv2
sh make.sh
```

## Train
Modify the configuration file and run:
```
sh train.sh
```

## Test
Model weight and testing results can be downloaded from : "https://pan.baidu.com/s/160kaWNhfq2ZTumRBDLiPSQ", extract code: fvrf

Modify the data path and run:
```
python test_film.py
python test_middle_bury.py
python ucf101.py
python test_vimeo.py
```

## Qualitative Comparison for 2x frame interpolation on Vimeo90K test set
![](./images/slow_motion_examples.png)

## Qualitative Comparison for 2x frame interpolation on SNU_FILM test set
![](./images/fast_motion_examples.png)

## Video demos
We test our model on Adobe240 testset. The original videos are 240FPS. For testing our model, we first sparsely sample them to 30FPS and then use our methods to interpolate them. 

Figures from left to right are the input videos with **30FPS**, interpolation videos with **60FPS, 120FPS and 240FPS**, respectively.
<p float="left">
  <img src=./demo_video/30fps.gif width=300 />
  <img src=./demo_video/60fps.gif width=300 />
  <img src=./demo_video/120fps.gif width=300 />
  <img src=./demo_video/240fps.gif width=300 />
</p>

<p float="left">
  <img src=./demo_video/30fps_v2.gif width=300 />
  <img src=./demo_video/60fps_v2.gif width=300 />
  <img src=./demo_video/120fps_v2.gif width=300 />
  <img src=./demo_video/240fps_v2.gif width=300 />
</p>

# Citation
If you find the code helpful in your research or work, please cite our paper.
```BibTeX
@ARTICLE{10164197,
  author={Lei, Pengcheng and Fang, Faming and Zeng, Tieyong and Zhang, Guixu},
  journal={IEEE Transactions on Multimedia}, 
  title={Flow Guidance Deformable Compensation Network for Video Frame Interpolation}, 
  year={2023},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TMM.2023.3289702}}
```

# Contact
If you have any questions, feel free to E-mail me with [pengchenglei1995@163.com].


