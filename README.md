<div align="center">
  <h1>CEAug: Crop-Expand Data Augmentation for
  <br>Self-Supervised Monocular Depth Estimation
  <br>[Paper ID: 5586]</h1>
</div>
<br>

    
# **Overview & Quantitative Result**

**1) Overview**

<div align="center">
  <img src="https://github.com/user-attachments/assets/cd00fb19-8305-43c3-9c53-65e017dae369" alt="그림1">
</div>
<br>
<br>


**2) Result on KITTI datase**

<div align="center">
  <img src="https://github.com/user-attachments/assets/2c17e76d-089f-4a09-b067-148b8d200f98">
</div>
Table 1. Comparison result of depth estimation performance on the KITTI eigen benchmark. In the type, M is monocular, MS is both
monocular and stereo. The evaluation was conducted at resolutions of 640x192 and 1024x320. The top results are in bold, and the second
result is underlined.
<br>
<br>

**3) Result on other networks**

<div align="center">
  <img src="https://github.com/user-attachments/assets/bcd1d66f-8637-4017-8c25-0ec38af3be45">
</div>
Table 2. Ablation experiment results. Results without * indicate the use of the network or framework, while result with * indicate the
combined use of the our data augmentation schems.

# Contents
1. **[Environment](#Environment)**
2. **[Dataset](#Dataset)**
    - **KITTI**
    - **Cityscapes**
3. **[Pretrained](#Pretrained)**
    - **Pretrained model list**
    - **Simple test**
4. **[Evaluation](#Evaluation)**
5. **[Training](#Training)**
6. **[Acknowledgement](#Acknowledgement)**

## Environment
You can set the environment as follows:
```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python==4.9.0.60
```
We experimented with versions of PyTorch 1.13.1, CUDA 11.6, and torchvision 0.14.1. If you use other versions, you may encounter issues in code, but they should be solvable.

## Dataset
- **KITTI**
  
You can easily find out how to download from [Monodepth2](https://github.com/nianticlabs/monodepth2?tab=readme-ov-file). Check out the `KITTI training data` section on the Monodepth2. Additionally, ground truth data is required for evaluation. Place the `gt_depth.npz` file in `splits/kitti/eigen/` using the following command:
```bash
python export_gt_depth.py --data_path /home/datasets/kitti_raw_data --split eigen
```
And if you want to eigen benchmark split, please download from this [link](https://www.dropbox.com/scl/fi/o5u5ed09oqp43kmnmdiec/gt_depths.npz?rlkey=akxiagrh1apal4btwnl7ag2fr&st=9p2pu5x9&dl=0) and Move it to the folder `split/kitti/eigen_benchmark/`.


- **Cityscapes**

We experimented with the Cityscapes dataset. To download a, I referred to [ManyDepth](https://github.com/nianticlabs/manydepth). In [website](https://www.cityscapes-dataset.com/) for Cityscapes download, Download files `leftImg8bit_sequence_trainvaltest.zip` and `camera_trainvaltest.zip`. And run the following command:
```bash
python prepare_cityscapes.py \
--img_height 512 \
--img_width 1024 \
--dataset_dir /your_path/cityscapes \
--dump_root /your_path/cityscapes_preprocessed \
--seq_length 3 \
--num_threads 8
```
you can download the ground truth file from the `pretrained weights and evaluation` section of [ManyDepth](https://github.com/nianticlabs/manydepth). Unzip it into the folder `split/cityscapes`
 

## Pretrained
We provide the pretrained model weights used in the paper. Click 'Our' on the table below to download and use the corresponding .pth file. And Download hrnet pre-trained .pth file `HRNet-W18-C (w/ CosineLR + CutMix + 300epochs)` with ImageNet [here](https://github.com/HRNet/HRNet-Image-Classification).
|Model|Dataset|Type|Resolution|Backbone|abs rel|sq rel|rmse|rmse log|a1|a2|a3|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|[Our](https://www.dropbox.com/scl/fi/swxs9vgyzbmybtgcrge68/CEAug_640x192_K.pth?rlkey=8fjyodd4qkngxcc8l3c9t18mg&st=6sp7forf&dl=0)|KITTI|M|640x192|HRNet18|0.094|0.618|4.198|0.171|0.904|0.968|0.985|
|[Our](https://www.dropbox.com/scl/fi/pnf61tksluqu521lb371s/CEAug_640x192_K_MS.pth?rlkey=clls7a19op35gj7199pa9ghav&st=4b17onbl&dl=0)|KITTI|MS|640x192|HRNet18|0.092|0.599|4.086|0.167|0.911|0.969|0.985|
|[Our](https://www.dropbox.com/scl/fi/7ljg8oxfmxnoyc528l6oy/CEAug_1024x320_K.pth?rlkey=90bsu5hicpq42mws1rg8a3dhe&st=7v9tsm1f&dl=0)|KITTI|M|1024x320|HRNet18|0.091|0.584|4.038|0.166|0.910|0.970|0.986|
|[Our](https://www.dropbox.com/scl/fi/z780zsvzi9w3pumr4nazm/CEAug_512x192_CS.pth?rlkey=kyo8vpnfu4fe3e4yxrs85bbmf&st=ucg27bxo&dl=0)|Cityscapes|M|512x192|HRNet18|0.106|0.970|5.791|0.159|0.880|0.968|0.990|


## Evaluation
For model evaluation, you should use file `evaluate_depth.py`. Prepare a pretrained model and run it as follows:

- **KITTI**
```bash
python evaluate_depth.py
--kitti_path /your_path/kitti_data
--backbone CEAug_network
--pretrained_path /your_pretrained_file_path/
--height 192 or 320
--width 640 or 1024
```
- **Cityscapes**
```bash
python evaluate_depth.py
--cityscapes_path /your_path/cityscapes/leftImg8bit_sequence_trainvaltest/
--backbone CEAug_network
--pretrained_path /your_pretrained_file_path/
--height 192
--width 512
```

## Training
For model training, you should use file `train.py`. You can reproduce the pretrained model we provided with the following command:

- **KITTI**
```bash
python train.py
--data_path your_data_path
--dataset kitti
--model_name your_model_name
--backbone CEAug_network 
--local_crop
--CEAug
--height 192 or 320
--width 640 or 1024
--resume 
--seed
```
- **Cityscapes**
(if you use preprocessed Cityscapes dataset, Set the input size to 512x192)
```bash
python train.py 
--data_path_pre your_preprocessed_data_path
--data_path_pre_test your_preprocessed_data_path_for_test
--dataset cityscapes 
--exp_name your_model_name 
--backbone CEAug_network 
--local_crop 
--CEAug 
--height 192
--width 512
--resume 
--seed
```

Additionally, you can train by changing the depth network as shown in the paper. If you want to change depth network to train, change the `--backbone` option to one of `CEAug_network, ResNet, HRNet, DIFFNet, RAdepth, HRdepth, BRNet, DNAdepth, SwinDepth`. You can use networks that are not on the list by adding them directly. Finally, data augmentation can be chosen for use. Select a data augmentation combination among `--CEAug, --local_crop, and --patch_reshuffle`.

## Acknowledgement
We used great open source projects [Monodepth2](https://github.com/nianticlabs/monodepth2?tab=readme-ov-file), [BDEdepth](https://github.com/LiuJF1226/BDEdepth/tree/master?tab=readme-ov-file#datasets), [Manydepth](https://github.com/nianticlabs/manydepth?tab=readme-ov-file), [RA-Depth](https://github.com/hmhemu/RA-Depth), [DIFFNet](https://github.com/brandleyzhou/DIFFNet), [HR-Depth](https://github.com/shawLyu/HR-Depth), [BRNet](https://github.com/wencheng256/BRNet), [DNA-Depth](https://github.com/boyagesmile/DNA-Depth), [SwinDepth](https://github.com/dsshim0125/SwinDepth). Thank you for the incredible project!
