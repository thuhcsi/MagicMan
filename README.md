<p align="center">

  <h2 align="center">MagicMan: Generative Novel View Synthesis of Humans with 3D-Aware Diffusion and Iterative Refinement </h2>
  <p align="center">
    <strong>Xu He</strong></a><sup>1</sup>
    · 
    <strong>Xiaoyu Li</strong></a><sup>2</sup>
    · 
    <strong>Di Kang</strong></a><sup>2</sup>
    ·
    <strong>Jiangnan Ye</strong></a><sup>1</sup>
    ·
    <strong>Chaopeng Zhang</strong></a><sup>2</sup>
    ·
    <strong>Liyang Chen</strong></a><sup>1</sup>
    ·
    <br>
    <strong>Xiangjun Gao</strong></a><sup>3</sup>
    ·
    <strong>Han Zhang</strong></a><sup>4</sup>
    ·
    <strong>Zhiyong Wu</strong></a><sup>1,5</sup>
    ·
    <strong>Haolin Zhuang</strong></a><sup>1</sup>
    ·
    <br>
    <sup>1</sup>Tsinghua University  &nbsp;&nbsp;&nbsp; <sup>2</sup>Tencent AI Lab &nbsp;&nbsp;&nbsp;
    <sup>3</sup>The Hong Kong University of Science and Technology  
    <br>
    <sup>4</sup>Standford University  &nbsp;&nbsp;&nbsp; <sup>5</sup>The Chinese University of Hong Kong
    <br>
    </br>
        <a href="https://arxiv.org/abs/2408.14211">
        <img src='https://img.shields.io/badge/arXiv-red' alt='Paper Arxiv'></a> &nbsp; &nbsp;  &nbsp; 
        <a href='https://thuhcsi.github.io/MagicMan/'>
        <img src='https://img.shields.io/badge/Project_Page-green' alt='Project Page'></a> &nbsp;&nbsp; &nbsp; 
        <!-- <a href='https://www.youtube.com/watch?v=mI8RJ_f3Csw'> -->
        <img src='https://img.shields.io/badge/YouTube-blue' alt='Youtube'></a>
  </p>
    </p>
<div align="center">
  <img src="./assets/teaser.png" alt="MagicMan: Generative Novel View Synthesis of Humans with 3D-Aware Diffusion and Iterative Refinement" style="width: 80%; height: auto;"></a>
</div>

<div align="left">
  Figure 1. Given a reference human image in different poses, outfits, or styles (i.e. real and fictional characters) as input, <strong>MagicMan</strong> is able to generate consistent high-quality novel view images and normal maps, which are well-suited for downstream multi-view reconstruction applications.
</div>


<div align="left">
  <br>
  This repository will contain the official implementation of <strong>MagicMan</strong>. For more visual results, please checkout our <a href="https://thuhcsi.github.io/MagicMan/" target="_blank">project page</a>.
</div>


## 📣 News & TODOs
- [x] **[2024.09.16]** Release inference code and pretrained weights!
- [x] **[2024.08.27]** Release paper and project page!
- [ ] Release reconstruction code.
- [ ] Release training code.

## 🧰 Models

|Model        | Resolution|#Views    |GPU Memery<br>(w/ refinement)|#Training Scans|Datasets|
|:-----------:|:---------:|:--------:|:--------:|:--------:|:--------:|
|magicman_base|512x512    |20        |23.5GB    |~2500|[THuman2.1](https://github.com/ytrock/THuman2.0-Dataset), [CustomHumans](https://github.com/custom-humans/editable-humans)|
|magicman_plus|512x512    |24        |26.5GB    |~5500|[THuman2.1](https://github.com/ytrock/THuman2.0-Dataset), [CustomHumans](https://github.com/custom-humans/editable-humans), [2K2K](https://github.com/SangHunHan92/2K2K), [CityuHuman](https://github.com/yztang4/HaP)|

Currently, we provide two versions of models: a base model trained on ~2500 scans to generate 20 views and an enhanced model trained on ~5500 scans to generate 24 views. 
Models can be downloaded [here](https://drive.google.com/drive/folders/1BpMikoiPP9cskpqdH_6T7u9QMgsAZqvt?usp=sharing). Both ```pretrained_weights``` and one version of ```magicman_{version}``` are needed to be downloaded and put under ```./ckpt``` as:
```
|--- ckpt/
|    |--- pretrained_weights/
|    |--- magic_base/ or magic_plus/
```


## ⚙️ Setup
### 1. Clone MagicMan
```bash
git clone https://github.com/thuhcsi/MagicMan.git
cd MagicMan
```

### 2. Installation
```bash
# Create conda environment
conda create -n magicman python=3.10
conda activate magicman

# Install PyTorch and other dependencies
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirements.txt

# Install PyTorch3D
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Install mmcv-full
pip install "mmcv-full>=1.3.17,<1.6.0" -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0.1/index.html

# Install mmhuman3d
pip install "git+https://github.com/open-mmlab/mmhuman3d.git"
```

### 3. Download required models and extra data
Register at [ICON's website](https://icon.is.tue.mpg.de/).

<img src="./assets/register.png" alt="Register" style="width: 50%; height: auto;">

Click **Register now** on all dependencies, then you can download them all using **ONE** account with:
```bash
cd ./thirdparties/econ
bash fetch_data.sh
```
Requied models and extra data are from [SMPL-X](https://github.com/vchoutas/smplify-x), [PIXIE](https://github.com/yfeng95/PIXIE), [PyMAF-X](https://github.com/HongwenZhang/PyMAF-X), and [ECON](https://github.com/YuliangXiu/ECON). 

<details><summary>Please consider cite these awesome works if they also help on your project.</summary>

```bibtex
@inproceedings{SMPL-X:2019,
  title = {Expressive Body Capture: 3D Hands, Face, and Body from a Single Image},
  author = {Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.},
  booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019}
}

@inproceedings{PIXIE:2021,
      title={Collaborative Regression of Expressive Bodies using Moderation}, 
      author={Yao Feng and Vasileios Choutas and Timo Bolkart and Dimitrios Tzionas and Michael J. Black},
      booktitle={International Conference on 3D Vision (3DV)},
      year={2021}
}

@article{pymafx2023,
  title={PyMAF-X: Towards Well-aligned Full-body Model Regression from Monocular Images},
  author={Zhang, Hongwen and Tian, Yating and Zhang, Yuxiang and Li, Mengcheng and An, Liang and Sun, Zhenan and Liu, Yebin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023}
}

@inproceedings{pymaf2021,
  title={PyMAF: 3D Human Pose and Shape Regression with Pyramidal Mesh Alignment Feedback Loop},
  author={Zhang, Hongwen and Tian, Yating and Zhou, Xinchi and Ouyang, Wanli and Liu, Yebin and Wang, Limin and Sun, Zhenan},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2021}
}

@inproceedings{xiu2023econ,
  title     = {{ECON: Explicit Clothed humans Optimized via Normal integration}},
  author    = {Xiu, Yuliang and Yang, Jinlong and Cao, Xu and Tzionas, Dimitrios and Black, Michael J.},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2023},
}
```
</details>

## 💫 Inference
```
python inference.py --config configs/inference/inference-{version}.yaml --input_path {input_image_path} --output_path {output_dir_path} --seed 42 --device cuda:0

# e.g.,
python inference.py --config configs/inference/inference-base.yaml --input_path examples/001.jpg --output_path examples/001 --seed 42 --device cuda:0
```

## 🙏 Acknowledgments

Our code follows several excellent repositories. We appreciate them for making their codes available to the public.
* [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone).
* [ECON](https://github.com/YuliangXiu/ECON).
* [HumanGaussian](https://github.com/alvinliu0/HumanGaussian). Thanks to the authors of HumanGaussian for additional advice and help!

## ✏️ Citing
If you find our work useful, please consider citing:
```BibTeX
@misc{he2024magicman,
    title={MagicMan: Generative Novel View Synthesis of Humans with 3D-Aware Diffusion and Iterative Refinement},
    author={Xu He and Xiaoyu Li and Di Kang and Jiangnan Ye and Chaopeng Zhang and Liyang Chen and Xiangjun Gao and Han Zhang and Zhiyong Wu and Haolin Zhuang},
    year={2024},
    eprint={2408.14211},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## 📢 Disclaimer
⚠️This is an open-source research exploration rather than a commercial product, so it may not meet all your expectations. Due to the variability of the  diffusion model, you may encounter failure cases. Try using different seeds and adjusting the denoising steps if the results are not desirable.
Users are free to create novel views using this tool, but they must comply with local laws and use it responsibly. The developers do not assume any responsibility for potential misuse by users.