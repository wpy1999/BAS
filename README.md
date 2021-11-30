# Background Activation Suppression for Weakly Supervised Object Localization

PyTorch implementation of ''Background Activation Suppression for Weakly Supervised Object Localization''. This repository contains PyTorch training code, inference code and pretrained models.

## 📋 Table of content
 1. [📎 Paper Link](#1)
 2. [💡 Abstract](#2)
 3. [📖 Method](#3)
 5. [📃 Requirements](#4)
 6. [✏️ Usage](#5)
    1. [Start](#51)
    2. [Download Datasets](#52)
    3. [Training](#53)
    4. [Inference](#54)
 8. [📊 Experimental Results](#6)
 11. [✉️ Statement](#7)
 12. [🔍 Citation](#8)

## 📎 Paper Link <a name="1"></a> 
> Background Activation Suppression for Weakly Supervised Object Localization ([link](https://arxiv.org/abs/xxx))
* Authors: Pingyu Wu*, Wei Zhai*, Yang Cao
* Institution: University of Science and Technology of China (USTC)

## 💡 Abstract <a name="2"></a> 
Weakly supervised object localization (WSOL) aims to localize the object region using only image-level labels as supervision. Recently a new paradigm has emerged by generating a foreground prediction map (FPM) to achieve the localization task. Existing FPM-based methods use cross-entropy (CE) to evaluate the foreground prediction map and to guide the learning of generator. We argue for using activation value to achieve more efficient learning. It is based on the experimental observation that, for a trained network, CE converges to zero when the foreground mask covers only part of the object region. While activation value increases until the mask expands to the object boundary, which indicates that more object areas can be learned by using activation value. In this paper, we propose a Background Activation Suppression (BAS) method. Specifically, an Activation Map Constraint module (AMC) is designed to facilitate the learning of generator by suppressing the background activation values. Meanwhile, by using the foreground region guidance and the area constraint, BAS can learn the whole region of the object. Furthermore, in the inference phase, we consider the prediction maps of different categories together to obtain the final localization results. Extensive experiments show that BAS achieves significant and consistent improvement over the baseline methods on the CUB-200-2011 and ILSVRC datasets.

## 📖 Method <a name="3"></a> 

<p align="center">
    <img src="./img/pipeline12.png" width="750"/> <br />
    <em> 
    </em>
</p>

**The architecture of the proposed BAS.** In the training phase, the class-specific foreground prediction map $F^{fg}$ and the coupled background prediction map $F^{bg}$ are obtained by the generator, and then fed into the activation map constraint module together with the feature map $F$. In the inference phase, we utilize Top-k to generate the final localization map.

## 📃 Requirements <a name="4"></a> 
  - python 3.6.10 
  - pytorch 1.4.0
  - opencv 4.5.3

## ✏️ Usage <a name="5"></a> 

### Start <a name="51"></a> 

```bash  
git clone https://github.com/wpy1999/BAS.git
cd BAS
```

### Download Datasets <a name="52"></a> 

* CUB ([http://www.vision.caltech.edu/visipedia/CUB-200-2011.html](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html))
* ILSVRC ([https://www.image-net.org/challenges/LSVRC/](https://www.image-net.org/challenges/LSVRC/))

### Training <a name="53"></a> 

We will release our training code upon acceptance.

### Inference <a name="54"></a> 

To test the CUB models, you can download the trained models from
[ [Google Drive (VGG16)]( https://drive.google.com/file/d/1phmgYfoLrUU1W5Dr8S1sdlIdW7BVf726/view?usp=sharing) ],
[ [Google Drive (Mobilenetv1)]( https://drive.google.com/file/d/1RjS0_tXVaERJHnZ25RvHD_EmCsXZQxRS/view?usp=sharing) ],
[ [Google Drive (ResNet50)](https://drive.google.com/file/d/1NHs1DOooovaFX7VTZi6LlXyLLIMarJN8/view?usp=sharing) ],
[ [Google Drive (Inceptionv3)](https://drive.google.com/file/d/1Mj_lGjcFwlzYXe4TcJNeXhE0a2iFw9qR/view?usp=sharing) ],
then run `BAS_inference.py`:
```bash  
python xxx.py
```

To test the ILSVRC models, you can download the trained models from [ [Google Drive](https://drive.google.com/file/d/1BLaGwXJHOg3sGFGwqKCl0LA8cSsfk7RS/view?usp=sharing) ], then run `BAS_inference.py`:
```bash  
python xxx.py
```

## 📊 Experimental Results <a name="6"></a> 

<p align="center">
    <img src="./img/PADresult.png" width="750"/> <br />
    <em> 
    </em>
</p>

## ✉️ Statement <a name="7"></a> 
This project is for research purpose only, please contact us for the licence of commercial use. For any other questions please contact [wpy364755620@mail.ustc.edu.cn](wpy364755620@mail.ustc.edu.cn) or [wzhai056@mail.ustc.edu.cn](wzhai056@mail.ustc.edu.cn).


## 🔍 Citation <a name="8"></a> 

```
@inproceedings{BAS,
  title={Background Activation Suppression for Weakly Supervised Object Localization},
  author={Pingyu Wu and Wei Zhai and Yang Cao},
  booktitle={xxx},
  year={2021}
}
```

