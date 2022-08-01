Noisy Labels in Computer Vision
=========

A curated list of papers that study learning with noisy labels.

---
<!--ts-->
   * [Image Classification](#image-classification)
     * [GitHub Repository](#github-repository)
     * [Survey](#survey)
     * [Distinguished Researchers and Team](#distinguished-researchers-and-team)
   * [Object Detection](#object-detection)
   * [Segmentation](#segmentation)
   * [Object Counting](#object-counting)
   
<!--te-->
---

Image Classification
====================

GitHub Repository
---
* [[Awesome-Learning-with-Label-Noise]](https://github.com/subeeshvasu/Awesome-Learning-with-Label-Noise) ![GitHub Repo stars](https://img.shields.io/github/stars/subeeshvasu/Awesome-Learning-with-Label-Noise?style=social)

Survey
---

* [**2014 TNNLS**] Classification in the Presence of Label Noise: A Survey [[paper]](https://ieeexplore.ieee.org/document/6685834%5C%22)
* [**2019 KBS**] Image Classification with Deep Learning in the Presence of Noisy Labels: A Survey [[paper]](https://arxiv.org/abs/1912.05170)
* [**2020 MIA**] Deep learning with noisy labels: exploring techniques and remedies in medical image analysis [[paper]](https://www.sciencedirect.com/science/article/pii/S1361841520301237)
* [**2020 ArXiv**] A Survey of Label-noise Representation Learning: Past, Present and Future [[paper]](https://arxiv.org/abs/2011.04406) [[code]](https://github.com/bhanML/label-noise-papers) ![GitHub Repo stars](https://img.shields.io/github/stars/bhanML/label-noise-papers?style=social)
* [**2022 TNNLS**] Learning from Noisy Labels with Deep Neural Networks: A Survey [[paper]](https://arxiv.org/abs/2007.08199) [[code]](https://github.com/songhwanjun/Awesome-Noisy-Labels) ![GitHub Repo stars](https://img.shields.io/github/stars/songhwanjun/Awesome-Noisy-Labels?style=social)

Distinguished Researchers and Team
---
* [Tongliang Liu](https://tongliang-liu.github.io/)
* [Bo Han](https://bhanml.github.io/)
* [RIKEN-AIP](https://aip.riken.jp/labs/generic_tech/imperfect_inf_learn/)


Object Detection
================

2022
----

* [**CVPR 2022**] **NLTE**: Xinyu Liu, Wuyang Li, Qiushi Yang, Baopu Li, Yixuan Yuan.  
  "Towards Robust Adaptive Object Detection under Noisy Annotations."
[[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Towards_Robust_Adaptive_Object_Detection_Under_Noisy_Annotations_CVPR_2022_paper.pdf)
[[code]](https://github.com/CityU-AIM-Group/NLTE)
![GitHub Repo stars](https://img.shields.io/github/stars/CityU-AIM-Group/NLTE?style=social)

* [**ECCV 2022**] **OA-MIL**: Chengxin Liu, Kewei Wang, Hao Lu, Zhiguo Cao, Ziming Zhang.  
  "Robust Object Detection With Inaccurate Bounding Boxes."
[[paper]](https://arxiv.org/pdf/2207.09697.pdf) 
[[code]](https://github.com/cxliu0/OA-MIL)
![GitHub Repo stars](https://img.shields.io/github/stars/cxliu0/OA-MIL?style=social)

* [**ECCV 2022**] **W2N**: Zitong Huang, Yiping Bao, Bowen Dong, Erjin Zhou, Wangmeng Zuo.  
   "W2N: Switching From Weak Supervision to Noisy Supervision for Object Detection."
   [[paper]](https://arxiv.org/pdf/2207.12104.pdf)
   [[code]](https://github.com/1170300714/w2n_wsod)
   ![GitHub Repo stars](https://img.shields.io/github/stars/1170300714/w2n_wsod?style=social)

* [**Remote Sensing 2022**] Maximilian Bernhard, Matthias Schubert.  
  "Correcting Imprecise Object Locations for Training Object Detectors in Remote Sensing Applications." [[paper]](https://www.mdpi.com/2072-4292/13/24/4962)

* [**ArXiv 2022**] Shaoru Wang, Jin Gao, Bing Li, Weiming Hu.  
  "Narrowing the Gap: Improved Detector Training with Noisy Location Annotations." [[paper]](https://arxiv.org/pdf/2206.05708.pdf)

2021
----

* [**TIP 2021**] **MRNet**: Youjiang Xu, Linchao Zhu, YiYang, Fei Wu.  
  "Training Robust Object Detectors From Noisy Category Labels and Imprecise Bounding Boxes."
[[paper]](https://ieeexplore.ieee.org/document/9457066)

* [**BMVC 2021**] Jiafeng Mao, Qing Yu, Yoko Yamakata, Kiyoharu Aizawa.  
  "Noisy Annotation Refinement for Object Detection" [[paper]](https://www.bmvc2021-virtualconference.com/assets/papers/0778.pdf)

* [**IEICE TIS 2021**] Jiafeng Mao, Qing Yu, Kiyoharu Aizawa.  
  "Noisy Localization Annotation Refinement for Object Detection."
[[paper]](https://www.jstage.jst.go.jp/article/transinf/E104.D/9/E104.D_2021EDP7026/_pdf)


2020
----

* [**CVPR 2020**] Yunhang Shen, Rongrong Ji, Zhiwei Chen, Xiaopeng Hong, Feng Zheng, Jianzhuang Liu, Mingliang Xu, Qi Tian.  
  "Noise-Aware Fully Webly Supervised Object Detection."
[[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shen_Noise-Aware_Fully_Webly_Supervised_Object_Detection_CVPR_2020_paper.pdf) 
[[code]](https://github.com/shenyunhang/NA-fWebSOD) 
![GitHub Repo stars](https://img.shields.io/github/stars/shenyunhang/NA-fWebSOD?style=social)

* [**CVPR 2020**] Hengduo Li, Zuxuan Wu, Chen Zhu, Caiming Xiong, Richard Socher, Larry S. Davis.  
  "Learning From Noisy Anchors for One-Stage Object Detection."
[[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Learning_From_Noisy_Anchors_for_One-Stage_Object_Detection_CVPR_2020_paper.pdf)

* [**CVPRW 2020**] Aybora Koksal, Kutalmis Gokalp Ince, A. Aydin Alatan.  
  "Effect of Annotation Errors on Drone Detection with YOLOv3."
[[paper]](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w69/Koksal_Effect_of_Annotation_Errors_on_Drone_Detection_With_YOLOv3_CVPRW_2020_paper.pdf)

* [**ICIP 2020**] Jiafeng Mao, Qing Yu, Kiyoharu Aizawa.  
  "Noisy Localization Annotation Refinement For Object Detection." [[paper]](https://ieeexplore.ieee.org/document/9190728)

* [**ArXiv 2020**] Junnan Li, Caiming Xiong, Richard Socher, Steven Hoi.  
  "Towards Noise-resistant Object Detection with Noisy Annotations." [[paper]](https://arxiv.org/pdf/2003.01285.pdf)


2019
----

* [**ICCV 2019**] **NOTE-RCNN**: Jiyang Gao, Jiang Wang, Shengyang Dai, Li-Jia Li, Ram Nevatia.  
  "NOTE-RCNN: NOise Tolerant Ensemble RCNN for Semi-Supervised Object Detection."
[[paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gao_NOTE-RCNN_NOise_Tolerant_Ensemble_RCNN_for_Semi-Supervised_Object_Detection_ICCV_2019_paper.pdf)

* [**AAAI 2019**] **SD-LocNet**: Xiaopeng Zhang, Yang Yang, Jiashi Feng.  
  "Learning to Localize Objects with Noisy Labeled Instances."
[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/4957)

* [**IV 2019**] Simon Chadwick, Paul Newman.  
"Training object detectors with noisy data." [[paper]](https://arxiv.org/pdf/1905.07202.pdf)

Segmentation
============

2022
----
* [**CVPR 2022 oral**] Sheng Liu, Kangning Liu, Weicheng Zhu, Yiqiu Shen, Carlos Fernandez-Granda.  
  "Adaptive Early-Learning Correction for Segmentation from Noisy Annotations."
[[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Adaptive_Early-Learning_Correction_for_Segmentation_From_Noisy_Annotations_CVPR_2022_paper.pdf)
[[code]](https://github.com/Kangningthu/ADELE)
![GitHub Repo stars](https://img.shields.io/github/stars/Kangningthu/ADELE?style=social)

* [**AAAI 2022**] Yaoru Luo, Guole Liu, Yuanhao Guo, Ge Yang.  
  "Deep Neural Networks Learn Meta-Structures from Noisy Labels in Semantic Segmentation."
[[paper]](https://www.aaai.org/AAAI22Papers/AAAI-12729.LuoY.pdf)
[[code]](https://github.com/YaoruLuo/Meta-Structures-for-DNN)
![GitHub Repo stars](https://img.shields.io/github/stars/YaoruLuo/Meta-Structures-for-DNN?style=social)

2021
----
* [**CVPR 2021**] Youngmin Oh, Beomjun Kim, Bumsub Ham.  
  "Background-Aware Pooling and Noise-Aware Loss for Weakly-Supervised Semantic Segmentation."
[[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Oh_Background-Aware_Pooling_and_Noise-Aware_Loss_for_Weakly-Supervised_Semantic_Segmentation_CVPR_2021_paper.pdf)
[[code]](https://github.com/cvlab-yonsei/BANA)
![GitHub Repo stars](https://img.shields.io/github/stars/cvlab-yonsei/BANA?style=social)

* [**ICCV 2021 oral**] Shuquan Ye, Dongdong Chen, Songfang Han, Jing Liao.  
  "Learning with Noisy Labels for Robust Point Cloud Segmentation."
[[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Ye_Learning_With_Noisy_Labels_for_Robust_Point_Cloud_Segmentation_ICCV_2021_paper.pdf)
[[code]](https://github.com/pleaseconnectwifi/PNAL)
![GitHub Repo stars](https://img.shields.io/github/stars/pleaseconnectwifi/PNAL?style=social)

* [**ICCV 2021**] Yuxi Wang, Junran Peng, Zhaoxiang Zhang.  
  "Uncertainty-aware Pseudo Label Refinery for Domain Adaptive Semantic Segmentation."
[[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Uncertainty-Aware_Pseudo_Label_Refinery_for_Domain_Adaptive_Semantic_Segmentation_ICCV_2021_paper.pdf)

* [**IJCV 2021**] Zhedong Zheng, Yi Yang.  
  "Rectifying Pseudo Label Learning via Uncertainty Estimation for Domain Adaptive Semantic Segmentation."
[[paper]](https://link.springer.com/article/10.1007/s11263-020-01395-y)
[[code]](https://github.com/layumi/Seg-Uncertainty)
![GitHub Repo stars](https://img.shields.io/github/stars/layumi/Seg-Uncertainty?style=social)
  * [**IJCAI 2020 Conference Version**] Unsupervised Scene Adaptation with Memory Regularization in vivo [[paper]](https://arxiv.org/pdf/1912.11164.pdf)

2020
----
* [**ECCV 2020**] Longrong Yang, Fanman Meng, Hongliang Li, Qingbo Wu, Qishang Cheng.  
  "Learning with Noisy Class Labels for Instance Segmentation."
[[paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590035.pdf)
[[code]](https://github.com/longrongyang/Learning-with-Noisy-Class-Labels-for-Instance-Segmentation)
![GitHub Repo stars](https://img.shields.io/github/stars/longrongyang/LNCIS?style=social)

* [**NeurIPS 2020**] Le Zhang, Ryutaro Tanno, Mou-Cheng Xu, Chen Jin, Joseph Jacob, Olga Ciccarelli, Frederik Barkhof, Daniel C. Alexander.  
  "Disentangling Human Error from the Ground Truth in Segmentation of Medical Images."
[[paper]](https://proceedings.neurips.cc/paper/2020/file/b5d17ed2b502da15aa727af0d51508d6-Paper.pdf)
[[code]](https://github.com/moucheng2017/Learn_Noisy_Labels_Medical_Images)
![GitHub Repo stars](https://img.shields.io/github/stars/moucheng2017/Learn_Noisy_Labels_Medical_Images?style=social)

* [**MICCAI 2020**] Minqing Zhang, Jiantao Gao, Zhen Lyu, Weibing Zhao, Qin Wang, Weizhen Ding, Sheng Wang, Zhen Li, Shuguang Cui.  
  "Characterizing Label Errors: Confident Learning for Noisy-labeled Image Segmentation."
[[paper]](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_70)
[[code]](https://github.com/502463708/Confident_Learning_for_Noisy-labeled_Medical_Image_Segmentation)
![GitHub Repo stars](https://img.shields.io/github/stars/502463708/Confident_Learning_for_Noisy-labeled_Medical_Image_Segmentation?style=social)

2019
----
* [**CVPR 2019**] Yi Zhu, Karan Sapra, Fitsum A. Reda, Kevin J. Shih, Shawn Newsam, Andrew Tao, Bryan Catanzaro.  
  "Improving Semantic Segmentation via Video Propagation and Label Relaxation."
[[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Improving_Semantic_Segmentation_via_Video_Propagation_and_Label_Relaxation_CVPR_2019_paper.pdf)
[[code]](https://github.com/NVIDIA/semantic-segmentation/tree/sdcnet)


Object Counting
===============

* [**CVPR 2022**] Zhi-Qi Cheng, Qi Dai, Hong Li, Jingkuan Song, Xiao Wu, Alexander G. Hauptmann.  
  "Rethinking Spatial Invariance of Convolutional Networks for Object Counting."
[[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Cheng_Rethinking_Spatial_Invariance_of_Convolutional_Networks_for_Object_Counting_CVPR_2022_paper.pdf)
[[code]](https://github.com/zhiqic/Rethinking-Counting)
![GitHub Repo stars](https://img.shields.io/github/stars/zhiqic/Rethinking-Counting?style=social)

* [**NeurIPS 2020**] Jia Wan, Antoni B. Chan.  
  "Modeling Noisy Annotations for Crowd Counting."
[[paper]](https://proceedings.neurips.cc/paper/2020/file/22bb543b251c39ccdad8063d486987bb-Paper.pdf)
[[code]](https://github.com/jia-wan/NoisyCC-pytorch)
![GitHub Repo stars](https://img.shields.io/github/stars/jia-wan/NoisyCC-pytorch?style=social)

