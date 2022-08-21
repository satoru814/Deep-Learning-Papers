# Deep Learning Papers in Computer Vision.
A paper list of deep learning in computer vision.

## Cascade
- **[HTC, HTC++] Hybrid Task Cascade for Instance Segmentation [2020]** [[arxiv]](https://arxiv.org/abs/1901.07518)<br>
The Cascade network architecture for instance object detection. It's can fix the drowback of Cascade-RCNN. 
Authors point out parallel processing of cascade-RCNN that prevent effectivery use of box-head and mask-head. They addisonary propose some cascade architecture related to effectivery use of features in backbone and heads.<br>  

## Loss Function
- **[RetinaNet] Focal Loss for Dense Object Detection [2017]** [[arxiv]](https://arxiv.org/abs/1708.02002)<br>
One-Stage object detection(e.g. yolo, SSD) is suffer from foreground-background imbalance that cause meaningless loss from  easy background examples. Focal loss and RetinaNet is designed to address class imbalance by down-weighting easy examples.<br>  

## RNN
- **Pixel Recurrent Neural Networks [2016]** [[arxiv](https://arxiv.org/abs/1601.06759)

## Transfomer
- **Image Transformer [2018]** [[arxiv]](https://arxiv.org/abs/1802.05751)
- **Generating Long Sequences with Sparse Transformers [2019]** [[arxiv]](https://arxiv.org/abs/1904.10509)

## ViT
- **[Swin-Transfomer] Swin Transformer: Hierarchical Vision Transformer using Shifted Windows [2020]** [[arxiv]](https://arxiv.org/abs/2103.14030)<br>
Shift window Transfomer for effective self-attention computation in dense tasks is proposed. [ViT](https://arxiv.org/abs/2010.11929) adapted transfomer to image classification task by looking image as parted tokens. However, for dense tasks(e.g. object detection and semantic segmentation), their quandaric computation is impractical. Swin-transfomer can linear self-attention computation as image height and width and keep connection between image tokens.<br>  
- **ConTNet: Why not use convolution and transformer at the same time? [2021]** [[arxiv]](https://arxiv.org/abs/2104.13497)[READ]
- **Involution: Inverting the Inherence of Convolution for Visual Recognition [2021]** [[arxiv]](https://arxiv.org/abs/2103.06255)
- **Emerging Properties in Self-Supervised Vision Transformers[2021]** [[arxiv]](https://arxiv.org/abs/2104.14294)
- **How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers[2021]** [[arxiv]](https://arxiv.org/abs/2106.10270)
- **Rethinking Spatial Dimensions of Vision Transformers [2021]** [[arxiv]](https://arxiv.org/abs/2103.16302)
- **[DeiT]Training data-efficient image transformers & distillation through attention [2021]** [[arxiv]](https://arxiv.org/abs/2103.16302)

## VQ
- **Neural Discrete Representation Learning[2017]** [[arxiv]](https://arxiv.org/abs/1711.00937)
- **Vector-quantized Image Modeling with Improved VQGAN [2021]** [[arxic]](https://arxiv.org/abs/2110.04627)[READ]

## CNN
- **Deformable Convolutional Networks [2017]** [[arxiv]](https://arxiv.org/abs/1703.06211)<br>  
- **GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond [2019]** [[arxiv]](https://arxiv.org/abs/1904.11492)<br>
It's important to understand long-range dependency for visual tasks. NLNet(Non-Local Networks) and SENet(Squeeze-and-Exicitation Networks) can understand the global context using non local windows. In this papers, autors point out redundant computation of NLNet and fix it by meet SENet architecture to it.<br>  
- **[VAN] Visual Attention Network [2021]** [[arxiv]](https://arxiv.org/abs/2202.09741)<br>
Attention mechanism can be regard as adaptive selecting process based on input features. Self-attention(e.g. Transfomer) is unsutable for adaptation in channel To further improve convolution-base-attention, they decompose a convolution into three parts(depth-wise conv, depth-wise dilated conv, point-wise conv).<br>   

## Object Detection
- **Faster R-CNN: Towards Real-Time ObjectDetection with Region Proposal Networks [2016]** [[arxiv]](https://arxiv.org/abs/1506.01497)
Region-proposal and detection is the components of an two-stage object detetion(e.g. RCNNs, Fast-RCNNs). In Fast-RCNNs, Region-proposal is the bottleneck for real-time object detection. In this paper, Region Proposal Networks(RPNs) that share convolutional layers with detection is proposed.<br>  
- **FCOS: Fully Convolutional One-Stage Object Detection [2019]** [[arxiv]](https://arxiv.org/abs/1904.01355)
- **[DETR] End-to-End Object Detection with Transformers [2021]** [[arxiv]](https://arxiv.org/abs/2005.12872)
- **Deformable DETR: Deformable Transformers for End-to-End Object Detection [2021]** [[arxiv]](https://arxiv.org/abs/2010.04159)
- **DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection** [[arxiv]](https://arxiv.org/abs/2203.03605v2)

## GAN
- **[VQGAN] Taming Transformers for High-Resolution Image Synthesis[2021]** [[arxiv]](https://arxiv.org/abs/2012.09841v1)

## VAE
- **[VQVAE] Discrete Variational Autoencoders** [[arxiv]](https://arxiv.org/abs/1609.02200)[READ]
- **Generating Diverse High-Fidelity Images with VQ-VAE-2 [2019]** [[arxiv]](https://arxiv.org/abs/1906.00446)
- **Zero-Shot Text-to-Image Generation** [[arxiv]](https://arxiv.org/abs/2102.12092)[READ]

## MIM
- **Generative Pretraining From Pixels**[[Semantic Scholar]][(https://www.semanticscholar.org/paper/Generative-Pretraining-From-Pixels-Chen-Radford/bc022dbb37b1bbf3905a7404d19c03ccbf6b81a8)
- **BEiT: BERT Pre-Training of Image Transformers[2021]** [[arxiv]](https://arxiv.org/abs/2106.08254)<br>
It's difficult to directly adapt BERT ideas to pretrain for vision transfomer by pixel level low-dependency of a image. To overcome this dificulty, they use dVAE tokenize image patches to discrete laten codes. 
- **[MAE]Masked Autoencoders Are Scalable Vision Learners[2021]** [[arxiv]](https://arxiv.org/abs/2111.06377)[READ]
- **mc-BEiT: Multi-choice Discretization for Image BERT Pre-training[2022]** [[arxiv]](https://arxiv.org/abs/2203.15371) 
- **[VQ-KD]BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers[2022]** [[arxiv]](https://arxiv.org/abs/2208.06366)[READ]

## KD
- **Masked Feature Prediction for Self-Supervised Visual Pre-Training** [[arxiv]](https://arxiv.org/abs/2112.09133)[READ]

## CNN Attention 
- **CBAM: Convolutional Block Attention Module[2018]** [[arxiv]](https://arxiv.org/abs/1807.06521)
- **An Attention Module for Convolutional Neural Networks** [[arxiv]](https://arxiv.org/abs/2108.08205)<br>

## Augmentation
- CutMix
- Mixup
- RandAug

## ??
- **[LV-ViT][tokens-labelling] All Tokens Matter: Token Labeling for Training Better Vision Transformers[2021]** [[arxiv]](https://arxiv.org/abs/2104.10858)[READ]
## Other modules
