# Deep Learning Papers in Computer Vision.
A paper list of deep learning in computer vision.

## Distributed training
- **Fast Algorithms for Convolutional Neural Networks[2015]** [[arxiv]](https://arxiv.org/abs/1509.09308)<br>
Introduction of fast comvolution operations as a deep learning layer. Winograd algorithm and FFT is discribed.
- **Performance Modeling and Evaluation of Distributed Deep Learning Frameworks on GPUs[2018]** [[arxiv]](https://arxiv.org/abs/1711.05979)<br>
Comparing the performance of deep learning frameworks(Caffe MPI, Tensorflow, MXNet etc..) in single-GPU and multiple-GPUs each other.
- **Communication-Efficient Distributed Deep Learning: A Comprehensive Survey [2020]** [[arxiv]](https://arxiv.org/abs/2003.06307)<br>

## Multimodal
- **Illustrative Language Understanding:Large-Scale Visual Grounding with Image Search[2018]** [[pdf]](http://www.cs.toronto.edu/~hinton/absps/picturebook.pdf)
- **[VSE++]VSE++: Improved Visual-Semantic Embeddings[2018]** [[arxiv]](https://arxiv.org/abs/1707.05612v1)<br>
- **[CLIP] Learning Transferable Visual Models From Natural Language Supervision** [[arxiv]](https://arxiv.org/abs/2103.00020)<br>
- **[CoCa] Contrastive Captioners are Image-Text Foundation Models [2022]** [[arxiv]](https://arxiv.org/abs/2205.01917)[READ]
- **[MetaLM] Language models are general-purpose interfaces [2022]** [[arxiv]](https://arxiv.org/abs/2206.06336)
- **[BEiT-3] Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks[2022]** [[arxiv]](https://arxiv.org/abs/2208.10442v1)[READ]<br>
- **[ALIGN] Scaling Up Visual and Vision-Language Representation LearningWith Noisy Text Supervision[2022]** [[arxiv]](https://arxiv.org/abs/2102.05918)<br>
Making web scale noisy dataset of text and image with simple filtring. Contrassive learning with dual encoder loss.
- 
## Learning in large dataset
- **Conceptual captions: A cleaned, hypernymed, image alt-textdataset for automatic image captioning[2018]** [[pdf]](https://aclanthology.org/P18-1238.pdf)<br>
- **[OpenImages] The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection at scale[2018]** [[arxiv]](https://arxiv.org/abs/1811.00982)<br>
- **[YFCC-100M] YFCC100M: The New Data in Multimedia Research** [[arxiv]](https://arxiv.org/abs/1503.01817)<br>
A early work on making and evaluating large dataset(100M images) for various deep learninig tasks.
- **[JFT-300M] Revisiting Unreasonable Effectiveness of Data in Deep Learning Era [2017]** [[arxiv]](https://arxiv.org/abs/1707.02968)<br>
The effect of dataset size is investigated. JFT-300M is made from web search and labeling algolithm. Accordings to their ablative study, larger dataset pre-training improve model performance logarithmically.
- **Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour [2017]** [[arxiv]](https://arxiv.org/abs/1706.02677)<br>
Training with big minibatch(8k)
- **Exploring the Limits of Weakly Supervised Pretraining [2018]** [[arxiv]]](https://arxiv.org/abs/1805.00932)<br>
learning in Instagrams data
- **[ViT-giant] Scaling Vision Transformers [2021]** [[arxiv]](https://arxiv.org/abs/2106.04560)<br>
learning in JFT data


## Cascade
- **[HTC, HTC++] Hybrid Task Cascade for Instance Segmentation [2020]** [[arxiv]](https://arxiv.org/abs/1901.07518)<br>
The Cascade network architecture for instance object detection. It's can fix the drowback of Cascade-RCNN. 
Authors point out parallel processing of cascade-RCNN that prevent effectivery use of box-head and mask-head. They addisonary propose some cascade architecture related to effectivery use of features in backbone and heads.<br>  

## Loss Function
- **[RetinaNet] Focal Loss for Dense Object Detection [2017]** [[arxiv]](https://arxiv.org/abs/1708.02002)<br>
One-Stage object detection(e.g. yolo, SSD) is suffer from foreground-background imbalance that cause meaningless loss from  easy background examples. Focal loss and RetinaNet is designed to address class imbalance by down-weighting easy examples.<br>  

## Transfomer
- **Image Transformer [2018]** [[arxiv]](https://arxiv.org/abs/1802.05751)<br>
- **Generating Long Sequences with Sparse Transformers [2019]** [[arxiv]](https://arxiv.org/abs/1904.10509)<br>
Extend transfomer to image to use sparse transfomers.
- **VLMo: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts [2021]** [[arxiv]](https://arxiv.org/abs/2111.02358)<br>
Multiway Transfomer

## ViT
- **[Swin-Transfomer] Swin Transformer: Hierarchical Vision Transformer using Shifted Windows [2020]** [[arxiv]](https://arxiv.org/abs/2103.14030)<br>
Shift window Transfomer for effective self-attention computation in dense tasks is proposed. [ViT](https://arxiv.org/abs/2010.11929) adapted transfomer to image classification task by looking image as parted tokens. However, for dense tasks(e.g. object detection and semantic segmentation), their quandaric computation is impractical. Swin-transfomer can linear self-attention computation as image height and width and keep connection between image tokens.<br> 
- **[LV-ViT][tokens-labelling] All Tokens Matter: Token Labeling for Training Better Vision Transformers[2021]** [[arxiv]](https://arxiv.org/abs/2104.10858)[READ]<br>
Token labeling(assign label for each patch of ViT and compute loss for all of them in ViT) is proposed for more context-rich training. 
- **[ConTNet] ConTNet: Why not use convolution and transformer at the same time? [2021]** [[arxiv]](https://arxiv.org/abs/2104.13497)[READ]<br>
- **Involution: Inverting the Inherence of Convolution for Visual Recognition [2021]** [[arxiv]](https://arxiv.org/abs/2103.06255)<br>
- **Emerging Properties in Self-Supervised Vision Transformers[2021]** [[arxiv]](https://arxiv.org/abs/2104.14294)<br>
- **How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers[2021]** [[arxiv]](https://arxiv.org/abs/2106.10270)<br>
- **Rethinking Spatial Dimensions of Vision Transformers [2021]** [[arxiv]](https://arxiv.org/abs/2103.16302)<br>
- **[DeiT]Training data-efficient image transformers & distillation through attention [2021]** [[arxiv]](https://arxiv.org/abs/2103.16302)<br>
- **[Tokens-to-Token ViT] Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet [2021]**[[arxiv]](https://arxiv.org/abs/2101.11986v1)
- **[GC-ViT] Global Context Vision Transformers [2022]** [[arxiv]](https://arxiv.org/abs/2206.09959v1)
- **[Hydra-attention] Hydra Attention: Efficient Attention with Many Heads[2022]** [[arxiv]](https://arxiv.org/abs/2209.07484)
many attention mechanism for computational efficiency.

## VQ
- **Neural Discrete Representation Learning[2017]** [[arxiv]](https://arxiv.org/abs/1711.00937)<br>
- **[ViT-VQGAN]Vector-quantized Image Modeling with Improved VQGAN [2021]** [[arxic]](https://arxiv.org/abs/2110.04627)[READ]<br>

## CNN
- **Deformable Convolutional Networks [2017]** [[arxiv]](https://arxiv.org/abs/1703.06211)<br>  
- **[GCNet] GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond [2019]** [[arxiv]](https://arxiv.org/abs/1904.11492)<br>
It's important to understand long-range dependency for visual tasks. NLNet(Non-Local Networks) and SENet(Squeeze-and-Exicitation Networks) can understand the global context using non local windows. In this papers, autors point out redundant computation of NLNet and fix it by meet SENet architecture to it.<br>  
- **[VAN] Visual Attention Network [2021]** [[arxiv]](https://arxiv.org/abs/2202.09741)<br>
Attention mechanism can be regard as adaptive selecting process based on input features. Self-attention(e.g. Transfomer) is unsutable for adaptation in channel To further improve convolution-base-attention, they decompose a convolution into three parts(depth-wise conv, depth-wise dilated conv, point-wise conv).<br>   

## Object Detection
- **[Faster R-CNN] Faster R-CNN: Towards Real-Time ObjectDetection with Region Proposal Networks [2016]** [[arxiv]](https://arxiv.org/abs/1506.01497)<br>
Region-proposal and detection is the components of an two-stage object detetion(e.g. RCNNs, Fast-RCNNs). In Fast-RCNNs, Region-proposal is the bottleneck for real-time object detection. In this paper, Region Proposal Networks(RPNs) that share convolutional layers with detection is proposed.<br>  
- **[FCOS] FCOS: Fully Convolutional One-Stage Object Detection [2019]** [[arxiv]](https://arxiv.org/abs/1904.01355)<br>
One-stage object detection using FCN.
- **[DETR] End-to-End Object Detection with Transformers [2021]** [[arxiv]](https://arxiv.org/abs/2005.12872)<br>
Object detection using transfomer.
- **Deformable DETR: Deformable Transformers for End-to-End Object Detection [2021]** [[arxiv]](https://arxiv.org/abs/2010.04159)<br>
- **[DINO] DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection [2021]** [[arxiv]](https://arxiv.org/abs/2203.03605v2)<br>

## GAN
- **[PixelSNAIL] PixelSNAIL: An Improved Autoregressive Generative Model [2017]** [[arxiv]](https://arxiv.org/abs/1712.09763)<br>
sota of autoregressive generative models in 2017
- **not-so-BigGAN: Generating High-Fidelity Images on Small Compute with Wavelet-based Super-Resolution [2020]**[[arxiv]](https://arxiv.org/abs/2009.04433)<br>
GAN with wavelet combarsion for a image contexts
- **[VQGAN] Taming Transformers for High-Resolution Image Synthesis[2021]** [[arxiv]](https://arxiv.org/abs/2012.09841v1)[READ]<br>
They use transfomer to improve to get more large-scale context of image for latent code efficiency from VQVAE.

## AutoRegressive model
- **[NADE] Neural Autoregressive Distribution Estimation** [[arxiv]](https://arxiv.org/abs/1605.02226)
- **Generative Image Modeling Using Spatial LSTMs [2015]** [[arxiv]](https://arxiv.org/abs/1506.03478)<br>
- **[PixelCNN] Pixel Recurrent Neural Networks [2016]** [[arxiv]](https://arxiv.org/abs/1601.06759)[READ]<br>
Several types of RNN and CNN for image generation is introduced.

## VAE
- **[VQVAE] Discrete Variational Autoencoders** [[arxiv]](https://arxiv.org/abs/1609.02200)[READ]
- **Generating Diverse High-Fidelity Images with VQ-VAE-2 [2019]** [[arxiv]](https://arxiv.org/abs/1906.00446)<br>[READ]
Extend VQVAE to use hierarchy of learned representations. 
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
- **RandAugment: Practical automated data augmentation with a reduced search space [2019]** [[arxiv]](https://arxiv.org/abs/1909.13719) [READ]
Random select and magnitude of transforms(augmentation) is suggested in place of previous data augmentation strategy(e.g. AutoAug, FastAug)

## Optimizer
- **Large Batch Optimization for Deep Learning: Training BERT in 76 minutes[2020]** [[arxiv]](https://arxiv.org/abs/1904.00962)

## Quality Assessment
- **Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment[2017]** [[pdf]](https://ieeexplore.ieee.org/document/8063957)

## Other
- **L2D2: Learnable Line Detector and Descriptor [2021]** [[arxiv]](https://www.researchgate.net/publication/355340221_L2D2_Learnable_Line_Detector_and_Descriptor)
Line paires dataset in 2d from 3d point cloouds. Learnable desctiptions for lines.
- **Pure Transformers are Powerful Graph Learners [2022]** [[arxiv]](https://arxiv.org/abs/2207.02505) [READ]
- **On the Spectral Bias of Neural Networks [2018]** [[arxiv]](https://arxiv.org/abs/1806.08734)
