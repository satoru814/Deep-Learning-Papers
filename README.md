# Deep Learning Papers in Computer Vision.
A paper list of deep learning in computer vision.

## Cascade
- **[HTC, HTC++] Hybrid Task Cascade for Instance Segmentation [2020]** [[arxiv]](https://arxiv.org/abs/1901.07518)<br>
The Cascade network architecture for instance object detection. It's can fix the drowback of Cascade-RCNN. 
Authors point out parallel processing of cascade-RCNN that prevent effectivery use of box-head and mask-head. They addisonary propose some cascade architecture related to effectivery use of features in backbone and heads.

## Loss Function
- **[RetinaNet] Focal Loss for Dense Object Detection [2017]** [[arxiv]](https://arxiv.org/abs/1708.02002)<br>
One-Stage object detection(e.g. yolo, SSD) is suffer from foreground-background imbalance that cause meaningless loss from  easy background examples. Focal loss and RetinaNet is designed to address class imbalance by down-weighting easy examples.


## Transfomers
- **[Swin-Transfomer] Swin Transformer: Hierarchical Vision Transformer using Shifted Windows [2020]** [[arxiv]](https://arxiv.org/abs/2103.14030)<br>
Shift window Transfomer for effective self-attention computation in dense tasks is proposed. [ViT](https://arxiv.org/abs/2010.11929) adapted transfomer to image classification task by looking image as parted tokens. However, for dnese tasks(e.g. object detection and semantic segmentation), Their quandaric computation is impractical. Swin-transfomer can linear self-attention computation as image height and width and keep connection between image tokens.
- **ConTNet: Why not use convolution and transformer at the same time? [2021]** [[arxiv]](https://arxiv.org/abs/2104.13497)
- **Involution: Inverting the Inherence of Convolution for Visual Recognition [2021]** [[arxiv]](https://arxiv.org/abs/2103.06255)
- **BEiT: BERT Pre-Training of Image Transformers[2021]** [[arxiv]](https://arxiv.org/abs/2106.08254)
- **mc-BEiT: Multi-choice Discretization for Image BERT Pre-training[2022]** [[arxiv]](https://arxiv.org/abs/2203.15371) 

## CNN
- **Deformable Convolutional Networks [2017]** [[arxiv]](https://arxiv.org/abs/1703.06211)<br>
- **GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond [2019]** [[arxiv]](https://arxiv.org/abs/1904.11492)<br>
It's important to understand long-range dependency for visual tasks. NLNet(Non-Local Networks) and SENet(Squeeze-and-Exicitation Networks) can understand the global context using non local windows. In this papers, autors point out redundant computation of NLNet and fix it by meet SENet architecture to it.
- **[VAN] Visual Attention Network [2021]** [[arxiv]](https://arxiv.org/abs/2202.09741)<br>
Attention mechanism can be regard as adaptive selecting process based on input features. Self-attention(e.g. Transfomer) is unsutable for adaptation in channel To further improve convolution-base-attention, they decompose a convolution into three parts(depth-wise conv, depth-wise dilated conv, point-wise conv). 

## Object Detection
- **Faster R-CNN: Towards Real-Time ObjectDetection with Region Proposal Networks [2016]** [[arxiv]](https://arxiv.org/abs/1506.01497)
Region-proposal and detection is the components of an two-stage object detetion(e.g. RCNNs, Fast-RCNNs). In Fast-RCNNs, Region-proposal is the bottleneck for real-time object detection. In this paper, Region Proposal Networks(RPNs) that share convolutional layers with detection is proposed.
- **[DETR] End-to-End Object Detection with Transformers [2021]** [[arxiv]](https://arxiv.org/abs/2005.12872)
- **Deformable DETR: Deformable Transformers for End-to-End Object Detection [2021]** [[arxiv]](https://arxiv.org/abs/2010.04159)

## Survey 
- **An Attention Module for Convolutional Neural Networks** [[arxiv]](https://arxiv.org/abs/2108.08205)<br>
 
## Other modules
