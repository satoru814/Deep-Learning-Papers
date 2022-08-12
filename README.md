# Deep learning object detection

A paper list of object detection using deep learning.

## Cascade
- **Hybrid Task Cascade for Instance Segmentation(HTC)** [[arxiv]](https://arxiv.org/abs/1901.07518)<br>
The Cascade network architecture for instance object detection. It's can fix the drowback of Cascade-RCNN. 
Authors point out parallel processing of cascade-RCNN that prevent effectivery use of box-head and mask-head. They addisonary propose some cascade architecture related to effectivery use of features in backbone and heads.

## Loss function
- **Focal Loss for Dense Object Detection** [[arxiv]](https://arxiv.org/abs/1708.02002)<br>

## Transfomers
- **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows(Swin-Transfomer)** [[arxiv]](https://arxiv.org/abs/2103.14030)<br>
Shift window Transfomer for effective self-attention computation in dense tasks is proposed. [ViT](https://arxiv.org/abs/2010.11929) adapted transfomer to image classification task by looking image as parted tokens. However, for dnese tasks(e.g. object detection and semantic segmentation), it's quandaric computation is impractical. Swin-transfomer can linear self-attention computation as image height and width and keep connection between image tokens.

## Other modules
