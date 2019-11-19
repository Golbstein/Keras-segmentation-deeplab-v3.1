# Keras DeepLab V3.1+

**DeepLab V3+ for Semantic Image Segmentation With Subpixel Upsampling Layer Implementation in Keras**

***Added Tensorflow 2 support - Nov 2019***

DeepLab is a state-of-art deep learning model for semantic image segmentation.
Original DeepLabV3 can be reviewed here: [DeepLab Paper](https://arxiv.org/pdf/1606.00915)
with the original model [implementation](https://github.com/tensorflow/models/tree/master/research/deeplab).

## Features
1. Conditional Random Fields (CRF) implementation as post-processing step to aquire better contour that is correlated with nearby pixels and their color. See here: [Fully-Connected CRF](https://github.com/lucasb-eyer/pydensecrf)
2. Custom image generator for semantic segmentation with large augmentation capabilities.

## New Features That Are Not Included In The Paper
1. Keras Subpixel (Pixel-Shuffle layer) from: [Keras-Subpixel](https://github.com/tetrachrome/subpixel/blob/master/keras_subpixel.py) for efficient upsampling and more accurate segmentation
2. ICNR Initializer for subpixel layer (removing checkerboard artifact) [ICNR](https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf)
3. Comparisson of the original Deeplab model with my Deeplab+subpixel+CRF
4. Fast training - transfer learning from paper's proposed model to a better model within ~1 hour with 1-1080Ti GPU
5. Jaccard (mIOU) monitoring during training process for multi-class segmentation tasks.
6. Adaptive pixel weights.

## Results
**I've compared the segmentation visual results and the IOU score between paper's model and mine, as well as the outcome of applying CRF as a post-processing step.**

Below depicted few examples:


![alt text](https://github.com/Golbstein/deeplabv3_keras/blob/master/examples/exp1.JPG)
![alt text](https://github.com/Golbstein/deeplabv3_keras/blob/master/examples/exp3.JPG)
![alt text](https://github.com/Golbstein/deeplabv3_keras/blob/master/examples/exp2.JPG)
![alt text](https://github.com/Golbstein/deeplabv3_keras/blob/master/examples/exp4.JPG)

**And the IOU score amid the classes:**

![alt text](https://github.com/Golbstein/deeplabv3_keras/blob/master/examples/iou.JPG)

I didn't receive a significant improvement of the IOU scores, perhaps due to low number of epochs. However I believe this method can eventually outperform the original model for a bigger dataset and more epochs.

## Dependencies
* Python 3.6
* Keras>2.2.x
* pydensecrf
* tensorflow > 1.11
