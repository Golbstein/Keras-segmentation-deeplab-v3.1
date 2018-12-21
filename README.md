# Keras DeepLab V3.1+

**DeepLab V3+ for Semantic Image Segmentation With Subpixel Upsampling layer Implementation in Keras**

Original DeepLabV3 can be reviewed here: [DeepLab Paper](https://arxiv.org/pdf/1606.00915)
with the original model [implementation](https://github.com/tensorflow/models/tree/master/research/deeplab).

## Proposed New Features

1. Keras Subpixel (Pixel-Shuffle layer) from: [Keras-Subpixel](https://github.com/tetrachrome/subpixel/blob/master/keras_subpixel.py) for efficient upsampling and more accurate segmentation
2. Conditional Random Fields (CRF) implementation as post-processing step to aquire better contour that is correlated with nearby pixels and their color. See here: [Fully-Connected CRF](https://github.com/lucasb-eyer/pydensecrf)
3. Comparisson of the original Deeplab model with my Deeplab+subpixel+CRF
