# CHSWI
This is the Pytorch Implementation of "3D (C)lothed (H)uman Model Reconstruction Based on (S)ingle-view In-the-(W)ild (I)mage Data".  In this work, a modular model is designed to perform \textcolor{red}{the 3D} clothed human reconstruction with high quality based on one single-view in-the-wild image.

**The whole code will be released when the paper is accepted.**

The contribution of this work can be summarized as below:
- Adaptive integration of convolution and multi-head attention is introduced into the feature extractor to extract the latent codes of the clothes from the input image
- Adjustment of the segmentations to filter the background information for better supervision.

The visualization of our demo (compared with the state-of-the-art methods):
!()[figs/vis.jpg]

