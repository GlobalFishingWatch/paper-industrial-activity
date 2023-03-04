# Vessel classification and length regression model

A neural network to classify objects in dual-channel SAR thumbnails as "vessels" or "noise" and estimate their length.

_See paper for further description of the data and the training-validation-test cycle_:

[Satellite mapping reveals extensive industrial activity as sea](http://#)

## Architecture

The model is a single-input/multi-output Convolutional Neural Network, consisting of a modified ResNet-50 [1] backbone, with 32 grouped convolutions [2] and a stage compute ratio (convolution layers per stage) of 4:4:5:3. The number of filters (layer depth) increases per stage as 96, 192, 384, 768. The backbone is preceded by a stem (down sampling) block containing three convolution layers with 24 filters and a max pooling layer. The network's head contains a binary classifier and a regressor layer. We use the SiLU activation function [3], and a custom "scaled sigmoid" activation for the regression head. We use binary crossentropy and squared error losses for the classification and regression tasks, respectively. We regularize the network with L2 Norm, and use Batch Normalization [4]. 

## Training

We train the network for 300 epochs using Stochastic Gradient Descent with Momentum [5]. We perform three cycles of cosine annealing [cite], reducing the learning rate from 5e-3 to 5e-6 each time. We use a batch size of 16 dual-channel 80 x 80 pixel images with respective objects' lengths. For data augmentations, we adopt common schemes including shifts, flips, transpose, scaling and crop with associated length scaling.

## Evaluation

We test the model on a holdout set (20%) that is spatially segregated from the training and validation sets. We tested additional model versions – that differed in the type of regularization used (Drop Block [6]), loss function (Evidential Loss [7]), and shim layers prior the classification head. We also tested weighting samples by length classes. Our best model achieved a F1 score of 0.97 (accuracy = 97.5%) for the classification task and a R2 score of 0.84 (RMSE = 21.9 m, or about 1 image pixel) for the length estimation task.

---

[1] https://arxiv.org/abs/1512.03385   
[2] https://doi.org/10.1145/3065386  
[3] https://arxiv.org/abs/1702.03118  
[4] https://arxiv.org/abs/1502.03167  
[5] https://arxiv.org/abs/1809.04564  
[6] https://arxiv.org/abs/1810.12890  
[7] https://arxiv.org/abs/1806.01768  
