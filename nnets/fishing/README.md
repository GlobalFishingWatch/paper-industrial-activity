# Fishing and non-fishing classification model

A neural network to classify environmental rasters (11-channel images + a scalar length value) as "fishing vessels" or "non-fishing vessels".

_See paper for further description of the data and the training-validation-test cycle_:

[Satellite mapping reveals extensive industrial activity as sea](http://#)

## Architecture

The model is a multi-input-mixed-data/single-output Convolutional Neural Network, consisting of a ConvNeXt [1] backbone with a stage compute ratio (convolution layers per stage) of 3:3:9:3. The number of filters (layer depth) increases per stage as 96, 192, 384, 768. The output features from the backbone are then concatenated and passed through a fully connected block (with two dense layers) prior to the network's head, containing the classifier layer with softmax activation. We use the GeLU [2] activation function throughout the net, and use categorical crossentropy loss. We regularize the network with Dropout [3] and Label Smoothing [4], and apply Layer Scale [5] and Batch Normalization [6]. 

## Training

We train the network for 120 epochs using Adam with Weight Decay [7]. We perform three cycles of cosine annealing [8], reducing the learning rate from 5e-3 to 5e-6 and the weight decay from 5e-6 to 5e-9 each time. We use a batch size of 64 11-channel 100 x 100 pixel images. We normalize the data by the respective order of magnitude of each channel. For data augmentations, we adopt common schemes including flips, transpose, random rotations, coarse dropout, channel dropout, magnitude scaling and log transformation. We divide the training data into two spatially independent sets and train two independent models. Our final predictions result from a two-model ensemble. We also calibrate the prediction scores for three (fishing:non-fishing) labels ratios: 1:3, 1:1, 3:1, providing a possible range for the predictions.

## Evaluation

We test the models on a single holdout set (12%) that is spatially segregated from the two sets of training and validation data (one set per model). We tested additional model versions – that differed in the type of regularization used, such as Drop Block [9] and Stochastic Depth [10]. We chose standard Dropout for deployment based on its superior performance and simplicity. Our best model ensemble achieved a F1 score of 0.91 (accuracy = 90.5%) for the classification task. 

---

[1] https://arxiv.org/abs/2201.03545  
[2] https://arxiv.org/abs/1606.08415  
[3] https://dl.acm.org/doi/10.5555/2627435.2670313  
[4] https://arxiv.org/abs/1906.02629  
[5] https://arxiv.org/abs/2103.17239  
[6] https://arxiv.org/abs/1502.03167  
[7] https://arxiv.org/abs/1711.05101  
[8] https://arxiv.org/abs/1608.03983v5  
[9] https://arxiv.org/abs/1810.12890  
[10] https://arxiv.org/abs/1603.09382  
