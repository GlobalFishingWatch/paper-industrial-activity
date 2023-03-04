# Infrastructure classification model

A neural network to classify objects in dual-channel SAR and four-channel optical thumbnails as "oil structures", "wind structures", "other structures", or "noise".

_See paper for further description of the data and the training-validation-test cycle_:

[Satellite mapping reveals extensive industrial activity as sea](http://#)

## Architecture

The model is a multi-input/single-output Convolutional Neural Network, consisting of two identical ConvNeXt [1] backbones in separate branches, one for SAR and another for optical image tiles, with a stage compute ratio (convolution layers per stage) of 3:3:9:3. The number of filters (layer depth) increases per stage as 96, 192, 384, 768. The output features from both branches are then concatenated and passed through a dense layer prior to the network's head, containing the (multi-class) classifier layer with softmax activation. We use the GeLU [2] activation function throughout the net, and use categorical crossentropy loss. We regularize the network with Dropout [3] and Label Smoothing [4], and apply Layer Scale [5] and Batch Normalization [6]. 

## Training

We train the network for 300 epochs using Adam with Weight Decay [7]. We perform three cycles of cosine annealing [8], reducing the learning rate from 5e-3 to 5e-6 and the weight decay from 5e-6 to 5e-9 each time. We use a batch size of 128 two- and four-channel 100 x 100 pixel images. For data augmentations, we adopt common schemes including shifts, flips, transpose, random rotations, and magnitude scaling.

## Evaluation

We test the model on a holdout set (20%) that is spatially segregated from the training and validation sets. We tested additional model versions – that differed in the type of regularization used (Drop Block [9] and Stochastic Depth [10]), loss function (Evidential Loss [11]), and input configuration (single mixed image stack vs two independent image stacks). Our best model achieved a combined F1 score of 0.99 (accuracy = 98.9%) for the multiclass problem.

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
[11] https://arxiv.org/abs/1806.01768  
