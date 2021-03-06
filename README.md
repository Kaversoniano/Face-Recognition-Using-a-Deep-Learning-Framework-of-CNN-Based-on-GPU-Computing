### Python
* Libraries: `theano, cv2, os, numpy, pandas, matplotlib.pyplot`
* Modules: `NN_CNN.py, training_expand.py`
* Projects: `Yale.py, Yale_expand.py`

### Dataset
* Name: Extended Yale Face Database B (B+) - Cropped Images 
* URL: http://vision.ucsd.edu/~iskwak/ExtYaleDatabase/ExtYaleB.html
* Datatype: 192x168 greyscale pixels
* Sample Size: 2414
* Labels: 38 different people

### Model Configurations
* Framework: a hybrid of convolutional layer, pooling layer, fully-connected layer, softmax layer
* Activation Function: Rectified Linear Unit
* Cost Function: Log-Likelihood
* Local Receptive Field: 31x31
* Feature Maps: 20
* Pooling Window: 2x2
* Other Hyperparameters:
  * mini batch size = 10
  * training epochs = 50
  * learning rate = 0.005
  * L2 regularization parameter = 0.5
  * dropout = 0.5

### Results
* in project `Yale.py`, best validation accuracy to date: 98.06% (47th epoch), its corresponding test accuracy: 97.78%
* in project `Yale_expand.py`, best validation accuracy to date: 98.34% (47th epoch), its corresponding test accuracy: 98.34%

### Artificial Expansion of Training Set
* training set size from 1690 to 6760
* translation and rotation
