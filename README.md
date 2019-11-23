# pytorch-to-tensorflow

This repository depends on [this repository](https://github.com/leonidk/pytorch-tf/blob/master/pytorch-tf.ipynb), but since it is not complete and has some issues, I make this one with a complete example on resnet18 to use in my master's thesis.

The issue in the repo above was that the convolution operation is not correct in all cases.<br>
The order of convolution and padding isdifferent in tensorflow and pytorch, so I take this into consideration when I use conv operator and do the padding first then apply convolution on the padded feature maps instead of using the convolution operator in tensorflow.
