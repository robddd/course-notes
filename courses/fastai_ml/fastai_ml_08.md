# Fastai - Machine Learning
## Lesson 8 - Gradient Descent and Logistic Regression


Random forests only offered a nearest neighbour type approach. They cannot extrapolate to things that it hasn’t seen before

Neural networks allow us to calculate in very complicated ways.

This course on ML will only be fully connected neural network stuff for how they work.

Its very hard to win a Kaggle competition with a random forest. But, it is pretty easy to get in the top 10%.

You can unzip things in python with gzip

Pickle works with almost all python objects. But its not optimized for any python objects.

Feather is good because it is highly optimized to pandas, feather can also be used with non python tools.

Some encoding differences between python 2 and python 3

De-structuring - when you assign data into tuples to make them into separate objects.

Flattening a tensor -> turning a tensor into a lower rank tensor than what you started with.

Deep Learning and Math `(rows, columns)`
Image people PIL and TV `(cols, rows)` eg VGA (640x480)

Normalising Data

Subtract by the mean and divide by the standard deviation.

`x = ( x - mean ) / std`

**Random forests only care about the sort order of the independent variables not about the scale. That’s why they are wonderfully immune to outliers.**

Spearmans correlation - Is a rank correlation
AUROC - only cares about order

Deep Learning - We need to use the mean and std of the training data, and use this exact mean for the val and for the test set.

When you normalise using the train mean and standard deviation. You can check your val and test sets once normalised to ensure the mean and std are close to 0, 1

For colour images you usually normalise by channel, one number each for R, G, B.

**In general you need a different normalisation co-efficient for things that you would expect to behave differently.**

How do you encode categories?

Reshape an image:

![Alt text](images/L8_reshape.png?raw=true)

-1 is a better thing to use so that if there is a change then the code wont break.

Open CV is Blue, Green, Red. Everything else is RGB. 

Spend a lot of time practicing reshaping, slicing and ordering dimensions. The best way is to create some small tensors yourself and experiment.

Almost all Deep Learning libraries are something like `(example, row, column)`

A number has to have a consistent meaning across all numbers when you use it.

Pictures and audio are things your brain is really good at interpreting, so **look at them to explore as you go**, as well as looking at the numbers.

A neural network can approximate any other function arbitrarily closely. It is a universal approximator.

A neural network is a lot of matrix multiplies, which are just linear functions. Also added are non-linearities

Why you (yes, you) should blog - By Rachel Thomas
https://medium.com/@racheltho/why-you-yes-you-should-blog-7d2544ac1045

Building NNs - With PyTorch
*It is like numpy that can run on the GPU (which can be 10s or hundreds of times faster)

CUDA is a framework universally used in deep learning. Nvidia is the only type of GPU with General Purpose support.

The derivatives get done for you. It is useful to know what a derivative is and the chain rule. But you do not need to hand derive functions.

Homework this week, go through the notebook and the pytorch stuff again for a refresher. If there is some fast ai stuff here re-write it in pytorch

`Negative Log Likelihood loss == cross entropy`

Categorical cross entropy, just looks at the prediction for the true class. Can be done with if statements. It doesn’t really matter which class it predicts if it isnt predicting the correct class.

`argmax` returns the index of the highest value

Pytorch uses 'criterion' to mean loss function

Universal Approximation Theorem
http://neuralnetworksanddeeplearning.com/chap4.html

The archictecture of a neural network can approximate any function. All we need to do is find the parameters that approimate the function.

A pytroch module is either a neural net or a layer in a neural net.

![Alt text](images/L8_pytorch_LR.png?raw=true)

This will make `nn.Module` a sub class of `LogReg`, it is crucial to run `super().__init__(self):` to construct this class.

If you use random numbers which are normally distributed and divided by the number of rows, number of classes?

In PyTorch use `.view` to do `.reshape` It is pretty much the same thing though.

We use softmax to get the probabilities of each class. Probabilities should be between zero one. They behave like probabilities and they force one of the probabilities to be very high. They are a great function to use to help the neural net to map to the output that we want.
