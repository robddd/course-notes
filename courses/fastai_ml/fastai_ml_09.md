# Fastai - Machine Learning
## Lesson 9 - Regularization, Learning Rates and NLP

https://www.youtube.com/watch?v=PGC0UxakTvM

Theory of human perception - What is the human brain good at percieving? We are best at perceiving differences in color. Very important for data viz.

http://structuringtheunstructured.blogspot.com/

Posting stuff out to the world is intimidating the first time you do it. But then when you get some good feedback its helpful

The key things that we get from pytorch are :
*writing python code that runs on the GPU
*autograd

We can do everything else ourselves

Can start using all of the help and libraries and build more and more of it ourselves

We must build our NN as OO. We must inherit `nn.Module`

Construct the object - dunder innit

`super().__init__()` construct the superclass

**When rebuilding something, get it working as is and then rebuild parts of it one at a time, making it simpler and simpler.**

Pytorch classes need to be under a method called `forward()`

Softmax:

Take `exps ^ output` for each class, then divide by the sum of the exps

We don’t strictly need a softmax - but it helps the NN learn.

Pytorch uses the log of softmax for numerical stability

We use sigmoid if we want to do multi class prediction, if examples have 0 to many labels

When you wrap things in `nn.Parameter()` PyTorch knows that this is something that needs to be optimised.

PyTorch data loader - uses fastai one, which is a generator to get another then another then another of something. There is a close relationship between generators and iterators.

You can use `iter()` to turn a generator into an iterator, you can use `next(<iterator>)` to get next thing in the iterator.

`next()` is essentially just looping through an iterator

PyTorch `variable()` is a superset of a tensor, but it also allows the variable to have gradients calculated automatically.

An activation is a value which is calculated in a layer. It is not a weight.

Non-linear activation functions. Are what is applied after the linear matmul.

For the hidden layers, you pretty much always want to use Relu, the next most popular one is leaky ReLU.

Choice of activation function for hidden layers does not matter much, been shown you can use arbitrary functions like a sine wave.

Pytorch has a `.max` which returns the max value and the argmax, you get the max vals by indexing `[0]` and the argmax by indexing `[1]`

Broadcasting - possibly the most important programming concept in all of Machine Learning

PyTorch also has elementwise operations just like numpy.

In the old days you would have had to have used a 'for' loop.

A for loop in python is 1000 or 10000 times slower than one in C.

You don’t just want this, you want SIMD  (Single Instruction Multiple Vector) depending on size of data type, you can get 4 or 8 times faster using this

Also you have multiple processors, multiple calls.

If you do something in C with SIMD and 4 cores you might be 32000 times faster…

Better still in PyTorch with a GPU, you can do about 10000 things at a time.

Using vectorised code is crucial for getting performant code.

Broadcasting is how different shapes are treated during arithmetic operations

Rank 0 tensor is called a scalar

Broadcasting is copying one or more axis of a tensor to allow it to be the same shape (it doesn’t really do that, it stores an internal thing which makes it act as if) -- the concept is setting the stride

```
c = np.arange(3)

c[None] # turn a row vector to a column vector
c[:, None] # reverse

np.broadcast_to(c, (3, 3))`
```

The rules of broadcasting.
*start with the trailing dimensions
*look for compatible dimensions
*2 things will be compatible when the dimensions are equal
*if the next dimension is missing we insert a 1
*with the 1 it is compatible

Eg if you perform an operation on a [3, 3] with a [1, 3]

It will check the two 3s are compatible tick, yes then the next dimension, 1, and 3 is compatible so it will work and fill in the 1 dims with repeating of the same

Example, if you want to normalise an image by RGB layer, subtract the mean from each layer, will find each mean of each layer which will have 3 elements. Then it will start with layers which is a match, then for height and width it is 1 x 1 so will just fill it in

Very few people in DS understand broadcasting well. For example, people write loops over the channels to normalise layers of an image

J language, can express very complex mathematical ideas like broadcasting.

Fundamental ways to think about math and to do programming

`np.ogrid[0:5, 0:5]` make a grid

J goes very very deep with this, but there is enough in numpy to go pretty deep.

We can use broadcasting to write matrix multiplication ourselves.

Tensor regression, tensor decomposition - being developed a lot at the moment. Taking higher ranked tensors and turn them into rows, columns and faces. To deal with high dminemsional spaces with little memory and computation. A library called "Tensorly" which does this.

Useful to understand this sort of thing to build new and interesting things.

Can use the `@` sign to do a matrix multiply. Can do it in PyTorch as well.

The `*` uses broadcasting, and does an element wise multiplication

Matrix multiplication.xyz
http://matrixmultiplication.xyz/

To get a mini batch at a time, we can wrap a data loader in `iter()`, then use `next()` for each iteration.

PyTorch `variables()` know how they were calculated, therefor you can get the gradient automatically.

SGD plot the loss against a random starting point. Take the derivative of this function, take a step in the downward direction.

The Learning Rate is the most important hyper parameter to set, too small and it will take too long to converge. Too large and it will diverge rather than converge.

Zero grad # set gradients to 0
Backward prop
Optimizer step

This video has great info about building a neural network in PyTorch.
