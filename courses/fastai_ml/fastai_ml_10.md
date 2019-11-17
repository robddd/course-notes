# Fastai - Machine Learning
## Lesson 10 - More NLP and Columnar Data


https://www.youtube.com/watch?v=37sFIak42Sc

Text Normalisation Challenge
Alvira - Class-wise Processing

Good example all feature engineering, uses regex.

Porto seguro competition 1st place with representation learning. Read these blogs!

https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629

An auto-encoder: Your dependent variable is your independent variable. It requires that your data learns important relationships within itself:

Using list comprehensions with `next()` to loop through an iterator

Stream processing - "I want the next thing" event streaming, grab the next event. "Give me the next batch of data"

A generator is python solution to stream processing.

In general you want to keep as little in memory in time, you can pipe data across

`<variable>.data` takes the tensor out of a variable in pytorch

`zero_()` gradient reset, it takes all of the 

Every function has an underscore version to do it inplace.

Full breakdown of pytorch code.

DL course has momentum from scratch.

Use the fast ai library to see the pytorch code for how to do things like data loading.

LSTM autoencoder:

Learning rate annealing - decrease learning rate and take smaller steps. Fast AI has `set_lrs()`

Learning rate schedules.

Deep Learning course has way more information about learning rates, schedulers and for Adam optimizer etc.

The PyTorch data loader. Quick version:

Very nice design in pytorch: Create a dataset that looks like a list, in has a len() and the ability to index into it. You start with a dataset, then you can take a dataset and pass it into a constructor for a data loader. That gives you something which is now iterable. You can now say iter(dl)

Pytorch shuffles everything and an epoch covers everything once.

`net.parameters` returns a list which is a tensor of parameter for each layer

`p.numel()` in pytorch tells you how many elements are in each layer

If we do this we see we have ~100K parameters. Here we may run risk of over fitting so we might want to use regularisation.

L2 regularization is very common. It means penalising parameters that are not 0. L1 is the absolute value of the weights. L2 is the weights squared.

2 ways to add regularisation. The normal way of adding it to the loss function.

Or: weight decay, you can subtract 2aw from the gradients.

Pytorch calls weight decay the L2 loss function modification.

You would expect training error to be worse when you regularize. 

Weight decay can make the function be more well behaved and smooth it out. It can help it train more quickly, but the final number should be more. So it is reasonable to expect better training scores earlier in training for the regularised model. But ultimately a better training score for the non regularised (over fitted model)

Old style of thinking is using a minimum number of parameters, new is use a lot of parameters, for random forest we use bagging, for a NN we use regularisation. No-one has done structured data interpretation libraries and blog posts. Been done a bit for random forests but is new for structured.

IMDB sentiment.

Linear results are close to SOTA for this task. For longer pieces of test throwing away order is ok when you are looking for sentiment, because the words average out quite well.

A Term document matrix is a bag of words representation.

1:08 - Section on tokenizing

Sklearn has CountVectorisation
*`veczr.fit_transform()` for train
*`veczr.transform()` for validation

Sparse matrices, store where the non-zeros are.

Add the 'ones' to the calculation because nothing is ever infinitely unlikely. Black swan.

Bayes rule

`P(c=1|d) = ( P(d|c=1) . P(c=1) ) / P(d)`

Some good numpy code for getting the naïve bayes formula.

Nicer to use the log, because we can add things together rather than multiply them

Binarized naïve bayes, `.sign()` replaces everything with -1, 0 and 1. This bag of words example doesn’t get helped by word counts, 1 is enough.

Method|Results
---|---:
Naïve bayes|80%
Binarized naïve bayes|82.5%
LR|85.5%
Binarized LR and regularization|88.4%

Mostly in the real world is better to learn your co-efficients than to try to work them out.

For LR if it is wider than it is tall `dual=True` will make it run way faster

`LogisticRegression(c=<regularisation parameter>)`

Bi-gram and Tri-gram features help us out a lot with taking our models a lot further.

LSTM seq2seq
https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
