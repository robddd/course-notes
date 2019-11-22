# Fastai - Machine Learning
## Lesson 11 - Embeddings

https://www.youtube.com/watch?v=XJ_waZlJU8g

Linear Regression
![Alt text](images/L11_linear_regression.png?raw=true)

When extending to hidden layers, there is no difference to how derivatives are calculated, just the chain rule gets longer, the calculus is no different.

Good information about backprop at the start of this lesson.

Some good ways to look at it using the `.grad` and `.backprop` methods

No new concepts above high school math, but takes a while to get a feel for how the different shapes move around when you are back propagating through the layers.

Almost all of machine learning usually boils down to a tree or a matrix product.

We use the loss function to take a derivate and do gradient descent.

Cross entropy - Specifically designed for classification

![Alt text](images/L11_logx.png?raw=true)

In practice with ML, you want to try both.

For Kaggle you should just use the same loss function as Kaggles evaluation metric.

Regularisation - Use a lot of parameters, but let the regularisation term decide which ones are useful.

The R in the nb + lr model, is similar to a prior.

The regularisation term now when we are starting with the NB prior is saying, lets keep this term unless its really wrong, in which case it has some room to move. 

We can use traditional, theoretical techniques by scaling the data, that will mean you wont have to regularise as much.

Regularisation weighting is the reciprocal for Logistic Regression

This technique was originally came up with in 2012 my C. Manning in a paper "Baselines and Bigrams: Simple, Good Sentiment and Topic Classification"

ULMfit can get > 94%

The best ML practitioners are tenacious, theyre stubben and bloody minded. Theyre also all very good coders, they can easily turn their ideas into code.

Fast ai NBSVM++

Adds a weight adjustment in the forward section, this will regularise to a 0.4 value, it gets that for free, and needs to work to get away from that.

Regularisation, wants the weights to be zero. 

This new addition of 0.4 will use the reg term to push the model to 0.4r. We know r is useful, so lets use it if it makes sense.

Naïve bayes model. We think that `r = p/q` is a good model. We think that `rx + b` is a decent model. We think 1 is a bit confident, but somewhere between 0-1 is good.

Sparse matricies are much more efficient. Mathematically identical is to have a list of all of the elements which are non zero and look them up rather than doing a matrix operation.

Embedding means make a matrix multiplcation faster by replacing it with a simple array lookup.

`nn.Embedding` uses a list of indices rather than a one hot encoded representation.

Example in the code of using `nn.Embedding` to represent information.

It is just the same as tokenising a sentence into a list of numeric indexes to tokens!

### location of the notebook
Dl1/lesson3 for the rossman notebook with the embeddings.

Mistake by Rossman, when designing this competition to use external data. When youre trying to predict next weeks sales, you don’t have next weeks weather or next weeks google trends. This 3rd place getter entry did almost no feature engineering.

Seeing what the winner did in a Kaggle competition, that is the bit where you learn the most, maybe you’ve thought of that and not tried it, maybe you made a mistake and you learn these lessons for how to do it. Or maybe its something you’ve never thought of, now you know that this is a really awesome thing to do.

Kaggle competitions are helpful for many reasons, one of them is for seeing what the winners did!

Jeremy always uses left joins with pandas merge. You need to do checks before and after merges to make sure that they make sense.

Inside a pandas series `.str.` Gives you access to all of the python string functions, `.cat.` Gives you all the cat functions, `.dt.` All of the date time functions.

Check all pre-processing steps and make sure that they improve your validation score.

NN model needs to know which columns you treat as categorical, and which ones as continuous.

Months open could be treated as a continuous variable, although where possible, it is best to treat things as categorical variables. The reason for that is that every level can be treated totally differently, say you have between 0 and 24 months, there is a massive difference between something been open zero months and 1 month. But hard to fit a function to this part of it.

As long as the cardinality isn't too high, treat it as categorical.

The 3rd place rossman getters used the genuine continuous measurements as continuous (eg distance to nearest store) and the rest as categorical.
