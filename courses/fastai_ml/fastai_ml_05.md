# Fastai - Machine Learning
## Lesson 5 - Extrapolation and Random Forest from Scratch

Generalisation is the key unique piece of Machine Learning. If we don’t know how well we generalise, then we know nothing.

If you just split into test and train. And then try 50 things until you get a good score on the test set. What have you done? Have you found a good generalizable mode or have you just gotten lucky. To get around this use train, val and test. So that you arent just getting lucky on the test set.

You need to measure this generalisability very carefully.

With RF's you can use the OOB score. This is just the same as K-Fold validation and ensembling?

OOB score is on average less good than your validation score, because each score is a score on a model trained on a small subset of the data, val score can use all the remaining data to train a model.

Kaggle is great because it tests for generalisability with the public/private leaderboard.

Can you do a 5-fold validation from time for data, split by day where some sets are at the end middle and start? It could run into problems. Since you are predicting the future, and the world changes. The older the data, the more that things would have changed by now. Also having information newer than what you are predicting is impossible.

Use `train` (is older than) `validation` (is older than) `test` splits.

Most M.L. models have an ability to assign a weight to each row, so that the most recent rows have a higher probability of being selected.

Jeremy will tune on the train and score on validation, then once finished, he will combine train and valid and then score on the test set.

Make 5 models of varying effectiveness, don’t tune them using the test set and maybe not from the validation set either. Then look at the linearity between the validation set and the test set. Keep changing your validation set until you find a validation set that is indicative of your test set results.

The test set has to be as close to production as possible. Try different validation sets and look at how they compare. For Kaggle, the test set is the Kaggle public leaderboard.

**The test and validation set is the most important step in a machine learning project. If you screw up everything and get this correct. Then you know that you have screwed up.**

Why is cross validation good/bad?
* you can use all of the data
* you now have 5 models which you can ensemble
* time -> downside, it will take a long time to train each model
* No good way to do cross validation with temporal data. You want to have your val set as close as possible to your test set.
* The benefit is the using all of the data some more, however being that short on data is not really a problem these days usually
* AND if you do use it, the validation sets will not match as closely to the test set.

Waterfall chart on Excel 2016.

Make one for matplotlib, has it been done already?

Bias is always going to be the mean sale price for a random forrest

`np.argsort` does sort each item but it shows where it would be if it was sorted.

Tree interpreter is something that is very useful when doing real life predictions, not so useful for a Kaggle kernel.

Extrapolation:
Take train and test, remove dependent variable and then make a label for train or test set and try to predict this given the features using random forest, if you CAN predict it well, it must not be a random sample. If you can then look at the feature importance for predicting whether it is in train or validation. This also will show you what are time dependent variables. To make your model generalise better, try taking out the features one at a time for.

See how it improves and then try and remove all at once, Then at the end run a full model with heaps of trees.

You want to get your validation score higher than your OOB score. 

How exactly does reset_rf samples work?

If we want to re-impliment something that always exists, maybe in another language or something. Knowing whether you’ve got it right or wrong is very important. You need to find a good way to test. EG if you make a random forest, use an existing implementation to test results against.

All computerised random number generation is not really random, but it is psuedo random. You can set a seed and make the randomness repeatable.

The "Please move your mouse and type random stuff" approach is user made entropy, to add some randomness.

He makes code breadth first, makes the obvious stuff and then assigns a function. Makes the function later on

You can pass `np.mean(<a normal list>, axis=0])` to get mean across an ordinary list.

List comprehensions allow you to write code in the way that you think.

`idxs = np.random.permutation(<integer>)` will give you a shufflied list of integers upto that number

Distributed computing has a very high I/O overhead. Sometimes can be hundreds of times slower than a single computer.

Being as good as possible on a single machine is always going to be more interactive and more iterative.

His style of coding is to write code so you are delaying the difficult part as long as possible.

"If there was the exact API which I needed - how would I use it" 

Keep going down and down until you reach something which is pretty simple and build that or use something which already exists.

OOP - Object Oriented Programming

A constructor is something that makes an object

`def __init__(self, x, y)` is the constructor

Dunder (double underscore) dunner init
First argument everyone uses 'self', everything you assign to self, inside the class gets remembered as part of the class.

Hit tab inside jupyter notebooks to get available methods and attributes

You can instantiate things inside methods which are not yet available, and it doesn’t look them up until you use it.

A lot of stuff requires OOP. For example PyTorch models need a class.

The good news is that this is all that you need to know:

![Alt text](images/L5_class.png?raw=true)
