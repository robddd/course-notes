# Fastai - Machine Learning
## Lesson 2 - Random Forest Deep Dive

`Fastai/courses/ml1/` has a sym link to `fastai/`

Creating sym links:
```
ln -s ../../fastai <name of desination ('./' will just use destination folder name)>
```
On mac its known as an alias, a shortcut on windows

Some info at top of notebook about RMSE and RMSLE. You can change the target to the log of the target to optimise for this metric

Have a look at the category datatype in pandas:
```
df.<cat_col>.cat.<attributes and methods>
```

R-squared or coefficient of determination [Kahn Academy Video](https://www.khanacademy.org/math/ap-statistics/bivariate-data-ap/assessing-fit-least-squares-regression/v/r-squared-or-coefficient-of-determination)

There is a line that minimizes the squared distances to the points:

![Alt text](images/L2_rsq.png?raw=true)

R-squared - how much variation in y is described by the variation in x

1 - (S.E. of the line / S.E. of y)

You can have a r squared value of anything less that 1. If your model is terrible. 

If you predict the mean you get 0? If you predict less than zero you predict worse than predicting the mean.

R squared is a good measure because you can see how your prediction is vs predicting the mean.

**Creating a validation set is the most important thing you need to do in the modelling part of machine learning. You need to come up with a dataset which is representative of how well you model is going to do in the real world.**

The bluebook dataset has a good validation split, with the test set being made up of all newer values than the training set. We need to do the same thing with our validation set.

For this example they use the same size validation set as Kaggles test set.

You really need to get a hold out set and completely remove this data and hide it away, give it to someone, and don’t let them let you look at it until you are finished.

Bluebook has a good example of why you want your validation set to be a separate time period (usually after) than your training set. If you make validation set on any random date, your model has already seen examples on the day that your validation data is on. This is not realistic.

RMSE

A lot of value of being able to quickly look at something and see what is going on.

Software engineers usually do it different, longer variable names, documentation etc.

%time will tell you how long it took to finish, how many cores etc, CPU times

If something takes longer than 10s its too long to interact with it and experiment.

One way to speed things up is to pass in a subset of rows.

Single tree - in sklearn `RandomForestRegressor(n_estimators=< >)`

A single tree consists of a sequence of binary splits

MSE - the mean of the squared errors for each prediction

To find where to make your split at each point of a tree. Try each variable and take a weighted average of the RMSE
![Alt text](images/L2_bin_tree.png?raw=true)
Minimise the (MSE * sample size of split 1) + (the MSE * sample size of split 2)

We try all possible split points.

This tree has found that coupler system is 'the' best individual variable for predicting the target.

Binary tree's are optimal, because you can get the same result by splitting twice.

`max_depth` (how many splits)
Otherwise, you stop when your leaf nodes only have one thing in them

This example you get a training score of 1.0 and a test score of 0.73

Random Forest is a way of bagging trees

What is bagging? If we make 5 different somewhat predictive and not correlated with each other. Each model would need to find different insights. Then you can average the results. This is a technique of ensembling.

Make a big deep, massively over fitting tree. Do it 100 times. All of the trees on their own are going to be better than nothing, they all over fit terribly, but all in different ways on different things. They all have errors but the errors are of different things. The average of many different errors is zero.

We get the different random tree each time using a different random subset of rows.

Sklearn random forest uses bootstrapping, picks out a random n rows at random with replacement. 

**The whole point of machine learning is working out which variables matter the most. And how do they relate to the target variable and to each other. -> Which variables are important, and how do they interact together to make a good prediction of your target variable.**

A machine learning model which is good, finds relationships in the data and makes good predictions as well as generalizes as well as possible.

More recent advances in ML have found that what is more important than getting more accurate trees, is getting trees which are as uncorrelated as possible.

There is a model in sklearn which doesn’t try all of the combinations to find a split, it just tries some random trees and splits and chooses the best, wont get as good individual trees but will be much faster.

For the tree model you get the MSE by looking at the average of each nodes target vs the actual value.

Your CPU performance is measure in gigahertz - which is billions of clock cycles per second. And it has multiple cores, and each core as something called SIMD (single instruction multiple data) and GPU is measured in trillions of operations per second.

Experimenting, dig inside objects and see how the work inside, does it do what you expected it to do?

When Jeremy is trying things he will use 20 to 30 trees, then at the end when hes happy with everything he will run it on 1000 trees and run it over night.

Out of Bag Score - unique to random forests. In each tree there are rows which don’t get used, we can get scores for each row and average them. This gets a score which is very close to the measure of what you would get on a validation set.

`set_rf_sample()` is this a fast ai thing? Yes

By setting the rf samples, you can use the whole data set, get better results. But it will be much quicker. Sklearn doesn’t support this out of the box but fastai has a hack which can do this.

By using small subsets and a simpler model, such as RF with less estimators you can get the same insight, but you can do much much more experimentation.

`min_samples_leaf` - stop training the tree further when your leaf node has 3 or less samples in it) in practice you have half the number of decisions needed to make vs min_samples_leaf = 1. Usually good values for this [1, 3, 5, 10, 25]. Sometimes if you have a really really big dataset and your not using a subsample, you may use hundreds or thousands. Upping this can get a better score and will go faster

`max_features` - for each binary split you only look at a subset of columns. You can imagine if there is a really really good column each tree will look at it. 0.5 will randomly choose half of them. The default is to use all of them. Usually good values are [1, 0.5, log2, sqrt]

At this point we are 14th. With a totally brainless random forest with some totally brainless random tuning.

Random Forest is often a great first step and often the only step needed for a competition. Linear models can really throw you off because there are a lot of assumptions that need to be met.

RFs are very easy to use and very resilient compared to linear regression.

Next Lesson: How to analyze the model to learn about your dataset

Key insight from this lesson.

Machine Learning is all about finding the variables that have the strongest relationship to the target variable. And finding the relationships between these variables. By simplifying the problem (using a subset of the data and less trees (in the case of an RF)) you can still learn this information and not have to wait long to get the results.

Jacob Neilson - 10 seconds when your brain switches context ~ Same as recomendation above to make a simple model that trains in 10 seconds for testing.
