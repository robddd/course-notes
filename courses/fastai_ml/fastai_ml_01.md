# Fastai - Machine Learning
## Lesson 1 - Introduction to Random Forests

There is no way other than creating a Kaggle competition to really know what is possible. In terms of score.

Jeremy has learned more from Kaggle competitions than from anything else.

By picking an area that we are not familiar with it’s a good test of if we can build an understanding about the data. Otherwise your intuition can get in the way of what the data is saying.

Format strings in python 3.6
Note: they don’t care about what type of data it is, it will convert it to a string!

```
name = 'Bert'
age = 68

f'Hello {name}, you are {age} years old'
```
Pandas read_csv
```
pd.read_csv(filename, low_memory=False, parse_dates=[' … '])
```

Low_memory False will read more of the file to decide what data types each column is, parse dates, to use pandas dates for those columns

Can use .transpose to a dataframe to help display, good for looking at columns with head etc

```
df.head().transpose()
```

Generally, look at the data to make sure you’ve imported it correctly and not too much more to begin with. Look at the data and see what are you trying to predict and how is it evaluated?

Video 'Deep Learning Workshop' video has an intro to numpy.

Random Forests:
* Can be used for anything
* Easy to deal with over fitting
* Easy for feature engineering
* Does not require a validation set
* Does not have many assumptions
* Does not need feature scaling or engineering
* Generally a great place to start

Curse of Dimensionality and No free lunch theorem. Largely meaningless and mis-understood.

KNN works really well with high dimensions.

In the 90s theory took over and a lot of practicality was taken out of it. Now days the word of machine learning has become very empirically driven.

No free lunch theorem. No model works well on two different random data sets. Mathematically true. Although, now days, empirically. There are techniques that do work much better than other techniques for nearly all of the data sets that you look at.

Sklearn is the biggest and widest covering library machine learning in python, pretty good but not the best at everything

Tab completion in jupyter notebooks is very useful

Regression is predicting any continuous dependent variable

Linear regression is a type of regression.

```
df.drop(<col>, axis=1) 
```
will return a dataframe with specified columns removed.

Trick of analysing a stack trace (error message) is to start at the bottom and see where the error is. 

There is no harm in adding more columns, nearly all of the time.

When using decision tree's, re ordering a category to a natural order will help a little bit. E.g. ['High', 'Medium', 'Low'] these are Ordinal Categorical Variables.

Checking missing values:
```
display_all( df_raw.isnull().sum().sort_index() / len(df_raw) )
```
Fastest way to read and to save:
```
df_raw.to_feather('tmp/raw')
```
Feather is an incredibly fast format

Using a *Tmp* folder is good for as you're going along type stuff you are working on

Need to conda or pip install this library.

fastai library's **fix_missing** handles missing numeric values by setting it to the median

They add 1 to the categorical encodings so that 0 is missing and categories start from 1

fastai library's **proc_df** handles all of missing values etc

Random forests are trivially parallelizable. It will split the data across CPU's and more or less linearly scale the job with number of CPUs

r-squared value is useful, because you can see how good your model is vs predicting the mean.

With this default model and preparation we get into the top 25% of the blue book competition.

Homework - open up the lesson 1 notebook and have a muck around, maybe try and run the latest competition with it.
