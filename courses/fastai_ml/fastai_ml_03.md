# Fastai - Machine Learning
## Lesson 3 - Performance, Validation and Model Interpretation

Remember raw defaults often give a very good result. Quite often you can go along way without doing any tweaks at all

How to interpret your model:

Unstructured data - where all data points represent a single thing, a word, a character a pixel. You almost always want to try deep learning.

With structured data you always want to try a RF first. The question is in which situations do you also want to try other things?

Feather format is almost exactly the same format that lives in ram, so it is ridiculously fast.

proc_df() now returns df, y, nas 

Nas now returns a dictionary of information about the NA values. It allows you to process the test set the way that you process the train set.

Really large data sets.

Kaggle groceries competition. 

Corporacion Favorita

**Your ability to explain the problem that you are working on is really really important, when you are not feeling confident at that, practise. The key thing to say is what are the independent variables and what is the dependent vartiable**

To limit the amount of space that you use, you pass in a dictionary of columns and dtypes and pass in `dtype=<dictionary>`

This file can be read in, in 1 minute and 48 seconds and there are 128 million rows.

Python itself isnt fast, but everything that we want to run in python is run in C, cython, assembly languages or fortran. There are layers and layers of optimisation that made behind the scenes

csv in python would take 1000's of times longer.

Key performance thing when specifying datatype is to use the smallest number of bits to represent the comment.

|Integer | n |
|--------|--------|
|2 ^ 8 | 256 |
|2 ^ 16 | 65,536 |
|2 ^ 32 | 4E9 (4 billion) |
|2 ^ 64 | 1E19 (10,000,000,000,000,000,000) |

With large datasets, the key performance consideration is reading and writing to to ram. Often the key thing is to have the right data type, especially if you can use SIMD (Single Instruction Multiple Data) Vectorized Code. Can fit more numbers into a single vector and run faster.

Search forum for `shuf` a unixx command to get a random sample of your data which you can read in.

In general, do as much work as possible on a sample until you understand the data and really need the larger chunk of the data.

A boolean that also does have NAs needs to be an object to start,  then if you replace the NAs with T or F you can convert it to a boolean. 

Now its optimised, you can save to feather in 5 seconds and run summary stats of all columns in 20 seconds.

For Dates: You must understand that your test set are dates after your train set. If you want to use a subset. Use the most recent to the test set training data. You will miss something, like seasonal patterns, but we are just looking for an initial and easy model. Later on, we may want to weight more recent dates more highly as they may be more relevant.

`np.log1p()` will get the log+1 of numbers, because in RMSLE log of 0 doesn’t make sense.

sklearn: `n_jobs` arg in random forest is the number of cores you are using, `-1` is use all, sometimes using more is worse because of overheads.

`%prun` # a profiler for ipython

A profiler is one of the most important things for a software engineer, for a data scientist, profilers are massivlely under appreciated.

You cant use OOB score if you do use set_rf_samples() because it will use all of the rows.

A random forest ability to understand exactly what is going on is limited. It only sees item numbers and doesn’t know that this item is gasoline and that is bread, and that these stores are close to each other and those other stores are far away from each other

If you take, store, item and on promotion and then you mean across date. You get to 30th on the leaderboard.

If you know this, your job is, how can you start with that model and make it better.

You can try to capture trends with things like, average sale for this month etc. The details are very important and somewhat difficult.

**Coding for machine learning is incredibly difficult. Usually if you get a detail wrong, you usually wont get an exception, you are just doing slightly less good than you could be otherwise.**

Its fine to make these mistakes, as long as you have a way to find out. You always just have to think about it as you go.

One way to check is if you have a pretty good model and something that makes a lot of sense, you can put the two sets of predictions on a scatter plot and they should just about form a line, and if they don't. You’ve got something wrong.

Outside data is frequently used to assist.

In general one way to tackle this type of problem is to make shit loads of new columns with averages of w/e on holidays or whatever it is.

**Look at the Kaggle blog of what the winners did.**

For Rossman: The 1st place getting was unusually a domain expert who made tonnes of new columns that they knew from experience would be useful. The 3rd place getter used deep learning but made one big oversight which probably cost them victory.

Plot Kaggle score vs Validation score.

**If you don’t have a good validation set, its hard, if not impossible to build a good model.**

Usually you should not use your test set until right at the end when you have finished building your model.

But there is one thing that you can use it for, which is validate your validation set. Build four simple models that are different from each other and score on both test and validation to check how the scores line up.

Ideally a validation set will lie on the y=x line, but as long as they have a good relationship it doesn’t really matter because if the relationship is strong, then you can trust the result 

![Alt text](images/L3_valset_val.png?raw=true)

The one on the right is not good, because the two blue points in bottom LH corner, the relationship is backwards between 2 points.

Pass in `na_dict=nas` if you want to pass in what is there before, used in case of when you impute values for training, to use the same values, means or medians etc on the new data.

We are less confident of a prediction if our model hasn’t seen many rows similar to that one. 

If a row is uncommon, there will be high variance in the predictions from each of the trees in the forest. So if we look at the standard deviation of the predictions we can get some kind of gauge of the confidence of our prediction.

This code isnt in sklearn or anything else, so we can make it

**We can use 20-50K size samples. We don’t need massively accurate samples. We just need indications. The way we can check is if we run on the same data several times if the results are close.**

Python code running to itself is serial. Compare CPU time to wall time. There is a fast AI code to run things in parallel. parallel_trees()

Writing code that runs in parallel is super useful for data science.

Pandas in built plotting is very useful and worth spending some time on it

The process of learning here, is we don’t know what something does, we run it and look at it, and find out what it is later on.

We can use this idea of standard deviations to look at subsets of our data where the model is less confident making predictions.

In smaller groups you will generally do a less good job of predicting

In production you can look at specific rows and get an idea of confidence.

How could you do this for Deep Learning? Make X models with slight differences even if its training data, random initiations of layers, whatever and find out where there is the greatest variance or standard deviations.

Feature Importance - Almost always will look like this:

![Alt text](images/L3_feat_imp_plot.png?raw=true)

A few columns which are super important and a long tail of ones which add little to nothing. The features that come in high on feature importance, you need to spend time looking at them and digging into them.

The most important thing to do with feature importance is to spend a lot of time digging into the features which have the strongest prediction.

Example of data leakage, predicting grant applications, Jeremy found that certain rows, what was very important was if they were missing or not. Since if they weren't successful people were less likely to bother entering them into the database. It is data leakage because the university wouldn’t have had this information at the time that they were making the decision.

Co-linearity - a certain feature can be powerful for indirect reasons, since they are indicative of something else which is more predictive.

You can remove redundant columns, ones which are very low on feature importance, it usually wont make your predictions any worse, but may make them a little better, AND it will make it a bit faster and get you to focus on what is important.

When you remove redundant columns, you are also removing sources of co-linearity. By stripping out these columns which may have co-linearity you can get a more accurate feature importance plot. If 2 columns are co-linear, their feature importance will be split between the two.

We can do feature importance for any type variable. We train a model using all features, then, one by one we remove features (by randomizing the values in that column) and see how much the scores are affected. We use the decay in score as a rank of which columns are important. You can remove all combos of two at a time also, to see which pairs of variables (to look at interactions) are important. In practice this is computationally intensive, there are better ways to do this (which he said he may go into)

Feature engineering deas for strings: Are there a hyphen or dot you can split it to make more columns? Is there some type of ordering you can use to make it perform better?
