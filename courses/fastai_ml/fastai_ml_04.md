# Fastai - Machine Learning
## Lesson 4 - Feature Importance, Tree Interpreter

Hyper parameters.

set_rf_samples -> how many rows are used in each tree. The depth of the tree doesn’t get too affected by this parameter, formula for max depth is

Log 2 (n_rows)

Example with 20K rows max depth is 14, with 1M rows it is 20

Setting RF samples, means you will overfit less and you will have weaker individual trees.

Purpose of random forest, each estimator is as accurate as possible, and each estimator (tree) is as least correlated as possible to the other trees. By using set_rf_samples. We make individual trees less accurate, but we also make the trees less correlated.

Reset_rf_samples() # will cause it to bootstrap, to sample a new data set as big as the original, with replacement.

Min_samples_leaf - each time we double the min samples leaf, we remove one layer of depth of the tree.

If we have a 20k sample and a min samples leaf of 2, we will have a max depth of log b2 (20K) - 1

The number of max leaves is 20k / min samples, so in this case of man sample leaf of 2, max leaves is 10K

Max_features - the column sub sample for each decision point. This will again mean the individual trees will be less accurate, but the trees will be more varied, think of if there is one super powerful variable which every tree uses to split at the start

N_jobs - how many cores you are using, default is to only use one. Seems to be diminishing returns after using 8ish cores.

Oob score - out of bag score. May take some more time and get a val score without a val set.

Fast ai library is not available in the kaggle kernels, but if you look inside of the fastai library, you can see that the functions are just a few lines of code, so they can be easily copied

Run some more hyper parameter cross validations on the santander dataset, run correlations across different variables

Interactions between variables is massively important. For example, we have year made and sale date, if we lose one of these columns, we lose the ability to calculate the age of the machinery at sale time.

If we know the exact variables you need,  interactions between variables and exact transformations needed for the variables, ahead of time. We can easily create a LR which is as good as any random forest. But… We never know that. 

The truth is there are many many things interacting in very subtle ways.

Having 2 or more variables that are correlated strongly to each other, will mean that this feature importance method will underestimate how important each variable is, for example if you have year twice, and you take one away, you will not lose much, if you take away both then you will lose a lot.

With interaction if you take one away you will lose a lot, maybe looking at combos will do a better job of finding correlated variables than of finding variable interactions… Also once you get a handle on two variables. Could look at 3 or more?

If you can see removing a model keeps your r-squared and your error score at the same or even improves it, then you can remove the variable, with the benefit of making your model simpler. Jeremy's example was by removing some weak variables, looking at how strong the variables were, after removing a lot of weak ones, the #1 variable became much much stronger than before.

Getting a good grasp of feature importance is great because it tells you where you can focus your feature engineering time. You can take what you have learned and ask domain knowledge experts, hey I found that this variable was very influential on the price, why do you think that might be?

If your validation set has gotten much worse, then you must be over fitting. Unless you use some really parameters, such as only one estimator. With 10+ tree's it is hard to overfit too much.

When you normalise the data first, Linear Regression co-efficient show the feature importance of each variable.

RF - few if any statistical assumptions. 

Linear regression and logistic regression. The co-efficients are going to be biased by the pre-processing. Either that or if it was a good job, you see co-efficients of a some variable which has been reduced by PCA or some other dimensionality reduction technique.

For a categorical variable, maybe only one of the categories is useful,

For example [Very Low, Low, Medium, Unknown, High, Very High] - maybe its just 'Unknown' that is useful. So what we can do is one hot encode.

For redundant columns, linear models hate colinearity, so you need to exclude one of the o.h.v. columns. But for random forests, it doesn’t matter and may help, because the model can get to that info easier.

This example uses one hot encoding for columns with more than 6 categories. Its another thing to play around with, you need to experiment and see what happens in the results.

Another benefit of one hot encoding is that it, will give feature importance of the individual categories.

Dendrogram is to remove redundant columns. 

Hierarchical (iglomorative) clustering, we look at every pair of objects and say which two objects are the closest, we start at the closes two points and replace them with the average.

Correlation co-efficient is almost the same as r-squared but it is of two variables. The problem with a correlation co-efficient is that it only works with linearity. Instead we can use rank correlation. This will make a non linear relationship linear

![Alt text](images/L4_rank_corr.png?raw=true)

Rank correlations are more analogous to a random forest, where traditional correlation is more analogous to linear regression.

Spearmans-R is a rank correlation.

With the dendrogram, he just looks at the stuff which is super close to the right hand side and doesn’t worry about anything else. Particularly in this case there is a bunch of stuff right on the RHS but then a big gap before the next thing

Spearman r correlation matrix is the best way to see all similarieties between all individual variables.

Didn’t work well on the santander data set (dendrogram), maybe spearman will be more interesting.

Once you have some candidates for correlated variables, you can remove them all one at a time to see if they have much effect on the score. This time you want to train a new model each time.

If the scores are the same or even very slightly worse we can probably drop them, the next step is dropping one of the variable from each of the correlation all at the same time and compare scores. Again if it drops a very small amount that is fine and simpler is better.

Partial Dependence

Look at the features which are important and find our how do they relate to the dependent variable.

Ggplot (the grammar of graphics) - R version has much better documentation than python. In ggplot a + adds a chart element, a smoother makes it easier to see than a scatter plot. Se=True standard error, shows confidence interval

We see a of missing values in the year made.

Problem when looking at a univariate relationship is there is a whole lot of information lost.

![Alt text](images/L4_univariate_relation.png?raw=true)

For that dip, is it that bulldozers of those ages are worth less or was there a ressision or is something else going on.

What we want to do is look at the relationship between the two variables, all other things being equal.

To do this we use a partial dependence plot

```
import pdp
```

We take out sample and then make a prediction for each row, but we make each of the year row replaced with a constant. We take the average of all of the sale prices

![Alt text](images/L4_partial_dep.png?raw=true)

Each blue line is one of the rows at each year. And the yellow / dark line is the average.

We see now it’s a pretty straight line, and generally more recent vehicles are more expensive. Due to inflation and the vehicles being newer.

Steps:
 - First look at feature importance to see what is important
 - Then look at 

Can do a cluster analysis on the different rows with a column being put in as a different constant. There might be different types of clusters of certain behaviors of different types of data.

Partial dependence is much more for real life than for Kaggle Competitions.

Useful for getting an actual idea on how one variable affects the outcome. Isolating a certain variable and figuring out how you would change it to get a better result.

Another way to interpret is, you can get a better indication of real value of something, the univariate plot will be more affected by other co-occurring features.

Partial dependence is much better but its not perfect, a lot of the other features may be different, for example a 2010 bulldozer vs a 1960 bulldozer.

When you see something interesting in the data, jump into google or ask someone about what categories are.

Sometimes adding an interaction will make score a bit 

Tree interpreter

Almost pointless for Kaggle competitions but super useful for real life.

A lot of vis in Jupiter notebooks don’t know how to save and you will have to rerun it.

Tree interpreters look at the average sale price at each node. And look at all of the decisions it made and its effect on the average price at each decision. You can see exactly why was it that you predicted 10.2.

You can then look at a prediction and see exactly how much 
