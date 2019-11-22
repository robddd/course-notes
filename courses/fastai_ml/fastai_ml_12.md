# Fastai - Machine Learning
## Lesson 12 - Complete Rossman, Ethical Issues


https://www.youtube.com/watch?v=5_xFdhfUnvQ

Categorical variables, thanks to embeddings have more flexibility in how a neural network can use them.

Apply functions are very slow, they just use arbitrary python code, should only be used if you cant find a vectorized solution.

Lambda creates a function just for the purpose of telling `df.apply` what to use.

These two cells are the same thing, either you can define in place or you can define the function and pass it right into apply

![Alt text](images/L12_apply_lambda.png?raw=true)

Most time series at some level will attempt to represent some event.

Example, if there is a public holiday, it is quite likely that sales will be higher before and after a holiday.

We don’t need to tell a Neural Net that a holiday is coming up, but it can help us get better results with whatever limited data and limited compute we have.

One example is creating columns, days till next public holiday, days after last public holiday.

It is much faster to iterate through a numpy array than use `df.iterrows()`

![Alt text](images/L12_zip.png?raw=true)

`zip` is an important function to practice with, you can loop through a bunch of lists at the same time.

ReLUs can handle ridiculously large outliers.

Nice trick with getting befores and afters, you can write a function and run it through a sorted dataframe sorted ascending and descing to get the time before and time after.

Rolling is how in pandas we look at windows, for example if we look at average sales in 7 day windows, this is a generic version of the moving average function.

If you are trying to predict the future, you cant include a window which includes the future.

Pandas has these functions inside `.rolling`

Pandas time series page has a tonne of time series functions, the creator of this was originally in hedge fund training. Time series computations are possibly the strongest part of Pandas.

Very worthwhile to learn all of the pandas functionality than trying to write it yourself. Pandas time series API is about as strong as any timeseries API out there.

A lot longer to write by hand, plus Pandas is highly optimized C code.

**Event counters (days before / after events) are probably always a good idea when working with time series.**

Next, split into continuous and categorical variables.

All of our categorical variables, we will turn them into pandas categorical variables then apply the same mappings to the test set.

Fastai has a function `apply_cats`

Joined samp cells, to run on a small sample and test things and make it all work first.

`proc_df` has an argument - `doscale=True`, returns a mapper so that the test and val sets can be normalised in the same way

Most comps use RMSE or log of the RMSE.

`ColumnarModelData` creates a pytorch data loader for structured columnar data.

A dimension of 600 is about what you need to capture all of the information of a word.

Human language is one of the most complex things that we model, 600 is going to be around the max dimensions of any one variable.

You could in theory know how complex a variable is and how many elements should represent it. In practise this is very hard to do. And a way to do it is use a rule of thumb then try a little higher and a little lower and see how that affects the result.

His rule of thumb, half the number of unique categories with a max of 50.

Old ML controls over fitting by restricting number of parameters, new ML controls over fitting by using regularization.

This way you can be generous on the side of the number of parameters.

Example of his NN, top 10% in public leaderboard, 5th (out of 3300) in the private leaderboard. Public leaderboard can be entirely useless. Have a very carefully chosen validation set and work to that.

2 Ways to train a model, building trees or SGD.

If you try to solve equations with colinearity analytically, you have a problem.

SGD doesn’t care. You don’t hacve to worry about things like colinearity to the same degree. A tree will complain about even less things than SGD.

You can use the same techniques with trees to interpret neural nets.

Mostly now, models will be statistically significant because they have tonnes of data.

**Statistical significance is much more important with small data sets. If you need it you can always get it by bootstrapping. Randomly sample your data and create a model many times, then look at the variance of the models**

Bag of little bootstraps -> stat significance

If cardinality is really high, one hot encoding doesn’t work too well, generally cuts it off at 6 or 7 levels.

### Ethics and Data Science

The programmer for VW who implimented cheating the emissions scheme went to jail.

Think about the ethical issues beforehand. If you get too deep into it, its very hard to get out of.

Facebook did not set out to maximize the genocide of Rohingya people. They set out to maximise clicks and likes and that happens by increasing sensationalist articles to come up.

Facebook fired human editors, algorithm immediately posts fake news.

People have different incentives, a lot of people are de-sensitized to ethical issues, because they can be at the expense of profits.

Meetup.com think a lot about ethics. Example is meetups that a lot of men go to, if there is a recommender system, you can end up with a run away feedback loop.

When thinking about what to work on, think about the problems that you are trying to solve.

The more you click on a certain topics, the more you get those topics on facebook, for example if you head down a conspiracy theory path, you can get deep into it.

Bias in Machine Learning Algorithms are usually made from bias in the input data.

Example, a ML algo to make you better looking, using white people as the training data. You end up running on a dark persons face and they make it more white.

Although these mistakes are innocent, people don’t know how ML works, so when they build a classifier and it classfies a person with dark skin as a gorilla. People think that the coders are racist.

Classic NLP bias is using historical data. "He is a Dr, she is a nurse".

Bias in ML and ethical considerations are business opportunities. 

A focus on technical excellence is sometimes at odds with ethics and 2nd and 3rd order thinking.

Once you’ve had your first job the next one is an order of magnitude easier. If you have in demand skills you can avoid things that you don’t want to do and have some control over what you do.

Most effective thing for avoiding bias in ML is having diverse an bias taken out of data and diverse teams to work on the data from different backgrounds, people from humanities, sociology or psychology could be useful too because theyre trained to think about these feedback loops

Weapons of mass destruction has some more info.

Questions to ask yourself to reduce bias:
* What bias may be in the data?
* How diverse is the team that built it?
* Can the code and data be audited?
* What are error rates for different sub groups?
* What is the accuracy of a simple rule based alternative?