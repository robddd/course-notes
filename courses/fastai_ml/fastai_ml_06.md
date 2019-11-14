# Fastai - Machine Learning
## Lesson 6 - Data Products and Live Coding


![Alt text](images/L6_1_marketing.png?raw=true)

https://www.oreilly.com/ideas/drivetrain-approach-data-products

Use models to say who is going to do X and why people are going to do X

You want to understand the technical model building deeply, and to understand solving business problems deeply.

![Alt text](images/L6_2_risk.png?raw=true)
![Alt text](images/L6_3_HR.png?raw=true)
![Alt text](images/L6_4_Horizontal.png?raw=true)
![Alt text](images/L6_5_healthcare.png?raw=true)
![Alt text](images/L6_6_retail.png?raw=true)

The vast majority of machine learning applications are things like these. What google and facebook do gets a disproportionate amount of press for their applications.

![Alt text](images/L6_7_travel.png?raw=true)
![Alt text](images/L6_8_industry.png?raw=true)

https://www.oreilly.com/ideas/drivetrain-approach-data-products

Random Forest Interpretation Methods.

Confidence based on tree variance. Minimised variance for predictions amongst trees will mean greater confidence, compared to examples with more variance amongst trees.

Can also do for groups, if we look at the variance for examples between different categories, how much does it vary?

Also this method is really good because we can look at best and worst case scenarios. For example X, what is the mean minus 2 S.D's

For feature importance - Look at the relative importance of different features, looks at the feature or features which are by far more important than the others and start with them.

Build a random forest and then do feature importance, evaluate if your variables should be there, is there data leakage or something else artificially boosting your score.

One outcome of feature importance is that you can see that you need to improve methods of collecting a certain variable, if its seen to be important and worth collecting precisely and rigorously.

Partial Dependence - Most often you see a univariate chart which can miss a lot of interactions. What we actually want to know is all other things being equal, what is the relationship between year and sale price.

The way to do partial dependence is similar to feature importance, although, instead of randomly shuffling a column, we replace all the values with a constant. And repeat for all range of values and do a decent (but not perfect) job of isolating that one variable.

We can plot the median, but we can also do a cluster analysis to see if there are some different paths which can happen.

Start with the data, don’t talk to anyone, do a model, look at feature importance, throw away nothing. Then talk to people after where you need to poke around.

Think about what you expect to see before you plot, then plot and see if it was you expected.

Partial dependence is useful for features that you care about operationally, they are important and they are a level that you can pull, or something like ZIP code, where you can gain insight and pull other leavers to target the right place.

Great technical communication - Use an as specific example as possible "A hospital re-admission" take an analogy to something that you already understand.

Tree Interpreter is a great example of a package which takes in an object (model) and a row, and uses the object to get something else out.

Waterfall charts are very important in business. The world is full of stuff which should exist but doesn’t. Python for example has a lot for research and science, but not much for busienss.

Github/hub makes pull requests extremely easy to do.

If you have a great github with thoughtful pull requests which are being accepted to peoples libraries, this is something extremely valuable. You can specifically mention it I am the person who created x, y, z. I am a contributed to x, y, z.

Sometimes missing values are by far the most important things in the data, maybe they are missing for a reason which is predictive of the dependent variable.

'*' can mean interacted with 'Year made * colour', 'time of day * trip cluster'

As at this time, there are no interaction feature importance libraries.

Create a synthetic dataset to show some of these ideas. Make something which has the underlyging function and some randomness aswell, categories, continuous.

np.linspace() can create a synthetic dataset. An array spaced out linearly

@ 1:26 creates a synthetic dataset