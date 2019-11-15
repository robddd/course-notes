# Fastai - Machine Learning
## Lesson 7 - Random Forest from Scratch and Gradient Descent


GBMs have a feature interaction package.

Size of test and validation set, comes down to how accurate do you need to have your validation scores? To within 1%? To 4 decimal places?

Improving from 97% to 98% Is a massive deal. The old one is 50% less accurate than the new one, there are an extra 50% more errors for the new one.

**Look at percentage change of the error.**

T distribution is the normal distribution for small data sets. Once we have more than 22 observations we can start to have decent approximation of the normal distribution.

If there is less than 22 of any class within any of the sets, then it is getting pretty unstable.

Binomial distribution
N = number of observations
P = probability

```
mean = n * p
std = n * p * (1 - p)
```

Standard error (stdev mean) = std / sqrt(n)

Set size depends on how uncommon your least common class is and how accurate you need it to be.

**In general your test and validation sets should have the distribution of what you are going to see in the real world.**

AND Your training set should have an even mix of samples from each class. And if you don’t, just replicate your least common classes until you do have an equal number.

`.iloc` makes pandas DF behave just like a numpy array

When we code, we want to do it to try and avoid having to think too much

`@property` this is a decorator and it means you can tell python more about how you can use your method.

If you are more advanced, you can write your own decorators. 

How to split on a tree deision - minimise the weighted average of the standard deviation of the 2 groups that are left after the split.

Another benefit of using column subsampling is that you can get better combos of splits where maybe the first split isnt the best possible single split, but may allow a stronger split by the time it gets to the 2nd split.

Calculating comutational complexity:
 - Is there a loop? Then we are doing this atleast n times.
 - is there a loop inside the loop? Then it will be n^2
 - is there a sort in there? Sort is n log n

N squared is not great, how do we try and make it better?

You can get standard deviation

Mean of the squares - square of the mean

Trick in the code to turn this into an order(n) algorithm.

You want to be improving your code by making it into as efficient as possible. Write it as fast as you can, run it and if it is too slow, profile it.

You can add a method to a class by assigning
```
DecisionTree.find_better_split = find_better_split
```
Very important to know how the name space works in a programming language, for example in python a method of a class can be the same name as a function in the global name space

Try and test at every step to see if it matches up with what you want

Python is dynamic, you can add methods to a class on the fly

When you do recursion with classes the attributes can have attributes.

How do we do predictions? We loop through each row and provide a prediction for each.

For numpy, we can loop through an array of any rank. It will loop through  the leading dimension. If you loop through a matrix, it will loop through each rows.

Predict row again is accidently recursive, you don’t need to know how it works underneath, you just need to ask it to do what you want.

Ternary operator
By doing if else on one line, 
```
T = self.lhs if xi[self.var_idx] <= self.split else self.rhs
```
Alpha for all graphics packages set how transparent something is to see how things are sitting on top of each other, add this to the notebook on features.

Numpy is extremely optimized code. But if you are calling it many times in a loop you have the inefficiencies similar to I/O?

To use cython in a notebook
```
%load_ext Cython
# then run
%% cython 
```

If you use C data types you can 

Reference:
How to ask for help: http://wiki.fast.ai/index.php/How_to_ask_for_Help

Jupyter extensions
 - Gist it
 - Collapseable headings
Can pip install nbextensions if you don’t have it.