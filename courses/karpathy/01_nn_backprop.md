# Andrey Karpathy
## The spelled-out intro to neural networks and backpropagation: building micrograd
[YouTube Link to Video](https://www.youtube.com/watch?v=VMj-3S1tku0)

This video was excellent, I probably spent 15-20 hours going through it all and coding up all the examples alongside watching it. As well as making Neural Nets simple to understand there are also some good explanations and examples of Object Oriented Programming (OOP) in Python.

### Examples of Manual Backprop
### Simple Example of Taking Derivatives

Imports
```
import math
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```
A random function
```
def f(x):
    return 3*x**2 - 4*x + 5

f(3.0)
```
Out
```
20.0
```
Plot the function
```
xs = np.arange(-5, 5, 0.25)
ys = f(xs)
plt.plot(xs, ys)
```
Out

![Alt text](images/00_function_plot.png?raw=true)

A derivate is if you slightly bump up a value, what is the slope
at that point?

General formula - (LIM x->0) - (f(x + h) - f(x)) / h

Derive f(x):

f(x) = 3x^2 - 4x + 5

f'(x) = 6x - 4

Try by slightly bumping 'x' to find the derivative. Do this by looking at the slope between 'x' and 'x+h'
```
h = 0.000001 # amount to bump 'x'
x = -4
(f(x + h) - f(x)) / h
```
Out
```
-27.999997001870724
```
Another example
```
x = 3
(f(x + h) - f(x)) / h
```
Out
```
14.000003002223593
```
Another example - Where the slope is zero
```
x = 2/3
(f(x + h) - f(x)) / h
```
Out
```
2.999378523327323e-06
```
We can see that these examples match up with our f'(x) derivative and also with the slopes shown on the plot. Note the 'x' and 'y' axis are on different scales in the plot. This is why the slopes look flatter than the values above.

### Derivatives of a More Complex Example

```
a = 2.0
b = -3.0
c = 10.0
d = a*b + c
print(d)
```
Out
```
4.0
```
Use same method as above to find derivatives
```
h = 0.0001

a = 2.0
b = -3.0
c = 10

d1 = a*b + c
a += h
d2 = a*b + c
print('d1', d1)
print('d2', d2)
print('slope', (d2 - d1)/h)
```
Out
```
d1 4.0
d2 3.999699999999999
slope -3.000000000010772
```
And can use calculus, if you differentiate a*b + c
* wrt 'a' you get just 'b' which is negative 3
* wrt 'b' you get just 'a' which is 2
* wrt 'c' you get 1 

Next: We define a value class which can store values, performa simple operations and keep track of the graph of operations used in an expression. In the video it is built up slowly.
```
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        return out
```
Now: Lets create an expression
```
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d*f; L.label = 'L'
L
```
Out
```
Value(data=-8.0)
```
Our value class stores the previous values as well as the operation which was used
```
L._prev, L._op
```
Out
```
({Value(data=-2.0), Value(data=4.0)}, '*')
```
Now we have some code to visualise a graph
```
from graphviz import Digraph

def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name=uid, label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name=uid+n._op, label=n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)
    
    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot
```

Using this code we can visualise the expression
```
draw_dot(L)
```
Out
![Alt text](images/00_graph_01.svg?raw=true)

### Manual Backprop Example 1

We can calculate all of the gradients of our expression with respect to the inputs

Here we show the derivative of L with respect to d

L = d * f

dL / dd = f

And with working shown:

(f(d+h) - f(x)) / h

((d+h)*f - d*f) / h

(df + hf - df) / h

hf / h

f

Finding the other derivates...

dd / dc = 1.0

dd / de = 1.0

d = c + e

We know:

dL / dd

dd / dc

We want:

dL / dc = (dL / dd) * (dd / dc)

dL / dc = -2.0 * 1.0

node derivates are just 1

The next level back...

dL / de = -2.0

de / da = b = -3.0

de / db = a = 2.0

dL / da = 6.0

dL / db = -4.0

Next, update the `.grad` attribute of the the Value objects
```
L.grad = 1.0
d.grad = -2.0
f.grad = 4.0
c.grad = -2.0
e.grad = -2.0
a.grad = 6.0
b.grad = -4.0
```
Now we can make the graph again with our gradients filled in
```
draw_dot(L)
```
Out
![Alt text](images/00_graph_02.svg?raw=true)

Write a function to do a numerical gradient check
```
def lol():
    """numerical gradient check"""
    h = 0.0001

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b; e.label = 'e'
    d = e + c; e.label = 'd'
    f = Value(-2.0, label='f')
    L = d*f; L.label = 'L'
    L1 = L.data

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    
    # Update this line to get gradient for any value
    # also must move the line to directly before the value is used in an operation
    a.data += h
    
    c = Value(10.0, label='c')
    e = a*b; e.label = 'e'
    d = e + c; e.label = 'd'
    f = Value(-2.0 , label='f')
    L = d*f; L.label = 'L'
    L2 = L.data

    print((L2 - L1)/h)

lol()
```
Out
```
6.000000000021544
```
6.0 matches up with our hand derived example 

### Activation Functions
Another part of neural networks are activation functions. Here we look at using tanh
```
plt.plot(np.arange(-5, 5, 0.2), np.tanh(np.arange(-5, 5, 0.2)))
plt.grid()
```
Out

![Alt text](images/00_tanh.png?raw=true)

### Manual Backprop Example 2
```
# inputs x1, x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1, w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias
b = Value(6.8813735870195432, label='b')
# x1w1 + x2w2 + b
x1w1 = x1*w1; x1w1.label = 'x1w1'
x2w2 = x2*w2; x2w2.label = 'x2w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1 + x2w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh(); o.label = 'o'
```
Gradients

o = tanh(n)

derivative of tanh = 1 - tanh(x) ** 2

do/dn = 1 - o**2 = 0.5
```
o.grad = 1.0
n.grad = 0.5
x1w1x2w2.grad = 0.5
b.grad = 0.5
x1w1.grad = 0.5
x2w2.grad = 0.5
x1.grad = w1.data * x1w1.grad
w1.grad = x1.data * x1w1.grad
x2.grad = w2.data * x2w2.grad
w2.grad = x2.data * x2w2.grad
```
Visualise
```
draw_dot(o)
```
Out

![Alt text](images/00_graph_03.svg?raw=true)
