# Andrey Karpathy
## The spelled-out intro to neural networks and backpropagation: building micrograd
## Part 2 - Auto Backprop
[YouTube Link to Video - Time stamped to where these notes start](https://youtu.be/VMj-3S1tku0?t=4141)

Here in part 2 we implement automatic back propogration. These notes start from 1:09:00 in the video.

### The Code
Imports
```
import math
import random

import torch
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```
The Value class completely built out for supporting more operations and backward passes
```
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supports int/float for now"
        out = Value(self.data**other, (self, ), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other-1)) * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):
        """
        Reverse add, if 'a + b' errors, it will try 'b + a'
        where here 'a' is a int or float, and 'b' is a Value object
        """
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
```
### Python Special Methods
`__init__()` - Initialisation - Runs at instantiation

`__add__()` - when the '+' operator is used effectively does `class.__add___(<what ever is to the right of the '+'>)`

`__radd__()` - if `a + b` errors and 'b' is of this class, will try `b.__radd__(a)` allows you to run `int(2.0) + Value(3.0)`

Related to above, to ensure `Value(3.0) + int(2.0)` runs, we add this line to `__add__()`
```
other = other if isinstance(other, Value) else Value(other)
```

Code for making graphs
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
### Autobackprop Example 1
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

o.backward()

draw_dot(o)
```
Out
![Alt text](images/00_pt2_autograd_ex1_graph.svg?raw=true)

### Autobackprop Example 2 - with a more atomic version of tanh
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
# HERE doing tanh in a more atomic way
e = (2*n).exp()
o = (e - 1) / (e + 1)
# o = n.tanh(); o.label = 'o'

o.backward()
draw_dot(o)
```
Out
![Alt text](images/00_pt2_autograd_ex2_graph.svg?raw=true)

### Example recreate micrograd in pytorch
Using single element tensors
```
x1 = torch.Tensor([2.0]).double(); x1.requires_grad = True
x2 = torch.Tensor([0.0]).double(); x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double(); w1.requires_grad = True
w2 = torch.Tensor([1.0]).double(); w2.requires_grad = True
b = torch.Tensor([6.8813735870195432]).double(); b.requires_grad = True
n = x1*w1 + x2*w2 + b
o = torch.tanh(n)

print(o.data.item())
o.backward()

print('---')
print('x1', x1.grad.item())
print('x2', x2.grad.item())
print('w1', w1.grad.item())
print('w2', w2.grad.item())
```
Out
```
0.7071066904050358
---
x1 -1.5000003851533106
x2 0.5000001283844369
w1 1.0000002567688737
w2 0.0
```

### Building a Multi Layer Perceptron
A `Neuron()` contains one weight for each input to the neuron + a bias value

A `Layer()` contains all of the neurons in the layer

An `MLP()` contains all of the layers in the MLP
```
class Neuron:

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
    
    def __call__(self, x):
        # w * x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        """
        generate list of all parameters in the neuron
        """
        return self.w + [self.b]

class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        """
        generate flat list of all parameters in the layer
        """
        return [p for neurons in self.neurons for p in neurons.parameters()]

class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        """
        generate flat list of all parameters in the MLP
        """
        return [p for layer in self.layers for p in layer.parameters()]
```
Implement an MLP with:
 - 3 inputs
 - 2 hidden layers of 4 neurons
 - one output
```
x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
n(x)

draw_dot(n(x))
```
Now, give the NN 4 labelled examples
```
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]
```
ypreds can be used against ys to check how well
the NN is performing, by seeing how far away the preds
are from the labels. 

We can define a function for the loss based on these 4 results
```
ypred = [n(x) for x in xs]
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
loss
```
Out
```
Value(data=6.0193840394612455)
```
Note: Above loss will changed because there is a random initialisation of weights that does not have a fixed seed. Same goes for looking at weights and grads below.

Currently the NN is not performing well

Lets look at layer 0, neuron 0's 0th weight
```
n.layers[0].neurons[0].w[0]
```
Out
```
Value(data=-0.48250447769839555)
```
Lets look at layer 0, neuron 0's 0th gradient
```
n.layers[0].neurons[0].w[0].grad
```
Out
```
0.0
```
Above - We forgot to run loss.backward() so we have the initial grad of 0.0

Lets run loss.backward()
```
loss.backward()

n.layers[0].neurons[0].w[0].grad
```
Out
```
-0.10505485267030025
```
Let's check the architecture of the MLP
```
print(f'Number of Layers {len(n.layers)}')
for i, l in enumerate(n.layers):
    print(f'Layer {i} has {len(l.neurons)} neurons')
```
Out
```
Number of Layers 3
Layer 0 has 4 neurons
Layer 1 has 4 neurons
Layer 2 has 1 neurons
```
We can look at the graph here but it is huge
```
draw_dot(loss)
```
Out
![Alt text](images/00_pt2_MLP_graph.svg?raw=true)

Now look at all of the parameters of the model
```
n.parameters()
```
Out
```
[Value(data=-0.48250447769839555),
 Value(data=-0.5121070718779797),
 Value(data=0.08712873458475001),
 Value(data=-0.8013012903701442),
 Value(data=-0.04341913348395665),
 Value(data=-0.4108643571290975),
 Value(data=0.15260352146458356),
 Value(data=0.45172184889002476),
 Value(data=0.47868912872033165),
 Value(data=0.9685163902149281),
 Value(data=0.5315775317746942),
 Value(data=-0.12331345893846635),
 Value(data=0.8279305339005452),
 Value(data=0.34198861513035195),
 Value(data=0.14947410351958634),
 Value(data=0.6815397437779895),
 Value(data=0.5981032731539744),
 Value(data=-0.11196648294897682),
 Value(data=0.968336935393721),
 Value(data=0.5761068949377972),
 Value(data=0.23718279459262326),
 Value(data=0.04596461776359151),
 Value(data=-0.03905821442907942),
 Value(data=-0.7433519596574283),
 Value(data=-0.6974511822612328),
 Value(data=-0.926513368967216),
 Value(data=0.007306818485270705),
 Value(data=-0.46416579252904966),
 Value(data=0.8500740049917666),
 Value(data=0.45252914214668793),
 Value(data=0.9984726521036289),
 Value(data=-0.862103898715961),
 Value(data=0.4363602510436324),
 Value(data=-0.6578410935706094),
 Value(data=-0.9862321617766492),
 Value(data=0.07064282203681582),
 Value(data=-0.8678107262716812),
 Value(data=-0.8350773620391136),
 Value(data=-0.3267594726206424),
 Value(data=-0.20239373201260613),
 Value(data=0.9217709387742457)]
```

An aside, for the parameters methods, we flatten a list, here's how to do it in Python with a list comprehension
```
params = [param for neuron in layer for param in neuron]
```
Now, a key part it training is updating the bweights based on the gradients we have calculated, this is done by
```
for p in n.parameters():
    p.data += -0.001 * p.grad
```
Then, we can recalculate the loss
```
ypred = [n(x) for x in xs]
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
loss.backward()
loss
```
And, finally, put all together in an update loop for training, we have:
```
steps = 2000
lr = 0.3
print(f"Initialised State - Loss {loss.data}")
for k in range(steps):
    # forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    # backward pass
    for p in n.parameters():
        # zero out gradients
        p.grad = 0.0
    loss.backward()
    # update
    for p in n.parameters():
        p.data += -1 * lr * p.grad
    
    if (k+1) % 100 == 0:
        print(f"Step {k+1} - Loss {loss.data}")
```
Out
```
Initialised State - Loss 6.277983103176016
Step 100 - Loss 0.0006500110242058011
Step 200 - Loss 0.00024981103153411707
Step 300 - Loss 0.00016013558284893044
Step 400 - Loss 0.00011896315722607197
Step 500 - Loss 9.501182524912765e-05
Step 600 - Loss 7.925407405863246e-05
Step 700 - Loss 6.80642934693096e-05
Step 800 - Loss 5.969163337114627e-05
Step 900 - Loss 5.318301751426277e-05
Step 1000 - Loss 4.7973749163459454e-05
Step 1100 - Loss 4.370733845887875e-05
Step 1200 - Loss 4.014734445162035e-05
Step 1300 - Loss 3.713061396891505e-05
Step 1400 - Loss 3.4540845968457264e-05
Step 1500 - Loss 3.229285309961101e-05
Step 1600 - Loss 3.0322781188833194e-05
Step 1700 - Loss 2.8581807280929536e-05
Step 1800 - Loss 2.7031951374078414e-05
Step 1900 - Loss 2.564321728861113e-05
Step 2000 - Loss 2.439159456846185e-05
```
Our loss is decreasing - We are winning!

Look at our predictions at this point
```
ypred
```
Out
```
[Value(data=0.9979085127787597),
 Value(data=-0.9981362872222674),
 Value(data=-0.996540008822785),
 Value(data=0.9978617035968881)]
```
And our labels
```
ys
```
Out
```
[1.0, -1.0, -1.0, 1.0]
```
We have trained a model that fits well to these data points...

End of first video
