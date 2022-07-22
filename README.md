# BijecTorch
Bijectors (differentiable and injective function defined on an open subset of R^n) in PyTorch

The goal of this project is to implement bijective transformations in PyTorch. 
Our usecase in mind is currently focused on (conditional) normalizing flows.
At the moment the `torch.distribution` interface is modelled after `tensorflow.probability` and implements a lot of different distributions.
The `bijector` part of `tensorflow.probability` is missing. We want to port that part to `torch`. 

This package generally follows the design of the TensorFlow Distributions package, but it is not limited to it.

Pull Requests are generally warmly welcomed. Just pick your favorite bijector and port it (or document it)!

