+++
author = "Laychiva Chhout"
title = "Gradient Descent"
date = "2023-12-24"
description = "From gradient descent to AdamW."
math = "true"
tags = [
    "ai",
    "ml",
    "dl",
    "optimisation"
]
categories = [
    "Artificial Intelligence",
    "Deep Learning",
    "Optimisation"
]
series = ["Themes Guide"]
aliases = ["migrate-from-jekyl"]
image = "gd_photo.png"
+++

{{< math.inline >}}
{{ if or .Page.Params.math .Site.Params.math }}
<!-- KaTeX -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/katex.min.css" integrity="sha384-zB1R0rpPzHqg7Kpt0Aljp8JPLqbXI3bhnPWROx27a9N0Ll6ZP/+DiW/UqRcLbRjq" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/katex.min.js" integrity="sha384-y23I5Q6l+B6vatafAwxRu/0oK/79VlbSz7Q9aiSZUvyWYIYsd+qj+o24G5ZU2zJz" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>
{{ end }}
{{</ math.inline >}}

## 1. Gradient Descent

Gradient descent is a fundamental optimization algorithm used in machine learning and artificial intelligence. It's particularly useful for minimizing a function, often a cost or loss function in machine learning models. The basic idea behind gradient descent is to find the minimum of a function by moving in the direction of the steepest descent as defined by the negative of the gradient.

Here's a simple overview of the gradient descent algorithm:
1. **Initialization**: Start with an initial point (initial guess) for the parameters you're trying to optimize.
2. **Gradient Calculation**: Compute the gradient of the function at the current point. The gradient is a vector of partial derivatives and points in the direction of the steepest ascent.
3. **Update Parameters**: Adjust the parameters in the opposite direction of the gradient. This is typically done by subtracting a fraction of the gradient from the current parameter values. The fraction is determined by the learning rate, a hyperparameter that controls how big a step we take.
4. **Iteration:** Repeat steps 2 and 3 until the algorithm converges to a minimum. Convergence is often defined by the gradient being close to zero or by reaching a maximum number of iterations.

The mathematical formula for updating each parameter in gradient descent is:
$$
\theta_{\text {new }}=\theta_{\text {old }}-\alpha \cdot \nabla J(\theta)
$$

Where:
- $\theta$ represents the parameters of the function.
- $\alpha$ is the learning rate.
- $\nabla J(\theta)$ is the gradient of the function $J$ with respect to the parameters $\theta$.

### 1.1. Batch Gradient Descent (BGD):

Batch Gradient Descent is like the methodical explorer in the world of optimization. It meticulously evaluates the entire training dataset in each iteration, making it a reliable choice for finding the global minimum of the loss function.

- **Advantages**: BGD converges to the global minimum of the loss function, if it exists, ensuring a thorough search.
It's generally more stable than other techniques due to its comprehensive evaluation of all training samples.
- **Disadvantages:** However, BGD can be computationally expensive for large training datasets, akin to carrying a heavy load.
It's also sensitive to the initial values of the model parameters, making the starting point crucial.

The mathematical formula for updating each parameter in gradient descent is: $$\theta=\theta-\alpha \cdot \nabla_\theta J(\theta)$$
**Description**: In batch gradient descent, the entire dataset is used to compute the gradient of the cost function for each iteration. Here, $\theta$ represents the parameters, $\alpha$ is the learning rate, and $J(\theta)$ is the cost function. The gradient $\nabla_\theta J(\theta)$ is computed over the whole dataset.

### 1.2. Stochastic Gradient Descent (SGD):

Stochastic Gradient Descent is like the agile sprinter among optimization methods. It quickly makes progress, especially when dealing with extensive training datasets, but it might take shortcuts along the way.

- **Advantages**: SGD's speed allows it to converge rapidly, a big plus when you have lots of data to process.
It's less sensitive to the initial parameter values, providing some flexibility.
- **Disadvantages**: However, SGD can be a bit reckless, potentially converging to a local minimum instead of the global one.
It's also noisier than BGD as it updates parameters using just one random training sample at a time.

The mathematical formula for updating each parameter in gradient descent is: $$\theta=\theta-\alpha \cdot \nabla_\theta J\left(\theta ; x^{(i)} ; y^{(i)}\right)$$
**Description**: SGD updates the parameters using only a single training example $x^{(i)}$ and its corresponding target $y^{(i)}$ at each iteration. This means that for each iteration, the gradient is calculated using just one data point. While this can make SGD faster and able to be used on large datasets, it can also make the convergence path more erratic.

### 1.3. Mini-batch Gradient Descent

Mini-batch Gradient Descent is the middle ground, balancing between BGD's thoroughness and SGD's speed. It's like the "just right" choice in the optimization story.

- **Advantages**: Mini-batch GD combines the strengths of BGD and SGD. It's faster than BGD for large datasets but more stable than pure SGD. It strikes a balance and is less likely to get stuck in local minima compared to SGD.
- **Disadvantages**: While it provides a balanced approach, mini-batch GD can still occasionally converge to a local minimum instead of the global one.

The mathematical formula for updating each parameter in gradient descent is: $$\theta=\theta-\alpha \cdot \nabla_\theta J\left(\theta ; X_{\text {mini-batch }} ; Y_{\text {mini-batch }}\right)$$
**Description**: Mini-batch gradient descent is a compromise between batch and stochastic versions. It uses a small, random subset of the data - a mini-batch - at each iteration to compute the gradient. The size of the mini-batch, typically ranging from 10 to 1000 , is an important hyperparameter. This approach seeks to balance the efficiency of batch gradient descent with the stochastic nature of SGD.

### 1.4. Problems of Gradient Descent
- Problems
    - does not guarantee good convergence
    - neural network = highly non-convex error functions
    - avoid getting trapped in their numerous suboptimal local minima or saddle points (points where one dimension slopes up and another slopes down) which are usually surrounded by a plateau
    - choosing a proper learning rate can be difficult, need to adapt the learning rate to dataset's characteristics
    - each parameter may require a different learning rate (sparse data)
- Alternatives to Gradient Descent
    - first-order methods: Momentum, Nesterov (NAG), Adagrad, Adadelta/RMSprop, Adam
    - second-order methods: Newton

## 2. Alternative to gradient descent

### 2.1. Momentum in Gradient Descent

**Goal:**
The main objective of Momentum is to accelerate gradient descent in the relevant direction while dampening oscillations. This helps in faster convergence of the optimization algorithm.

**How it Works:**
Momentum modifies the update rule for the gradient descent by incorporating a fraction of the previous update. This modification can be explained in two steps:
1. Calculation of Velocity $V_{d \theta}$ :
    - On iteration $t$, first compute the gradient $\frac{\partial \mathscr{L}\left(\theta^{[t-1]}, x, y\right)}{\partial \theta}$ on the current mini-batch.
    - Update the velocity $V_{d \theta}$ using the formula:
    $$
    V_{d \theta}^{[t]}=\beta V_{d \theta}^{[t-1]}+(1-\beta) \frac{\partial \mathscr{L}\left(\theta^{[t-1]}, x, y\right)}{\partial \theta}
    $$
    - Here, $\beta$ is a hyperparameter that represents the momentum coefficient, typically chosen as 0.9.
2. Updating Parameters $\theta$ :
    - Update the parameters using the velocity:
    $$
    \theta^{[t]}=\theta^{[t-1]}-\alpha V_{d \theta}^{[t]}
    $$
    - $\alpha$ is the learning rate.

**Explanation of the Momentum Term:**
- The term $V_{d \theta}$ acts as a velocity for the parameters, accumulating the gradients of the past steps.
- The momentum term increases for dimensions whose gradients consistently point in the same direction, allowing faster movement in those dimensions.
- It reduces updates for dimensions where gradients frequently change directions, helping to dampen oscillations.
- The parameter $\beta$ functions as a friction parameter, controlling the influence of past gradients on the current update.
- The gradient $\frac{\partial \mathscr{L}}{\partial \theta}$ acts as an acceleration, determining the direction and magnitude of the parameter update.

### 2.2. Nesterov Accelerated Gradient (NAG)
**Problem**: Imagine a ball rolling down a hill, following the slope without any awareness of its surroundings. This scenario is similar to standard gradient descent, where the updates are based solely on the current gradient, without considering the future direction. This approach can be inefficient as it doesn't account for the upcoming changes in the slope.

**Solution**: Ideally, we want a more intelligent approach. Imagine a 'smarter' ball that has an understanding of its trajectory. Such a ball would slow down before reaching the bottom or an upcoming ascent, anticipating the change in slope. This concept is analogous to Nesterov Accelerated Gradient (NAG), an improved version of the momentum method in gradient descent.

Momentum Update in NAG:
In NAG, the parameter updates are modified to anticipate the future position of the parameters. The steps are as follows:
1. Forward-Looking Update:
    - Instead of computing the gradient at the current position, $\theta^{[t-1]}$, NAG first makes a preliminary step in the direction of the accumulated velocity (momentum), $\beta V_{d \theta}^{[t-1]}$
    - The derivative of the loss is then calculated at this lookahead position: $\theta^{[t-1]}-$ $\alpha \beta V_{d \theta}^{[t-1]}$.
    - This gives an approximation of where the parameters are likely to be in the next step, providing a more informed gradient calculation.
2. Velocity and Parameter Update:
    - The velocity is updated using this lookahead gradient:
    $$
    V_{d \theta}^{[t]}=\beta V_{d \theta}^{[t-1]}+(1-\beta) \frac{\partial \mathscr{L}\left(\theta^{[t-1]}-\alpha \beta V_{d \theta}^{[t-1]}, x, y\right)}{\partial \theta}
    $$
    - The parameters are then updated using this new velocity:
    $$
    \theta^{[t]}=\theta^{[t-1]}-\alpha V_{d \theta}^{[t]}
    $$

### 2.3. AdaGrad (Adaptive Gradient Algorithm)
**Goal:** AdaGrad's primary goal is to adapt the learning rate for each parameter individually. The idea is to adjust the learning rate differently for different features, depending on their frequency and importance in the dataset.
- For frequently occurring features, AdaGrad aims for smaller updates (lower learning rate).
- For infrequent features, it seeks larger updates (higher learning rate).

This adaptive approach helps in dealing more effectively with sparse data and different scales of features.

**Mechanism and Notation:**
- Let $d \theta_i^{[t]}=\frac{\partial \mathscr{L}\left(\theta^{[t]}, x, y\right)}{\partial \theta_i}$ be the gradient of the loss function with respect to the parameter $\theta_i$ at time step $t$.
- In standard Stochastic Gradient Descent (SGD), the parameter $\theta_i$ is updated as follows:
$$
\theta_i^{[t]}=\theta_i^{[t-1]}-\alpha d \theta_i^{[t-1]}
$$
where $\alpha$ is the constant learning rate.

**AdaGrad Update Rule:**
- AdaGrad modifies the update rule by scaling the learning rate for each parameter based on the sum of squares of past gradients for that parameter.
- Compute the sum of squares of past gradients for $\theta_i$ :
$$
G_{i, i}^{[t]}=\sum_{\tau=0}^t d \theta_i^{[\tau]^2}
$$
- Update the parameter $\theta_i$ using the modified learning rate:
$$
\theta_i^{[t]}=\theta_i^{[t-1]}-\frac{\alpha}{\sqrt{G_{i, i}^{[t-1]}+\epsilon}} d \theta_i^{[t-1]}
$$
Here, $\epsilon$ is a small constant added for numerical stability.

**Problem with AdaGrad:**

- The main issue with AdaGrad is that the gradients are accumulated from the beginning of the training process, which can lead to an overly aggressive decrease in the learning rate.
- As training progresses, the accumulated squared gradients in the denominator cause the learning rate to shrink and become infinitesimally small. This makes the algorithm converge too slowly in the later stages of training.

To solve this problem, we introduce a variant of AdaGrad which is AdaDelta and RMSprop.

#### 2.3.1. AdaDelta
AdaDelta modifies the AdaGrad approach by limiting the accumulation of past gradients to a fixed-size window. This is achieved by introducing a decay factor $\gamma$, which reduces the influence of older gradients.

**AdaDelta Update Rule:**
1. Accumulating Gradients Over a Window:
    - Instead of summing up all past squared gradients, AdaDelta maintains a running average of the recent squared gradients.
    - The expectation $\mathbb{E}\left[d \theta^2\right]^{[t]}$ of the squared gradients is calculated as follows: $\mathbb{E}\left[d \theta^2\right]^{[t]}=\gamma \mathbb{E}\left[d \theta^2\right]^{[t-1]}+(1-\gamma) d \theta^{[t]^2}$
    - Here, $\gamma$ is a decay constant, similar to the momentum term in other optimization algorithms, and $d \theta^{[t]}$ is the gradient at time $t$.
2. Parameter Update:
    - The parameters are updated using a modified learning rate that considers the running average of squared gradients:
    $$
    \theta^{[t]}=\theta^{[t-1]}-\frac{\alpha}{\sqrt{\mathbb{E}\left[d \theta^2\right]^{[t-1]}+\epsilon}} d \theta^{[t-1]}
    $$
    - $\alpha$ is the learning rate, and $\epsilon$ is a small constant added for numerical stability.

**Advantages of AdaDelta:**
- AdaDelta addresses the diminishing learning rate problem in AdaGrad by preventing the continual growth of the denominator.
- The use of a decay factor $\gamma$ helps to maintain a more balanced and adaptive learning rate throughout the training process.
- By focusing on a window of recent squared gradients, AdaDelta adapts more effectively to the data's changing patterns, leading to better convergence behavior in practice.

#### 2.3.2. RMSprop(Root Mean Square Propagation) 
RMSprop is another optimization algorithm designed to address some of the shortcomings of AdaGrad, especially in the context of training deep neural networks.

**Algorithm**

**Core Idea:**
RMSprop, like AdaGrad, adjusts the learning rate for each parameter, but it does so in a way that's better suited for training deep networks. It aims to speed up learning in directions with small gradients and slow down learning in directions with large gradients.

**RMSprop Update Rule:**
1. Compute Gradient on Current Mini-Batch:
    - On iteration $t$, calculate the gradient $d \theta$ based on the current mini-batch.
2. Update Rule:
    - Calculate the exponentially weighted moving average of the squared gradients:
    $$
    S_{d \theta}^{[t]}=\gamma S_{d \theta}^{[t-1]}+(1-\gamma) d \theta^{[t]^2}
    $$
    - Update the parameters using an element-wise division by the square root of $S_{d \theta}$ :
    $$
    \theta^{[t]}=\theta^{[t-1]}-\frac{\alpha}{\sqrt{S_{d \theta}^{[t-1]}+\epsilon}} d \theta^{[t-1]}
    $$
    - Here, $\gamma$ is a decay constant (typically close to 1), $\alpha$ is the learning rate, and $\epsilon$ is a small constant to ensure numerical stability.

**Speeding Up and Slowing Down Learning:**
- Speed Up in Horizontal Direction (e.g., Weight $W$ ):
    - If the squared gradient $S_{d W}^{[t]}$ is small, the term $\frac{1}{\sqrt{S_{d W}^{[t]}+\epsilon}}$ becomes large, leading to a larger update step. This speeds up learning when the changes in the parameter $W$ are small.
- Slowing Down Oscillations in Vertical Direction (e.g., Bias $b$ ):
    - If the squared gradient $S_{d W}^{[t]}$ is large, the term $\frac{1}{\sqrt{S_{d W}^{[t]}+\epsilon}}$ becomes small, resulting in a smaller update step. This helps in damping oscillations in parameters like the bias $b$, where the changes are large.

### 2.4. Adam (Adaptive moment estimation)

Adam is an optimization algorithm that combines the ideas of Momentum and RMSprop/AdaDelta. It's popular in training deep neural networks due to its effective handling of sparse gradients and adaptive learning rate mechanics.

**Adam Algorithm**

Combining Momentum and AdaDelta/RMSprop:
Adam integrates the concepts of Momentum (first moment) and RMSprop (second moment) to optimize the learning process.

**Adam Update Rule:**
1. Compute Gradient on Current Mini-Batch:
    - On iteration $t$, calculate the gradient $d \theta$ based on the current mini-batch.
2. Momentum and RMSprop Updates:
    - Update the biased first moment estimate (similar to Momentum):
    $$
    V_{d \theta}^{[t]}=\beta_1 V_{d \theta}^{[t-1]}+\left(1-\beta_1\right) d \theta^{[t-1]}
    $$
    - Update the biased second moment estimate (similar to RMSprop):
    $$
    S_{d \theta}^{[t]}=\beta_2 S_{d \theta}^{[t-1]}+\left(1-\beta_2\right) d \theta^{[t-1]^2}
    $$
3. Parameter Update:
    - Update the parameters using a combination of both moments:
    $$
    \theta^{[t]}=\theta^{[t-1]}-\alpha \frac{V_{d \theta}^{[t]}}{\sqrt{S_{d \theta}^{[t]}+\epsilon}}
    $$
    - Here, $\alpha$ is the learning rate, and $\epsilon$ is a small constant added for numerical stability.

**Hyperparameters:**
- Learning Rate $\alpha$ : This is a crucial hyperparameter that needs tuning. It determines the step size at each iteration.
- Momentum Term $\beta_1$ : Typically set to 0.9. It controls the first moment estimate, which is akin to Momentum.
- RMSprop Term $\beta_2$ : Usually set to 0.999. It manages the second moment estimate, similar to RMSprop.
- Stabilizing Term $\epsilon$ : A very small number, often $10^{-8}$, to prevent division by zero in the update.

### 2.5. AdamW
AdamW is a modification of the Adam optimization algorithm, which addresses an issue in the way Adam integrates weight decay (regularization). AdamW separates the weight decay term from the gradient updates, leading to better training dynamics and generalization performance.

**AdamW Algorithm**

AdamW revises the Adam update rule by decoupling the weight decay component. In standard Adam, weight decay is implicitly included in the gradient updates, which can interfere with the adaptive learning rate mechanism. AdamW separates these two aspects, applying weight decay directly to the weights rather than through the gradient.

**AdamW Update Rule:**
1. Compute Gradient on Current Mini-Batch:
    - On iteration $t$, compute the gradient $d \theta$ based on the current mini-batch.
2. Momentum and RMSprop Updates:
    - Update the biased first moment estimate (similar to Momentum):
    $$
    V_{d \theta}^{[t]}=\beta_1 V_{d \theta}^{[t-1]}+\left(1-\beta_1\right) d \theta^{[t-1]}
    $$
    - Update the biased second moment estimate (similar to RMSprop):
$$
S_{d \theta}^{[t]}=\beta_2 S_{d \theta}^{[t-1]}+\left(1-\beta_2\right) d \theta^{[t-1]^2}
$$
3. Decoupled Weight Decay:
    - Apply weight decay directly to the weights:
    $$
    \theta=\theta-\lambda \theta
    $$
    - Here, $\lambda$ represents the weight decay coefficient.
    
4. Parameter Update:
    - Update the parameters using a combination of both moments, excluding the weight decay:
    $$
    \theta^{[t]}=\theta^{[t-1]}-\alpha \frac{V_{d \theta}^{[t]}}{\sqrt{S_{d \theta}^{[t]}+\epsilon}}
    $$
    - $\alpha$ is the learning rate, and $\epsilon$ is a small constant for numerical stability.

**Hyperparameters:**

- Learning Rate $\alpha$ : Determines the step size at each iteration.
- Momentum Term $\beta_1$ : Typically 0.9, controls the first moment estimate (momentum).
- RMSprop Term $\beta_2$ : Usually 0.999, manages the second moment estimate (RMSprop).
- Stabilizing Term $\epsilon$ : A small number like $10^{-8}$, to prevent division by zero.
- Weight Decay $\lambda$ : Controls the amount of weight decay, applied separately from the gradient update.

**PyTorch code to load different optimizers:**

```python
import torch.optim as optim
# Different optimizers
optimizers = {
    "SGD": optim.SGD(model.parameters(), lr=0.01),
    "Adam": optim.Adam(model.parameters(), lr=0.001),
    "RMSprop": optim.RMSprop(model.parameters(), lr=0.01),
    "Adagrad": optim.Adagrad(model.parameters(), lr=0.01),
    "AdamW": optim.AdamW(model.parameters(), lr=0.001),
}
```

### Reference

Lecture slides of Professor: **Geoffroy Peeter**, **Télécom Paris.**
