### Optimization Interview Questions

# Optimization Algorithms

Optimization algorithms in machine learning can be broadly classified into two primary categories: **First-order Methods** and **Second-order Methods**.

---

### First-order Methods (Derivative-based)

These methods leverage the gradient of the objective function to optimize parameters effectively.

These methods are sensitive to the choice of learning rate

- **Gradient Descent**
- **Stochastic Gradient Descent (SGD)**
- **Momentum**
- **AdaGrad**
- **RMSprop**
- **Adam**

---

### Second-order Methods

Second-order methods utilize second derivatives, allowing for potentially faster convergence, albeit at a higher computational cost.

- **Newton's Method**
- **L-BFGS** (Limited-memory Broyden-Fletcher-Goldfarb-Shanno)
- **Conjugate Gradient**
- **Hessian-Free Optimization**

### 1) Explain mathematically what is gradient in optimization? And if you want to minimize the loss function why do you update the model parameters in opposite direction of gradient, and how it ensures that loss function is minimized.

### 2) What is Learning rate ? And how to decide the size of steps to reach a local minima?

### 3) Explain the variants in gradient descent . And how they all differ ?

---

- Batch Gradient Descent
- stochastic Gradient Descent
- Mini Batch Gradient Descent

They differ in how  much data we use to compute the gradient of the objective function.



## 3.1 Batch Gradient Descent

Update the parameters for the entire dataset:

$$
\theta = \theta - \eta \cdot \nabla_{\theta} J(\theta)
$$

---

## 3.2 Stochastic Gradient Descent (SGD)

Update the parameters for each individual example  $\(x^{(i)}, y^{(i)}\)$

$$
\theta = \theta - \eta \cdot \nabla_{\theta} J(\theta; x^{(i)}, y^{(i)})
$$

---

## 3.3 Mini-Batch Gradient Descent

Update the parameters for each mini-batch of $n$ training examples:

$$
\theta = \theta - \eta \cdot \nabla_{\theta} J(\theta; x^{(i:i+n)}, y^{(i:i+n)})
$$

### 4) What are the challenges in above mentioned optimization Algorithms


Vanilla mini-batch gradient descent, however, does not guarantee good convergence, but offers a few challenges that need to be addressed:

- Choosing a proper learning rate can be difficult. A learning rate that is too small leads to painfully slow convergence, while a learning rate that is too large can hinder convergence and cause the loss function to fluctuate around the minimum or even to diverge.
- Learning rate schedules try to adjust the learning rate during training by methods such as annealing, where the learning rate is reduced according to a pre-defined schedule or when the change in objective between epochs falls below a threshold. However, these schedules and thresholds need to be defined in advance, limiting their adaptability to the characteristics of a given dataset.
- The same learning rate applies to all parameter updates. In cases where data is sparse or features have widely differing frequencies, it may be preferable to adjust the extent of updates differently—such as performing larger updates for rarely occurring features.
- Another challenge of minimizing the highly non-convex error functions common in neural networks is avoiding being trapped in suboptimal local minima. Research by Dauphin et al. argues that the issue often arises from saddle points, which are points where one dimension slopes up while another slopes down. These saddle points are typically surrounded by a plateau of similar error, making it difficult for SGD to escape as the gradient is close to zero in all dimensions.


### 5) What is momentum method? And how it accerlates Gradient Descent ? And how the update equation looks like in momentum algorithm?


SGD has trouble navigating ravines, i.e., areas where the surface curves much more steeply in one dimension than in another, which are common around local optima. In these scenarios, SGD oscillates across the slopes of the ravine while only making hesitant progress along the bottom towards the local optimum. Momentum is a method that helps accelerate SGD in the relevant direction and dampens oscillations. It achieves this by adding a fraction, $\gamma$, of the update vector from the past time step to the current update vector:

$$
v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta)
$$

$$
\theta = \theta - v_t
$$

The momentum term, $( \gamma)$ is typically set to 0.9 or a similar value.


### What are adaptive methods in optimization ?

### 6) What is Adagrad?


Adagrad is an algorithm for gradient-based optimization that adapts the learning rate to the parameters, allowing for larger updates for infrequent parameters and smaller updates for frequent ones. This makes it well-suited for dealing with sparse data. Dean et al. found that Adagrad significantly improved the robustness of SGD and used it for training large-scale neural networks at Google, which, among other things, learned to recognize cats in YouTube videos. Additionally, Pennington et al. utilized Adagrad to train GloVe word embeddings, as infrequent words require much larger updates than frequent ones.

Adagrad uses a different learning rate for each parameter $ \theta_i $ at every time step $ t $. The gradient of the objective function with respect to the parameter $ \theta_i $ at time step $ t $ is represented as:

$$
g_{t,i} = \nabla_{\theta_t} J(\theta_{t,i})
$$

The SGD update for every parameter $ \theta_i $ at each time step $ t $ is given by:

$$
\theta_{t+1,i} = \theta_{t,i} - \eta \cdot g_{t,i}
$$

Adagrad modifies the general learning rate $ \eta $ at each time step $ t $ for every parameter $ \theta_i $ based on the past gradients computed for $ \theta_i $:

$$
\theta_{t+1,i} = \theta_{t,i} - \frac{\eta}{\sqrt{G_{t,ii}} + \epsilon} \cdot g_{t,i}
$$

Here, $ G_t \in \mathbb{R}^{d \times d} $ is a diagonal matrix where each diagonal element $ G_{t,ii} $ is the sum of the squares of the gradients with respect to $ \theta_i $ up to time step $ t $, while $ \epsilon $ is a smoothing term that avoids division by zero (usually on the order of $ 1 \times 10^{-8} $). Interestingly, without the square root operation, the algorithm performs much worse.



Since $ G_t $ contains the sum of the squares of the past gradients with respect to all parameters $ \theta $ along its diagonal, we can vectorize our implementation by performing an element-wise matrix-vector multiplication between $ G_t $ and $ g_t $:

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} \odot g_t
$$

One of Adagrad main **benefits is that it eliminates the need to manually tune the learning rate**. Most implementations use a default value of $ 0.01 $ and leave it at that.

However, Adagrad main **weakness lies in the accumulation of the squared gradients in the denominator**. Since every added term is positive, the accumulated sum keeps growing during training. This leads to a decrease in the learning rate, which can eventually become infinitesimally small. At this point, the algorithm can no longer acquire additional knowledge. 

$$


$$

### 7) What is Adadelta and how it is different from Adagrad?

Adadelta is an extension of Adagrad that aims to reduce its aggressive, monotonically decreasing learning rate. Instead of accumulating all past squared gradients, Adadelta restricts the window of accumulated past gradients to a fixed size $ w $.


### 8) What is RMS prop?

RMSprop is an adaptive learning rate method proposed by Geoff Hinton. It was developed to address Adagrad’s diminishing learning rates. RMSprop is identical to the first update vector of Adadelta:


RMSprop (Root Mean Square Propagation) is an optimization algorithm that helps manage the learning rate during training, making it particularly useful for non-convex problems like training deep neural networks.

RMSprop is a variant of Stochastic Gradient Descent (SGD) and shares similarities with AdaGrad.

**Key Components:**

- **Squaring of Gradients:** The current gradient is divided by the root mean square of past gradients. This effectively adjusts the learning rate based on an estimate of the variance, helping to reach the optimal point more efficiently.
- **Leaky Integration:** RMSprop uses exponential smoothing of the squared gradient, acting as a leaky integrator to prevent vanishing learning rates.

$$
E[g^2]_t = 0.9 E[g^2]_{t-1} + 0.1 g^2_t
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t} + \epsilon} g_t
$$

Hinton suggests setting $ \gamma $ to $ 0.9 $, with a default learning rate $ \eta $ of $ 0.001 $.


### 9) Adam 


Adaptive Moment Estimation (Adam) is a method that computes adaptive learning rates for each parameter. It stores an exponentially decaying average of past squared gradients $ v_t $ like Adadelta and RMSprop, and it also keeps an exponentially decaying average of past gradients $ m_t $, similar to momentum:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

Here, $ m_t $ and $ v_t $ estimate the first moment (the mean) and the second moment (the uncentered variance) of the gradients, respectively. Since $ m_t $ and $ v_t $ are initialized as vectors of zeros, they tend to be biased towards zero, especially in the initial time steps, particularly when the decay rates are small (i.e., when $ \beta_1 $ and $ \beta_2 $ are close to 1).

To counteract these biases, the authors compute bias-corrected first and second moment estimates:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$

$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

They use these estimates to update the parameters, leading to the Adam update rule:

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

The authors propose default values of $ 0.9 $ for $ \beta_1 $, $ 0.999 $ for $ \beta_2 $, and $ 10^{-8} $ for $ \epsilon $. They demonstrate that Adam performs well in practice and compares favorably to other adaptive learning methods.


### 10) Which optimizer to use?

When choosing an optimizer, consider the nature of your input data. If the data is sparse, adaptive learning-rate methods tend to yield the best results. A significant advantage of these methods is that they usually require no manual tuning of the learning rate, as the default values often work well.

To summarize:

- **RMSprop** is an extension of Adagrad that addresses its issue of rapidly diminishing learning rates. It is effectively identical to Adadelta, except that Adadelta uses the RMS of parameter updates in the numerator of its update rule.
- **Adam** builds upon RMSprop by incorporating bias-correction and momentum. Studies by Kingma et al. show that this bias-correction enables Adam to slightly outperform RMSprop towards the end of optimization, especially as gradients become sparser.

Given their similarities and performance in comparable situations, **Adam** may be the best overall choice.

Interestingly, many recent studies still use vanilla SGD without momentum and a simple learning rate annealing schedule. While SGD can eventually find a minimum, it often takes significantly longer than adaptive optimizers, relies heavily on robust initialization and annealing schedules, and can become trapped in saddle points rather than local minima.

**Therefore, if fast convergence is a priority and you're training a deep or complex neural network, it's advisable to opt for one of the adaptive learning rate methods.**
