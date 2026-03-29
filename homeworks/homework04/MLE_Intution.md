## MLE with a one-parameter example

A clean one-parameter example is: estimate the mean $\mu$ of iid normal data $y_1,\dots,y_T$, assuming the variance $\sigma^2$ is known.

We assume

$$
y_t \sim N(\mu,\sigma^2), \qquad t=1,\dots,T.
$$

Here the **only unknown parameter** is $\mu$.

### What MLE does

For any candidate value of $\mu$, the model tells us how plausible each observed $y_t$ is.  
That plausibility is measured by the density

$$
f(y_t\mid \mu)=\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\!\left(-\frac{(y_t-\mu)^2}{2\sigma^2}\right).
$$

If the observations are independent, the sample likelihood is

$$
L(\mu)=\prod_{t=1}^T f(y_t\mid \mu).
$$

Usually we maximize the **log-likelihood** instead:

$$
\log L(\mu)=\sum_{t=1}^T \log f(y_t\mid \mu),
$$

because logs are easier to work with and give the same optimizer.

### Intuition

If we try a candidate value like $\mu=10$, but the observed data are all around 1, then the observed values are far from the center of the assumed normal distribution. Their densities are small, so the likelihood is small.

If we try a candidate like $\mu=1$, and the data are clustered around 1, then the observed values are close to the center of the distribution. Their densities are larger, so the likelihood is larger.

So MLE chooses the value of $\mu$ that makes the observed sample **most likely** under the model.

### Why the sample mean comes out

For the normal model,

$$
\log L(\mu)
= \text{constant} - \frac{1}{2\sigma^2}\sum_{t=1}^T (y_t-\mu)^2.
$$

So maximizing the log-likelihood is equivalent to minimizing

$$
\sum_{t=1}^T (y_t-\mu)^2.
$$

That means the MLE of $\mu$ is

$$
\hat{\mu}=\frac{1}{T}\sum_{t=1}^T y_t,
$$

which is just the **sample mean**.

## Tiny numeric example

Suppose

$$
y=(0.8,\;1.1,\;1.3)
$$

and $\sigma^2=1$.

Compare two candidate values.

### Candidate 1: $\mu=0$

$$
(0.8-0)^2+(1.1-0)^2+(1.3-0)^2
=0.64+1.21+1.69=3.54
$$

### Candidate 2: $\mu=1$

$$
(0.8-1)^2+(1.1-1)^2+(1.3-1)^2
=0.04+0.01+0.09=0.14
$$

Since the sum of squared deviations is much smaller for $\mu=1$, the log-likelihood is much higher there.

The sample mean is

$$
\bar y = \frac{0.8+1.1+1.3}{3}\approx 1.067,
$$

so that is the MLE.

## Connection to your homework

In your homework, the idea is the same, but now there are **three parameters** instead of one:

$$
(\beta_0,\beta_1,s).
$$

The model is

$$
y_t = \beta_0 + \beta_1 x_t + u_t,
\qquad
u_t \sim \text{Laplace}(0,s).
$$

So MLE means:

1. Try candidate values of $\beta_0,\beta_1,s$.
2. Compute how likely the observed data are under that model.
3. Choose the values that maximize the likelihood.

## One-sentence summary

**MLE chooses the parameter value that makes the observed data most plausible under the assumed probability model.**