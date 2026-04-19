# Toy GARCH(1,1) Example

Here is a small worked **GARCH(1,1)** example that mirrors the course workflow.

We keep the mean model as simple as possible: assume the return has **zero mean**, so the residual is just the return itself,

$$
u_t = r_t.
$$

The GARCH(1,1) variance equation is

$$
\sigma_t^2 = \omega + \alpha u_{t-1}^2 + \beta \sigma_{t-1}^2,
$$

with the usual restrictions

$$
\omega > 0, \qquad \alpha \ge 0, \qquad \beta \ge 0, \qquad \alpha + \beta < 1.
$$

---

## 1. Data

Take the following toy return series:

$$
r_1 = 0.020, \qquad
r_2 = -0.010, \qquad
r_3 = 0.030, \qquad
r_4 = -0.025, \qquad
r_5 = 0.015.
$$

These correspond to:

- $2.0\%$
- $-1.0\%$
- $3.0\%$
- $-2.5\%$
- $1.5\%$

Because we assume zero mean, we have

$$
u_t = r_t.
$$

---

## 2. Pick a starting variance

To start the recursion, choose an initial variance. A simple choice is the sample variance of the residuals:

$$
\sigma_0^2 \approx \operatorname{Var}(u_t) = 0.000414.
$$

Also set

$$
u_0 = 0
$$

just to launch the recursion.

---

## 3. Try one parameter guess

Suppose the optimizer tries the following candidate parameters:

$$
\omega = 0.00001, \qquad
\alpha = 0.10, \qquad
\beta = 0.85.
$$

These satisfy the GARCH restrictions.

---

## 4. Run the recursion

Now compute the conditional variances one by one.

### At \( t = 1 \)

$$
\sigma_1^2
= 0.00001 + 0.10 \cdot u_0^2 + 0.85 \cdot \sigma_0^2
= 0.00001 + 0 + 0.85(0.000414)
= 0.0003619.
$$

### At \( t = 2 \)

$$
\sigma_2^2
= 0.00001 + 0.10(0.020)^2 + 0.85(0.0003619)
= 0.000357615.
$$

### At \( t = 3 \)

$$
\sigma_3^2
= 0.00001 + 0.10(-0.010)^2 + 0.85(0.000357615)
= 0.000323973.
$$

### At \( t = 4 \)

$$
\sigma_4^2
= 0.00001 + 0.10(0.030)^2 + 0.85(0.000323973)
= 0.000375377.
$$

### At \( t = 5 \)

$$
\sigma_5^2
= 0.00001 + 0.10(-0.025)^2 + 0.85(0.000375377)
= 0.000391570.
$$

This already shows the GARCH mechanism: after a large shock such as $u_3 = 3\%$, the next conditional variance rises.

---

## 5. Compute the log-likelihood

Assume conditional normality:

$$
u_t \mid \mathcal{F}_{t-1} \sim \mathcal{N}(0, \sigma_t^2).
$$

Then the period-\(t\) log-likelihood contribution is

$$
\ell_t
=
-\frac{1}{2}
\left[
\ln(2\pi)
+ \ln(\sigma_t^2)
+ \frac{u_t^2}{\sigma_t^2}
\right].
$$

Using the variances above gives approximately

$$
\ell_1 = 2.4905, \qquad
\ell_2 = 2.9093, \qquad
\ell_3 = 1.7095, \qquad
\ell_4 = 2.1924, \qquad
\ell_5 = 2.7164.
$$

So the total log-likelihood is

$$
\ell(\omega, \alpha, \beta) \approx 12.018.
$$

---

## 6. Compare with another parameter guess

Now try a second candidate:

$$
\omega = 0.00005, \qquad
\alpha = 0.25, \qquad
\beta = 0.60.
$$

Running the same recursion and likelihood calculation gives a total log-likelihood of about

$$
\ell \approx 11.849.
$$

Since

$$
12.018 > 11.849,
$$

the **first candidate fits this toy sample better**.

That is what MLE does:

1. choose trial values for $(\omega, \alpha, \beta)$,
2. recursively compute $\sigma_t^2$,
3. compute the total log-likelihood,
4. update the trial values,
5. repeat until the likelihood is maximized.

---

## 7. Forecast the next variance

Once you have final estimates, forecasting is straightforward.

Using the **first candidate** as if it were the estimated model, the next variance forecast is

$$
\sigma_6^2
= 0.00001 + 0.10(0.015)^2 + 0.85(0.000391570)
= 0.000365335.
$$

So the next-period conditional standard deviation forecast is

$$
\sigma_6 = \sqrt{0.000365335} \approx 0.0191,
$$

which is about **1.91\%**.

---

## 8. Main intuition

The important distinction is:

- the **GARCH recursion** tells you how volatility evolves **given parameters**;
- **MLE** tells you which parameter values make the observed data most plausible.

So the workflow is:

1. choose a mean model,
2. define the residuals $u_t$,
3. choose starting values,
4. guess $(\omega, \alpha, \beta)$,
5. recursively compute $\sigma_t^2$,
6. compute the log-likelihood,
7. optimize,
8. then use the fitted model to forecast volatility.

---

## 9. One-line summary

**Recursion computes volatility for given parameters; MLE learns the parameters from the data.**
