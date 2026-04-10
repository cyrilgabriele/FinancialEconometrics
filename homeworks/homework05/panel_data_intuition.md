## Panel data — short summary

**Main problems:** panel data combines **units** $i$ and **time** $t$, so errors may show **cross-sectional correlation**, **autocorrelation**, and **unit heterogeneity** from different intercepts $\alpha_i$.

### 1. Pooled OLS
Use
$$
y_{it} = \alpha + x_{it}^{\prime}\beta + u_{it}.
$$
This is the baseline model. It imposes **one common intercept** $\alpha$ and **one common slope vector** $\beta$ for all units.  
So yes: in pooled OLS, there is **only one alpha and one beta** shared by all units.  
Inference must account for the covariance structure of the errors; depending on the problem, use **White**, **clustered**, or **Driscoll-Kraay** type standard errors.

### 2. Fixed effects (within estimator)
Use
$$
y_{it} = \alpha_i + x_{it}^{\prime}\beta + u_{it}.
$$
This allows each unit to have its **own intercept** $\alpha_i$, while the slope vector $\beta$ is still common across units.  
So in fixed effects, unit 1 has $\alpha_1$, unit 2 has $\alpha_2$, and so on.

This is important when the unit-specific effect $\alpha_i$ is correlated with the regressors.

Estimation can be done in two equivalent ways:
- **Dummy-variable approach:** include one dummy for each unit
- **Within transformation:** demean each variable within unit and then run pooled OLS on the transformed data

The within transformation is:
$$
\bar{y}_i = \frac{1}{T}\sum_{t=1}^T y_{it}, \qquad
\bar{x}_i = \frac{1}{T}\sum_{t=1}^T x_{it},
$$

$$
y_{it}^* = y_{it} - \bar{y}_i, \qquad
x_{it}^* = x_{it} - \bar{x}_i,
$$

and then estimate
$$
y_{it}^* = {x_{it}^*}^{\prime}\beta + u_{it}^*.
$$

Important: you do **not** run a separate OLS for each unit.  
It is still **one overall regression**, either with unit dummies or with demeaned data.

After estimating $\hat{\beta}$, the unit-specific intercepts can be recovered as
$$
\hat{\alpha}_i = \bar{y}_i - \bar{x}_i^{\prime}\hat{\beta}.
$$

### 3. Time fixed effects
Use
$$
y_{it} = \lambda_t + \alpha_i + x_{it}^{\prime}\beta + u_{it}.
$$
This adds a separate intercept for each time period and handles shocks common to all units at a given date.

### 4. First differences
Use
$$
\Delta y_{it} = \Delta \lambda_t + \Delta x_{it}^{\prime}\beta + \Delta u_{it}.
$$
This is another way to remove time-constant unit effects.

### 5. Differences-in-differences
A special case of the first-difference setup with a treatment dummy. It identifies the treatment effect under the **common trend** assumption.

### 6. Fama-MacBeth
Estimate a cross-sectional regression in each period, then average the coefficients over time. This targets the **average cross-sectional effect** and handles cross-sectional dependence better than naive pooled OLS.

### One-line intuition
- **Pooled OLS:** one common intercept and slope for everyone
- **Robust / clustered SEs:** fix inference under dependence
- **Fixed effects / first differences:** remove time-constant heterogeneity
- **DiD:** treatment-effect setup
- **Fama-MacBeth:** repeated cross-sections over time