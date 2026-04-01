### Why autocorrelation in the residuals is bad in an AR(p) model

An AR(p) model is built to explain the time dependence in the series:

\[
y_t = c + a_1 y_{t-1} + \dots + a_p y_{t-p} + u_t
\]

The idea is:

- the lagged \(y\)'s capture the **predictable part**
- the residual \(u_t\) is what is **left over**
- this leftover should be an **innovation / shock**
- therefore, in a well-specified AR(p) model, the residuals should be approximately **white noise**

White noise means in particular:

- mean zero
- constant variance
- **no autocorrelation**

So the logic is:

1. Fit the AR(p) model.
2. Remove the predictable component from \(y_t\).
3. Look at the residuals.
4. If the residuals are still autocorrelated, then there is still predictable structure left in the data.
5. Therefore, the AR(p) model has **not fully captured the dynamics**.

So if a Ljung-Box (or Box-Pierce) test gives a very low p-value, we reject:

\[
H_0:\text{ no autocorrelation in the residuals}
\]

This is **not a good result**, because it means the residuals are not white noise and the model is likely misspecified or missing lags.

### Short exam-style version

This is not a good result because in a correctly specified AR(p) model, the residuals should be approximately white noise. A very low p-value means we reject the null of no autocorrelation in the residuals, so there is still serial dependence left unexplained by the model.