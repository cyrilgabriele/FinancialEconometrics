# Cross-check of lab comments

Scope note: this cross-check uses the course lecture notes, lecture slides, and fact sheet as the reference base. It can verify the *econometric content and course-topic alignment*, but it cannot verify the exact `labs/...` line references or R-specific implementation details unless the lab scripts themselves are provided. The course files confirm that the labs are part of the examinable material and that they cover statistics, least squares/testing, time series, maximum likelihood, and volatility models. 

Legend:
- **Verified (course-aligned):** clearly supported by the lecture notes/slides.
- **Partially verified:** the econometric idea is course-aligned, but the exact R/package/tool comment is not documented in the course files.
- **Not directly verifiable from course files:** implementation-specific and not covered in the lecture notes/slides.

## Lab 01

- **Partially verified.** Environment comments explain why to set a working directory, load add-on packages, and convert the imported tab-delimited table into an `xts` object for time-series operations (`labs/lab01/s1.R:3`, `labs/lab01/s1.R:21`, `labs/lab01/s1.R:28`, `labs/lab01/s1.R:33`).  
  The *time-series-data* aspect fits the course, but working directories, package loading, and `xts` are R-implementation details rather than lecture-note content.

- **Partially verified.** Plotting notes contrast `autoplot` and manual `ggplot` layering, caution about mixing scales, and describe how vertical/horizontal reference lines plus `geom_smooth(method="lm")` shape scatter plots (`labs/lab01/s1.R:47`, `labs/lab01/s1.R:55`, `labs/lab01/s1.R:64`, `labs/lab01/s1.R:214`).  
  Scatter plots, fitted lines, and visual interpretation are course-aligned; the specific `ggplot2`/`autoplot` distinctions are tooling-specific.

- **Partially verified.** Transformation guidance covers computing percentage log returns, naming conventions (uppercase levels vs lowercase returns), creating subsamples with `xts` date slices, and de-annualizing the risk-free rate by dividing by 12 (`labs/lab01/s1.R:78`, `labs/lab01/s1.R:88`, `labs/lab01/s1.R:96`).  
  Returns rather than prices are strongly course-aligned; naming conventions, `xts` slicing, and the exact code style are not documented in the lecture files.

- **Verified (course-aligned).** Performance-testing comments walk through the Sharpe ratio loop, correlation matrix interpretation, and one-sample/paired t-tests (including manual test-statistic and confidence-interval calculations) to compare returns with or without the risk-free leg (`labs/lab01/s1.R:102`, `labs/lab01/s1.R:117`, `labs/lab01/s1.R:125`, `labs/lab01/s1.R:149`).  
  This matches the course treatment of sample means, covariance/correlation, hypothesis testing, and Sharpe-ratio testing.

- **Verified (course-aligned).** Distribution diagnostics summarize skewness/kurtosis meaning, Jarque–Bera hypothesis interpretation, QQ-plot behavior, and how histogram overlays assess normality (`labs/lab01/s1.R:164`, `labs/lab01/s1.R:182`, `labs/lab01/s1.R:194`).  
  This matches the notes/slides on skewness, kurtosis, Jarque–Bera, histograms, and QQ plots.

## Lab 02

- **Partially verified.** Library comments justify pulling in `dygraphs`, `tseries`, `forecast`, `lmtest`, and `sandwich`, and note the need to drop the first observation once returns are computed to avoid `NA` values (`labs/lab02/s2.R:5`, `labs/lab02/s2.R:7`, `labs/lab02/s2.R:9`, `labs/lab02/s2.R:19`).  
  The residual-testing and forecasting ideas are course-aligned; the package list and `NA` handling are implementation details.

- **Partially verified.** Visualization and regression notes explain the role of each `ggplot2` layer in the scatter plot and remind that additional regressors are added with `"+"` in `lm()` formulas (`labs/lab02/s2.R:21`, `labs/lab02/s2.R:27`, `labs/lab02/s2.R:31`, `labs/lab02/s2.R:35`).  
  Scatter-plot and multiple-regression logic fit the course; `ggplot2` and `lm()` syntax are not covered in the lecture files.

- **Verified (course-aligned).** Goodness-of-fit commentary defines \(R^2\) vs adjusted \(R^2\), how RSS/ESS/TSS relate, and how the ANOVA / residual-standard-error output maps back to manual calculations (`labs/lab02/s2.R:40`, `labs/lab02/s2.R:55`, `labs/lab02/s2.R:59`, `labs/lab02/s2.R:64`).  
  The course slides explicitly discuss \(R^2\), adjusted \(R^2\), and model-selection tradeoffs; mapping software output back to formulas is consistent with the lecture style.

- **Verified (course-aligned).** Residual-diagnostics comments recommend plotting residual levels and squares (including interactive views) plus Jarque–Bera/histogram checks to assess normality departures (`labs/lab02/s2.R:73`, `labs/lab02/s2.R:77`, `labs/lab02/s2.R:80`, `labs/lab02/s2.R:84`).  
  This matches the course emphasis on residual diagnostics, heteroskedasticity, and normality checks.

- **Verified (course-aligned).** Autocorrelation guidance explains the meaning of ACF confidence bands, why occasional spikes can appear by chance, and how the Ljung–Box test evaluates joint serial dependence (`labs/lab02/s2.R:87`, `labs/lab02/s2.R:92`, `labs/lab02/s2.R:94`, `labs/lab02/s2.R:98`, `labs/lab02/s2.R:102`, `labs/lab02/s2.R:106`).  
  The ACF interpretation is clearly course-aligned. The exact mention of Ljung–Box is not prominent in the retrieved snippets, but it is fully consistent with the course unit on autocorrelation diagnostics.

- **Verified / partially verified mix.** Later comments cover Breusch–Pagan vs White tests, using Newey–West HAC errors via `coeftest`, the intuition behind SUR (correlated error systems), and chi-square hypothesis testing on linear restrictions (`labs/lab02/s2.R:116`, `labs/lab02/s2.R:119`, `labs/lab02/s2.R:127`, `labs/lab02/s2.R:133`, `labs/lab02/s2.R:140`, `labs/lab02/s2.R:145`).  
  White tests, Newey–West, systems of regressions, and chi-square/Wald tests are directly course-aligned. `coeftest` and the exact R workflow are implementation-specific. Breusch–Pagan is plausible and standard in this context, but the retrieved course snippets support White much more directly than BP.

## Lab 03

- **Verified (course-aligned).** ACF/PACF notes describe how slow-decaying autocorrelation indicates non-stationarity and why both direct (ACF) and partial effects (PACF) matter when diagnosing unit roots (`labs/lab03/s3.R:16`, `labs/lab03/s3.R:22`, `labs/lab03/s3.R:27`, `labs/lab03/s3.R:31`, `labs/lab03/s3.R:33`, `labs/lab03/s3.R:39`).  
  This is directly supported by the time-series lecture material.

- **Verified (course-aligned).** Unit-root testing comments restate the hypotheses for ADF (non-stationary null) and KPSS (stationary null) and how to interpret test statistics relative to critical values (`labs/lab03/s3.R:45`, `labs/lab03/s3.R:47`, `labs/lab03/s3.R:54`, `labs/lab03/s3.R:61`, `labs/lab03/s3.R:67`).  
  The notes explicitly state the different null hypotheses for ADF and KPSS.

- **Partially verified.** Forecasting remarks explain `Arima(order = c(p,d,q))`, why the reported date corresponds to the out-of-sample period, and how `qnorm` provides 95% confidence bounds (`labs/lab03/s3.R:70`, `labs/lab03/s3.R:74`, `labs/lab03/s3.R:76`, `labs/lab03/s3.R:80`, `labs/lab03/s3.R:85`).  
  Forecasting, AR/MA/ARMA structure, and confidence bands are course-aligned; the exact `Arima(...)` function syntax and `qnorm` usage are coding details.

- **Verified (course-aligned).** Rolling forecast comments note that each looped AR(1) or factor regression produces a \(t+1\) forecast, highlight difficulties capturing conditional volatility, and define RMSE as the square root of mean squared errors (`labs/lab03/s3.R:96`, `labs/lab03/s3.R:98`, `labs/lab03/s3.R:103`, `labs/lab03/s3.R:114`, `labs/lab03/s3.R:118`).  
  This matches the forecasting section and the course’s discussion of volatility dynamics.

- **Verified / partially verified mix.** Model-comparison comments outline how to forecast with lagged factors, compute competing RMSEs, apply Diebold–Mariano tests (with alternative choices), and interpret negative out-of-sample \(R^2\) when a benchmark beats the factor model (`labs/lab03/s3.R:123`, `labs/lab03/s3.R:132`, `labs/lab03/s3.R:139`, `labs/lab03/s3.R:143`, `labs/lab03/s3.R:146`, `labs/lab03/s3.R:156`).  
  Out-of-sample \(R^2\), benchmark comparison, and forecast evaluation are directly course-aligned. Diebold–Mariano is cited in the lecture notes bibliography and is fully plausible here, but it is not prominent in the retrieved teaching snippets, so that part is best treated as only partially verified from the provided course files.

## Lab 04

- **Verified / partially verified mix.** Early comments contrast MLE and OLS, show how `mle()` minimizes a negative log-likelihood for a normal-error market model, and include a commented AR(1) MLE template as context (`labs/lab04/s4.R:11`, `labs/lab04/s4.R:13`, `labs/lab04/s4.R:17`, `labs/lab04/s4.R:22`, `labs/lab04/s4.R:26`, `labs/lab04/s4.R:32`).  
  MLE, likelihood functions, and the contrast with OLS are course-aligned. The exact `mle()` function call and template structure are implementation-specific.

- **Verified (course-aligned).** Custom GARCH notes explain computing the unconditional variance, iterating conditional variances inside a loop, and maximizing the resulting likelihood (`labs/lab04/s4.R:46`, `labs/lab04/s4.R:49`, `labs/lab04/s4.R:52`, `labs/lab04/s4.R:55`, `labs/lab04/s4.R:62`).  
  This matches the lecture-note treatment of ARCH/GARCH and unconditional variance \(\omega / (1 - \alpha - \beta)\).

- **Partially verified.** `rugarch` comments define the `ugarchspec` components (`variance.model`, `mean.model`, `distribution.model`) and point out built-in plotting options for sigma paths and QQ plots (`labs/lab04/s4.R:65`, `labs/lab04/s4.R:67`, `labs/lab04/s4.R:69`, `labs/lab04/s4.R:72`, `labs/lab04/s4.R:76`, `labs/lab04/s4.R:78`, `labs/lab04/s4.R:80`).  
  The underlying GARCH specification, conditional sigma paths, and QQ diagnostics are course-aligned; the `rugarch` API is package-specific.

- **Verified / partially verified mix.** Further variance-model notes introduce the GJR specification and detail how `out.sample` feeds rolling volatility forecasts, including manual reconstruction of residuals/sigmas and confirmation via `ugarchforecast()` (`labs/lab04/s4.R:82`, `labs/lab04/s4.R:89`, `labs/lab04/s4.R:91`, `labs/lab04/s4.R:95`, `labs/lab04/s4.R:99`, `labs/lab04/s4.R:107`, `labs/lab04/s4.R:112`, `labs/lab04/s4.R:116`).  
  GARCH variants with asymmetric responses and rolling volatility forecasting are course-aligned. The exact `out.sample` and `ugarchforecast()` workflow is implementation-specific.

- **Verified (course-aligned).** Risk commentary illustrates computing sample and GARCH-based VaR at the 10% tail using `qnorm` and the package’s `quantile()` helper (`labs/lab04/s4.R:119`, `labs/lab04/s4.R:121`, `labs/lab04/s4.R:124`, `labs/lab04/s4.R:126`).  
  VaR from normal quantiles and GARCH-based conditional volatility is directly aligned with the risk-measures material.

- **Partially verified.** Optional sections mention fitting ARMA-GARCH structures and including an external regressor (market factor) within the mean equation, pointing to the `mxreg1` coefficient for beta (`labs/lab04/s4.R:135`, `labs/lab04/s4.R:137`, `labs/lab04/s4.R:140`, `labs/lab04/s4.R:143`, `labs/lab04/s4.R:145`, `labs/lab04/s4.R:147`).  
  ARMA-GARCH and mean-equation regressors are course-consistent; the exact parameter naming (`mxreg1`) is package-specific.

## Bottom line

- The **econometric content** of the four lab summaries is **overall well aligned** with the course structure:  
  **Lab 01** = review of statistics and distributions;  
  **Lab 02** = least squares, testing, diagnostics, and systems/HAC ideas;  
  **Lab 03** = time series, unit roots, and forecasting;  
  **Lab 04** = maximum likelihood, ARCH/GARCH, and VaR.

- The main limitation is that several items in your list are **R/package-specific comments** (`xts`, `ggplot2`, `dygraphs`, `forecast`, `lmtest`, `sandwich`, `rugarch`, `mle()`), which are **not documented in the lecture notes/slides**. Those are best marked as **partially verified** rather than fully verified.

- The exact `labs/...` **line references cannot be checked** without the actual lab scripts.
