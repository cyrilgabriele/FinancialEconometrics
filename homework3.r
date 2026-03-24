setwd("/Users/nilufarismayilova/Downloads")

library(ggplot2)
library(xts)
library(forecast)
library(urca) #for unit root tests
data <- read.table("s3_data.txt",header = TRUE,sep="\t")
date <- as.Date(data$date, format = "%Y-%m-%d")
ts <- xts(x = data[-1], order.by = as.Date(date))
remove(data,date)

ts$sp500 <- 100*diff(log(ts$SP500))-ts$rf
ts$financ <- 100*diff(log(ts$FINANC))-ts$rf
ts <- ts[-1,] #leave out the first observation (no SP500/FINANC return data:"NA")

# Filtering from 2000 onward
ts <- ts['2000/'] # keep only 2000 onwards

# Q1: Correlogram on training sample
#----------------------------------------------------
financ <- ts$financ

ggAcf(financ['/2008-12'], lag.max = 15)
ggPacf(financ['/2008-12'], lag.max = 15)

# Q2: AR(p) model selection based on AIC
#----------------------------------------------------
ar.mle(as.ts(financ['/2008-12']))

# Q3: Fit AR(5) on training data and check residuals
#----------------------------------------------------
fit_ar <- Arima(financ['/2008-12'], order = c(5,0,0), method = "ML")
fit_ar
checkresiduals(fit_ar)

# Q4: Forecast 2009:01 with 90% CI
#----------------------------------------------------
forecast(fit_ar, h=1, level=90)

# Q5: Multiple AR(5) forecasts & RMSE
ts$f_ar5 <- NA

for (i in nrow(ts['/2009-01']):nrow(ts))
{fit <- Arima(financ[1:i-1], order=c(5,0,0), method="ML")
ts$f_ar5[i] <- forecast(fit,1)$mean}

# RMSE
fe_ar5 <- financ - ts$f_ar5
RMSE_ar5 <- sqrt(mean((fe_ar5)^2, na.rm=TRUE))
RMSE_ar5

# Q6: Price dividend ratio
#----------------------------------------------------
ts$pd <- log(ts$SP500 / ts$SPDIV)

# ADF test - include constant, no trend
adf_pd <- ur.df(ts$pd['2000/'], type = "drift")
summary(adf_pd)

# KPSS test - include constant
kpss_pd <- ur.kpss(ts$pd['2000/'], type = "mu")
summary(kpss_pd)

# Q7: Factor model with OLS
#----------------------------------------------------
# Q7 - Factor model OLS estimation and forecast for 2009:01
X <- cbind(ts$sp500, ts$eurusd, ts$VIX, ts$pd)

fit_fac <- lm(financ['/2008-12'] ~ lag.xts(X['/2008-12']))
summary(fit_fac)

# Forecast for 2009:01 using 2008:12 predictor values
forecast(fit_fac, as.data.frame(X['2008-12']), level=95)

# Q8: Factor model expanding-window factor-model forecasts
#----------------------------------------------------
ts$f_fac <- NA #creating forecast variable
for (i in nrow(ts['/2009-1']):nrow(ts))
{fit <- lm(financ[1:i-1] ~ lag.xts(X[1:i-1]))
ts$f_fac[i] <- forecast(fit,as.data.frame(X[i-1]))$mean}

#RMSE
fe_fac <- financ-ts$f_fac #forecast error series
RMSE_fac <- sqrt(mean((fe_fac)^2,na.rm=TRUE))
RMSE_fac


# Plot predicted vs realized
series <- cbind(ts$financ, ts$f_fac)
autoplot(series, facets=NULL)

# Q9 - Diebold-Mariano test (two-sided)
dm.test(fe_ar5, fe_fac, alternative = "two.sided")




