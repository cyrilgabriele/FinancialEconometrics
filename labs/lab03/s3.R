library(ggplot2)
library(xts)
library(forecast)
library(urca) #for unit root tests

data <- read.table("s3_data.txt",header = TRUE,sep="\t")
date <- as.Date(data$date, format = "%Y-%m-%d")
ts <- xts(x = data[-1], order.by = as.Date(date))
remove(data,date)

ts$sp500 <- 100*diff(log(ts$SP500))-ts$rf

ts <- ts[-1,] #leave out the first observation (no SP500 return data:"NA")


#Q2 ACF to check for a unit root 
SP500 <- ts$SP500
sp500 <- ts$sp500
autoplot(SP500) #series with trends or seasonality => non-stationary
autoplot(sp500)
ggAcf(SP500,lag.max=10) #strong serial dependence with a slow decay => non-stationary
ggAcf(sp500,lag.max=10)
ggPacf(SP500,lag.max=10) #strong serial dependence with a slow decay => non-stationary

#Explanation: ggAcf(): This function generates an autocorrelation plot (ACF) for 
#the given time series data. Autocorrelation measures the correlation between a time 
#series and a lagged version of itself at different lags. The lag.max parameter specifies 
#the maximum lag to be considered in the plot.


#ggPacf(): This function generates a partial autocorrelation plot (PACF) for the 
#given time series data. Partial autocorrelation measures the correlation between 
#a time series and a lagged version of itself, controlling for the intermediate lags. 
#The lag.max parameter specifies the maximum lag to be considered in the plot.







#Q3 Formal unit root tests
#SP500
adf <- ur.df(SP500,type="drift")
summary(adf)
kpss <- ur.kpss(SP500,type="mu")
summary(kpss)
#sp500
adf <- ur.df(sp500,type="drift") 
summary(adf)
kpss <- ur.kpss(sp500,type="mu") 
summary(kpss) 


#Explanation: ADF test H0 = unit root. If the test statistic is less than the critical 
#value at a chosen significance level (e.g., 5%), we fail to reject the null hypothesis. 
#This suggests that the time series has a unit root and is non-stationary.
#If the test statistic is greater than the critical value, we reject the null hypothesis. 
#This indicates that the time series does not have a unit root and is stationary.

#KPSS test H0 = stationary (constant mean, variance and autocovariance). 



#Q4 AR(1) forecast
#Arima() function from the forecast package (very similar to arima() from stats):
fit <- Arima(sp500['/2005-05'], order = c(1,0,0), method = "ML")
fit
a <- fit$coef[1]
mu <- fit$coef[2]
#one step-ahead forecast (the date in the result should in fact be interpreted as 2005-06, 
#because we predict out-of-sample, using the last value of sp500):
pred <- mu + a * (sp500['2005-05']- mu)
pred
CI95.h <- pred + qnorm(0.975)*sqrt(fit$sigma2) #could also use 1.96 instead of qnorm()
CI95.l <- pred - qnorm(0.975)*sqrt(fit$sigma2)
forecast(fit, h=1) #same results using forecast(). h = nb of steps ahead
forecast(fit, h=1)$mean #to only extract the forecast

#Explanation: Arima order is (p,i,q). P refers to AR, i to I, and q to MA. 
#qnorm() is a function in R that calculates quantiles of the normal distribution







#Q5 AR(1) model checking
checkresiduals(fit)
#fitted vs actual returns:
fitted <- xts(fitted(fit),order.by=index(sp500['/2005-05'])) #converting the fitted values to an xts object
series <- cbind(fitted,sp500['/2005-05'])
autoplot(series,facets=NULL) #conditional volatility (heteroskedasticity) hard to capture with an AR(1)!










#Q6 Multiple AR(1) forecasts
ts$f_ar1 <- NA #creating forecast variable
for (i in nrow(ts['/2005-06']):nrow(ts))
{fit <- Arima(sp500[1:i-1], order= c(1,0,0), method="ML")
ts$f_ar1[i] <- forecast(fit,1)$mean}

#Explanation: the forecast is always for the time point t+1. 





#Q7 Plot
series <- cbind(ts$sp500,ts$f_ar1)
autoplot(series,facets=NULL)








#Q8 RMSE
fe_ar1 <- sp500-ts$f_ar1 #forecast error series
RMSE_ar1 <- sqrt(mean((fe_ar1)^2,na.rm=TRUE))

#Explanation: RMSE provides a measure of the model's prediction accuracy in the 
#same units as the target variable. Lower RMSE values indicate better agreement 
#between predicted and actual values, while higher RMSE values indicate poorer performance.





#Q9 Factor model forecast
X <- cbind(ts$eurusd,ts$VIX)
fit <- lm(sp500['/2005-05'] ~ lag.xts(X['/2005-05']))
summary(fit)
#when using forecast() with an lm() model, you need to specify
#an object with which you make your prediction (X['2005-5'] in our case):
forecast(fit,as.data.frame(X['2005-5']))


#Q10 Factor model: multiple forecasts
ts$f_fac <- NA #creating forecast variable
for (i in nrow(ts['/2005-6']):nrow(ts))
{fit <- lm(sp500[1:i-1] ~ lag.xts(X[1:i-1]))
ts$f_fac[i] <- forecast(fit,as.data.frame(X[i-1]))$mean}

#RMSE
fe_fac <- sp500-ts$f_fac #forecast error series
RMSE_fac <- sqrt(mean((fe_fac)^2,na.rm=TRUE))
#accuracy(as.ts(ts$f_fac),as.ts(ts$sp500))






#Q11 Diebold-Mariano test
dm.test(fe_ar1, fe_fac, alternative = "greater") #DM test with quadratic loss


#Explanation: DM test is used to compare the forecast accuracy of two competing 
#forecasting models. It evaluates if one significantly outperforms the otehr in terms 
# of accuracy. 

#Alternative can be = greater, less or two.sided. 




#Q12 R2oos
R2oos <- 1 - RMSE_fac^2/RMSE_ar1^2
#the R2oos is negative, indicating that the fac model cannot beat the ar1. 
#This is confirmed by the one-sided Diebold-Mariano test, where we fail to reject the null.

