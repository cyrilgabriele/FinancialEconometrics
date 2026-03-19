
library(ggplot2)
library(xts)
library(forecast) 
library(stats4) #for mle()
library(rugarch) #for GARCH models

data <- read.table("s4_data.txt",header = TRUE,sep="\t")
ts <- xts(x = data[-1], order.by = as.Date(data$date))
remove(data)

#Q1 MLE (normal distribution)
negLL <- function(b0, b1, s) {
  logdens = dnorm(ts$enrgy, b0 + b1*ts$mkt, abs(s), log = TRUE)
  -sum(logdens)
}
#The mle() function minimizes the negative of the log-likelihood (equivalent to maximizing the log-likelihood)
fit.norm <- mle(negLL, start = list(b0=0.5, b1=1, s=0.85)) 
summary(fit.norm)

# #AR(1) by MLE: example
# r <- ts$enrgy
# T <- nrow(r)
# negLL <- function(a, mu, s) {
#   logdens = dnorm(r[2:T], mu + a*(r[1:T-1]-mu), abs(s), log = TRUE)
#   -sum(logdens)
# }
# fit.ar1 <- mle(negLL, start = list(a=1, mu=1, s=1))
# summary(fit.ar1)
# 
# Arima(r, c(1,0,0), method="ML") #with Arima()

#Q2 OLS
fit.ols <- lm(enrgy ~ mkt, data=ts)
summary(fit.ols)

#Q3 MLE (t-distribution)
negLL <- function(b0, b1, v) {
  logdens = dt(ts$enrgy - b0 - b1*ts$mkt, abs(v), log = TRUE)
  -sum(logdens)
}
fit.t <- mle(negLL, start = list(b0=0.2, b1=1, v=5))
summary(fit.t)

#Q4 GARCH(1,1): the long way

#creating our function
garch_negLL <- function(mu, omega, alpha, beta){
  #loglik and s2 initialization
  loglik=0
  s2=omega/(1-alpha-beta) #unconditional variance 
  # Loop
  r = coredata(ts$enrgy) #notation
  for (i in 2:length(r)){
    s2=omega+alpha*(r[i-1]-mu)^2+beta*s2
    loglik=loglik+dnorm(r[i],mu,sqrt(abs(s2)),log=TRUE)
  }
  -loglik
}
#applying MLE
fit.garch11 <- mle(garch_negLL, start = list(mu=0, omega=0.05, alpha=0.4, beta=0.4)) #takes a few seconds to run
summary(fit.garch11)

#Q5 GARCH(1,1) using rugarch
spec1 <- ugarchspec(variance.model = list(model="sGARCH",garchOrder=c(1,1)), 
                   mean.model = list(armaOrder=c(0,0)), 
                   distribution.model = "norm")
fit.g11 <- ugarchfit(spec = spec1, data = ts$enrgy)
fit.g11

#Q6 Plots
sigma <- sigma(fit.g11)
autoplot(sigma)

windows()
plot(fit.g11, which=3) #conditional sd (sigma)
plot(fit.g11, which=9) #qqplot of standardized residuals
windows()
plot(fit.g11, which="all") #overview of the other plots available

#Q7 GJR
spec2 <- ugarchspec(variance.model = list(model="gjrGARCH",garchOrder=c(1,1)),
                    mean.model = list(armaOrder=c(0,0)),
                    distribution.model = "norm")
fit.gjr <- ugarchfit(spec = spec2, data = ts$enrgy)
fit.gjr

#Q8 out-of-sample forecasts of the volatility

#The "out.sample" option takes out the last two observations, leaving us with
#a sample ending at time T - 2 (where T is the number of observations in ts$enrgy)
newfit <- ugarchfit(spec = spec1, data = ts$enrgy, out.sample=2) 

#Estimated parameters from newfit
coef <- coef(newfit)
# In our model, the mean is constant so the forecast
# of ts$enrgy for any period is simply mu
mu <- as.numeric(coef["mu"])
#last two observations of ts$enrgy 
ret <- as.numeric(last(ts$enrgy, n=2))
#residuals
uTm2 <- as.numeric(last(residuals(newfit)))
uTm1 <- ret[1]-mu
uT <- ret[2]-mu
#sigmas
sTm2 <- as.numeric(last(sigma(newfit)))
sigfun <- function(u_0,s_0) {
  s_1 = sqrt(coef["omega"] + coef["alpha1"]*u_0^2 + coef["beta1"]*s_0^2)
  return(s_1)
}
sTm1 <- as.numeric(sigfun(uTm2,sTm2))
sT <- as.numeric(sigfun(uTm1,sTm1))
sTp1 <- as.numeric(sigfun(uT,sT))
#Obtain the same results with the ugarchforecast() function
fitfor <- ugarchforecast(newfit, n.ahead = 1, n.roll = 2, out.sample = 2)
sigma(fitfor)

#Q9 VaR
#sample VaR
r <- ts$enrgy[1:(nrow(ts)-2)]
VaR_sample <- mean(r) + sd(r)*qnorm(0.10)
#out-of-sample VaRs (based on GARCH)
VaR_oosample <- mu +sigma(fitfor)*qnorm(0.10)
#or using the built-in function from rugarch
quantile(fitfor,0.10)








#OPTIONAL: ARMA(1,1) with GARCH errors
auto.arima(ts$enrgy) #see ?auto.arima. We get an ARMA(1,1)
spec <- ugarchspec(variance.model = list(model="sGARCH",garchOrder=c(1, 1)),
                   mean.model = list(armaOrder=c(1, 1)), #ARMA(1,1)
                   distribution.model = "norm")
fit.arma.g11 <- ugarchfit(spec = spec, data = ts$enrgy)
fit.arma.g11

#Market model with GARCH errors
spec <- ugarchspec(variance.model = list(model="sGARCH",garchOrder=c(1, 1)),
                   mean.model = list(armaOrder=c(0, 0), external.regressors = ts$mktrf), #explanatory variable: ts$mktrf
                   distribution.model = "norm")
fit.reg.g11 <- ugarchfit(spec = spec, data = ts$enrgy)
fit.reg.g11 #beta is the coefficient on the line 'mxreg1' (alpha is on the line 'mu')

