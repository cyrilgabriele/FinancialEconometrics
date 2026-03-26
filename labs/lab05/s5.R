library(ggplot2)
library(xts)
library(sandwich) 
library(lmtest) 
library(forecast)
library(boot) #for bootstrapping
library(plm) #for panel data

data <- read.table("s5_data.txt",header = TRUE,sep="\t")
ts <- xts(x = data[-1], order.by = as.Date(data$date))
remove(data)

#excess return
ts$rx_q <- ts$fund4_qoq - ts$rf_qoq
    
#Q1 OLS
fit <- lm(rx_q ~ mktrf_qoq, data=ts)
summary(fit)
autoplot(as.xts(fit$res)) #quick look at the residuals (high autocorrelation)
#robust vcv
m <- floor(0.75*nrow(ts)^(1/3))
coeftest(fit, vcov = NeweyWest(fit, lag=m, prewhite = F, adjust = T)) 

##Q2 Resampling
#the sample() function: example
x <- c(1:8)
sample(x)
sample(x)
sample(x,replace=T) #replacement allowed => an observation can be chosen more than once. needed for bootsrapping
#estimation after resampling
res <- residuals(fit)
# sample(res,replace=T) are the bootstraped residuals (from the in Q1 computed residuals)
y_sim <- fitted(fit) + sample(res,replace=T)
fit.resamp <- lm(y_sim ~ ts$mktrf_qoq)
summary(fit.resamp)

##Q3 Bootstrap
#The boot() function from the boot package requires a user-supplied function (get.coef in our case)
R <- 10000 #nb of replicates
get.coef <- function(x,indx){
  # x: the data to be resampled
  # idx: an index created by the boot() function in order to apply random resampling with replacement
  y_sim <- fitted(fit) + x[indx]
  return(lm(y_sim ~ ts$mktrf_qoq)$coef)
}
boot.out <- boot(res, get.coef, R)
boot.out #bias = (average bootstrapped coef) - (original coef)
         #std. error: bootstrapped standard error
head(boot.out$t) #series of bootstrapped alpha and beta 
apply(boot.out$t,2,mean) #average bootstrapped coef

#Confidence intervals
boot.ci(boot.out,index=1,type="perc",conf=0.95) #CI alpha
boot.ci(boot.out,index=2,type="perc",conf=0.95) #CI beta

# p-values: Two options for calculating a one-sided bootstrapped p-value:
#(1) If H0: beta < 0 => p1 = sum(beta < 0) / R = percentage of betas smaller than 0
#(2) If H0: beta > 0 => p2 = sum(beta > 0) / R = percentage of betas higher than 0
#The two-sided bootstrapped p-value (H0: beta = 0) is twice the smaller of p1 and p2:
pval.alpha <- 2 * sum(boot.out$t[,1] < 0) / R
pval.beta <- 2 * sum(boot.out$t[,2] > 0) / R

##Q4 Block bootstrap
tsboot.out <- tsboot(res, get.coef, R=10000, sim="fixed", l=length(res)^(1/3))
#the option 'sim="fixed"' allows us to specify the length of our blocks (l)
tsboot.out
apply(tsboot.out$t,2,mean) 
boot.ci(tsboot.out,index=1,type="perc",conf=0.95) 
boot.ci(tsboot.out,index=2,type="perc",conf=0.95) 
pval.ts.alpha <- 2 * sum(tsboot.out$t[,1] < 0) / R
pval.ts.beta <- 2 * sum(tsboot.out$t[,2] > 0) / R

#Q5 Importing the data
data <- read.table("s5_data_panel.txt",header = TRUE,sep="\t")
pdata <- pdata.frame(data,c("id","date")) #define individuals (variable "id") and time periods (variable "date")
remove(data)

#Q6 Boxplots
pdata$rx <- pdata$performance - pdata$rf
# windows()
ggplot(data=as.data.frame(pdata), aes(x = id, y = rx)) + geom_boxplot()

#Q7 Pooled OLS
fit.pool <- plm(rx~mktrf, model="pooling", data=pdata)
summary(fit.pool)

#Q8 Pooled OLS - Robust errors
T <- length(unique(pdata$date))
m <- floor(0.75*T^(1/3)) #rule of thumb for nb of lags
coeftest(fit.pool, vcov = vcovSCC(fit.pool, maxlag = m)) #robust (Driscoll-Kraay) errors

#Q9 FE = Fixed Effects (Lecture 12, page 22)
fit.fe<-plm(rx~mktrf, model="within", data=pdata)
summary(fit.fe)

#Q10 Dummy
fit.dum <- lm(rx~mktrf + id -1, data=pdata)
#If a factor variable ("id" in our case) is supplied to lm(), 
#a dummy variable is automatically created and used in the regression.
#"-1" is used to exclude an intercept from our regression.
#Note that if an intercept is included, one of the dummies will automatically be dropped ("id122" in our case) 
summary(fit.dum)
