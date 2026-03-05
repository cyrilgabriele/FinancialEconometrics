setwd("/Users/cyrilgabriele/Documents/School/00_Courses/03_FinancialEconometrics/02_Exercises/code/FinancialEconometrics/labs/lab02")

library(xts) 
library(ggplot2)
library(dygraphs) #for interactive charts
library(tseries) #for JB test
library(forecast) #for plotting: gghistogram(), ggAcf(), checkresiduals()
library(lmtest) #to perform regressions with HAC errors: coeftest()
library(sandwich) #to calculate HAC standard errors: NeweyWest()
# library(systemfit) #to perform seemingly unrelated regressions (SUR)

data <- read.table("s2_data.txt",header = TRUE,sep= "\t") 
ts <- xts(x = data[-1], order.by = as.Date(data$date,format="%d/%m/%Y"))
remove(data)

ts$dist <- 100*(ts$DIST/lag(ts$DIST)-1)-ts$rf
ts$mac <- 100*(ts$MAC/lag(ts$MAC)-1)-ts$rf

ts <- ts[-1,] #leave out the first observation (no return data for the computed returns:"NA")

#Q1 Scatter plot
scplot <- ggplot(ts, aes(mktrf,mac)) + geom_point() + 
  geom_vline(xintercept=0) + geom_hline(yintercept=0) +
  geom_smooth(method="lm") 
scplot

#Explanation: geom_point adds points to the plot. geom_vline adds a vertical line at the xintercept, 
#geom_hline adds a horizontal line at the yintercept, geom_smooth adds a smoothed line (method=lm specifies using a linear regression).


#Q2 OLS model
fit <- lm(mac ~ mktrf, data=ts)
summary(fit)

#Explanation: fitting a linear regression. You can have multiple independent vars by adding a "+". Exmaple: lm(mac ~ x1 + x2 + x3, data=ts) 


#Q3 Sum of squares, R2 and ANOVA table

#R-squared (R2) is a measure of how well the independent variables explain the variability 
#of the dependent variable in a regression model. It ranges from 0 to 1, with higher values 
#indicating a better fit. Adjusted R-squared (Adjusted R2) adjusts R-squared for the number 
#of predictors in the model, penalizing for the inclusion of irrelevant variables. It provides 
#a more accurate assessment of the model's goodness of fit, particularly when comparing models 
#with different numbers of predictors.


res <- as.xts(resid(fit)) #y - yhat
T <- length(res) #No. observations
k <- length(coef(fit)) #No. coefficients
y <- ts$mac
yhat <- fitted(fit)
ybar <- mean(y)

rss <- sum(res^2) #residual sum of squares
ess <- sum((yhat - ybar)^2) #explained sum of squares 
tss <- sum((y - ybar)^2) #total sum of squares = ess + rss

#'Multiple R-squared' in summary(fit):
R2 <- 1 - (rss/T)/(tss/T) #1 - v(res)/v(y)
#We just used here maximum likelihood estimates of variance (i.e. dividing by T)
#Replacing the variances with their unbiased estimates gives the Rbar2:
#'Adjusted R-squared' in summary(fit):
Rbar2 <- 1 - (rss/(T-k))/(tss/(T-1)) 
Rb2 <- 1 - (1-R2)*(T-1)/(T-k) #other way to calculate it (as in the Lecture notes)
#'Residual standard error' in summary(fit)
sqrt(rss/(T-k))
#ANOVA table
anova(fit)

#Q4: infer from regression output

#Q5 Residuals plots
autoplot(res)
autoplot(res^2)
#Interactive visualizations of a time series:
dygraph(res)
dygraph(res^2)

#Q6 Normality Test
jarque.bera.test(res)
gghistogram(res,add.normal=TRUE)

#Explanation: JB test is a goodness-of-fit test that checks if a given sample comes from a normal distribution. A p-value below your chosen significance level means
#that you'd reject the null hypothesis that the data comes from a normal distribution. 

#Q7 Autocorrelation
ggAcf(res, lag.max = 10) + ggtitle("Correlogram of residuals")
#The pair of dashed lines represents confidence bounds of 2 standard errors (± 2/sqrt(T)).
#Note that a 95% confidence interval would be ± 1.96/sqrt(T).
#The ACF is only slighlty significant at lag 1

#Ljung-Box test:
Box.test(res, lag = 10, type="Lj")
#Explanation: Ljung-box test assesses if there is autocorrelation in the residuals. The p-value is large => the residuals are not distinguishable from a white noise

#(Q5+6+7)
checkresiduals(fit,test=FALSE)

#Q8 White test
#Breusch-Pagan test
bptest(fit)
#Breusch-Pagan test is different from White test in that there is no squared regressor or 
#cross-product of the regressors (just the regressors themselves). So we add them back to get White test:
bptest(fit, ~ mktrf + I(mktrf^2), data = ts)
#The use of the I operator specifies that the ^ exponentiation is a mathematical operator, not the ^ formula operator (factor crossing).

#Q9 HAC standard errors
m <- floor(0.75*T^(1/3)) #rule of thumb for nb of lags
#coeftest function (from the lmtest package), where we can specify our var-cov matrix (vcov):
fit.rob <- coeftest(fit, vcov = NeweyWest(fit, lag=m, prewhite = F, adjust = T)) 
fit.rob

#Q10 Seemingly unrelated regressions
r1 <- ts$mac ~ ts$mktrf + ts$smb + ts$hml 
r2 <- ts$dist ~ ts$mktrf + ts$smb + ts$hml
eqSystem <- list(macreg = r1,distreg = r2)
fitsur <- systemfit(eqSystem, "SUR")
summary(fitsur)

#Explanation: Here we perform Seemingly Unrelated Regressions (SUR). SUR is a method used when 
#there are multiple regression equations with correlated error terms. The aim is to 
#estimate the coefficients of each equation simultaneously while accounting for the 
#correlation between the error terms across equations.

#Q11
#First, define the restriction for the hypothesis test. Here, the restriction is if both intercepts are jointly equal to zero. 
restriction <- c("macreg_(Intercept)","distreg_(Intercept)")
#Then, do a hyothesis test using linearHypothesis() from the "car" package. This is using the Chi-squared test. 
linearHypothesis(fitsur, restriction, test = "Chisq")
#Finally, this is to calculate our critical value for a 5% SL. and 2 degrees of freedom.
qchisq(0.95, 2)
