
# Q1
# setwd("") #set your working directory using the "Session" menu

##Explanation: 
#You can set a working directory either by using the setwd("") function, or, alternatively, 
#at the top by clicking on Session --> Set Working Directory --> To Source File (or to Files Pane) directory. 

#This serves the purpose of defining the default location where R will look for files to import and save. 
#For example: in Q3 we read from a .txt file without specifying where the file is located.


#Q2
library(ggplot2) #package needed for nice plots
library(forecast) #package used to plot our histogram
# library(qqplotr) #package used to plot our QQ-plot
library(xts) #package needed in order to create/use time series objects
library(fBasics) #package for detailed summary statistics
library(tseries) #package for JB test

#Explanation: You use the library() to access additional functions that are not included with base R. If you want to use a library for the very first time 
#on a given computer, you have to run install.packages("xyz"). 


#Q3
data <- read.table("s1_data.txt",header = TRUE,sep= "\t") #import the data (a tab delimited txt file with headers)

#Explanation: 
#This code reads data from a file named "s1_data.txt" into a variable called "data", 
#assuming the file has a header row, and uses tabs ("\t") as the separator between columns.

#Q4
data = na.omit(data)
#Let's first get rid of the missing observations. 

class(data) #class or type of an object
str(data) #evaluate the object structure and the data types
date <- as.Date(data$date, format = "%d/%m/%Y" ) # $ sign is used to extract a variable/column
ts <- xts(x = data[-1], order.by = date) #we leave out the first column of data which is the date ([-1])
remove(data,date) #we don't need this data anymore (we have it in ts)
start(ts) #start date of our time series
end(ts) #end date of our time series
periodicity(ts) #tells us the periodicity/frequency (daily, weekly, monthly) => ts = monthly


#Q5 plot of SP500 & USDMXN
series <- cbind(ts$SP500,ts$USDMXN) #cbind() is used to combine vectors
#autoplot() returns a time series plot that can later be customized using the ggplot grammar
autoplot(series) 
plot1 <- autoplot(series, facets = NULL) #we need to specify "facets = NULL" to draw on a single axis
plot1 
plot2 <- plot1 + labs(x="Time (year)", y="Index value") + theme_bw() #if we want to change the default theme_gray
plot2

#other option: use ggplot()
plot <- ggplot(ts, aes(x=time(ts))) + 
  geom_line(aes(y=SP500,colour = "SP500")) +
  geom_line(aes(y=USDMXN,colour = "USDMXN")) + 
  scale_color_discrete(name="Legend") +
  labs(x="Time (year)", y="Index value") + 
  theme_bw()
plot

#Explanation: ggplot2 and autoplot are both functions used for creating plots in R, 
#but they are associated with different packages and have different capabilities.

#ggplot2 requires users to specify plot aesthetics and layers explicitly, offering a great deal of flexibility but requiring more detailed coding.
#autoplot is designed to automatically generate plots for specific classes of objects, such as time series, regression models, and other statistical outputs.
#You can use either for plotting USDMXN and SP500.

#Since SP500 and USDMXN have such different scales on the y-axis, you could think about plotting them separately instead. You would then run: 
#autoplot(ts#SP500)
#and separately: 
#autoplot(ts$USDMXN)
#You can see that the y-axis scale for USDMXN became much smaller. 


#Q6 logreturns (in %)
ts$sp500 <- 100*diff(log(ts$SP500))
ts$usdmxn <- 100*diff(log(ts$USDMXN))
ts$aapl <- 100*diff(log(ts$AAPL))

#The above calculated the percentage change in the logarithm of any given stock, and then multiplies it by 100.
#Specifically: log(ts#SP500) calculates the natural log of SP500. Diff() computes the difference between consecutive elements of the time series. 
#100*.... multiplies the results obtained, effectively converting the differences to percentage changes. ts$sp500 <- ... assigne the calculated result to a new column called sp500 
#in the ts dataframe. Here, we use lower case letters (sp500) to denote returns and upper case letters (SP500) to denote price levels. 

#Q7 subsample
ts <- ts['2011-01/']

#Explanation: This will create a subsample that starts at 2011-01 and goes until the end of the time series. 
#If we had instead written ts['/2011-01'], (note the placement of the /) this would have started at the first available date in the original date in the time series 
#and stopped at 2011-01. 


#Q8 deannualizing rf
ts$RF <- ts$USRF/12

#Explanation: since the risk-free  rate is annualized, we divide by 12 to obtain monthly values. 


#Q9 Sharpe ratios
ret <- cbind(ts$sp500,ts$usdmxn,ts$aapl,ts$RF) #combining several vectors with cbind()
SR <- data.frame(0,0,0) #initializing our dataframe for our SRs
colnames(SR) <- colnames(ret[,1:3]) #give names to SR columns
for (i in 1:3)  SR[,i] <- mean(ret[,i]-ts$RF)/sd(ret[,i]) #we select only the first three columns of ret (1:3)
SR

#Explanation: The for loop calculates the Sharpe ratio for each asset using the formula: 
#(mean(return - risk-free rate)) / (standard deviation of return). 
#It iterates over the first three columns of 'ret', calculates the Sharpe ratio for each column, 
#and assigns the result to the corresponding column in the 'SR' data frame.
#The Sharpe ratio measures the risk-adjusted return of an investment or portfolio, 
#with higher values indicating better risk-adjusted performance.


#Q10 correlation matrix
cor(ret)

#Explanation: calculates the correlation matrix for the data stored in the 'ret' object. 
#A correlation of an asset with itself is always 1. It can take any value from -1 (perfect negative correlation), to 1 (perfect correlation). A value of 0 indicates
#no correlation between the two assets. 


#Q11 t-test for sp500
t.test(as.vector(ts$sp500), alternative="two.sided",
       mu=0,conf.level=0.99)

#Explanation: The code conducts a two-sample t-test on the 'sp500' data with a two-sided alternative hypothesis, 
#assuming a hypothesized mean difference of 0, and calculates a 99% confidence interval for the difference between 
#the sample means. We evaluate the statistical significance of the test by comparing the p-value obtained from the 
#test to the chosen significance level (alpha). For example, if alpha is set to 0.01, we reject the null hypothesis 
#if the p-value is smaller than 0.01. Rejecting the null hypothesis indicates that there is sufficient evidence to 
#conclude that the population means of the two samples are different, based on the chosen significance level.


#if we want to check manually...
T <- nrow(ts$sp500)
se <- sd(ts$sp500)/sqrt(T)
xbar <- mean(ts$sp500)
tstat <- xbar/se
qt <- qt(.995, df=T-1)  #quantile for the t-distribution (99% two-tailed test => 99.5% quantile)
ci_hi <- xbar + qt*se #we could also have used 2.58 (the normal quantile) instead of qt, which is a good approximation
ci_lo <- xbar - qt*se




#Q12 t-test for sp500 and RF
exret <- ts$sp500-ts$RF
t.test(as.vector(exret), alternative="two.sided",
       mu=0,conf.level=0.99)

#Explanation: We now test if the SP500 returns are different from risk-free returns at the 1% significance level. One way to do this is to first compute excess returns, 
#and then check if excess returns are different from 0. We can do that as seen above. Another alternative is to use a paired sample t-test, as seen below. 
#Here, instead of our mu=0, we give as input the second sample (risk-free rate).
#Both tests yield the same result. 

#option 2 (yielding the same result): paired sample t-test:
t.test(as.vector(ts$sp500),as.vector(ts$RF),alternative="two.sided",
       conf.level=0.99,paired=TRUE)


#Q13 summary statistics + p-value
basicStats(ret) #summary statistics from fBasics
#Note that the kurtosis computed by basicStats() is in fact the excess kurtosis. Add 3 to get the real kurtosis.
pval <- data.frame(0,0,0)
colnames(pval) <- colnames(ret[,1:3])
for (i in 1:3) pval[,i] <- pnorm(min(ret[,i]), 
                                 mean=mean(ret[,i]),sd=sd(ret[,i]))
pval

#explanation: some basic statistics. Skewness measures the asymmetry of the distribution of a dataset around its mean, 
#where positive skewness indicates a longer tail on the right side and negative skewness indicates a longer tail on the left side. 
#Kurtosis measures the peakedness or flatness of the distribution, with higher kurtosis indicating sharper or more peaked distributions, 
#and lower kurtosis indicating flatter distributions.

#In a normal distribution, the skewness is close to 0, indicating symmetry, and the kurtosis is close to 3, which represents the level of 
#peakedness similar to that of a normal distribution. Many statistical packages report excess kurtosis (which is kurtosis - 3), and hence you'd expect a value of 0 
#in a normal distribution. However, slight deviations from these values are generally acceptable, especially in large datasets.

#Q14 testing normality
usdmxn <- ts$usdmxn
#easy way to plot a histogram: gghistogram() from the forecast package.
hist <- gghistogram(usdmxn, add.normal = T)  
hist
jarque.bera.test(usdmxn) #Jarque-Bera test (tseries package)

#Explanation: JB-test h0 = data are normally distributed. H1 = data are not normally distributed. t-stat: follows chi-square distr. p-value: indicated 
#the probability of observing the test statistic or a more extreme value, assuming that the null is true. Interpretation: 
#if p-val. is less than a chosen significance level (e.g. 0.05). we reject the null hypothesis and assume that the data do not come from a normally distributed 
#population. If p-val > sig. level, we fail to reject the null hypothesis. 

# Q15 Normal QQ plot
usdmxn_vec <- as.numeric(na.omit(ts$usdmxn))

gg <- ggplot(data.frame(usdmxn = usdmxn_vec), aes(sample = usdmxn)) +
  stat_qq() +
  stat_qq_line() +
  labs(x = "Theoretical Quantiles", y = "Sample Quantiles") +
  theme_bw()
gg
# qqnorm(smi) 
# qqline(smi) 

#Explanation: A QQ plot, or quantile-quantile plot, is a graphical tool used to assess whether 
#a dataset follows a certain theoretical distribution, such as the normal distribution. In a QQ plot:
#The x-axis represents the quantiles of the theoretical distribution.The y-axis represents the quantiles of the observed data.
#Each point on the plot represents a quantile pair: one from the theoretical distribution and one from the observed data.
#If the observed data follow the theoretical distribution closely, the points on the Q-Q plot will fall approximately along a straight line.
#Departures from this straight line indicate deviations from the assumed distribution, helping to identify potential discrepancies 
#between the observed data and the theoretical distribution.

#Q16 Scatter plot
scplot <- ggplot(ret, aes(x=usdmxn,y=sp500)) +
  geom_vline(xintercept = 0) + geom_hline(yintercept = 0)
scplot
scplot1 <- scplot + geom_point()
scplot1
scplot2 <- scplot1 + geom_smooth(method="lm")
#By default, geom_smooth(method="lm") plots a 95% confidence level interval around the regression line.
scplot2


#Explanation: geom_vline(...) adds a vertical line at x + 0. geom_hline(...) adds a horizontal line at y = 0. geom_smooth(...) adds a linear regression line to the plot. 

