setwd("E:\\Excelr\\clustering")
library(readr)
library(readxl)
crime<-read.csv("crime_data.csv",1)
View(crime)
attach(crime)
crime$X <- as.numeric(X)
View(crime)
normalized <- scale(crime[,2:5])
View(normalized)



wss = (nrow(normalized)-1)*sum(apply(normalized, 2, var))		 # Determine number of clusters by scree-plot 
for (i in 2:8) wss[i] = sum(kmeans(normalized, centers=i)$withinss)
plot(1:8, wss, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares")   # Look for an "elbow" in the scree plot #
title(main = "K-Means Clustering Scree-Plot")
####therefore k value =3 

fit <- kmeans(normalized,3)
str(fit)
table(fit$cluster)
final2<- data.frame(crime, fit$cluster) # append cluster membership
View(final2)

library(data.table)
setcolorder(final2, neworder = c("fit.cluster"))
View(final2)
aggregate(crime[,2:5], by=list(fit$cluster), FUN=mean)
###  Group.1    Murder  Assault UrbanPop     Rape
##1       1 13.937500 243.6250  53.75000   21.41250
##2       2  4.734483 111.8276  64.10345   15.82069
##3       3 10.815385 257.3846  76.00000   33.19231
### in group1 and group 3, the murder and rape rates are comparitively high.