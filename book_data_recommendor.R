#Installing and loading the libraries
setwd("E:/Excelr/recommendation")
install.packages("recommenderlab", dependencies=TRUE)
install.packages("Matrix")
library("recommenderlab")
library(caTools)
#book rating data
book_rate_data <- read.csv("E://Excelr//recommendation//book.csv")
View(book_rate_data)
class(book_rate_data)
book_rate_data <- book_rate_data[-1]
View(book_rate_data)

#metadata about the variable
str(book_rate_data)
table(book_rate_data$Book.Title)
#rating distribution
hist(book_rate_data$Book.Rating)
#the datatype should be realRatingMatrix inorder to build recommendation engine
book_rate_data_matrix <- as(book_rate_data, 'realRatingMatrix')
#Popularity based 
book_recomm_model1 <- Recommender(book_rate_data_matrix, method="POPULAR")
#Predictions for two users 
recommended_items1 <- predict(book_recomm_model1, book_rate_data_matrix[413:414], n=5)
as(recommended_items1, "list")


## Popularity model recommends the same books for all users , we need to improve our model
#using # # Collaborative Filtering
#User Based Collaborative Filtering
book_rate_data <- book_rate_data[-1]
View(book_rate_data)
book_recomm_model2 <- Recommender(book_rate_data_matrix, method="UBCF")
#Predictions for two users 
recommended_items2 <- predict(book_recomm_model2, book_rate_data_matrix[415:420], n=5)
as(recommended_items2, "list")
                                            
