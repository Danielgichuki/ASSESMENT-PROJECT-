# Load necessary libraries
library(tidyverse)     # For data manipulation and visualization
library(caret)         # For model training and evaluation
library(randomForest)  # Random Forest algorithm
library(e1071)         # Support Vector Machine (SVM)
library(MASS)          # Linear Discriminant Analysis (LDA)
library(xgboost)       # XGBoost algorithm
# Load the dataset
df <- read.csv("city_day.csv")

# View the structure of the data
str(df)

# Check the first few rows of the dataset
head(df)
# Data Cleaning - Remove rows with missing data
df_clean <- df %>%
  na.omit()

# Confirm that there are no missing values remaining
sum(is.na(df_clean))

# Quick Summary of Data after Cleaning
summary(df_clean)
# Feature Engineering - Extract month and year from 'Date' column
df_clean$Date <- as.Date(df_clean$Date, format="%Y-%m-%d")
df_clean$month <- format(df_clean$Date, "%m")
df_clean$year <- format(df_clean$Date, "%Y")

# Optional: Convert AQI into categorical values if you want to do classification
df_clean$AQI_category <- cut(df_clean$AQI,
                             breaks = c(0, 50, 100, 200, 300, 500),
                             labels = c("Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Hazardous"))
# Data Transformation - Scaling the numerical features
num_features <- c("PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "AQI")
df_clean[num_features] <- scale(df_clean[num_features])
# Feature Selection - Correlation Matrix
cor_matrix <- cor(df_clean[num_features])
print(cor_matrix)

# Optional: Use Random Forest to select important features
set.seed(123)
rf_model <- randomForest(AQI ~ PM2.5 + PM10 + NO + NO2 + NOx + NH3 + CO + SO2 + O3, data=df_clean, importance=TRUE)
importance(rf_model)
varImpPlot(rf_model)
# Benchmark Model - Linear Regression
set.seed(123)
trainIndex <- createDataPartition(df_clean$AQI, p = 0.8, list = FALSE)
train <- df_clean[trainIndex,]
test <- df_clean[-trainIndex,]

# Linear Regression
lm_model <- lm(AQI ~ PM2.5 + PM10 + NO + NO2 + NOx + NH3 + CO + SO2 + O3, data = train)
lm_pred <- predict(lm_model, test)

# Evaluate the linear regression model
lm_rmse <- sqrt(mean((lm_pred - test$AQI)^2))
print(paste("Linear Regression RMSE: ", lm_rmse))
# Random Forest Model
set.seed(123)
rf_model <- randomForest(AQI ~ PM2.5 + PM10 + NO + NO2 + NOx + NH3 + CO + SO2 + O3, data = train)
rf_pred <- predict(rf_model, test)

# Evaluate Random Forest Model
rf_rmse <- sqrt(mean((rf_pred - test$AQI)^2))
print(paste("Random Forest RMSE: ", rf_rmse))
# Support Vector Machine (SVM)
train <- na.omit(train)
test <- na.omit(test)
set.seed(123)
svm_model <- svm(AQI ~ PM2.5 + PM10 + NO + NO2 + NOx + NH3 + CO + SO2 + O3, data = train)
svm_pred <- predict(svm_model, test)

# Evaluate SVM Model
svm_rmse <- sqrt(mean((svm_pred - test$AQI)^2))
print(paste("SVM RMSE: ", svm_rmse))
# XGBoost Model
set.seed(123)
train_matrix <- xgb.DMatrix(data = as.matrix(train[num_features]), label = train$AQI)
test_matrix <- xgb.DMatrix(data = as.matrix(test[num_features]))

xgb_model <- xgboost(data = train_matrix, max_depth = 6, nrounds = 100, objective = "reg:squarederror", verbose = 0)
xgb_pred <- predict(xgb_model, test_matrix)

# Evaluate XGBoost Model
xgb_rmse <- sqrt(mean((xgb_pred - test$AQI)^2))
print(paste("XGBoost RMSE: ", xgb_rmse))
# RMSE Comparison
rmse_results <- data.frame(
  Model = c("Linear Regression", "Random Forest", "SVM", "XGBoost"),
  RMSE = c(lm_rmse, rf_rmse, svm_rmse, xgb_rmse)
)

print(rmse_results)
