# Convert Species column to a factor (if not already)
iris$Species <- as.factor(iris$Species)

# Split data into training (80%) and testing (20%) sets
set.seed(123)  # Set seed for reproducibility
trainIndex <- createDataPartition(iris$Species, p = 0.8, list = FALSE)
trainData <- iris[trainIndex, ]
testData <- iris[-trainIndex, ]

# Check data distribution
table(trainData$Species)
table(testData$Species)

# Train a logistic regression model
logit_model <- train(Species ~ ., data = trainData, method = "multinom", trControl = trainControl(method = "cv", number = 5))

# Model summary
summary(logit_model)

# Train a decision tree model
rf_model <- randomForest(Species ~ ., data = trainData, ntree = 100)

# Print model summary
print(rf_model)

# Predict on test data
logit_preds <- predict(logit_model, testData)

# Confusion Matrix
confusionMatrix(logit_preds, testData$Species)

# Predict on test data
rf_preds <- predict(rf_model, testData)

# Confusion Matrix
confusionMatrix(rf_preds, testData$Species)

# Plot variable importance
varImpPlot(rf_model)

ggplot(testData, aes(x = Sepal.Length, y = Sepal.Width, color = rf_preds)) +
  geom_point(size = 3) +
  labs(title = "Predictions by Random Forest", x = "Sepal Length", y = "Sepal Width") +
  theme_minimal()
s
