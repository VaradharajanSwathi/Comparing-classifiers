#install.packages("rpart")
#install.packages("e1071")
#install.packages("class")
#install.packages(c("neuralnet","ada","randomForest","ipred"))


library(rpart)
library(e1071)
library(class)
library(neuralnet)
library(ada)
library(randomForest)
library(ipred)

args <- commandArgs(TRUE)
dataURL<-as.character(args[1])
header<-as.logical(args[2])
d<-read.csv(dataURL,header = header)
d<- d[rowSums(is.na(d))==0, ]

# create 10 samples
set.seed(123)
for(i in 1:10) {
  cat("Running sample ",i,"\n")
  sampleInstances<-sample(1:nrow(d),size = 0.9*nrow(d))
  trainingData<-d[sampleInstances,]
  testData<-d[-sampleInstances,]
  # which one is the class attribute
  Class<-d[,as.integer(args[3])]
  testClass <- testData[,as.integer(args[3])]

  #Decision trees
  method <- "Decision tree"
  decision_model <- rpart( as.formula(paste0("as.factor(", colnames(d)[as.integer(args[3])], ") ~ .")), data = trainingData, method = 'class', parms = list(split = "information"))
  pruned_decision_tree <- prune(decision_model, cp = 0.010000)
  predict_decision_tree <- predict(pruned_decision_tree,testData,type="class")
  decision_tree_accuracy_table <- table(predict_decision_tree,testClass)
  decision_accuracy = sum(diag(decision_tree_accuracy_table))/ sum(decision_tree_accuracy_table)*100;
  cat("Method = ", method,", accuracy= ", decision_accuracy,"\n")
  
  #Support vector machines
  method <- " Support Vector Machines"
  svm_model <- svm(as.formula(paste0("as.factor(", colnames(d)[as.integer(args[3])], ") ~ .")),data = trainingData)
  predict_svm <- predict(svm_model,testData,type="class")
  svm_accuracy_table <- table(predict_svm,testClass)
  svm_accuracy = sum(diag(svm_accuracy_table))/ sum(svm_accuracy_table)*100;
  cat("Method = ", method,", accuracy= ", svm_accuracy,"\n")
  
  # Naive Bayes
  method <- "Naive Bayes"
  bayes_model <- naiveBayes(as.formula(paste0("as.factor(", colnames(d)[as.integer(args[3])], ") ~ .")),data = trainingData)
  predict_bayes <- predict(bayes_model,testData,type="class")
  bayes_accuracy_table <- table(predict_bayes,testClass)
  bayes_accuracy = sum(diag(bayes_accuracy_table))/ sum(bayes_accuracy_table)*100;
  cat("Method = ", method,", accuracy= ", bayes_accuracy,"\n")
  
  #kNN
  method <- "Knn"
  trainClass<-trainingData[,as.integer(args[3])]
  knn_model <- knn(trainingData, testData,   trainClass, k= 3, prob = TRUE)
  knn_accuracy_table <- table(knn_model,testClass)
  knn_accuracy = sum(diag(knn_accuracy_table))/ sum(knn_accuracy_table)*100;
  cat("Method = ", method,", accuracy= ", knn_accuracy,"\n")
  
  
  #Logistic Regression
  method <- " Logistic Regression"
  logistic_model <- glm(as.formula(paste0("as.factor(", colnames(d)[as.integer(args[3])], ") ~ .")),data = trainingData, family = "binomial")
  predict_logistic <- predict(logistic_model,testData,type="response")
  threshold=0.65
  prediction<-sapply(predict_logistic, FUN=function(x) if (x>threshold) 1 else 0)
  actual<-testData$admit[1:nrow(testData)]
  accuracy <- sum(actual==prediction)/length(actual) * 100
  cat("Method = ", method,", accuracy= ", accuracy,"\n")
  
  
  
  #Random Forest
  method <- "Random Forest"
  random_forest_model <- randomForest(as.formula(paste0("as.factor(", colnames(d)[as.integer(args[3])], ") ~ .")),data = trainingData)
  predict_random_forest <- predict(random_forest_model,testData)
  random_forest_accuracy_table <- table(predict_random_forest,testClass)
  random_forest_accuracy = sum(diag(random_forest_accuracy_table))/ sum(random_forest_accuracy_table)*100;
  cat("Method = ", method,", accuracy= ", random_forest_accuracy,"\n")
  
  
  #Neural network
  method = "Neural networks"
  c<-colnames(d)[as.integer(args[3])]
  formula_neu_net <- as.formula(paste0(c,' ~ ', paste(names(trainingData[!names(trainingData) %in% c]),collapse = '+')))
  neural_network_model <- neuralnet(formula_neu_net, trainingData, hidden = 4, lifesign = "minimal",linear.output = FALSE, threshold = 0.1)
  temp_test <- testData[,-1]
  neural_network_pred <- compute(neural_network_model, temp_test)
  results <- data.frame(actual = testClass, prediction = neural_network_pred$net.result)
  results$prediction <- round(results$prediction)
  accy <- sum(results$actual==results$prediction)/length(results$actual) *100
  cat("Method = ", method,", accuracy= ", accy,"\n")
  
  #Bagging 
  method <- "Bagging"
  bagging_model <- bagging(as.formula(paste0("as.factor(", colnames(d)[as.integer(args[3])], ") ~ .")), data = trainingData, coob=TRUE)
  bagging_pred <- predict(bagging_model, testData)
  bagging_accuracy_table <- table(bagging_pred, testClass)
  bagging_accuracy <-     (sum(diag(bagging_accuracy_table))/sum(bagging_accuracy_table) *100)
  cat("Method = ", method,", accuracy = ", bagging_accuracy,"\n")
  
  #Boosting
  method <- "Boosting"
  boosting_model <- ada( as.formula(paste0("as.factor(", colnames(d)[as.integer(args[3])], ") ~ .")), data = trainingData, iter=20, nu=1, type="discrete")
  predict_boosting <- predict(boosting_model,testData)
  boosting_accuracy <- sum(testClass==predict_boosting)/length(predict_boosting) * 100
  cat("Method = ", method,", accuracy = ", boosting_accuracy,"\n")

  
  
  

  
}

