#Load Pacman and required libraries
if("pacman" %in% rownames(installed.packages())==FALSE){install.packages("pacman")}

pacman::p_load("xgboost","dplyr","caret",
               "ROCR","lift","glmnet","MASS","e1071"
               ,"mice","partykit","rpart","randomForest","dplyr"   
               ,"lubridate","ROSE","smotefamily","DMwR","beepr","MLmetrics",
               "caretEnsemble","mlbench","gbm","gdata")
df_train <- read.csv("diabetes_train.csv")
df_test <- read.csv(("diabetes_test.csv"))

# Checking for any missing data
lapply(df_train, function(x) sum(is.na(x)))
lapply(df_test, function(x) sum(is.na(x)))
# No missing data

# Adding diabetes column to the test data before joining both train and test datasets
df_test$diabetes <- 0

# Joining or combining both datasets
df <- rbind(df_train,df_test)

str(df)
head(df)

# Checking for correlation among the features
library(corrplot)
# Grab only numeric columns
num.cols <- sapply(df_train, is.numeric)
# Filter to numeric columns for correlation
cor.data <- cor(df_train[,num.cols])
cor.data
# Plotting the correlation
par(mfrow=c(1,1))
corrplot(cor.data)

# Changing diabetes to categorical in both combined and train data
df$diabetes <- as.factor(df$diabetes)
df_train$diabetes <- as.factor(df_train$diabetes)
table(df_train$diabetes)
# Target variable is imbalanced with 34.5% having diabetes.

# Feature engineering
df$obese_highglucose <- ifelse((df$BMI > 30 & df$plasma_glucose>126),1,0)
df$obese_highglucose <- as.factor(df$obese_highglucose)
df$pressure_insulin <- ifelse((df$DBP>90 & df$serum_insulin>126),1,0)
df$pressure_insulin <- as.factor(df$pressure_insulin)

# Splitting into the training set and prediction set based on ID
training.set <- subset(df,Id <= 576)
predict.set <- subset(df,Id > 576)

# Removing the ID column
training.set <- training.set[,-1]

# Splitting the training set into train and test(validation) in 80:20 ratio
set.seed(42) #set a random number generation seed to ensure that the split is the same everytime
inTrain <- createDataPartition(y = training.set$diabetes,
                               p = 0.8, list = FALSE)

training <- training.set[ inTrain,]
testing <- training.set[ -inTrain,]

table(training$diabetes)

# Data balancing on the training data
#Smote Balancing
data_balance<-SMOTE(diabetes~.,data=training,perc.over=100)
table(data_balance$diabetes)

#### Random Forest ####

model_forest <- randomForest(diabetes~ ., data=training, 
                             type="classification",
                             importance=TRUE,
                             ntree = 100,           # hyperparameter: number of trees in the forest
                             mtry = 5,             # hyperparameter: number of random columns to grow each tree
                             nodesize = 10,         # hyperparameter: min number of datapoints on the leaf of each tree
                             maxnodes = 10,         # hyperparameter: maximum number of leafs of a tree
                             cutoff = c(0.5, 0.5)   # hyperparameter: how the voting works; (0.5, 0.5) means majority vote
) 
plot(model_forest)  
varImpPlot(model_forest)

###Finding predicitons: probabilities and classification
forest_probabilities<-predict(model_forest,newdata=testing,type="prob") 
forest_classification<-rep("1",115)
forest_classification[forest_probabilities[,2]<0.5]="0" 
forest_classification<-as.factor(forest_classification)

confusionMatrix(forest_classification,testing$diabetes, positive="1") 

####ROC Curve
forest_ROC_prediction <- prediction(forest_probabilities[,2], testing$diabetes) 
forest_ROC <- performance(forest_ROC_prediction,"tpr","fpr") 
plot(forest_ROC) 

####AUC (area under curve)
AUC.tmp <- performance(forest_ROC_prediction,"auc") 
forest_AUC <- as.numeric(AUC.tmp@y.values) 
forest_AUC 

F1_Score(forest_classification,testing$diabetes)

# Predicting for the prediction or test data
forest_probabilities<-predict(model_forest,newdata=predict.set,type="prob") 
forest_classification<-rep("1",192)
forest_classification[forest_probabilities[,2]<0.5]="0" 
forest_classification<-as.factor(forest_classification)

write.csv(forest_classification,"diabetes_Prediction_RF.csv")
#write.table(cbind(predict.set$Id,forest_classification),file = "file_RF.csv",row.names = F,col.names = c("Id","Prediction"))

#Logistic
logistic_fit<-glm(diabetes~.,data=data_balance,family=binomial(link = "logit"))

summary(logistic_fit)

logisitc_AIC_fit<-stepAIC(logistic_fit,direction = "both",trace=1)

summary(logisitc_AIC_fit)

par(mfrow=c(1,4))
plot(logisitc_AIC_fit) 
par(mfrow=c(1,1))

log_predict<-predict(logisitc_AIC_fit,newdata = testing,type="response")
log_classification<-rep("1",115)
log_classification[log_predict<0.65]="0"
log_classification<-as.factor(log_classification)


###Confusion matrix  
confusionMatrix(log_classification,testing$diabetes,positive = "1") #Display confusion matrix

####ROC Curve
logistic_ROC_prediction <- prediction(log_predict, testing$diabetes)
logistic_ROC <- performance(logistic_ROC_prediction,"tpr","fpr") 
plot(logistic_ROC) 

####AUC (area under curve)
auc.tmp <- performance(logistic_ROC_prediction,"auc") 
logistic_auc_testing <- as.numeric(auc.tmp@y.values) 
logistic_auc_testing

#### Lift chart
plotLift(log_predict, testing$diabetes, cumulative = TRUE, n.buckets = 10) 

write.csv(log_classification,'diabetes_Prediction_logistic.csv')

F1_Score(log_classification,testing$diabetes)

## Random Forest gives the best confusion matrix and F1 Score

