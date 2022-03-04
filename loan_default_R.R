install.packages("caret")
install.packages("sampler")
install.packages("ggpubr")
install.packages("doParallel")
library(caret)      
library(ggplot2)    
library(tidyverse)   
library(sampler)    
library(ggpubr) 
library(doParallel)

##### Parallel processing #####
detectCores() #determine number of cores on pc
cl <- makeCluster(8) #assign number of cores
registerDoParallel(cl) #start parallel processing
stopCluster(cl) # ONLY run to stop parallel processing

# load the data
loandata <- read.csv("default_loan_data.csv")

# view summary stats/data structure
str(loandata)      
summary(loandata) 
names(loandata)


# convert "default.payment.next.month"(direct variable) to a factor 
loandata$default.payment.next.month <- factor(loandata$default.payment.next.month, 
                                              levels = c(0, 1), 
                                              labels = c("No Default", "Default"))

table(loandata$default.payment.next.month) #count of direct variable
prop.table(table(loandata$default.payment.next.month)
) %>% round(digits = 2) # % of direct variable

# check for missing data
sum(is.na(loandata)) 
colSums(is.na(loandata))

#######
# EDA 
#######

# bar chart - distribution of direct variable
ggplot(data = loandata, aes(x = default.payment.next.month, y = ..count.., 
                            fill = default.payment.next.month)) +
  geom_bar() +
  geom_text(stat='count', aes(label=..count..), vjust=-0.5) +
  #scale_fill_manual(values = c("gray50", "orangered2")) +
  labs(title = "Count of Customers Default/No Default") +
  theme_bw() +
  theme(legend.position = "bottom")

# plot for variable: sex
ggplot(data = loandata, aes(x = as.factor(SEX), group=default.payment.next.month)) +
  geom_bar(aes(y=..prop.., fill = factor(..x..)), stat="count") +
  geom_text(aes(label=scales::percent(..prop..), y=..prop..),
            stat="count", vjust = -.5) +
  
  scale_fill_discrete(labels = c("Male","Female")) +
                                 
  theme(axis.text.x = element_blank()) +
  labs(y= "Percent",x="Sex", fill="Sex") +
  labs(title = "Percentage of Default/No Default by Sex") +
  facet_grid(~default.payment.next.month) +
  scale_y_continuous(labels=scales::percent)



ggplot(data = loandata, aes(x = default.payment.next.month, y = ..count.., 
                            fill = as.factor(SEX))) +
  geom_bar() +
  #scale_fill_manual(values = c("blue", "red")) +
  labs(title = "Proportion of Default by Sex") +
  theme_bw() +
  theme(legend.position = "bottom")

# plot for variable: education
ggplot(data = loandata, aes(x = as.factor(EDUCATION), group=default.payment.next.month)) +
  geom_bar(aes(y=..prop.., fill = factor(..x..)), stat="count") +
  geom_text(aes(label=scales::percent(..prop..), y=..prop..),
            stat="count", vjust = -.5) +
  
  scale_fill_discrete(labels = c("Other(1)", "Graduate School", "University",
                                 "High School","Other(2)","Other(3)","Other(4)")) +
  theme(axis.text.x = element_blank()) +
  labs(y= "Percent",x="Education Level", fill="Education Level") +
  labs(title = "Percentage of Default/No Default by Education Level") +
  facet_grid(~default.payment.next.month) +
  scale_y_continuous(labels=scales::percent)

# plot for variable: marital status
ggplot(data = loandata, aes(x = as.factor(MARRIAGE), group=default.payment.next.month)) +
  geom_bar(aes(y=..prop.., fill = factor(..x..)), stat="count") +
  geom_text(aes(label=scales::percent(..prop..), y=..prop..),
            stat="count", vjust = -.5) +
  scale_fill_discrete(labels = c("Other", "Married", "Single","Divorced")) +
  theme(axis.text.x = element_blank()) +
  labs(y= "Percent",x="Marital Status", fill="Marital Status") +
  labs(title = "Percentage of Default/No Default by Marital Status") +
  facet_grid(~default.payment.next.month) +
  scale_y_continuous(labels=scales::percent)

############### 
# PreProcessing
##############

#sample out size: 3,000 observations (about 10%)
loandatasample <- loandata[sample(1:nrow(loandata),3000, replace = FALSE),]

# calculate the correlation of the continuous variables 
loanCOR <- cor(loandatasample[, -which(colnames(loandatasample
) =="default.payment.next.month")])
# variables with correlation > .90
hc <- findCorrelation(loanCOR, cutoff = 0.9, names = TRUE)

# remove highly correalted variables from the sample size
loandatasample1 <- loandatasample[, -which(colnames(loandatasample) %in% hc)]
names(loandatasample1)

str(loandatasample1)
ncol(loandatasample1) # number of columns: 21

#############################
# Split Train/Test sets
############################
set.seed(123)   
inTraining <- createDataPartition(y = loandatasample1$default.payment.next.month,
                                  p = 0.80,list = FALSE)
trainloan = loandatasample1[inTraining,]
testloan = loandatasample1[-inTraining,]

#################
# Training Models
################

# Define parameters for trainControl  
set.seed(123)
fitControl <- trainControl(method = 'repeatedcv',   
                           number = 10, repeats = 5,
                           returnResamp = "final", verboseIter = FALSE,
                           allowParallel = TRUE) 

###### Random Forest (rf) ######

# mtry: Number of random variables collected at each split

# Define tuning parameters of mtry: number randomly variable selected 
tunegrid_rf = expand.grid(.mtry = (1:12))

#fit model
set.seed(123)
rf_model <- train(default.payment.next.month ~., 
                  data = trainloan, 
                  method = 'rf', 
                  metric = "Accuracy",   
                  tuneGrid = tunegrid_rf, 
                  trControl = fitControl)
rf_model

# predict on test set
pred.rf_model <- predict(rf_model, testloan)
summary(pred.rf_model)

#Accuracy and Kappa performance measurements
results.rf_model <- postResample(pred.rf_model,testloan$default.payment.next.month)
accuracy.rf <- results.rf_model[1]
accuracy.rf
#Accuracy 
#0.8163606
kappa.rf <- results.rf_model[2]
kappa.rf
#Kappa 
#0.39436   

######  Support Vector Machines (svmLinear) ######

# single tuning parameter: C (Cost: cost of constraints violation)

# define the hyperparameters 
hyper_svm <- expand.grid(C=c(1, 20, 50, 100))

# fit model
set.seed(123)
svm_model <- train(default.payment.next.month~.,
                   data = trainloan, 
                   method = "svmLinear", 
                   metric = "Accuracy", 
                   tuneGrid = hyper_svm, 
                   trControl = fitControl)
svm_model

#predict on test set
pred.svm_model <- predict(svm_model, testloan)
summary(pred.svm_model)

#Accuracy and Kappa performance measurements
results.svm_model <- postResample(pred.svm_model,testloan$default.payment.next.month)
accuracy.svm <- results.svm_model[1]
accuracy.svm
#Accuracy 
#0.8063439 
kappa.svm <- results.svm_model[2]
kappa.svm
#Kappa 
#0.4020824   


###### Gradient Boosting (gbm) ######

# There are 4 tuning parameters in the algorithm: 
#   n.tree : boosting iterations
#   interaction.depth : max tree depth
#   shrinkage : (learning rate) control the influence that each model has on the set of the ensemble
#   n.minobsinnode : minimum terminal node size 

# define the hyperparameters
hyper_gbm <- expand.grid(n.trees = c(200, 500, 1000, 2000), 
                         interaction.depth = c(1, 2), 
                         shrinkage = c(0.001, 0.01, 0.1), 
                         n.minobsinnode = c(1, 2, 5, 15))

# fit model
set.seed(123)
gbm_model <- train(default.payment.next.month~.,
                   data = trainloan, 
                   method = "gbm", 
                   metric = "Accuracy", 
                   tuneGrid = hyper_gbm, 
                   trControl = fitControl)

gbm_model

#predict on test set
pred.gbm_model <- predict(gbm_model, testloan)
summary(pred.gbm_model)

#Accuracy and Kappa performance measurements
results.gbm_model <- postResample(pred.gbm_model,testloan$default.payment.next.month)
accuracy.gbm <- results.gbm_model[1]
accuracy.gbm
#Accuracy 
#0.8113523
kappa.gbm <- results.gbm_model[2]
kappa.gbm
#Kappa 
#0.4492928    

###################
# Model Comparison
##################

# list of models
models <- list(RF = rf_model, SVM = svm_model, GBM = gbm_model)
save(models, file = "models.RData")

Resamp_models <- resamples(models)
Resamp_models$values %>% head(10)

# metrics resamples
metric_resamples <- Resamp_models$values %>% 
  gather(key = "model", value = "value", -Resample) %>%
  separate(col = "model", into = c("model", "metric"),
           sep = "~", remove = TRUE)
metric_resamples %>% head()

metric_resamples %>% 
  group_by(model, metric) %>% 
  summarise(mean = mean(value)) %>%
  spread(key = metric, value = mean) %>%
  arrange(desc(Accuracy))


metric_resamples %>% filter(metric == "Accuracy") %>%
  group_by(model) %>%
  mutate(mean = mean(value, na.rm = TRUE)) %>%
  ungroup() %>%
  ggplot(aes(x = reorder(model, mean), y = value, color = model)) +
  geom_boxplot(alpha = 0.6, outlier.shape = NA) +
  geom_jitter(width = 0.1, alpha = 0.6) +
  scale_y_continuous(limits = c(0.7,0.9)) +
  theme_bw() +
  labs(title = "Validation: Accuracy Mean", 
       subtitle = "Models by their Accuracy Mean",
       x = "Model", y = "Accuracy Mean") +
  coord_flip() +
  theme(legend.position = "none")


# Prediction on the test dataset for all models together
prediction <- extractPrediction(models = models, 
                                testX = testloan %>% select(-default.payment.next.month),
                                testY = testloan$default.payment.next.month)
prediction %>% head()

metric_pred <- prediction %>%
  mutate(true = ifelse(obs == pred, TRUE, FALSE)) %>%
  group_by(object, dataType) %>%
  summarise(accuracy = mean(true))

metric_pred %>%
  spread(key = dataType, value = accuracy) %>%
  arrange(desc(Test))

##################################################################################
# The Random Forest model has the higher accuracy and kappa score vs other models
# and thus is the best model at predicting loan default
##################################################################################


########## Visual representation comparing all models for accuracy ############

#assign values for dataframe
Metric_Values <- c(accuracy.rf,kappa.rf,accuracy.svm,kappa.svm,accuracy.gbm,kappa.gbm)
Algorithms <- c(rep("RF",2),rep("SVM",2),rep("GBM",2))
Metrics_ <- c("Accuracy","Kappa","Accuracy","Kappa","Accuracy","Kappa")

#create dataframe
plotdata12 <- data.frame(Metric_Values,Algorithms,Metrics_)
#shorten decimal in column "Metric_Values" to 4 positions
plotdata12[,"Metric_Values"]=round(plotdata12[,"Metric_Values"],4)

#clustered bar chart displaying performance metrics 
ggplot(aes(x=Algorithms, y= Metric_Values, group=Metrics_,fill=Metrics_),data=plotdata12)+
  geom_bar(position="dodge", stat="identity")+
  scale_fill_manual(values = c("#364F6B","#3FC1C9"))+
  geom_text(aes(label=Metric_Values, fontface="bold"),hjust=0.5, vjust=0,position = position_dodge(.9))+
  scale_y_continuous(expand=c(0,0))+
  theme_classic()+
  ylim(0,1)+
  theme(legend.position = "left")+
  ggtitle("Model Comparison by Performance Metrics")+
  theme(plot.title = element_text(hjust=0.5))

