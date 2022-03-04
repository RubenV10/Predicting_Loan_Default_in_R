install.packages("dplyr")
install.packages("caret")
install.packages("doParallel")
library(dplyr)
library(caret)
library(tidyverse)
library(doParallel)

##### Parallel processing #####
detectCores() #determine number of cores on CPU
cl <- makeCluster(8) #assign number of cores
registerDoParallel(cl) #start parallel processing
stopCluster(cl) # ONLY run to stop parallel processing

# Load data 
data <- read.csv("wifiTrainingData.csv",  stringsAsFactors=FALSE)

################
# Evaluate data
################

head(data)
tail(data)

dim(data) #shape of the data: 19937 rows 529 columns
glimpse(data) #view last columns

colSums(is.na(data)) #check df for NAs: 0 NAs

#################
# PreProcessing
################

ncol(data) #number of columns: 529

#use the nearZeroVar function to remove features with zero variance
zeroVardata <- nearZeroVar(data, saveMetrics = TRUE)
summary(zeroVardata)

# view features with near zero variance
zeroVar <- zeroVardata[zeroVardata[,"zeroVar"] > 0, ]
nzv <- zeroVardata[zeroVardata[,"nzv"] > 0, ]

all_columns <- names(data)
nzv_columns <- row.names(zeroVar)
#new dataset with nzv columns removed
wifidata <- data[ , setdiff(all_columns, nzv_columns )]

ncol(wifidata) #number of columns: 474

#529 - 474 = 55 features were removed

# Consolidate position identifiers to create the LOCATIONID feature
wifidata$LOCATIONID <- as.integer(group_indices(wifidata, BUILDINGID, FLOOR, SPACEID))

#######################
# Split/Train 
######################

# Due to  the size of the dataset, the strategy will be to breakdown the dataset into 
# three smaller datesets by building and then take a sample of data from each building

set.seed(123)

inTraining <- createDataPartition(wifidata$LOCATIONID, p = .75, list = FALSE)
train <- wifidata[inTraining,]
test <- wifidata[-inTraining,]

# Separate the data by BUILDINGID for training  
bldg0 <- filter(train, BUILDINGID == 0) #Building 0
bldg1 <- filter(train, BUILDINGID == 1) #Building 1
bldg2 <- filter(train, BUILDINGID == 2) #Building 2

# Create dfs of the direct variable for each building
bldg0df <- data.frame(bldg0$LOCATIONID, bldg0[,1:465]) # Building 0
bldg1df <- data.frame(bldg1$LOCATIONID, bldg1[,1:465]) # Building 1
bldg2df <- data.frame(bldg2$LOCATIONID, bldg2[,1:465]) # Building 2

# Sample size of 2,000 transactions from each df
bldg0dfsample <- bldg0df[sample(1:nrow(bldg0df), 2000, replace = FALSE),] # Building 0
bldg1dfsample <- bldg1df[sample(1:nrow(bldg1df), 2000, replace = FALSE),] # Building 1
bldg2dfsample <- bldg2df[sample(1:nrow(bldg2df), 2000, replace = FALSE),] # Building 2

# Convert the direct variable: LOCATIONID to a factor
bldg0dfsample$bldg0.LOCATIONID <- as.factor(bldg0dfsample$bldg0.LOCATIONID)
bldg1dfsample$bldg1.LOCATIONID <- as.factor(bldg1dfsample$bldg1.LOCATIONID)
bldg2dfsample$bldg2.LOCATIONID <- as.factor(bldg2dfsample$bldg2.LOCATIONID)

#################
# Training Models
#################

set.seed(123)
rightmetric <- "Accuracy"
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)


##### BUILDING 0 #####
#### KNN ####
set.seed(123)
knnGrid <- expand.grid(.k=c(1:4))
knnfitbldg0 <- train(bldg0.LOCATIONID ~ ., 
                   bldg0dfsample,
                   method = "knn", metric=rightmetric,
                   tuneGrid = knnGrid,
                   trControl = fitControl)
                  
knnfitbldg0

#### Random Forest ####
set.seed(123)
rfGrid <- expand.grid(mtry=c(16,32,48))
rffitbldg0 <- train(bldg0.LOCATIONID ~ ., 
                    bldg0dfsample,
                    method = "rf", tuneGrid=rfGrid,
                    metric=rightmetric,
                    trControl = fitControl)
rffitbldg0           


#### C5.0 #### 
set.seed(998)
c50grid <- expand.grid( .winnow = FALSE, .trials=c(2,8,24), .model = "tree" )
c50fitbldg0 <- train(bldg0.LOCATIONID ~ ., 
                   bldg0dfsample,
                   method = "C5.0", tuneGrid=c50grid,
                   metric=rightmetric,
                   trControl = fitControl)
c50fitbldg0


######### BUILDING 1 ############
#### kNN ####
set.seed(123)
knnGrid <- expand.grid(.k=c(1:4))
knnfitbldg1 <- train(bldg1.LOCATIONID ~ ., 
                     bldg1dfsample,
                   method = "knn", metric=rightmetric,
                   tuneGrid = knnGrid,
                   trControl = fitControl)
                   
knnfitbldg1

#### Random Forest ####
set.seed(123)
rfGrid <- expand.grid(mtry=c(16,32,48))
rffitbldg1 <- train(bldg1.LOCATIONID ~ ., 
                    bldg1dfsample,
                    method = "rf", tuneGrid=rfGrid,
                    metric=rightmetric,
                    trControl = fitControl)
rffitbldg1

#### C5.0 ####
set.seed(123)
c50grid <- expand.grid( .winnow = FALSE, .trials=c(2,8,24), .model = "tree" )  
c50fitbldg1 <- train(bldg1.LOCATIONID ~ ., 
                   bldg1dfsample,
                   method = "C5.0", tuneGrid=c50grid,
                   metric=rightmetric,
                   trControl = fitControl)
c50fitbldg1

############# BUILDING 2 ###############
##### kNN #####
set.seed(123)
knnGrid <- expand.grid(.k=c(1:4))  
knnfitbldg2 <- train(bldg2.LOCATIONID ~ ., 
                   bldg2dfsample,
                   method = "knn", metric=rightmetric,
                   tuneGrid = knnGrid,
                   trControl = fitControl)
                   
knnfitbldg2

#### Random Forest ####
set.seed(123)
rfGrid <- expand.grid(mtry=c(16,32,48))
rffitbldg2 <- train(bldg2.LOCATIONID ~ ., 
                    bldg2dfsample,
                    method = "rf", tuneGrid=rfGrid,
                    metric=rightmetric,
                    trControl = fitControl)
                    
rffitbldg2 

#### C5.0 ####
set.seed(123)
c50grid <- expand.grid( .winnow = FALSE, .trials=c(2,8,24), .model = "tree" )  
c50fitbldg2 <- train(bldg2.LOCATIONID ~ ., 
                   bldg2dfsample,
                   method = "C5.0", tuneGrid=c50grid,
                   metric=rightmetric,
                   trControl = fitControl)
c50fitbldg2

################################################      
# Test Set Predictions and Confusion Matrices
##############################################

#This section will consist of three subsections
#1) Prepare the test set for testing our models
#2) Predictions, Confusion Matrices, and Accuracy and Kappa scores
#3) Calculate the combined weighted average Kappa and Accuracy score 
##  for each algorithm in the entire dataset


# @@@@@@@@@@@@@@@ Subsection (1) @@@@@@@@@@@@@@@@@@

# Separate the data by BUILDINGID for testing
bldg0test <- filter(test, BUILDINGID == 0)
bldg1test <- filter(test, BUILDINGID == 1)
bldg2test <- filter(test, BUILDINGID == 2)

# Create dfs of the direct variable for each building for testing
bldg0testdf <- data.frame(bldg0test$LOCATIONID, bldg0test[,1:465])
bldg1testdf <- data.frame(bldg1test$LOCATIONID, bldg1test[,1:465])
bldg2testdf <- data.frame(bldg2test$LOCATIONID, bldg2test[,1:465])

# Convert LOCATIONID to factor in test set
bldg0testdf$bldg0test.LOCATIONID <- as.factor(bldg0testdf$bldg0test.LOCATIONID)
bldg1testdf$bldg1test.LOCATIONID <- as.factor(bldg1testdf$bldg1test.LOCATIONID)
bldg2testdf$bldg2test.LOCATIONID <- as.factor(bldg2testdf$bldg2test.LOCATIONID)

# @@@@@@@@@@@@@@@@@@@@@@ Subsection (2) @@@@@@@@@@@@@@@@@@@@@@@@@@@@

#-------------- Building 0 ------------------#

#### Prediction Building 0 ####
#### KNN ####

#predict on test set
pred.knnfitbldg0 <- predict(knnfitbldg0, bldg0testdf)
summary(pred.knnfitbldg0)

#confusion matrix
conmtrxknnb0 <- confusionMatrix(pred.knnfitbldg0,bldg0testdf$bldg0test.LOCATIONID)
conmtrxknnb0$overall #confusion matrix stats

results.knnfitbldg0 <- postResample(pred.knnfitbldg0,bldg0testdf$bldg0test.LOCATIONID)
results.knnfitbldg0
#Accuracy     Kappa 
#0.5332326 0.5312219

accuracy.knnbldg0 <- results.knnfitbldg0[1]
accuracy.knnbldg0

kappa.knnbldg0 <- results.knnfitbldg0[2]
kappa.knnbldg0

#### Prediction Building 0 ####
#### RF ####
#predict on test set
pred.rffitbldg0 <- predict(rffitbldg0, bldg0testdf) 
summary(pred.rffitbldg0)

#confusion matrix
conmtrxrfb0 <- confusionMatrix(pred.rffitbldg0,bldg0testdf$bldg0test.LOCATIONID)
conmtrxrfb0$overall #confusion matrix stats


results.rffitbldg0 <- postResample(pred.rffitbldg0,bldg0testdf$bldg0test.LOCATIONID)
results.rffitbldg0
#Accuracy     Kappa 
#0.6613876 0.6599145

accuracy.rfbldg0 <- results.rffitbldg0[1]
accuracy.rfbldg0

kappa.rfbldg0 <- results.rffitbldg0[2]
kappa.rfbldg0

#### Prediction Building 0 ####
#### c5.0 ####

pred.c50fitbldg0 <- predict(c50fitbldg0, bldg0testdf)
summary(pred.c50fitbldg0)

#confusion matrix
conmtrxc50b0 <- confusionMatrix(pred.c50fitbldg0,bldg0testdf$bldg0test.LOCATIONID)
conmtrxc50b0$overall #confusion matrix stats

results.c50fitbldg0 <- postResample(pred.c50fitbldg0,bldg0testdf$bldg0test.LOCATIONID)
results.c50fitbldg0
#Accuracy     Kappa 
#0.5837104 0.5819001 
accuracy.c50bldg0 <- results.c50fitbldg0[1]
accuracy.c50bldg0

kappa.c50bldg0 <- results.c50fitbldg0[2]
kappa.c50bldg0

#---------------- Building 1  ----------------#

#### Prediction Building 1 ####
#### KNN ####
pred.knnfitbldg1 <- predict(knnfitbldg1, bldg1testdf)
summary(pred.knnfitbldg1)

#confusion matrix
conmtrxknnb1 < confusionMatrix(pred.knnfitbldg1,bldg1testdf$bldg1test.LOCATIONID)
conmtrxknnb1$overall #confusion matrix stats

results.knnfitbldg1 <- postResample(pred.knnfitbldg1,bldg1testdf$bldg1test.LOCATIONID)
results.knnfitbldg1
#Accuracy     Kappa 
#0.6920063 0.6890237

accuracy.knnbldg1 <- results.knnfitbldg1[1]
accuracy.knnbldg1

kappa.knnbldg1 <- results.knnfitbldg1[2]
kappa.knnbldg1

#### Prediction Building 1 ####
#### RF ####

pred.rffitbldg1 <- predict(rffitbldg1, bldg1testdf) 
summary(pred.rffitbldg1)

#confusion matrix
conmtrxrfb1 < confusionMatrix(pred.rffitbldg1,bldg1testdf$bldg1test.LOCATIONID)
conmtrxrfb1$overall #confusion matrix stats

results.rffitbldg1 <- postResample(pred.rffitbldg1,bldg1testdf$bldg1test.LOCATIONID)
results.rffitbldg1
#Accuracy     Kappa 
#0.8087774 0.8068995 

accuracy.rfbldg1 <- results.rffitbldg1[1]
accuracy.rfbldg1

kappa.rfbldg1 <- results.rffitbldg1[2]
kappa.rfbldg1

#### Prediction Building 1 ####
#### c5.0 ####

pred.c50fitbldg1 <- predict(c50fitbldg1, bldg1testdf)
summary(pred.c50fitbldg1)

#confusion matrix
conmtrxc50b1 < confusionMatrix(pred.c50fitbldg1,bldg1testdf$bldg1test.LOCATIONID)
conmtrxc50b1$overall #confusion matrix stats

results.c50fitbldg1 <- postResample(pred.c50fitbldg1,bldg1testdf$bldg1test.LOCATIONID)
results.c50fitbldg1
#Accuracy     Kappa 
#0.7319749 0.7293433
accuracy.c50bldg1 <- results.c50fitbldg1[1]
accuracy.c50bldg1

kappa.c50bldg1 <- results.c50fitbldg1[2]
kappa.c50bldg1

#------------------- Building 2 ----------------------#

#### Prediction Building 2 ####
#### KNN ####
pred.knnfitbldg2 <- predict(knnfitbldg2, bldg2testdf)
summary(pred.knnfitbldg2)

#confusion matrix
conmtrxknnb2 < confusionMatrix(pred.knnfitbldg2,bldg2testdf$bldg2test.LOCATIONID)
conmtrxknnb2$overall #confusion matrix stats

results.knnfitbldg2 <- postResample(pred.knnfitbldg2,bldg2testdf$bldg2test.LOCATIONID)
results.knnfitbldg2
#Accuracy     Kappa 
#0.5124106 0.5103216 

accuracy.knnbldg2 <- results.knnfitbldg2[1]
accuracy.knnbldg2

kappa.knnbldg2 <- results.knnfitbldg2[2]
kappa.knnbldg2
 
#### Prediction Building 2 ####
#### RF ####

pred.rffitbldg2 <- predict(rffitbldg2, bldg2testdf) 
summary(pred.rffitbldg2)

#confusion matrix
conmtrxrfb2 < confusionMatrix(pred.rffitbldg2,bldg2testdf$bldg2test.LOCATIONID)
conmtrxrfb2$overall #confusion matrix stats

results.rffitbldg2 <- postResample(pred.rffitbldg2,bldg2testdf$bldg2test.LOCATIONID)
results.rffitbldg2
#Accuracy     Kappa 
#0.6248949 0.6231863 
accuracy.rfbldg2 <- results.rffitbldg2[1]
accuracy.rfbldg2

kappa.rfbldg2 <- results.rffitbldg2[2]
kappa.rfbldg2

#### Prediction Building 2 ####
#### c5.0 ####

pred.c50fitbldg2 <- predict(c50fitbldg2, bldg2testdf)
summary(pred.c50fitbldg2)

#confusion matrix
conmtrxc50b2 < confusionMatrix(pred.c50fitbldg2,bldg2testdf$bldg2test.LOCATIONID)
conmtrxc50b2$overall #confusion matrix stats

results.c50fitbldg2 <- postResample(pred.c50fitbldg2,bldg2testdf$bldg2test.LOCATIONID)
results.c50fitbldg2
#Accuracy     Kappa 
#0.4764706 0.4741777 
accuracy.c50bldg2 <- results.c50fitbldg2[1]
accuracy.c50bldg2

kappa.c50bldg2 <- results.c50fitbldg2[2]
kappa.c50bldg2

# @@@@@@@@@@@@@@@@@@@@@@ Subsection (3) @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

##### KNN #####
KNN.Accuracy <- ( accuracy.knnbldg0 * nlevels(knnfitbldg0) + accuracy.knnbldg1 * nlevels(knnfitbldg1) +
                    accuracy.knnbldg2 * nlevels(knnfitbldg2)) / 
  (nlevels(knnfitbldg0) + nlevels(knnfitbldg1) + nlevels(knnfitbldg2))
KNN.Accuracy
#Accuracy for KNN
#0.5597705  

KNN.Kappa <- ( kappa.knnbldg0 * nlevels(knnfitbldg0) + kappa.knnbldg1 * nlevels(knnfitbldg1) +
                 kappa.knnbldg2 * nlevels(knnfitbldg2)) / 
  (nlevels(knnfitbldg0) + nlevels(knnfitbldg1) + nlevels(knnfitbldg2))
KNN.Kappa
#Kappa for KNN
#0.5575095 

###### Random Forest #####
RF.Accuracy <- ( accuracy.rfbldg0 * nlevels(rffitbldg0) + accuracy.rfbldg1 * nlevels(rffitbldg1) +
                   accuracy.rfbldg2 * nlevels(rffitbldg2)) / 
  (nlevels(rffitbldg0) + nlevels(rffitbldg1) + nlevels(rffitbldg2)) 
RF.Accuracy
#Accuracy for Random Forest
#0.678694 

RF.Kappa <- ( kappa.rfbldg0 * nlevels(rffitbldg0) + kappa.rfbldg1 * nlevels(rffitbldg1) +
                kappa.rfbldg2 * nlevels(rffitbldg2)) / 
  (nlevels(rffitbldg0) + nlevels(rffitbldg1) + nlevels(rffitbldg2))
RF.Kappa 
#Kappa for Random Forest
#0.67703  

##### C5.0 #####
c50.Accuracy <- ( accuracy.c50bldg0 * nlevels(c50fitbldg0) + accuracy.c50bldg1 * nlevels(c50fitbldg1) +
                    accuracy.c50bldg2 * nlevels(c50fitbldg2)) / 
  (nlevels(c50fitbldg0) + nlevels(c50fitbldg1) + nlevels(c50fitbldg2))
c50.Accuracy
#Accuracy for c5.0
#0.5710031

c50.Kappa <- ( kappa.c50bldg0 * nlevels(c50fitbldg0) + kappa.c50bldg1 * nlevels(c50fitbldg1) +
                 kappa.c50bldg2 * nlevels(c50fitbldg2)) / 
  (nlevels(c50fitbldg0) + nlevels(c50fitbldg1) + nlevels(c50fitbldg2))
c50.Kappa
#Kappa for c5.0
#0.5688035 

##############
# Conclusion
#############

#Based on the Accuracy and kappa performance metrics, the Random Forest algorithm is the
#best model to be used.

##################################
# Performance Metrics Visualized
##################################
 
# Plot(1) Weighted Average Model Comparison by Performance Metrics (Accuracy/Kappa) 

#data to convert into dataframe
Values = c(KNN.Accuracy,KNN.Kappa,RF.Accuracy,RF.Kappa,c50.Accuracy,c50.Kappa)
Algorithms = c(rep("KNN",2),rep("Random Forest",2),rep("c5.0",2))
Metric = c("Accuracy","Kappa","Accuracy","Kappa","Accuracy","Kappa")

#create dataframe
plotdata1 <- data.frame(Values,Algorithms,Metric)
#shorten decimal in column "Values" to 4 positions
plotdata1[,"Values"]=round(plotdata1[,"Values"],4) 

write.csv(plotdata1, "weighted_avg_performance_metrics.csv")

#bar chart displaying performance metrics (entire dataset)
mtrc_plot <- ggplot(aes(x=Algorithms, y= Values, group=Metric,fill=Metric),data=plotdata1)+
  geom_bar(position="dodge", stat="identity")+
  scale_fill_manual(values = c("#406882","#B1D0E0"))+
  geom_text(aes(label=Values, fontface="bold"),hjust=0.5, vjust=0,position = position_dodge(.9))+
  scale_y_continuous(expand=c(0,0))+
  theme_classic()+
  ylim(0,1)+
  theme(legend.position = "left")+
  ggtitle("Model Comparison by Performance Metrics")+
  theme(plot.title = element_text(hjust=0.5))

print(mtrc_plot) #print plot 

# Plot(2) Model Comparison by Building# for Accuracy

#data to convert into dataframes
Accuracy_Value = c(accuracy.knnbldg0,accuracy.rfbldg0,accuracy.c50bldg0, 
            accuracy.knnbldg1,accuracy.rfbldg1,accuracy.c50bldg1,
            accuracy.knnbldg2,accuracy.rfbldg2,accuracy.c50bldg2)
Kappa_Value = c(kappa.knnbldg0,kappa.rfbldg0,kappa.c50bldg0, 
                   kappa.knnbldg1,kappa.rfbldg1,kappa.c50bldg1,
                   kappa.knnbldg2,kappa.rfbldg2,kappa.c50bldg2)

Buildings = c(rep("Building 0",3),rep("Building 1",3),rep("Building 2",3))
Algorithms_ = c("KNN","Random Forest","c5.0",
                "KNN","Random Forest","c5.0",
                "KNN","Random Forest","c5.0")

#create dataframe
plotdata2 <- data.frame(Accuracy_Value,Buildings,Algorithms_)
#shorten decimal in column "Values2" to 4 positions
plotdata2[,"Accuracy_Value"]=round(plotdata2[,"Accuracy_Value"],4)

write.csv(plotdata2,"accuracy_metric_building.csv")

ggplot(aes(x= Buildings, y=Accuracy_Value, group=Algorithms_,fill=Algorithms_),data=plotdata2)+
  geom_bar(position="dodge", stat="identity")+
  scale_fill_manual(values = c("#D62828","#F77F00","#FCBF49"))+
  geom_text(aes(label=Accuracy_Value, fontface="bold"),hjust=0.5, vjust=0,position = position_dodge(.9))+
  scale_y_continuous(expand=c(0,0))+
  theme_classic()+
  ylim(0,1)+
  theme(legend.position = "left")+
  ggtitle("Accuracy by Algorithm and Building")+
  theme(plot.title = element_text(hjust=0.5))

# Plot(3) Model Comparison by Building# for Kappa

#create dataframe
plotdata3 <- data.frame(Kappa_Value,Buildings,Algorithms_)
#shorten decimal in column "Values2" to 4 positions
plotdata3[,"Kappa_Value"]=round(plotdata3[,"Kappa_Value"],4)

write.csv(plotdata3,"kappa_metric_building.csv")

ggplot(aes(x= Buildings, y=Kappa_Value, group=Algorithms_,fill=Algorithms_),data=plotdata3)+
  geom_bar(position="dodge", stat="identity")+
  scale_fill_manual(values = c("#D62828","#F77F00","#FCBF49"))+
  geom_text(aes(label=Kappa_Value, fontface="bold"),hjust=0.5, vjust=0,position = position_dodge(.9))+
  scale_y_continuous(expand=c(0,0))+
  theme_classic()+
  ylim(0,1)+
  theme(legend.position = "left")+
  ggtitle("Kappa by Algorithm and Building")+
  theme(plot.title = element_text(hjust=0.5))

  









