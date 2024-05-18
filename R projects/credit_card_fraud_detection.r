# ************************************************
# CREDIT CARD FRAUD DETECTION 
# - EDA, Preparing data for modeling, Predict using Models ,Results
#
# Based on real-world dataset
#
# DATE: 1st November 2022
# VERSION: v1.6
# AUTHOR: Adarsh Manoj,Piyush Patel,Abubakar Sani,Ramandeep Singh
#
#
# UPDATE
# 1.0 - 1/11/2022 -  Initial Version
# 1.1  - 8/11/2022 -  EDA
# 1.2  - 13/11/2022 - Pre Processing
# 1.3  - 16/11/2022 - Modeling
# 1.4  - 22/11/2022 - Tuning
# 1.5  - 24/11/2022 - Results
# 1.6  - 26/11/2022 - Final Changes and Commenting
# ************************************************

# clearing all global environment variables
rm(list=ls())

#Constant Varibles:
DATASET<-"creditcard.csv"    #name of dataset on which we will be working

HOLDOUT<-0.7  #% split of Training Data
SAMPLINGP<-0.5 # p value used in sampling

DISCRETE_BINS<-50 # no. of empty bins to determine discrete

CLASSTRUE<-1 # Class variable 1
CLASSFALSE<-0 # Class varibale 0

KFOLDNUMBER<-10 # folds for cross validation

#Defining Global Variables
Global_df<-NULL
Global_test<-NULL
Global_train<-NULL

#*********************************************************
#Define the libraries used in this project
# Library from CRAN          Version
#  dplyr                     1.0.10
#  ggplot2                   3.4.0
#  ROSE                      0.0-4
#  e1071                     1.7-12
#  caret                     6.0-93
#  rpart                     4.1.19
#  PRROC                     1.3.1              
#  reshape2                  1.4.4
#  pacman                    0.5.1


# ***********************************
#installLoadLibraries() : loading all the relevant Packages
# INPUT  : NONE
# OUPUT  : NONE
#
#**********************************
installLoadLibraries<-function(){
    #Libraries used in this project
    MYLIBRARIES<-c("dplyr","ggplot2","ROSE","e1071","caret","rpart","PRROC","reshape2")
    
    #installing pacman for managing packages
    install.packages("pacman",dependencies=TRUE)
    
    #installation and loading of packages
    library(pacman)
    pacman::p_load(char=MYLIBRARIES,install=TRUE,character.only=TRUE)
    print(" Libraries Loaded Succesfully Loaded ")
    
}

# Get the versions of all packages  
#sessionInfo()

# ***********************************
#readDataSet(filename) : loading the data file
# INPUT  : File Name
# OUPUT  : DataFrame Cotaining data
#
#**********************************
readDataSet<-function(filename){
    df<-read.csv(filename,header=TRUE)
    print(paste(" Dataset ",filename," Succesfully Loaded ","with ",nrow(df)," rows ","and ",ncol(df)," columns"))
    return(df)
}

# ********************** **************************
# mainEDA() :entry point EDA Credit Card Fraud Detection
# INPUT: None
# OUTPUT :None
# ************************************************
 
#This keeps all variables as local to this function
mainEDA<-function(){
    #Get data types
    str(Global_df)
    
    #Statistics of data
    summary(df)
    
    #Total fraud and non-fraud transactions
    totalFraud<-sum(Global_df$Class)
    totalNonFraud<-sum(Global_df$Class==CLASSFALSE)
    print(paste("Fraud = ",totalFraud))
    print(paste("Non Fraud = ",totalNonFraud))
    
    
    Amount<-Global_df$Amount
    Time<-Global_df$Time
    Amount<-data.frame(Amount)
    Time<-data.frame(Time)
    
    #seperating dataframe for fraud and non fraud transaction
    non_fraud <- Global_df %>% filter(Class == CLASSFALSE)
    df_fraud <- Global_df %>% filter(Class == CLASSTRUE)
    
    writeLines("\n\n")
    #Plotting fraud and non fraud transactions
    Class<-Global_df$Class
    Class<-data.frame(Class)
    
    writeLines("\n\n")
    print(ggplot(Class,aes(x=Class))+geom_bar()+labs(x = "Fraud and Non-Fraud Class",y = "Total No. of Transaction",title= "Distribution of Class"))
    
    #analysis of target variable
    targetVariableAnalysis()
    
    #check whether data is discrete or continuous
    
    #correlation Matrix
    correlationMatrix()
    
    #other graphs
    writeLines("\n\n Plotting the Graph Fraud against Amount")
    #plot the graph of fraud against amount
    print(ggplot(df_fraud,aes(Amount))+geom_histogram(bin=DISCRETE_BINS))
    
    
    writeLines("\n\n Plotting the Graph Non-Fraud against Amount")
    #plot the graph of non fraud against amount
    print(ggplot(non_fraud,aes(Amount))+geom_histogram(bin=DISCRETE_BINS))
    
    writeLines("\n\n Plotting the Fraud Transactions against Amount")
    #plotting the graph for fraud transaction against time and amount
    print(ggplot(df_fraud, aes(x=Time,y=Amount))+geom_point())
    
    writeLines("\n\n Plotting the Non-Fraud Transactions against Amount")
    #plotting the graph for non fraud transaction against time and amount
    print(ggplot(non_fraud, aes(x=Time,y=Amount))+geom_point())
    
    
    
    } # endof mainEDA()

# ********************** **************************
# targetVariableAnalysis() :Analysis of Target Variable used in EDA
# INPUT: None
# OUTPUT :None
# ************************************************

#plotting the Amount variable
targetVariableAnalysis<-function(){
    
    
    Amount<-Global_df$Amount
    Time<-Global_df$Time
    Amount<-data.frame(Amount)
    Time<-data.frame(Time)
    
 
    
    writeLines("\n Summary of Amount \n")
    summary(Amount)

    writeLines("\nNow we are gonna plot the Amount variable to check for distribution\n")
    print(ggplot(Amount,aes(Amount))+geom_histogram(bins=DISCRETE_BINS)+labs(x = "Amount in $ ",y = "Frequency",title= "Distribution of Transaction Amount"))

    writeLines("\nWe see that the data is skewed towards left side, but we need a normalised data.\n")
    print(ggplot(Amount,aes(Amount)) +geom_histogram(bins=DISCRETE_BINS)+scale_x_log10() +labs(x = "Amount in $ ",y = "Frequency",title= "Distribution of Transaction Amount"))
    
    
    #check whether data is discrete or continuous
    
    writeLines("\nPlot the data of Time distribution\n")
    print(ggplot(Time)+geom_qq(aes(sample=Time))+labs(x = "Time interval ",y = "Frequency",title= "Time Distribution"))

  
    
}# endof targetVariableAnalysis()

# ********************** ********************************************
# CorrelationMatrix() :Correlation Matrix of the dataset used in EDA
# INPUT: None
# OUTPUT :None
# ********************************************************************
correlationMatrix<-function(){
    #HeatMaps
    #Preparing the data
    #correlation Matrix
    corrmatrix<-round(cor(Global_df),2)
 
    #melted Correaltion matrix
    meltedcorrmat<-melt(corrmatrix)

    #print Melted Correaltion matrix
    head(meltedcorrmat)

    #we will try to find the pairs having coeficient greater than 0.5
    df_corr<-data.frame(corrmatrix)
    df_corr[(abs(df_corr) >= 0.5) & (abs(df_corr) !=1)]
    
    
    #Plotting the correlation matrix
    
    writeLines("\nCorrelation Matrix\n")
    print(ggplot(data = meltedcorrmat, aes(Var2, Var1, fill = value))+geom_tile(color = "white")+scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1,1), space = "Lab", name="Correlation") +theme_minimal()+ theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 12, hjust = 1))+coord_fixed()+geom_text(aes(Var2, Var1, label = value),color = "black", size = 2))

    
    
    
}# end of correlationMatrix()

# ********************** **************************
# dataAfterSampling() :Checking the data after sampling used in Pre-Processing
# INPUT: sampledData: Sampled Data Frame 
# OUTPUT :None
# ************************************************

#plotting the Amount variable
dataAfterSampling<-function(sampledData){
    
    data_balanced<-sampledData
    ## Check class distribution after sampling
    
    data_balanced %>% group_by(Class) %>%
    summarise(cnt = n()) %>%
    mutate(freq = round(cnt / sum(cnt), 5)) %>% 
    arrange(desc(freq))
    
    
    
    
    
}# end of dataAfterSampling()

# ********************** **************************
# dataSampling() :Using Sampling for class balance used in Pre-Processing
# INPUT: None
# OUTPUT :Sampled Data
# ************************************************

#plotting the Amount variable
dataSampling<-function(){
    ## now using ROSE for sampling
    sampledData<- ovun.sample(Class ~., data=Global_df,p=SAMPLINGP, seed=1,method="over")
    df<-sampledData$data
    return(df)
}# end of dataSampling()

# ************************************************
# removeDuplicateRows() :Removing the duplicate rows from the data frame
# INPUT: None
# OUTPUT :Data with duplicate rows removed
# ************************************************
removeDuplicateRows<-function(){
    
# Check for duplicate rows and removing them
    df<-Global_df[!duplicated(Global_df),]
    duplicateRows<-284807-nrow(df)                      
    writeLines(paste("\n\nTotal Number of Duplicate Rows Removed : ",duplicateRows))
    return(df)
}# end of removeDuplicateRows()

# ********************** **************************
# mainPreProcessing() :Pre-Processing of Data
# INPUT: None
# OUTPUT :None
# ************************************************
 
#This keeps all variables as local to this function
mainPreProcessing<-function(){
    head(Global_df)
    
    
    #Checking for any null values
    sm<-sum(is.null(df))
    writeLines(paste("\n\nTotal Number of Null Values = ",sm))
    
    #removing all the duplicate values
    df<-removeDuplicateRows()          
    
    
               
    #scale for feature engineering
    Global_df$Amount<-scale(Global_df$Amount)
    
    
    #sampling
    df<-dataSampling()
    
    #results of sampling
    dataAfterSampling(df)
    
    #Setting Target Variable to factor
    df$Class <- as.factor(df$Class)
    
    #Assigning sampled data to Global Data
    Global_df<<-df
    
    #reducing Dimentionality
    mainDimReduction()
    
    
} # endof mainPreProcessing()

#***********************************************************************
# pcaCharts(X): PCA attempts to find the charts of principal components (or features) as its names denote.
#
# INPUT : Class variable
#
# OUTPUT : charts of Proportions of variance w.r.t Principle Components
#
# *************************************************************************
# Function to check Variance explained
pcaCharts <- function(x) {
    x.var <- x$sdev ^ 2
    x.pvar <- x.var/sum(x.var)
    print("proportions of variance:")
    print(x.pvar)
    
    par(mfrow=c(2,2))
    plot(x.pvar,
         xlab="Principal component", 
         ylab="Proportion of variance explained", 
         ylim=c(0,1), type='b')
    plot(cumsum(x.pvar),
         xlab="Principal component", 
         ylab="Cumulative Proportion of variance explained", 
         ylim=c(0,1), type='b')
    screeplot(x)
    screeplot(x,type="l")
    par(mfrow=c(1,1))
    
}# end of PcaCharts()



# ********************** **************************
# mainDimReduction() :Dimensionality Reduction 
# INPUT: None
# OUTPUT :None
# ************************************************
 
#This keeps all variables as local to this function
mainDimReduction<-function(){
    
    
    # Seperating features from target variable
    X <- subset(Global_df, select = -c(Class)) 
    
    #error here 
    #Error in subset.default(df, select = -c(Class)): argument "subset" is missing, with no default


    
    
    #reducing dim of Class with PCA
    pca <- prcomp(X, scale. = TRUE)
    pcaCharts(pca)
    
    # Calculate Explained Variance
    pca_var <- round((pca$sdev^2)/sum(pca$sdev^2) * 100, 1) 
    n_pc <- seq(1, length(pca_var))
    pca_df <- data.frame(pca_var, n_pc)

    #Graph of explained variance
    pca_df %>% ggplot(aes(x = n_pc, y = pca_var)) +geom_bar(stat = 'identity') +labs(title = "PCA Variation Plot",xlab = "PCs",ylab = "Prop. of variance explained",ylim = c(0, 100))
    
    
    # SVD for dim reduction
    svd <- svd(X)
    
    # Calculate Explained Variance
    svd_var <- round(svd$d^2/sum(svd$d^2) * 100, 4) 
    n_pc <- seq(1, length(svd_var))
    svd_df <- data.frame(svd_var, n_pc)

    # Graph of explained variance
    svd_df %>% ggplot(aes(x = n_pc, y = svd_var)) +geom_bar(stat = 'identity') +labs(title = "SVD Variation Plot",xlab = "PCs",ylab = "Prop. of variance explained",ylim = c(0, 100))
    
    # Consolodiating PCA transformed dataframe
    X_pca <- as.data.frame(pca$x[, 1:5])
    y <- subset(Global_df, select = c(Class))
    df_pca <- cbind(X_pca, y)
    
    #Partitioning train-test split
    index <- createDataPartition(y = df_pca$Class, p = HOLDOUT, times = 1, list = FALSE)
    train <- df_pca[index,]
    test  <- df_pca[-index,]
    
    #Assigning test and train data to Global test and Global test respectivly.
    Global_test<<-test
    Global_train<<-train
    
} # endof Dim Reduction()




# ********************** **********************************
# logisticTrain(train) :train the model using training data 
# INPUT: training data
# OUTPUT :trained model
# **********************************************************
logisticTrain<-function(train){
    
    kfold_cv <- trainControl(method = "cv",  number = 3, savePredictions = T)
    
    logreg_model <- train(form=Class~., data = train,method="glm", family="binomial",trControl=kfold_cv, tuneLength=3)
    print(summary(logreg_model))
    return(logreg_model)
    
}# end of logisticTrain

# ********************** **********************************
# decisionTrain(train) :train the model using training data 
# INPUT: training data
# OUTPUT :trained model
# **********************************************************
decisionTrain<-function(train){
    tree <- rpart(Class ~ ., data = train, method = "class",control=rpart.control(xval=3))
    print(summary(tree))
    return(tree)
}# end of decisionTrain

# ********************** **********************************
# randomForestTrain(train) :train the model using training data 
# INPUT: training data
# OUTPUT :trained model
# **********************************************************
randomForestTrain<-function(train){
   kfold_cv <- trainControl(method = "cv",  number = KFOLDNUMBER, savePredictions = T)
    forest <- train(
        
        # Formula. We are using all variables to predict Species
        Class~., 
        
        # Source of data; remove the Species variable
        data=train, 
        
        # `rf` method for random forest
        method='rf', 
        
        #min tree
        mtry=3,
        
        maxdepth=10,
        
        
        # Add repeated cross validation as trControl
        trControl=kfold_cv,
        
        # Accuracy to measure the performance of the model
        metric='Accuracy')

    
    return(forest)
} # end of randomForestTrain

#
# ********************** **************************
# predictionTestLR(preds,test) :Predict the results using model and test data 
# INPUT: predictive Model,testing data
# OUTPUT :NONE
# ***********************************************
predictionTestLR<-function(pred_model,test){
    #logistic Regression    
    preds <- predict(pred_model, test, type="prob")[,2] #prob of positive class
    return(preds)
}# end of predictionTestLR()


# ********************** **************************
# predictionTestDT(preds,test) :Predict the results of Decision Tress using model and test data 
# INPUT: predictive Model,testing dat
# OUTPUT :NONE
# ***********************************************
predictionTestDT<-function(pred_model,test){
    pred <- predict(pred_model, test, type="class")
    return(pred)

}# end of predictionTestDT()

# ********************** **************************
# predictionTestDT(preds,test) :Predict the results of Random Forest using model and test data 
# INPUT: predictive Model,testing data
# OUTPUT :NONE
# ***********************************************
predictionTestRF<-function(pred_model,test){
    pred <- predict(pred_model, test)
    return(pred)
} # end of predictionTestRF()

#
# ********************** **************************
# predictionResultsLR(preds,test,modelName) :Display the results using model and test data 
# INPUT: predictive Model,testing data,model name
# OUTPUT :NONE
# ************************************************
predictionResultsLR<-function(preds,test,modelName){
    
   
    preds_pos <- preds[test[,6]==1] #preds for true positive class
    preds_neg <- preds[test[,6]==0] #preds for true negative class
   
    writeLines(paste("\n\n ROC-AUC - ",modelName))
    PRC <- pr.curve(preds_pos, preds_neg, curve=TRUE)
    print(plot(PRC))
    
    
    writeLines(paste("\n\nPrecision,Recall and F1-Score - ",modelName))
    dt_db_test<- factor(ifelse(preds> 0.50,1,0))
    
    precision_dtdb <- posPredValue(dt_db_test,test$Class,positive = 1)
    recall_dtdb <- sensitivity(dt_db_test,test$Class,positive = 1)
    F1_dtdb <- (2 * precision_dtdb * recall_dtdb) / (precision_dtdb + recall_dtdb)
    print(paste("Precsion : ",precision_dtdb))
    print(paste("Recall : ",recall_dtdb))
    print(paste("F1-Score : ",F1_dtdb))
    
    writeLines(paste("\n\nConfusion Matrix - ",modelName))
    print(confusionMatrix(dt_db_test, test$Class, positive = '1'))
    
}# end of predictionResultsLR()

# ********************** **************************
# predictionResults(preds,test,modelName) :Predict the results using model and test data 
# INPUT: training data,testing data and Model name
# OUTPUT :NONE
# ************************************************

predictionResults<-function(preds,test,modelName){
    
   
    preds_pos <- preds[test[,6]==1] #preds for true positive class
    preds_neg <- preds[test[,6]==0] #preds for true negative class
   
    writeLines(paste("\n\n ROC-AUC - ",modelName))
    PRC <- pr.curve(preds_pos, preds_neg, curve=TRUE)
    print(plot(PRC))
    
    
    writeLines(paste("\n\nPrecision,Recall and F1-Score - ",modelName))
    
    
    precision_dtdb <- posPredValue(preds,test$Class,positive = 1)
    recall_dtdb <- sensitivity(preds,test$Class,positive = 1)
    F1_dtdb <- (2 * precision_dtdb * recall_dtdb) / (precision_dtdb + recall_dtdb)
    print(paste("Precsion : ",precision_dtdb))
    print(paste("Recall : ",recall_dtdb))
    print(paste("F1-Score : ",F1_dtdb))
    
    writeLines(paste("\n\nConfusion Matrix - ",modelName))
    print(confusionMatrix(preds, test$Class, positive = '1'))
} # end of predictionResults()

  

#
# ********************** **************************
# modelDeclarationLR(train,test) :execute Logistic Regression Model 
# INPUT: training and testing data
# OUTPUT :NONE
# ************************************************
modelDeclarationLR<-function(train,test){  
            
    logreg_model<-logisticTrain(train)
    
    predLR<-predictionTestLR(logreg_model,test)
    
    resultLR<-predictionResultsLR(predLR, test, "Logistic Regression")
    
} #modelDeclarationLR

# ********************** **************************
# modelDeclarationDT(train,test) : execute Decision Tress Model
# INPUT: training and testing data
# OUTPUT :NONE
# ************************************************

modelDeclarationDT<-function(train,test){
    
    decision_model<-decisionTrain(train)

    predDT<-predictionTestDT(decision_model,test)

    resultDT<-predictionResults(predDT, test, "Decisition Tree")
    
}# end of modelDeclarationDT

# ********************** **************************
# modelDeclarationRF(train,test) : execute Random Forest Model
# INPUT: training and testing data
# OUTPUT :NONE
# ************************************************
modelDeclarationRF<-function(train,test){
    
    random_forest_model<-randomForestTrain(train)
    
    predRF<-predictionTestRF(random_forest_model,test)
    
    resultRF<-predictionResults(predRF, test, "Random Forest")
    
}#end of modelDeclarationRF()

# ********************** **************************
# mainModeling() :entry point of Modeling Credit Card Fraud Detection
# INPUT: None
# OUTPUT :None
# ************************************************
 
#This keeps all variables as local to this function
mainModeling<-function(){
    
   
    
    
    test<-Global_test
    train<-Global_train
    test<-data.frame(test)
    train<-data.frame(train)
    
    modelDeclarationLR(train,test)
    modelDeclarationDT(train,test)
    #modelDeclarationRF(train,test)
    
    
}# end of mainModeling()


# ********************** **************************
# mainPerformanceEvaluation() :Model Perfromance using Precision Recall and F1 score
# INPUT: None
# OUTPUT :None
# ************************************************
 
#This keeps all variables as local to this function
mainPerformanceEvaluation<-function(){
    #K-fold cross validation
   
precision_both <- posPredValue(dt_dbsmote_test,test$Class,positive = "yes")
recall_both <- sensitivity(dt_dbsmote_test,test$class,positive = "yes")
F1_both <- (2 * precision_dtdbsmote * recall_dtdbsmote) / (precision_dtdbsmote + recall_dtdbsmote)
} # endof mainPerformanceEvaluation()

# ********************** **************************
# mainResults() :entry point to execute the ML data analytics
# INPUT: None
# OUTPUT :None
# ************************************************
 
#This keeps all variables as local to this function
mainResults<-function(){
# AUC for each results 
} # endof mainResults()

# ************************************************
#  Start of Running the functions to run the model
# ************************************************

 #LOAD LIBRARIES
installLoadLibraries()

writeLines("\nload Dataset\n") 
df<-readDataSet(DATASET)
#Assigning global data variable
Global_df<<-df

writeLines("\nHead DATA\n")
print(head(df))
    

writeLines("\nEDA\n")
#EDA
mainEDA()


#Pre Processing
mainPreProcessing()

#Model Preparation and Results
mainModeling()


# **************************************************
# End of Notebook 
# **************************************************
