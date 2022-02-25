#clean the workspace
rm(list=ls())
#First we load the data and read the data
load("titanic_train.RDATA")
head(titanic.train)
#MAIN CODE
#FIRST WE WILL ELIMINATE CABIN AND TICKET

#We remove the cabin and ticket columns
titanic.train = titanic.train[-9]
titanic.train = titanic.train[-7]
head(titanic.train)

#-------------------------

{if(!require(foreach)){
  install.packages("foreach")
}
library("foreach")

if(!require(pracma)){
  install.packages("pracma")
}
library("pracma")

if(!require(doParallel)){
  install.packages("doParallel")
}
library("doParallel")}

#-------------------------

#SUBFUNCTIONS

#-------------------------
#RANDOM FOREST: 
random_forest = function(titanic.train){
  #First we install the required library
  if(!require(randomForest)){
    install.packages("randomForest")
  }
  library("randomForest")
  
  #USING PARALELL PROCESSING PART WITH THE PRACMA FUNCTION
  
  cl = makeCluster((detectCores()-1))
  registerDoParallel(cl)
  
  # To tune the hyper parameters we first create vectors and a grid
  
  vmtry = seq(1,5,1)  #The maximum number of variables cannot exceed 8 as we only have 9 in our dataset
  vntree = seq(100,10000,100) #There is no such limitation in ntree, we want to set it above a certain threshold to ensure that each variable gets predicted more than once
  
  grid = expand.grid(vmtry = seq(1,5,1), vntree = seq(100,10000,100))
  
  # We will leave the rest of the parameters in the default values
  #Now we partition the data into 4 disjoint sets, as integers
  inds=sample(as.integer(cut(1:nrow(titanic.train),4)), nrow(titanic.train), replace=FALSE)
  #Now for each of these sets we need to find the best fit, tunning hyperparameters
  #For all subgroups created sampling the data
  
  #This is the part of the parallel processing
  {tic()
    grid$accuracies = foreach(j=1:4, .combine = "+", .packages = "randomForest")%:%
      foreach(i=1:nrow(grid), .combine = "c")%dopar%{
        # Now we split those into data and training
        train=titanic.train[inds != j,]
        test=titanic.train[inds == j,]
        
        #Train the model store it's accuracy in the vector
        model=randomForest(Survived~., train, mtry = grid$vmtry[i], ntree = grid$vntree[i])
        mypredictions=predict(model,test[,-1], type = "Class")
        accuracies=sum(mypredictions==test[,1])/length(mypredictions)
        
        accuracies/4
      }
toc()} 
  stopCluster(cl)
  #This is the highest accuracy
  a = grid[which.max(grid$accuracies),]
  return(a)
}
#---------------
#KNN
knn_fun = function(titanic.train){
  #We load the required libraries
  if(!require("class")){
    install.packages("class")
  }
  library("class")
  
  cl = makeCluster((detectCores()-1))
  registerDoParallel(cl)
  #We sample the data
  inds=sample(as.integer(cut(1:nrow(titanic.train),4)), nrow(titanic.train), replace=FALSE)
  grid = expand.grid(k = 1:40)
  #Since we are using a distance based algorithm we need to convert all factor variables to numeric
  for(j in 1:8){
    titanic.train[,j]=as.numeric(titanic.train[,j])
  }
  
  {tic()
    grid$accuracies = foreach(j=1:4, .combine = "+", .packages = "class")%:%
      foreach(i = 1:nrow(grid), .combine = "c")%dopar%{
        #Now we divide the data in training and evaluation
        data_tr = titanic.train[inds != j,]
        data_eval = titanic.train[inds == j,]
        # We use a for to find the optimal k, as there is only one hyper parameter, we don't need a grid
        # We create a vector of 6 elements
        train=data_tr[,2:8]
        train = scale(train)
        cl=data_tr[,1]
        test=data_eval[,2:8]
        test = scale(test)
        predictions = knn(train, test, cl, k = grid$k[i])
        accuracies=sum(predictions==data_eval[,1])/length(predictions)
        
        accuracies/4
       }
toc()}
  stopCluster(cl)
  b = grid[which.max(grid$accuracies),]
  return(b)
}
#-----------------------------------
#DECISION_TREE
decisiontree = function(titanic.train){
  if(!require('rpart')){
    install.packages(rpart)
  }
  library("rpart")
  inds=sample(as.integer(cut(1:nrow(titanic.train),4)), nrow(titanic.train), replace=FALSE)
  #Now we are going to create a grid for the hyperparameters
  vdepth = 1:5
  vcp = seq(from=0,to=1,by=0.25)
  params = expand.grid('vdepth'=vdepth,'vcp'=vcp)
  {tic()
    #We loop to find the best combination of those parameters
    accuracies = NULL
    for (i in 1:nrow(params)){
      for (i in 1:max(inds)){
        aux=which(inds==i)
        data_tr=titanic.train[-aux,]
        data_eval=titanic.train[aux,]
        mytree=rpart(Survived~.,data_tr,method="class",control = rpart.control(maxdepth = params$vdepth[i], cp=params$vcp[i]))
        myprediction=predict(mytree,data_eval[,-1],type="class")  #method implies we are classifying a integer class, the [-1,] means that we are using all variables but number 1
        accuracies[i] = sum(myprediction==data_eval[,1])/length(myprediction)
      } 
    }
  toc()}
  baccur = max(accuracies)
  index = which(accuracies ==baccur)
  c=NULL
  c[1]=params[index,1]
  c[2]=params[index,2]
  c[3]=baccur
  return(c)
}
#-----------------------------------
#SUPPORT VECTOR MACHINE
supportvectormachine = function(titanic.train){
  #First the required library
  if (!require(e1071)) 
    install.packages("e1071") 
  library(e1071)
  
  #We may use ggplot to make a graph of our data and see if a kernel needs to be applied
  
  if(!require(ggplot2)){
    install.packages("ggplot2")
  }
  library(ggplot2)
  #Since this is a distance based algorithm as knn we need to convert all variables to numeric
  for(j in 2:8){
    titanic.train[,j]=as.numeric(titanic.train[,j])
  }
  #Data frame that'll be used for prediction
  data1 = data.frame(
    x1 = titanic.train[,2],
    x2 = titanic.train[,3],
    x3 = titanic.train[,4],
    x4 = titanic.train[,5],
    x5 = titanic.train[,6],
    x6 = titanic.train[,7],
    x7 = titanic.train[,8],
    z = titanic.train[,1]
  )
  #We create vectors in which we store, for every kernel, in the first one the accuracies for the range of parameters tested
  # and on the second one we store, first the optimal value for the parameter and then accuracy it yields
  accsLinear = NULL
  accsLinear[1]=-1
  accsPolynomial = NULL
  opPoly = NULL
  accsSigmoid = NULL
  opSigmoid = NULL
  accsRadial = NULL
  opRadial = NULL
  #We partition data1, in training and evaluation
  {tic()
    set.seed(1)
    data_train = sample(1:nrow(data1), floor(0.7*nrow(data1)))
    train = data1[data_train,] 
    test = data1[-data_train,]
    #We are going to find the best parameters for every type of kernel (linear, polynomial, radial basis and sigmoid) and then compare which one produces the best accuracy
    #for the linear kernel, there are no extra parameters
    classifier1 = svm(formula = z~., data = data1, type = 'C-classification',kernel = 'linear')
    prediction = predict(classifier1,test)
    accsLinear[2] = sum(prediction == test$z)/length(prediction)
    
    
    #for the polynomial, we iterate through different degrees, there is only one parameter so we do not need a grid
    for (i in 1:9){
      classifier2 = svm(z~., data = data1, type = "C-classification",kernel = "polynomial", degree = i)
      pred = predict(classifier2, test, type = "class")
      accsPolynomial[i] = sum(pred == test$z)/length(pred)
    }
    
    aux = max(accsPolynomial)
    opDeg = which(accsPolynomial==aux)
    #Optimal degree for the polynomial kernel
    opPoly[1]=opDeg
    #Precision corresponding to the optimal degree for said kernel
    opPoly[2]=aux
    #ggplot(grid) + aes(x = degree, y = accuracy)+geom_point(size = 0.3) a plot of accuracies 
    
    #for the sigmoid kernel there is only one parameter as well, gamma
    vgamma =2^(-5:5) 
    grid = expand.grid(vgamma)
    for (i in 1:nrow(grid)){
      # fit the model 
      classifier3 = svm(z~., train, type = "C-classification", kernel = "sigmoid", gamma = vgamma[i])
      # make the predictions
      pred = predict(classifier3, test, type = "class")
      # evaluation
      accsSigmoid[i] = sum(pred == test$z)/length(pred)
    }
    #aux is the maximum accuracy possible for the sigmoid kernel
    aux = max(accsSigmoid)
    #Optimal value of gamma
    opGamma = which(accsSigmoid==aux)
    opSigmoid[1] = opGamma
    opSigmoid[2] = aux
    
    
    #for the radial kernel
    grid = expand.grid(gamma = 2^(-7:10))
    for (i in 1:nrow(grid)){
      # fit the model 
      classifier4 = svm(z~., train, type = "C-classification",kernel = "radial", gamma = grid$gamma[i])
      #make the predictions
      pred = predict(classifier4, test, type = "class")
      # get the accuracy
      accsRadial[i] = sum(pred == test$z)/length(pred)
    }
    aux = max(accsRadial)
    opGammar = which(accsRadial == aux)
    opRadial[1]=opGammar
    opRadial[2]=aux
  toc()}
  #Now we compare the accuracies for each kernel
  d = max(accsLinear,opSigmoid[2],opPoly[2],opRadial[2])
  if (d == accsLinear[2]){
    # we return the accuracy for the linear kernel
    accsLinear[2] = 1
    m=accsLinear
  }
  if(d == opSigmoid[2]){
    #We return optimal Sigmoid
    opSigmoid[3] = 2
    m=opSigmoid
  }
  if(d == opPoly[2]){
    # We return optimal polynomial
    opPoly[3] = 3
    m=opPoly
  }
  if(d == opRadial[2]){
    #We return optimal radial
    opRadial[3] = 4
    m=opRadial
  }
  return(m)
}
#-------------------------
#MAIN FUNCTION
my_model = function(data){
  # We split the data in training and evaluation
  #Return is a vector of length three
  ##a = random_forest(data)
  set.seed(1)
  a= random_forest(data)
  #The return of the knn function is just a number as there's only one hyperparameter
  set.seed(1)
  b = knn_fun(data)
  #Return is a vector of length 3
  set.seed(1)
  c = decisiontree(data)
  #
  set.seed(1)
  d = supportvectormachine(data)
  #The following elements of the returned vectors are where the corresponding accuracies are stored
  e = max(a[1,3],b[1,2],c[3],d[2])
  if(e==a[1,3]){
    print('Random forest')
    #Now we must train the optimal random Forest to return its predictions
    model=randomForest(Survived~., titanic.train, mtry = a[1,1], ntree = a[1,2])
    predictions=predict(model,titanic.train, type = "Class")
    a[1,1]
    a[1,2]
    return(predictions)
  }
  if(e==b[1,2]){
    print('K- nearest neighbour')
    #Now we must train the optimal k-nearest neighbour in order to return its predictions
    # The return format is (optimalk, corresponding precision)
    for(j in 1:8){
      data[,j]=as.numeric(data[,j])
    }
    data = scale(data)
    predictions = knn(train=data[,2:8], cl=data[,1], test=data[,2:8], k=b[1])
    b[1]
    return(predictions)
  }
  if(e==c[3]){
    print("Decision tree")
    # Now we must train the optimal decision tree to return its predictions
    mytree=rpart(Survived~.,data,method="class",control = rpart.control(maxdepth = c[1], cp=c[2]))
    predictions=predict(mytree,data[,-1],type="class")
    c[1]
    c[2]
    return(predictions)
  }
  if(e==d[2]){
    print("SVM")
    #Now we must train an optimal svm, format of return is (optimal value of parameter, corresponding accuracy, kernel identifier)
    for(j in 1:8){
      data[,j]=as.numeric(data[,j])
    }
    # Linear = 1
    # Sigmoid = 2
    # Poly = 3
    # Radial = 4
    if(d[2]==1){
      # We train a knn with linear kernel
      classifier = svm(formula = z~., data = data, type = 'C-classification',kernel = 'linear')
      prediction = predict(classifier,data)
      # we return the predictions of the svm with that kernel
      return(prediction)
    }
    if(d[3]==2){
      classifier = svm(formula = z~., data = data, type = 'C-classification',kernel = 'Sigmoid', gamma = d[1])
      prediction = predict(classifier,data)
      # We return the prediction of the svm with that kernel
      d[1]
      return(prediction)
    }
    if(d[3]==3){
      classifier = svm(class~., data = data, type = "C-classification",kernel = "polynomial", degree = d[1])
      pred = predict(classifier, data, type = "class")
      # We return the prediction of the svm with that kernel
      d[1]
      return(prediction)
    }
    if(d[3]==4){
      classifier = svm(formula = z~., data = data, type = 'C-classification',kernel = 'radial', gamma = d[1])
      prediction = predict(classifier,data)
      # We return the prediction of the svm with that kernel
      d[1]
      return(prediction)
    }
    
  }
  
}
#----------------
{e = my_model(titanic.train)
print('Welcome')
print("The following script will perform  prediction algorithms for the titanic data set(random forest, decision tree, svm and k-nearest neighbour) and once all the hyperparameters of each model have been tuned we will print that with the maximum accuracy and an identifier of said model ")
print("The most precise prediction is provided by: Random Forest")
print("with a accuracy of")
acc=(sum(e==titanic.train$Survived)/length(e))
acc}



