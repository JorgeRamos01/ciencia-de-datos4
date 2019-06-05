rm(list=ls())
setwd("~/ciencia de datos/Tarea4")

###### El codigo debe correrse secuencialmente sino pueden presentarse errores

#Cargamos los datos
train<-read.table("oef.train")
test<-read.table("oef.test")


library(doParallel) #Esta la usamos para usar multi threading para intentar tener un poco mas de velocidad
library (MASS) 
library(neuralnet)
library(dplyr)
library(caret)   #Esta libreria la empleamos debido a que permite hacer la validación cruzada por k-folding mas facilmente
library(mlbench)
library(nnet)
library(lfda)
library("scales")
library(ggplot2)
library(caTools)
library(rpart)

ceroUno <- function(x)  #Esta funcion la empleamos para hacer que los datos que no son etiquetas se encuentren entre 0 y 1
{
  x<-(x-min(x))/(max(x)-min(x))
  return(x)
}

train.scale<-apply(train[,2:ncol(train)],2,ceroUno) 
test.scale<-apply(test[,2:ncol(test)],2,ceroUno)

#Generamos nuestras versiones escaladas
train =as.data.frame( cbind(train$V1,train.scale) )
test =as.data.frame( cbind(test$V1,test.scale) )

label.train = train[,1]
label.test = test[,1]

###### Clasificamos nuestros datos usando LDA con 10-fold

train[,1]<-as.factor(train[,1])

cl <- makePSOCKcluster(4)
registerDoParallel(cl)  
set.seed(100)
#Test
lda.fit = train(V1 ~ ., data=train, method="lda",
                trControl = trainControl(method = "cv"))

stopCluster(cl)

lda.fit

#probamos el modelo en el conjunto de prueba
pred.out = predict(lda.fit, test)
(sum(pred.out == test$V1)/length(test$V1))*100  #Nivel de precision

######## QDA con 10-fold
  
  label.train = train[,1]
  label.test = test[,1]
  train = train[,-1]
  test = test[,-1]
  
  #Agregamos ruido a ambos conjuntos para que se cumplan los requisitos para que funcione QDA
  train.qda<-apply(train,2, jitter)
  test.qda<-apply(test,2,jitter)
  train.qda<-data.frame(V1=label.train,train.qda)
  test.qda<-data.frame(V1=label.test,test.qda)
  set.seed(100)
  
  
  cl <- makePSOCKcluster(4)
  
  registerDoParallel(cl)
  
  fit = train(V1 ~ ., data=train.qda, method="qda",
                       trControl = trainControl(method = "cv"))
  
  ## When you are done:
  stopCluster(cl)
  
  fit #Precision en entrenamiento con 10-fold
  
  pred.out = predict(fit, test.qda)
  
  (sum(pred.out == test.qda$V1)/length(test.qda$V1))*100 #Nivel de precision en datos de prueba
  
  
######### Redes neutonales

  #Preparamos los datos para pasarselos a la red neuronal
  train.neuralnet<-as.data.frame(train)
  test.neuralnet<-as.data.frame(test)
  train.neuralnet<-as.data.frame(cbind(model.matrix(~as.factor(label.train)-1),train.neuralnet[,-1]))
  test.neuralnet<-as.data.frame(cbind(model.matrix(~as.factor(label.test)-1),test.neuralnet[,-1]))
  
  #Se generaron los nombres de las columnas extra 
  colnames(train.neuralnet)<-c("r0","r1","r2","r3","r4","r5","r6","r7","r8","r9",colnames(train.neuralnet[,11:266]))
  colnames(test.neuralnet)<-c("r0","r1","r2","r3","r4","r5","r6","r7","r8","r9",colnames(test.neuralnet[,11:266]))
  
  #Se prepara la formula y la función que genera los intervalos para hacer los k-folds
  formula<-"r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9~"
  f <- as.formula(paste(formula, paste(colnames(train.neuralnet[,11:266]), collapse = " + ")))
  sapply(train.neuralnet, class)
  k<-5
  
  #Funcion que genera los k-folds en los datos
  pedazos<-round(quantile(1:nrow(train),probs = seq(from=0,to=1,by = 1/k)))
  
  precision<-rep(1,k)
  pred.out<-list()
  models<-list()

  for(i in 1:k){
    if(i==1){
      validacion<-(pedazos[i]):(pedazos[i+1])
    }
    else{
      validacion<-(pedazos[i]+1):(pedazos[i+1])
    }
      ##Correr modelo y guardar respuesta
      set.seed(100)
      fit1<- neuralnet(f,data = train.neuralnet[-validacion,],
                                    hidden = c(10, 10,10),
                                    act.fct = "logistic",
                                    linear.output = FALSE,
                                    lifesign = "minimal")
      models[[i]]<-fit1
      pred.out[[i]]<-neuralnet::compute(fit1, train.neuralnet[validacion,11:266])[[2]]
      precision[i]<-sum(apply(pred.out[[i]],1,which.max)-1==label.train[validacion])/length(label.train[validacion])
    
  }

  precision[which.max(precision)]*100 #Precision de entrenamiento
  
  #Probamos nuestro modelo en los datos de prueba

  res<-neuralnet::compute(models[[which.max(precision)]], test.neuralnet[,11:266])[[2]]
  clases<-apply(res,1,which.max)
  clases<-clases-1
  (sum(clases==label.test)/length(label.test))*100 #Nivel de precisión para datos de prueba


  
#### Arboles
  train<-read.table("oef.train")
  test<-read.table("oef.test")
  
  train[,1] = as.factor(train[,1])
  test[,1] = as.factor(test[,1])
  
  cl <- makePSOCKcluster(4)
  registerDoParallel(cl)  
  
  tune.gridcart <- expand.grid(cp=seq(0.00005,0.05,0.005)) # Tunear usando parametro de complejidad
  
  cctrl1 <- trainControl(method = "cv", number = 5, returnResamp = "all")
  fit.tree <- train(V1 ~ ., data = train, 
                              method = "rpart", 
                              tuneGrid =tune.gridcart,
                              trControl = cctrl1
                              )
  #Stop after trainning
  stopCluster(cl)
  
  plot(fit.tree,xlab="Complejidad",ylab="Precisión", main="Arbol de decisión 5-fold",col="red")
  
 #Nivel de precisión datos de entrenamiento
  (sum(predict(fit.tree,train)==train$V1)/length(train$V1))*100
  
  #Nivel de precisión datos de prueba
  (sum(predict(fit.tree,test)==test$V1)/length(test$V1))*100


############## SVM
  library(e1071)
  
  set.seed(3233)
  sapply(train,class)
  
  train$V1 = as.factor(train[,1])
  test$V1 = as.factor(test[,1])
  
  svmTuneGrid <- data.frame(.C = 2^(-3:3))
  
  cl <- makePSOCKcluster(4)
  registerDoParallel(cl)  
  
  trctrl <- trainControl(method = "cv", number = 3)
  svm_Linear <- train(V1 ~., data = train, method = "svmLinear",
                      tuneGrid = svmTuneGrid,
                      trControl=trctrl)
  
  #Stop after trainning
  stopCluster(cl)
  
  plot(svm_Linear, xlab="Costo", ylab="Precisión", main="SVM lineal 3-fold",col="red")
  
  #Nivel de precision datos de entrenamiento
  (sum(predict(svm_Linear)==label.train)/length(label.train))*100
  
  #Nivel de precision datos de prueba
  (sum(predict(svm_Linear,test)==label.test)/length(label.test))*100
  
#### Adaboost
  
  cl <- makePSOCKcluster(5)
  registerDoParallel(cl)
  
  library(adabag)
  grid <- expand.grid(mfinal = c(18,24,30), maxdepth = c(6, 8, 10),
                      coeflearn = c("Zhu"))
  
  cctrl1 <- trainControl(method = "cv", number = 3, returnResamp = "all",
                         classProbs = TRUE)
  
  
  label.train<-as.factor(label.train)
  label.test<-as.factor(label.test)
  
  levels(label.train)<-c("r0","r1","r2","r3","r4","r5","r6","r7","r8","r9")
  levels(label.test)<-c("r0","r1","r2","r3","r4","r5","r6","r7","r8","r9")
                                    
  
  test_class_cv_model <- train(train[,-1], label.train, 
                               method = "AdaBoost.M1", 
                               trControl = cctrl1,
                               tuneGrid = grid)
  
  stopCluster(cl)
  
  plot(test_class_cv_model)

  #Nivel de precision datos de entrenamiento
  (sum(predict(test_class_cv_model,train)==label.train)/length(label.train))*100

  #nivel de precision datos de prueba
  (sum(predict(test_class_cv_model,test[,-1])==label.test)/length(label.test))*100
    