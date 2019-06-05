rm(list=ls())
library(imager)
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
library(grDevices)
setwd("~/ciencia de datos/Tarea4/data_fruits_tarea/train")

files <- list.files(path=".", pattern=".jpg",all.files=T, full.names=F, no.. = T) 
list_of_images = lapply(files, load.image)
list_of_images2=lapply(files, readJPEG)
  
mean.train<-as.data.frame(matrix(nrow = 1,ncol = 3))
quant.train<-as.data.frame(matrix(nrow = 1,ncol = 9))
mean.test<-as.data.frame(matrix(nrow = 1,ncol = 3))
quant.test<-as.data.frame(matrix(nrow = 1,ncol = 9))


#Transformamos las imágenes al espacio HSV y generamos las matrices a usar para entrenamiento
for (i in 1:length(list_of_images)){ 
  a<-RGBtoHSV(list_of_images[[i]])
  quant.train[i,1:9]<-c(quantile(a[,,,1])[2:4],quantile(a[,,,2])[2:4],quantile(a[,,,3])[2:4])
  mean.train[i,1]<-median(a[,,,1])
  mean.train[i,2]<-median(a[,,,2])
  mean.train[i,3]<-median(a[,,,3])
}
#Borramos variables innecesarias para dejar memoria libre
rm(list_of_images)
rm(files)
rm(a)
rm(i)

#Obtenemos los datos de prueba
setwd("~/ciencia de datos/Tarea4/data_fruits_tarea/test")

files <- list.files(path=".", pattern=".jpg",all.files=T, full.names=F, no.. = T) 
list_of_images = lapply(files, load.image)

#Transformamos las imágenes al espacio HSV y generamos las matrices a usar para prueba
for (i in 1:length(list_of_images)){ 
  a<-RGBtoHSV(list_of_images[[i]])
  quant.test[i,1:9]<-c(quantile(a[,,,1])[2:4],quantile(a[,,,2])[2:4],quantile(a[,,,3])[2:4])
  mean.test[i,1]<-median(a[,,,1])
  mean.test[i,2]<-median(a[,,,2])
  mean.test[i,3]<-median(a[,,,3])
}
#Borramos variables innecesarias para dejar memoria libre
rm(list_of_images)
rm(files)
rm(a)
rm(i)

#Generamos las etiquetas
etiquetas.test<-c(rep("Braeburn",20),rep("Golden",20),rep("Granny",20),rep("Apricot",20),rep("Avocado",20),rep("Carambula",20),rep("Cherry",20), rep("Huckleberry",20),rep("Kiwi",20),rep("Orange",20),rep("Peach",20),rep("Pineapple",20),rep("Strawberry",20))
etiquetas.train<-c(rep("Braeburn",80),rep("Golden",80),rep("Granny",80),rep("Apricot",80),rep("Avocado",80),rep("Carambula",80),rep("Cherry",80), rep("Huckleberry",80),rep("Kiwi",80),rep("Orange",80),rep("Peach",80),rep("Pineapple",80),rep("Strawberry",80))
mean.train1<-as.data.frame(cbind(etiquetas.train,mean.train))
quant.train1<-as.data.frame(cbind(etiquetas.train,quant.train))

##########Aplicamos LDA con 10-fold

###Caso mediana
cl <- makePSOCKcluster(4)
registerDoParallel(cl)  
set.seed(100)

lda.fit.median = train(etiquetas.train ~ ., data=mean.train1, method="lda",
                trControl = trainControl(method = "cv"))

stopCluster(cl)

lda.fit.median

#probamos el modelo en el conjunto de prueba
pred.lda.median = predict(lda.fit.median, mean.test)
(sum(pred.lda.median == etiquetas.test)/length(etiquetas.test))*100  #Nivel de precision

###Caso cuantiles centrales
cl <- makePSOCKcluster(4)
registerDoParallel(cl)  
set.seed(100)
#Test
lda.fit.quant = train(etiquetas.train ~ ., data=quant.train1, method="lda",
                       trControl = trainControl(method = "cv"))

stopCluster(cl)

lda.fit.quant

#probamos el modelo en el conjunto de prueba
pred.lda.quant = predict(lda.fit.quant, quant.test)
(sum(pred.lda.quant == etiquetas.test)/length(etiquetas.test))*100  #Nivel de precision


##########Aplicamos QDA con 10-fold

### Caso mediana
cl <- makePSOCKcluster(4)

registerDoParallel(cl)
set.seed(100)
fit.qda.median = train(etiquetas.train ~ ., data=mean.train1, method="qda",
            trControl = trainControl(method = "cv"))

## When you are done:
stopCluster(cl)

fit.qda.median #Precision en entrenamiento con 10-fold

pred.qda.median = predict(fit.qda.median, mean.test)

(sum(pred.qda.median == etiquetas.test)/length(etiquetas.test))*100 #Nivel de precision en datos de prueba

### Caso cuantiles centrales
#Agregamos un poco de ruido a la informacion para que se cumplan las condiciones del clasificador
train.qda.quant<-apply(quant.train,2, jitter)
train.qda.quant<-data.frame(etiquetas.train,train.qda.quant)
test.qda.quant<-apply(quant.test,2,jitter)

cl <- makePSOCKcluster(4)

registerDoParallel(cl)
set.seed(100)
fit.qda.quant = train(etiquetas.train ~ ., data=train.qda.quant, method="qda",
                       trControl = trainControl(method = "cv"))

## When you are done:
stopCluster(cl)

fit.qda.quant #Precision en entrenamiento con 10-fold

pred.qda.quant = predict(fit.qda.quant, test.qda.quant)

(sum(pred.qda.quant == etiquetas.test)/length(etiquetas.test))*100 #Nivel de precision en datos de prueba


########### Arboles de decision
##Caso mediana
cl <- makePSOCKcluster(4)
registerDoParallel(cl)  

tune.gridcart <- expand.grid(cp=seq(0.00005,0.05,0.005)) # Tunear usando parametro de complejidad

cctrl1 <- trainControl(method = "cv", number = 5, returnResamp = "all")
fit.tree.mean <- train(etiquetas.train ~ ., data = mean.train1, 
                  method = "rpart", 
                  tuneGrid =tune.gridcart,
                  trControl = cctrl1
)
#Stop after trainning
stopCluster(cl)

plot(fit.tree.mean,xlab="Complejidad",ylab="Precisión", main="Arbol de decisión 5-fold",col="red",sub="Mediana")

#Nivel de precisión datos de entrenamiento
(sum(predict(fit.tree.mean,mean.train)==etiquetas.train)/length(etiquetas.train))*100

#Nivel de precisión datos de prueba
(sum(predict(fit.tree.mean,mean.test)==etiquetas.test)/length(etiquetas.test))*100


##Caso mediana
cl <- makePSOCKcluster(4)
registerDoParallel(cl)  

fit.tree.quant <- train(etiquetas.train ~ ., data = quant.train1, 
                       method = "rpart", 
                       tuneGrid =tune.gridcart,
                       trControl = cctrl1
)
#Stop after trainning
stopCluster(cl)

plot(fit.tree.quant,xlab="Complejidad",ylab="Precisión", main="Arbol de decisión 5-fold",col="red",sub="Cuantiles centrales")

#Nivel de precisión datos de entrenamiento
(sum(predict(fit.tree.quant,quant.train)==etiquetas.train)/length(etiquetas.train))*100

#Nivel de precisión datos de prueba
(sum(predict(fit.tree.quant,quant.test)==etiquetas.test)/length(etiquetas.test))*100

################ SVM
#Caso mediana

library(e1071)

set.seed(100)
svmTuneGrid <- data.frame(.C = 2^(-3:3))

cl <- makePSOCKcluster(4)
registerDoParallel(cl)  

trctrl <- trainControl(method = "cv", number = 5)
svm_Linear.mean <- train(etiquetas.train ~., data = mean.train1, method = "svmLinear",
                    tuneGrid = svmTuneGrid,
                    trControl=trctrl)

#Stop after trainning
stopCluster(cl)

plot(svm_Linear.mean, xlab="Costo", ylab="Precisión", main="SVM lineal 5-fold",col="red",sub="Mediana")

#Nivel de precision datos de entrenamiento
(sum(predict(svm_Linear.mean,mean.train)==etiquetas.train)/length(etiquetas.train))*100

#Nivel de precision datos de prueba
(sum(predict(svm_Linear.mean,mean.test)==etiquetas.test)/length(etiquetas.test))*100

###Caso cuantiles centrales



cl <- makePSOCKcluster(4)
set.seed(100)
registerDoParallel(cl)  

svm_Linear.quant <- train(etiquetas.train ~., data = quant.train1, method = "svmLinear",
                         tuneGrid = svmTuneGrid,
                         trControl=trctrl)

#Stop after trainning
stopCluster(cl)

plot(svm_Linear.quant, xlab="Costo", ylab="Precisión", main="SVM lineal 5-fold",col="red",sub="Cuantiles centrales")

#Nivel de precision datos de entrenamiento
(sum(predict(svm_Linear.quant,quant.train)==etiquetas.train)/length(etiquetas.train))*100

#Nivel de precision datos de prueba
(sum(predict(svm_Linear.quant,quant.test)==etiquetas.test)/length(etiquetas.test))*100


###### Adaboost
###Caso mediana
cl <- makePSOCKcluster(5)
registerDoParallel(cl)

library(adabag)
grid <- expand.grid(mfinal = c(18,24,30), maxdepth = c(6, 8, 10),
                    coeflearn = c("Zhu"))

cctrl1 <- trainControl(method = "cv", number = 3, returnResamp = "all",
                       classProbs = TRUE)


label.train<-as.factor(etiquetas.train)
label.test<-as.factor(etiquetas.test)

levels(label.train)<-unique(label.train)
levels(label.test)<-unique(label.test)

set.seed(100)
fit.ada.mean <- train(mean.train1[,-1], label.train, 
                             method = "AdaBoost.M1", 
                             trControl = cctrl1,
                             tuneGrid = grid)

stopCluster(cl)

plot(fit.ada.mean,sub="Mediana")

#Nivel de precision datos de entrenamiento
(sum(predict(fit.ada.mean,mean.train)==label.train)/length(label.train))*100

#nivel de precision datos de prueba
(sum(predict(fit.ada.mean,mean.test)==label.test)/length(label.test))*100

###Caso cuantiles centrales
cl <- makePSOCKcluster(5)
registerDoParallel(cl)
set.seed(100)
fit.ada.quant <- train(quant.train1[,-1], label.train, 
                      method = "AdaBoost.M1", 
                      trControl = cctrl1,
                      tuneGrid = grid)

stopCluster(cl)

plot(fit.ada.quant,sub="Cuantiles centrales")

#Nivel de precision datos de entrenamiento
(sum(predict(fit.ada.quant,quant.train)==label.train)/length(label.train))*100

#nivel de precision datos de prueba
(sum(predict(fit.ada.quant,quant.test)==label.test)/length(label.test))*100

#############Fotos reales

####Se usara AdaBoost con los parametros obtenidos por la validacion cruzada dado que es el clasificador
#   con mejor precision

setwd("~/ciencia de datos/Tarea4/reales/frutas_reales")

files <- list.files(path=".", pattern=".jpg",all.files=T, full.names=F, no.. = T) 
list_of_images = lapply(files, load.image)

mean.test.real<-as.data.frame(matrix(nrow = 1,ncol = 3))
quant.test.real<-as.data.frame(matrix(nrow = 1,ncol = 9))

#Transformamos las imágenes al espacio HSV y generamos las matrices a usar para prueba
for (i in 1:length(list_of_images)){ 
  a<-RGBtoHSV(list_of_images[[i]])
  quant.test.real[i,1:9]<-c(quantile(a[,,,1])[2:4],quantile(a[,,,2])[2:4],quantile(a[,,,3])[2:4])
  mean.test.real[i,1]<-median(a[,,,1])
  mean.test.real[i,2]<-median(a[,,,2])
  mean.test.real[i,3]<-median(a[,,,3])
}

#Liberamos memoria
rm(list_of_images)
rm(files)
rm(i)

#Generamos las etiquetas
etiquetas.reales<-c("Peach",rep("Avocado",2),rep("Braeburn",3), "Cherry", "Granny","Kiwi", rep("Orange",3),"Pineapple")
label.test.real<-as.factor(etiquetas.reales)
#levels(label.test.real)<-unique(label.test)

#Probamos nuestro conjunto "real" de prueba en los modelos generados por AdaBoost
predict(fit.ada.quant,quant.test.real[13])
table(label.test.real,predict(fit.ada.quant,quant.test.real))

predict(fit.ada.mean,mean.test.real)
table(label.test.real,predict(fit.ada.mean,mean.test.real))
