rm(list=ls())
setwd("~/ciencia de datos/Tarea4")
source("ADAboost.R")
etiquetas <- read.csv("~/ciencia de datos/Tarea4/class_img_exp.dat", sep="")
colnames(etiquetas)<-c("files","semanticExp","fileExp")
library(tiff)
library(imager)
setwd("~/ciencia de datos/Tarea4/img_expression/first/train")
#Leyendo imagenes
path.train="~/ciencia de datos/Tarea4/img_expression/first/train"
path.test="~/ciencia de datos/Tarea4/img_expression/first/test"
files_train <- list.files(path=path.train, pattern=".tiff",all.files=T, full.names=F, no.. = T) 
files_test <- list.files(path=path.test, pattern=".tiff",all.files=T, full.names=F, no.. = T) 

#Generamos las etiquetas
etiquetas_train<- as.data.frame(files_train)
colnames(etiquetas_train)<-"files"
etiquetas_train <- merge(etiquetas_train,etiquetas,by="files")

etiquetas_test<- as.data.frame(files_test)
colnames(etiquetas_test)<-"files"
etiquetas_test <- merge(etiquetas_test,etiquetas,by="files")

#Generamos las imagenes de entrenamiento y prueba
images_train = lapply(files_train, readTIFF)
setwd("~/ciencia de datos/Tarea4/img_expression/first/test")
images_test = lapply(files_test, readTIFF)

#Transformando las imagenes a un formato imager y en escala de grises
for (i in 1:length(images_train)){
  images_train[[i]]<-grayscale(as.cimg(images_train[[i]]))
}


for (i in 1:length(images_test)){
  images_test[[i]]<-grayscale(as.cimg(images_test[[i]]))
}
#Generando la matriz de expresiones
matriz_expres<-matrix(0L,ncol=length(images_train),nrow=dim(images_train[[1]])[1]*dim(images_train[[2]])[1])
for (i in 1:length(images_train)){
  matriz_expres[,i]<-c(images_train[[i]])
}

#Generamos la media de las imagenes para centrar los datos
mean_images<-apply(matriz_expres,1,mean)
plot(as.cimg(matrix(mean_images, ncol=256, byrow=TRUE)))
#Centramos la matriz de expresiones
matriz_expres<-matriz_expres-mean_images
plot(as.cimg(matrix(matriz_expres[,1], ncol=256, byrow=TRUE)))
#Calculamos los valores propios
svd.matrix<-svd(matriz_expres)
#Primeras 5 eigenfaces
plot(as.cimg(matrix(svd.matrix$u[,1], ncol=256, byrow=TRUE)),axes=FALSE)
plot(as.cimg(matrix(svd.matrix$u[,2], ncol=256, byrow=TRUE)),axes=FALSE)
plot(as.cimg(matrix(svd.matrix$u[,3], ncol=256, byrow=TRUE)),axes=FALSE)
plot(as.cimg(matrix(svd.matrix$u[,4], ncol=256, byrow=TRUE)),axes=FALSE)
plot(as.cimg(matrix(svd.matrix$u[,5], ncol=256, byrow=TRUE)),axes=FALSE)

#Calculando el criterop de selección del número de componentes principales
sum.varianzas<-sum((svd.matrix$d)^2)
var.acum<-(cumsum(svd.matrix$d^2)/sum.varianzas)*100  #Calculamos el porcentaje de varianza que explica cada componente principal
eigenfaces<-svd.matrix$u[,1:22]  #Se usaron 22 componentes principales porque explican cerca de 88% de la varianza
imag_red<-t(eigenfaces) %*% matriz_expres

imag_red<-t(imag_red)

i=10 # Parametro para cambiar el número de iteraciones en ADAboost

#########Aplicamos ADAboost a los datos de entrenamiento para clasificarlos 
etiquetas_train_factor<-as.factor(as.numeric(as.factor(etiquetas_train[,2])))
a<-boost(imag_red,etiquetas_train_factor,M=i,K=5,maxdepth = 2)
b<-predict.boost(a, imag_red,etiquetas_train_factor)
table(b$class,etiquetas_train[,2])   # Tabla de confusion para datos de entrenamiento
(sum(b$class==etiquetas_train_factor)/dim(imag_red)[1])*100


###########ADAboost para los datos de prueba
#Generando la matriz de expresiones
matriz_expres_prueba<-matrix(0L,ncol=length(images_test),nrow=dim(images_test[[1]])[1]*dim(images_test[[2]])[1])
for (i in 1:length(images_test)){
  matriz_expres_prueba[,i]<-c(images_test[[i]])
}
#centramos la matriz de prueba
matriz_expres_prueba<-matriz_expres_prueba-mean_images
#Proyectamos los datos de prueba
imag_red_test<-t(eigenfaces) %*% matriz_expres_prueba
imag_red_test<-t(imag_red_test)

#Aplicamos ADAboost a los datos de prueba proyectados
etiquetas_test_factor<-as.factor(as.numeric(as.factor(etiquetas_test[,2])))
b<-predict.boost(a, imag_red_test,etiquetas_test_factor)
table(b$class,etiquetas_test[,2])   # Tabla de confusion para datos de entrenamiento
(sum(b$class==etiquetas_test_factor)/dim(imag_red_test)[1])*100 


######Aplicando ADAboost considerando las etiquetas tipo file.expression

#Aplicamos ADAboost a los datos de entrenamiento para clasificarlos 
etiquetas_train_factor2<-as.factor(as.numeric(as.factor(etiquetas_train[,3])))
a1<-boost(imag_red,etiquetas_train_factor2,M=i,K=6,maxdepth = 2)
b1<-predict.boost(a1, imag_red,etiquetas_train_factor2)
table(b1$class,etiquetas_train[,3])   # Tabla de confusion para datos de entrenamiento
(sum(b1$class==etiquetas_train_factor2)/dim(imag_red)[1])*100 


#Aplicamos ADAboost a los datos de prueba proyectados
etiquetas_test_factor2<-as.factor(as.numeric(as.factor(etiquetas_test[,3])))
b1<-predict.boost(a1, imag_red_test,etiquetas_test_factor2)
table(b1$class,etiquetas_test[,3])   # Tabla de confusion para datos de entrenamiento
(sum(b1$class==etiquetas_test_factor2)/dim(imag_red_test)[1])*100 

############ El siguiente codigo es solo para generar las graficas, no correr debido a que puede llevar mucho tiempo
########### Mas adelante se encuentra la parte de las fotos reales
matriz.comportamiento<-matrix(0L, ncol=4, nrow=100)
for (j in 1:100){
  a<-boost(imag_red,etiquetas_train_factor,M=j,K=5,maxdepth = 2)
  b<-predict.boost(a, imag_red,etiquetas_train_factor)
  matriz.comportamiento[j,1]<-(sum(b$class==etiquetas_train_factor)/dim(imag_red)[1])*100
  b<-predict.boost(a, imag_red_test,etiquetas_test_factor)
  matriz.comportamiento[j,2]<-(sum(b$class==etiquetas_test_factor)/dim(imag_red_test)[1])*100
  a1<-boost(imag_red,etiquetas_train_factor2,M=j,K=6,maxdepth = 2)
  b1<-predict.boost(a1, imag_red,etiquetas_train_factor2)
  matriz.comportamiento[j,3]<-(sum(b1$class==etiquetas_train_factor2)/dim(imag_red)[1])*100
  b1<-predict.boost(a1, imag_red_test,etiquetas_test_factor2)
  matriz.comportamiento[j,4]<-(sum(b1$class==etiquetas_test_factor2)/dim(imag_red_test)[1])*100
}

plot(matriz.comportamiento[,1],type="l",col="red",xlab="Iteraciones",ylab="Nivel de predicción",main="Entrenamiento: Maxdepth=2")
lines(matriz.comportamiento[,3],type="l",col="blue")
legend(50,70,col=c("red","blue"),lty=1, cex=0.8,legend=c("Semantic", "File"),box.lty=0)
plot(matriz.comportamiento[,2],type="l",col="red",xlab="Iteraciones",ylab="Nivel de predicción",main="Prueba: Maxdepth=2",ylim=c(25,65))
lines(matriz.comportamiento[,4],type="l",col="blue")
legend(50,50,col=c("red","blue"),lty=1, cex=0.8,legend=c("Semantic", "File"),box.lty=0)

matriz.comportamiento2<-matrix(0L, ncol=4, nrow=100)
for (j in 1:100){
  a<-boost(imag_red,etiquetas_train_factor,M=j,K=5,maxdepth = 3)
  b<-predict.boost(a, imag_red,etiquetas_train_factor)
  matriz.comportamiento2[j,1]<-(sum(b$class==etiquetas_train_factor)/dim(imag_red)[1])*100
  b<-predict.boost(a, imag_red_test,etiquetas_test_factor)
  matriz.comportamiento2[j,2]<-(sum(b$class==etiquetas_test_factor)/dim(imag_red_test)[1])*100
  a1<-boost(imag_red,etiquetas_train_factor2,M=j,K=6,maxdepth = 3)
  b1<-predict.boost(a1, imag_red,etiquetas_train_factor2)
  matriz.comportamiento2[j,3]<-(sum(b1$class==etiquetas_train_factor2)/dim(imag_red)[1])*100
  b1<-predict.boost(a1, imag_red_test,etiquetas_test_factor2)
  matriz.comportamiento2[j,4]<-(sum(b1$class==etiquetas_test_factor2)/dim(imag_red_test)[1])*100
}

plot(matriz.comportamiento2[,1],type="l",col="red",xlab="Iteraciones",ylab="Nivel de predicción",main="Entrenamiento: Maxdepth=3")
lines(matriz.comportamiento2[,3],type="l",col="blue")
legend(50,70,col=c("red","blue"),lty=1, cex=0.8,legend=c("Semantic", "File"),box.lty=0)
plot(matriz.comportamiento2[,2],type="l",col="red",xlab="Iteraciones",ylab="Nivel de predicción",main="Prueba: Maxdepth=3",ylim=c(30,70))
lines(matriz.comportamiento2[,4],type="l",col="blue")
legend(50,53,col=c("red","blue"),lty=1, cex=0.8,legend=c("Semantic", "File"),box.lty=0)


matriz.comportamiento3<-matrix(0L, ncol=4, nrow=100)
for (j in 1:100){
  a<-boost(imag_red,etiquetas_train_factor,M=j,K=5,maxdepth = 10)
  b<-predict.boost(a, imag_red,etiquetas_train_factor)
  matriz.comportamiento3[j,1]<-(sum(b$class==etiquetas_train_factor)/dim(imag_red)[1])*100
  b<-predict.boost(a, imag_red_test,etiquetas_test_factor)
  matriz.comportamiento3[j,2]<-(sum(b$class==etiquetas_test_factor)/dim(imag_red_test)[1])*100
  a1<-boost(imag_red,etiquetas_train_factor2,M=j,K=6,maxdepth = 10)
  b1<-predict.boost(a1, imag_red,etiquetas_train_factor2)
  matriz.comportamiento3[j,3]<-(sum(b1$class==etiquetas_train_factor2)/dim(imag_red)[1])*100
  b1<-predict.boost(a1, imag_red_test,etiquetas_test_factor2)
  matriz.comportamiento3[j,4]<-(sum(b1$class==etiquetas_test_factor2)/dim(imag_red_test)[1])*100
}


plot(matriz.comportamiento3[,1],type="l",col="red",xlab="Iteraciones",ylab="Nivel de predicción",main="Entrenamiento: Maxdepth=10")
lines(matriz.comportamiento3[,3],type="l",col="blue")
legend(50,80,col=c("red","blue"),lty=1, cex=0.8,legend=c("Semantic", "File"),box.lty=0)
plot(matriz.comportamiento3[,2],type="l",col="red",xlab="Iteraciones",ylab="Nivel de predicción",main="Prueba: Maxdepth=10",ylim=c(30,75))
lines(matriz.comportamiento3[,4],type="l",col="blue")
legend(50,55,col=c("red","blue"),lty=1, cex=0.8,legend=c("Semantic", "File"),box.lty=0)


##############  Probando con fotos reales el modelo
library(imager)
setwd("~/ciencia de datos/Tarea4/reales/facial_expresion")
files_reales <- list.files(path=".", pattern=".jpg",all.files=T, full.names=F, no.. = T) 
real_images = lapply(files_reales, load.image)

mat_realImag<-matrix(0L, ncol=19, nrow=256*256)
for (i in 1:length(real_images)){
  real_images[[i]]<-grayscale(real_images[[i]])
}

for (i in 1:length(real_images)){
  mat_realImag[,i]<-c(real_images[[i]])
}

#Reduciendo la matriz al espacio de eigenfaces
mat_realImag<-mat_realImag-mean_images   #Centramos la matriz
real_reduc<-t(eigenfaces) %*%mat_realImag

#generamos las etiquetas
real_seman<-c("ANG","ANG","DIS","DIS","DIS","HAP","HAP","HAP","HAP","SAD","SAD","HAP","SAD","SAD","SAD","SUR","SUR","SUR","SUR")
real_file<-c("ANG","ANG","DIS","DIS","DIS","HAP","HAP","HAP","HAP","NEU","NEU","NEU","NEU","SAD","SAD","SUR","SUR","SUR","SUR")
real_seman1<-as.factor(as.numeric(as.factor(real_seman)))
real_file1<-as.factor(as.numeric(as.factor(real_file)))

#vamos a considerar arboles con maxima profundidad de 10 y 50 iteraciones que son las que mostraron un nivel
#decente de precisión
a<-boost(imag_red,etiquetas_train_factor,M=50,K=5,maxdepth = 10)
b<-predict.boost(a, t(real_reduc),real_seman1)
table(b$class,real_seman) 
(sum(b$class==real_seman1)/dim(t(real_reduc))[1])*100

a1<-boost(imag_red,etiquetas_train_factor2,M=50,K=6,maxdepth = 10)
c<-predict.boost(a1, t(real_reduc),real_file1)
table(c$class,real_file) 
(sum(c$class==real_file1)/dim(t(real_reduc))[1])*100


