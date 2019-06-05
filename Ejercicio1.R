setwd("~/ciencia de datos/Tarea4")
source("ADAboost.R")

################## Conjunto de datos de vinos

wine.data<-read.csv("wine.data")
plot(wine.data[,2:3],col=wine.data[,1])   #Cargando los datos
ncol(wine.data)
sample_set<-sample(nrow(wine.data),round(nrow(wine.data)*.75) )
wine.data_train<-wine.data[sample_set,]
wine.data_test<-wine.data[-sample_set,]

a<-boost(wine.data_train[,-1], wine.data_train[,1],5,K=3)
b<-predict.boost(a,wine.data_train[,-1],wine.data_train[,1])
table(b$class,wine.data_train[,1])   # Tabla de confusion para datos de entrenamiento
(sum(b$class==wine.data_train[,1])/dim(wine.data_train)[1])*100  #Precision para datos de entrenamiento

c<-predict.boost(a,wine.data_test[,-1],wine.data_test[,1])
table(c$class,wine.data_test[,1])   # Tabla de confusion para datos de prueba
(sum(c$class==wine.data_test[,1])/dim(wine.data_test)[1])*100  #Precision para datos de prueba


################################## Prueba considerando el dataset iris
iris_prueba<-iris[,-5]
iris_y<-as.numeric(as.factor(iris[,5]))
iris_y<-as.factor(iris_y)

a<-boost(iris_prueba, iris_y,5, K=3)
b<-predict.boost(a,iris_prueba,iris_y)
table(b$class,iris_y)   # Tabla de confusion para datos de entrenamiento
(sum(b$class==iris_y)/dim(iris_prueba)[1])*100  #Precision para datos de entrenamiento

################################ Prueba usando el dataset letters
letter.recog<-read.csv("letter-recognition.data")
letter.prueba<-letter.recog[1:16000,-1]
letter.test<-letter.recog[16001:dim(letter.recog)[1],-1]
letter.etiqueta<-as.numeric(as.factor(letter.recog[,1]))
letter.etiqueta<-as.factor(letter.etiqueta)

a<-boost(letter.prueba, letter.etiqueta[1:16000],100, K=26, normalpha = TRUE,maxdepth = 7)
b<-predict.boost(a,letter.prueba, letter.etiqueta[1:16000])
table(b$class,letter.etiqueta[1:16000])   # Tabla de confusion para datos de entrenamiento
(sum(b$class==letter.etiqueta[1:16000])/dim(letter.prueba)[1])*100 #Precision para datos de entrenamiento

c<-predict.boost(a,letter.test,letter.etiqueta[16001:length(letter.etiqueta)])
table(c$class,letter.etiqueta[16001:length(letter.etiqueta)])   # Tabla de confusion para datos de prueba
(sum(c$class==letter.etiqueta[16001:length(letter.etiqueta)])/dim(letter.test)[1])*100  #Precision para datos de prueba

############### No correr, es muy tardado
############## Es solo para crear una grafica de precision con respecto a la profundidad del arbol
effect.maxdepth<-rep(1,20)
count=1
for (i in c(3,7)){
  for (j in seq(10,100,10)){
    a<-boost(letter.prueba, letter.etiqueta[1:16000],j, K=26, normalpha = TRUE,maxdepth = i)
    c<-predict.boost(a,letter.test,letter.etiqueta[16001:length(letter.etiqueta)])
    effect.maxdepth[count]<-round((sum(c$class==letter.etiqueta[16001:length(letter.etiqueta)])/dim(letter.test)[1])*100,2)
    count = count+1
  }
}



plot(seq(10,100,10),effect.maxdepth[1:10], type="o", col="blue",ylab="Precisión", xlab="Número de iteraciones",main="Letters dataset",ylim=c(40,100))
lines(seq(10,100,10), effect.maxdepth[11:20],type="o", pch=22, lty=1, col="red")
legend(45,85,legend=c("profundidad 3", "profundidad 7"), col = c("blue","red"),lty=1, cex=0.8,
       box.lty=0)
