library(Metrics)
library(MASS)
library(BART)
library(MLmetrics)

source("plot_func.R")

############# Set the running parameters #############

d_ind = 1 # data set index: 1 = body fat, 2 = baseball, 3 = auto risk
ref_effect = T # If True, the effect of reference model will be evaluated,
               # check Section 4.1.2 in the paper. If False, results of 
               # comparing interpretability utility with interpretability
               # prior approach will be generated, check Section 4.1.3 in the paper
num_run = 50

############# Load data #############

switch (d_ind,
  case_1 = {
    address = "./Data Sets/Body Fat/"
    target = "Bodyfat"
    data = as.matrix(read.csv(paste(address,"Bodyfat.csv",sep="")))
  },
  case_2 = {
    address = "./Data Sets/Baseball/"
    target = "Salary"
    data = as.matrix(read.table(paste(address,"baseball.dat",sep=""), header=TRUE))
  },
  case_3 = {
    address = "./Data Sets/Auto Risk/"
    target = "symboling"
    data = as.matrix(read.csv(paste(address,"Autorisk.csv",sep="")))
  }
)

nsamples = dim(data)[1]
nfeatures = dim(data)[2]

set.seed(as.numeric(1234))

############# Loop over num_run runs #############

for (run in 1:num_run) {

  ############# Split data into train and test #############

  trind = sample(c(1:nsamples),floor(0.75*nsamples))

  trdata = data[trind,]
  x.train = trdata[,!(colnames(data) %in% target)]
  y.train = trdata[,c(target)]

  tstdata = data[-trind,]
  x.test = tstdata[,!(colnames(data) %in% target)]
  y.test = tstdata[,c(target)]

  ############# Training data will be read later by the Python code #############
  write.csv(trdata,file = paste(address,"train_",run,".csv",sep=""),row.names=FALSE)
  write.csv(tstdata,file = paste(address,"test_",run,".csv",sep=""),row.names=FALSE)

  ############# Fit BART with ntree obtained with CV #############
  ############# check paper for best ntree for each data set #############

  ref.mdl.1 = wbart(x.train,y.train,x.test,power=1,ntree=40,
                    nskip=2000,ndpost=4000)

  ref1.mean.train = ref.mdl.1$yhat.train.mean
  ref1.var.train = apply(ref.mdl.1$yhat.train,2,var)

  write.table(RMSE(ref.mdl.1$yhat.test.mean,y.test),
              paste(address,"result_ref_1_",run,".txt",sep=""),
              row.names=FALSE,col.names=FALSE)

  ############# Generate training data for the proxy model #############

  train.proxy.mdl = cbind(x.train,ref1.mean.train,ref1.var.train)
  colnames(train.proxy.mdl) = c(colnames(trdata),"predictive_var")
  write.csv(train.proxy.mdl,file=paste(address,"proxy_train_1_",run,".csv",sep=""),row.names=FALSE)

  ############# Fit proxy models to the data and ref. model #############

  cmd = paste("python Fit_Proxy.py ",run," ",d_ind," ",0," ",1,sep="")
  system('cmd.exe',input=cmd)

  if(ref_effect){

    ############# Fit BART with ntree = 3 #############

    ref.mdl.2 = wbart(x.train,y.train,x.test,power=1,ntree=3,
                      nskip=2000,ndpost=4000)

    ref2.mean.train = ref.mdl.2$yhat.train.mean
    ref2.var.train = apply(ref.mdl.2$yhat.train,2,var)

    write.table(RMSE(ref.mdl.2$yhat.test.mean,y.test),
                paste(address,"result_ref_2_",run,".txt",sep=""),
                row.names=FALSE,col.names=FALSE)

    ############# Generate training data for the proxy model #############

    train.proxy.mdl = cbind(x.train,ref2.mean.train,ref2.var.train)
    colnames(train.proxy.mdl) = c(colnames(trdata),"predictive_var")
    write.csv(train.proxy.mdl,file=paste(address,"proxy_train_2_",run,".csv",sep=""),row.names=FALSE)

    ############### fit models with different alpha values ################

    cmd = paste("python Fit_Proxy.py ",run," ",d_ind," ",0," ",2,sep="")
    system('cmd.exe',input=cmd)
  }
}

if(ref_effect)
{
  ############# Fit GP as the reference model #############

  cmd = paste("python GP_Ref.py ",d_ind," ",num_run," ",0," ",3,sep="")
  system('cmd.exe',input=cmd)
}

if(!ref_effect){
  plot1(d_ind, num_run)
  }else{
  plot2(d_ind, num_run)
}

