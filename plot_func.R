library(BEST)
library(ggplot2)

plot1 <- function(d_ind, num_run)
{
  switch(d_ind,
         case1 = {data_set = "body fat"
                  address = "./Data Sets/Body Fat/"},
         case2 = {data_set = "baseball"
                  address = "./Data Sets/Baseball/"},
         case3 = {data_set = "auto risk"
                  address = "./Data Sets/Auto Risk/"}
         )
  RMSEs = array(0, c(3,num_run,20))
  for (run in 1:num_run) {
    RMSEs[3,run,] = as.numeric(read.table(paste(address,"result_ref_1_",run,".txt",sep="")))
    for (method in 0:1){
      result = as.matrix(read.table(paste(address,"result_proxy_",method,"_",run,".txt", sep="")))
      result = result[c(nrow(result):(nrow(result)-19)),1]
      RMSEs[method+1,run,] = result
    }
  }
  
  avg_rmse = apply(RMSEs, c(1,3), mean)
  sd_rmse = apply(RMSEs, c(1,3), sd)
  
  png(filename = paste("acc size ",data_set,".png",sep=""), width = 7, height = 7, units = "in", res = 300)
  plot(c(2:21),avg_rmse[1,], type="o", col="red", lwd=1.5, pch=16, xlab="Tree Size", ylab="Root Mean Squared Error", cex.axis = 1.6, cex.lab = 1.6, ylim=range(min(avg_rmse)-0.5,max(avg_rmse)+0.5),xaxt="n")
  axis(1,seq(2,21,by=1),las=2,cex.axis=1.6)
  lines(c(2:21), avg_rmse[2,], type="o", col="blue", lwd=1.5, pch=16)
  lines(c(2:21), avg_rmse[3,], col="green", lwd=3, lty=3)
  legend(6.5,max(avg_rmse)+0.5,c("CART fitted to training data","CART fitted to ref mdl", "ref mdl"), lwd=c(1.5,1.5,3), lty=c(1,1,3), col=c("red","blue","green"), pch=c(16,16,NA), y.intersp=1.5,cex = 1.7,bty="n")
  dev.off()
  
  bayes.t.test_stats = data.frame(muDiff = rep(0,20), HDILow = rep(0,20), HDIUp = rep(0,20))
  for (size in c(1:20)){
    # Run the Bayesian analysis using the default broad prior:
    mcmcChain = BESTmcmc(as.vector(RMSEs[2,,size]), as.vector(RMSEs[1,,size]), doPriorsOnly = FALSE, numSavedSteps=12000, burnInSteps = 20000, thinSteps=5)
    testSummary = summary(mcmcChain)
    bayes.t.test_stats$muDiff[size] = testSummary["muDiff", "mean"]
    bayes.t.test_stats$HDILow[size] = testSummary["muDiff", "HDIlo"]
    bayes.t.test_stats$HDIUp[size] = testSummary["muDiff", "HDIup"]
  }
  
  p = ggplot(bayes.t.test_stats, aes(x=c(2:21), y=muDiff)) +
    geom_line() + geom_point(size=3, shape=21, fill="white") +
    geom_errorbar(aes(ymin=HDILow, ymax=HDIUp),  width=.2) +
    theme_grey() +
    theme(plot.title = element_text(size=20, face="bold", hjust=0.5),
          axis.text.x = element_text(size = 18, colour = "black", angle = 90, vjust=0.5, hjust=1),
          axis.text.y = element_text(size = 18, colour = "black"),
          axis.title.x = element_text(size = 19, colour = "black"),
          axis.title.y = element_text(size = 19, colour = "black"),
          panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"),
          plot.margin = unit(c(0.5, 0.5, 0.5, 0.5), "cm"))
  p + labs(title= expression(paste("Mean and 95% HDI of the Distribution of ", mu[2] - mu[1], sep = "")), y = "", x = "Tree Size") + scale_x_time(breaks = c(2:21), labels = c(2:21))
  ggsave(filename = paste("Bayesian_T_Test_",data_set,".png",sep=""), width = 7, height = 7, units = "in", dpi = 300)
  # dev.copy(png,"Bayesian_T_Test.png")
  
  png(filename = paste("Sample distribution ",data_set,".png",sep=""), width = 7, height = 7, units = "in", res = 300)
  plot(mcmcChain)
  dev.off()
}

plot2 <- function(d_ind, num_run)
{
  switch(d_ind,
         case1 = {data_set = "body fat"
         address = "./Data Sets/Body Fat/"},
         case2 = {data_set = "baseball"
         address = "./Data Sets/Baseball/"},
         case3 = {data_set = "auto risk"
         address = "./Data Sets/Auto Risk/"}
  )
  RMSEs = array(0, c(3,num_run,20))
  RMSEs_ref = array(0, c(3,num_run,20))
  for (run in 1:num_run) {
    for (method in 1:3){
      result = as.matrix(read.table(paste(address,"result_proxy_",method,"_",run,".txt", sep="")))
      result = result[c(nrow(result):(nrow(result)-19)),1]
      RMSEs[method,run,] = result
      RMSEs_ref[method,run,] = as.numeric(read.table(paste(address,"result_ref_",method,"_",run,".txt",sep="")))
    }
  }
  
  avg_rmse = apply(RMSEs, c(1,3), mean)
  avg_rmse_ref = apply(RMSEs_ref, c(1,3), mean)
  
  png(filename = paste("ref effect ",data_set,".png",sep=""), width = 7, height = 7, units = "in", res = 300)
  
  min_val = min(min(avg_rmse_ref),min(avg_rmse))-0.5
  plot(c(2:21),avg_rmse[1,], type="o", col="blue", lwd=1.5, pch=16, xlab="Tree Size", ylab="Root Mean Squared Error", cex.axis = 1.6, cex.lab = 1.6, ylim=range(min_val,max(avg_rmse)+0.5),xaxt="n")
  axis(1,seq(2,21,by=1),las=2,cex.axis=1.6)
  lines(c(2:21), avg_rmse_ref[1,], col="blue", lwd=3, lty=3)
  
  lines(c(2:21), avg_rmse[2,], type="o", col="red", lwd=1.5, pch=15)
  lines(c(2:21), avg_rmse_ref[2,], col="red", lwd=3, lty=3)
  
  lines(c(2:21), avg_rmse[3,], type="o", col="green", lwd=1.5, pch=17)
  lines(c(2:21), avg_rmse_ref[3,], col="green", lwd=3, lty=3)
  
  legend(9,max(avg_rmse)+0.5,c("CART fitted to BART#3", "BART#3", "CART fitted to BART#40",
                  "BART#40", "CART fitted to GP", "GP"),
         lwd=c(1.5,3,1.5,3,1.5,3), col=c("red","red","blue","blue","green","green"), 
         pch=c(16,NA,15,NA,17,NA), lty=c(1,3,1,3,1,3), y.intersp=1.5,cex = 1.5,bty="n")
  dev.off()
}
