library(ggplot2)
library(latex2exp)
library(extrafont)
font_import()
par(family = "LM Roman 10")

###################################
# extract relevant data
###################################

files <- list.files(".", pattern = "2019")
filesFull <- lapply(files, function(x) list.files(x, pattern = "csv$", recursive = TRUE, full.names = TRUE))

# warning: these are old salvaging functions and will not work on new Results.csv files
salvage <- function(row){
  a <- lapply(row, function(x) as.numeric(strsplit(grep("weighted", strsplit(as.character(x), "\n")[[1]], value = TRUE), "\\s+")[[1]][3:5]))
  return(do.call("c",a))
}

salvage2 <- function(row){
  a <- lapply(strsplit(strsplit(as.character(row),"\n")[[1]][3:15],"\\s+"), function(x) x[5])
  return(as.numeric(do.call("c",a)))
}

salvage3 <- function(row){
  a <- lapply(strsplit(strsplit(as.character(row),"\n")[[1]][3:15],"\\s+"), function(x) x[6])
  return(as.numeric(do.call("c",a)))
}

# process simple data
dfMain = data.frame()
for(i in 1:length(filesFull)){
  first <- read.csv(filesFull[[i]][2], stringsAsFactors = FALSE)
  second <- read.csv(filesFull[[i]][1], stringsAsFactors = FALSE)
  tmp1 <- rbind(cbind(sub(".*m_", "m_",files[i]),first[,c("Counter","MinImprov","Method","LernR","Momentum","Decay","Regular.","Hidden")]))
  tmp2 <- lapply(1:nrow(second), function(i) cbind(second[i,1], t(salvage(second[i,-1]))))
  tmp2 <- do.call("rbind",tmp2)
  main <- merge(tmp1,tmp2,by.x=2,by.y=1)
  dfMain <- rbind(dfMain,main)
}

# process simple data
dfClass = data.frame()
for(i in 1:length(filesFull)){
  first <- read.csv(filesFull[[i]][2], stringsAsFactors = FALSE)
  second <- read.csv(filesFull[[i]][1], stringsAsFactors = FALSE)
  tmp1 <- rbind(cbind(sub(".*m_", "m_",files[i]),first["Counter"]))
  tmp2 <- lapply(1:nrow(second), function(i) c(second[i,1], salvage2(second[i,2]), salvage3(second[i,2])))
  tmp2 <- do.call("rbind",tmp2)
  main <- merge(tmp1,tmp2,by.x=2,by.y=1)
  dfClass <- rbind(dfClass,main)
}

names(dfMain)[c(2,10:18)] <- c("Model","Train_Precision","Train_Recall","Train_F1",
                               "Dev_Precision","Dev_Recall","Dev_F1",
                               "Test_Precision","Test_Recall","Test_F1")

labels <- c(0,3,6,7,8,9,10,11,13,14,16,17,19)
names(dfClass)[c(2:28)] <- c("Model",0,3,6,7,8,9,10,11,13,14,16,17,19,paste0(labels,"_freq"))

###################################
# plot combined PR curves
###################################

# input best model data
test <- dfMain[which(!dfMain[,"Model"] %in% c("m_1_push", "m_1_alt")),]
test <- gdata::drop.levels(test, reorder = FALSE)
fun <- function(x) {x/(5*x-1)}
levels(test$Model) <- c(TeX("$M_0$"),TeX("$M_1$"),TeX("$M_2$"),TeX("$M_3$"),TeX("$M_4$"),
                        TeX("$M_5$"),TeX("$M_6$"),TeX("$M_7$"),TeX("$M_8$"),TeX("$M_9$"),TeX("$M_{10}$"),
                        TeX("$M_{11}$"))

png("combined.png", width =1600, height=1800, res = 200)
p <- ggplot(data = test) + geom_point(aes(x=Test_Recall, y=Test_Precision, colour="Test"),alpha=0.4) +
  geom_point(aes(x=Dev_Recall, y=Dev_Precision, colour="Dev"),alpha=0.4) +
  geom_point(aes(x=Train_Recall, y=Train_Precision, colour="Train"),alpha=0.4) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", size = 0.2) +
  annotate("text", x=0.34, y=0.44, label=TeX("$F_1=0.4$"), size=2.5)+
  stat_function(fun = fun, size = 0.1)+
  theme_bw() + ylim(c(0.27,0.46)) + xlim(c(0.27,0.46)) +
  xlab("\nRecall") + ylab("Precision\n") + 
  scale_colour_manual("Legend",values=c("Test"="red", "Dev"="green", "Train"="blue"),
                      breaks = c("Train", "Dev", "Test")) +
  theme(legend.background = element_rect(fill="lightgray",
                                   size=0.5, linetype="solid", 
                                   colour ="black"),strip.text.x = element_text(size = 10))

q <- p + facet_wrap( ~ Model, ncol=3,labeller = label_parsed)
print(q)
dev.off()

###################################
# plot training f1 wrt. sample size
###################################

test <- data.frame(matrix(nrow=20))
test[,1] <- c(0:19)
test <- cbind(test,t(dfClass[1,23:42]))

i = 1
for(name in names(dfClass)[3:22]){
  test[i,3] <- mean(dfClass[,name])
  test[i,4] <- max(dfClass[,name])
  test[i,5] <- min(dfClass[,name])
  i = i + 1
}

names(test) <- c("x","y","mean","max","min")
row.names(test) <- NULL
test[,"mean"] <- test[,"mean"]

png("trainComp.png", width =1800, height=1200, res = 200)
g <- ggplot(test,aes(x=x,y=y)) + geom_bar(stat = "identity",colour = "black", fill = "blue", alpha = 0.5, size = 0.2) + theme_bw() + ylab("Training Instances") + xlab("Class") +
  theme(plot.margin=unit(c(0.5,0.5,0.5,0.5),"cm"), plot.title = element_text(hjust = 0.5), axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0))) +
  scale_x_continuous(breaks = seq(0,19, by = 1)) + ggtitle("Training Instance Distribution")
q <- ggplot(test,aes(x=x,y=y)) + geom_line(aes(x=x,y=mean),size=0.3) + theme_bw() + ylab(TeX("Mean Training $F_1$ score")) + xlab("Class") +
  theme(plot.margin=unit(c(0.5,0.5,0.5,0.5),"cm"), plot.title = element_text(hjust = 0.5),axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)))+ ggtitle(TeX("Training $F_1$ Distribution")) +
  scale_x_continuous(breaks = seq(0,19, by = 1))
res <- grid.arrange(g, q, ncol=2)
print(res)
dev.off()

###################################
# plot keras distribution run
###################################

add <- data.frame()
thing <- read.csv("m1_alt_push_keras_results.csv", stringsAsFactors = FALSE)

add <- cbind(thing["Counter"],"m_1_alt_push_2",NA,NA,NA,NA,NA,NA,NA,thing["Train.Precision"],thing["Train.Recall"],
             thing["Train.F1"],thing["Valid.Precision"],thing["Valid.Recall"],thing["Valid.F1"],
             thing["Test.Precision"],thing["Test.Recall"],thing["Test.F1"])
names(add) <- names(dfMain)

png("kerasRun.png", width =1600, height=1400, res = 200)
p <- ggplot(data = add) + geom_point(aes(x=Test_Recall, y=Test_Precision, colour="Test"),alpha=0.8) +
  geom_point(aes(x=Dev_Recall, y=Dev_Precision, colour="Dev"),alpha=0.8) +
  geom_point(aes(x=Train_Recall, y=Train_Precision, colour="Train"),alpha=0.8) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", size = 0.2) +
  annotate("text", x=0.35, y=0.44, label=TeX("$F_1=0.4$"), size=3.5)+
  stat_function(fun = fun, size = 0.1)+
  theme_bw() + ylim(c(0.20,0.46)) + xlim(c(0.20,0.46)) +
  ggtitle(TeX("PR curve for $M_{opt}$")) +
  xlab("\nRecall") + ylab("Precision\n") + 
  scale_colour_manual("Legend",values=c("Test"="red", "Dev"="green", "Train"="blue"),
                      breaks = c("Train", "Dev", "Test")) +
  theme(legend.background = element_rect(fill="lightgray",
                                         size=0.5, linetype="solid", 
                                         colour ="black"),strip.text.x = element_text(size = 8.5),plot.title = element_text(hjust = 0.5))
print(p)
dev.off()