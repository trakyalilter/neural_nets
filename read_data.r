data_faulty <- load("/home/ilter-cnc/Downloads/TEP_Faulty_Testing.RData")
ls()
#write.csv(data_faulty, file = "~/Desktop/NN_ws/TEP_Faulty_Testing.csv",
#append = FALSE, quote = TRUE, sep = ",")

load("/home/ilter-cnc/Downloads/TEP_FaultFree_Training.RData")
load("/home/ilter-cnc/Downloads/TEP_FaultFree_Testing.RData")
ls()