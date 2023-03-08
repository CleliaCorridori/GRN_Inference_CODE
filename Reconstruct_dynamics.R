args <- commandArgs(trailingOnly = T)
finit <- args[1]
fA <- args[2]
fout <- args[3]
tfnum <- as.integer(args[4])

init <- as.matrix(read.table(finit, sep="\t"))[1:tfnum,2]

A <- as.matrix(read.table(fA, sep=" ", header = FALSE))

tmp <- eigen(A)
U <- tmp$vectors
invU <- solve(U)
l <- tmp$values

time.len <-100
mean <- matrix(nrow=tfnum, ncol=time.len+1)

tmpt <- (0:time.len)*0.01
for(i in 1:length(tmpt)){
	t <- tmpt[i]
	eAt <- Re(U %*% diag(exp(l*t)) %*% invU)
	#write.table(dim(eAt), "prova.txt", sep="\t", col.names=F, row.names=F)
	mean[,i] <- eAt %*% init
}

write.table(rbind(tmpt, mean), fout, sep="\t", col.names=F, row.names=F)
