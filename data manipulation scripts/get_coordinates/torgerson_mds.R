library(phytools)
require(geiger)
require(maps)
library(ape)
library(matrixcalc)
library(igraph)
library(ade4)
library(rgl)
library(vegan)
library(ggplot2)
library(smacof)
#------------------load---------------------------------#
# The leaves are 307 because the variable unassigned was deleted

tree <- read.newick(file="~/Desktop/preprocessing308/RAxML_bestTree.raxml_ott.tre" )
is.rooted(tree) 

#Add taxonomy
table_correspondence_id_tax <- read.delim("~/Desktop/ThesisData/progettoR/table_correspondence_id_tax.txt")
 
 
hs <- read.delim("~/Desktop/ThesisData/progettoR/FBK_thesis/16S/otu_table_16S_L6_commsamp_HS.txt")
names <- as.data.frame(colnames(hs[-c(1,2)]))
table_correspondence_id_tax <- cbind(table_correspondence_id_tax,names)
tree$tip.label
index <- match(as.factor(tree$tip.label),table_correspondence_id_tax[,1])
tree$tip.label <- table_correspondence_id_tax[index,4]
write.tree(tree, file = "raxMl_label.tre", append = FALSE)

#----------Patristic Distance between leaves------------#
D <- cophenetic(tree)
is.positive.semi.definite(D) #FALSE
det(D)

#---------MDS PROCEDURE: MARK LAYER AND JOHN A.RHODES----#
#----------Convertion in Euclidean matrix----------------#
  D_eu <- sqrt(D)
  is.positive.semi.definite(D_eu)
  is.euclid(as.dist(D_eu), plot = FALSE, print = FALSE, tol = 1e-07)
  isSymmetric(D_eu)
  


#  F <-  diag(dim(D)[1])-(1/dim(D)[1])*(matrix(1,dim(D)[1],dim(D)[1]))
#  H <- -0.5*F%*%(D)^2%*%(F)
#  isSymmetric(H)
#  is.positive.semi.definite(round(H,3))
# 
# 
# # We have to find a matrix X of dimesion (n-1) x n s.t. 
# # H=X^tX that is the  Cholesky decomposition  
# # It is convinient to compute the eigenvalues 
#   
#   P <- eigen(H)
#   R <- P$vectors
#   Q <- diag(P$values)
#   
#   
#   
#   #configuration of points in the a euclidean space of dimension n-1 by col
#   X <- (Q[-dim(Q)[1],])^(0.5)%*%t(R) #leave out the last eigenvector that correspond to the eigevalue zero
#   h <- t(X)%*%X
#   
#   colnames(X) <- as.factor(row.names(D))
#  
#   
#   for( i in 1:dim(X)[2]){
#     if(colnames(X)[i]%in%annotation[,1]){
#       colnames(X)[i] <- as.character(annotation[which(annotation[,1]==colnames(X)[i]),2])
#       }
#   }
#   coordinate <- X
#   row.names(X) <- (1:dim(X)[1])
#   write.table(coordinate,file="coordinateR307.txt", quote = FALSE, sep="\t" ,row.names=T)
#   plot(t(X[1:2,]))
#  
#   
#   plot3d(X[1,], X[2,],X[3,],pch=30,size=5, col= 4)
# 
# # --------------------- R package for the MDS---------------------------------#
#   library("bios2mds")
# S <- mmds(sqrt(D_ucf),pc=250)
#   coordinate_bio <-  S$coord # scale factor sqrt(308)*X
#   scree.plot(S$eigen, lab = FALSE, title = NULL, xlim = NULL,
#            ylim = NULL, new.plot = TRUE, pdf.file = NULL)

#----------------------------------MDS------------------------------------------
X <- torgerson(D_eu, p=306) #307x306
#order variables as in the dataset
index_2 <- match( table_correspondence_id_tax[,4],as.data.frame(row.names(X))[,1])
X <- X[index_2,]
colnames(X) <- 1:306
write.table(X,file="coordinates_307x306.txt", quote = FALSE, sep="\t" ,row.names=T)
#--------------------- Coordinates for each dataset-------------------------#

#UCf
ucf <-  read.delim("~/Desktop/ThesisData/progettoR/HS_UCf/Sokol_16S_taxa_HS_UCflare_commsamp_training.txt", stringsAsFactors=F)[,-1]
names_ucf <- as.data.frame(colnames(ucf))
coord_names <- as.data.frame(row.names(X))
coord_ucf <- which(names_ucf[,1]%in%coord_names[,1])

#delete the coordinates that are not in the dataset 
row_del <-which(coord_names[,1]%in%names_ucf[,1]==F)
X_ucf <- X[-row_del,]
PCs <- 1:dim(X_ucf)[2]
X_ucf <- cbind(PCs, t(X_ucf))
write.table(X_ucf,file="coordinates_ucf.txt", quote = FALSE, sep="\t" ,row.names=F)

 
#---------------------------------UCr--------------------------------------------#
ucr <- read.delim("~/Desktop/ThesisData/progettoR/HS_UCr/Sokol_16S_taxa_HS_UCr_commsamp_training.txt")[-1]
names_ucr <- as.data.frame(colnames(ucr))
index <- which(names_ucr[,1]%in%coord_names[,1])
 
#delete the coordinates that are not in the dataset 
row_del <-which(coord_names[,1]%in%names_ucr[,1]==F)
X_ucr <- X[-row_del,]
PCs <- 1:dim(X_ucr)[2]
X_ucr <- cbind(PCs, t(X_ucr))
write.table(X_ucr,file="coordinates_ucr.txt", quote = FALSE, sep="\t" ,row.names=F)

#----------------------------------CDf----------------------------------------#
cdf <- read.delim("~/Desktop/ThesisData/progettoR/HS_CDf/Sokol_16S_taxa_HS_CDflare_commsamp_training.txt")[-1]
names_cdf <- as.data.frame(colnames(cdf))
index <- which(names_cdf[,1]%in%coord_names[,1])
 
#delete the coordinates that are not in the dataset 
row_del <-which(coord_names[,1]%in%names_cdf[,1]==F)
X_cdf <- X[-row_del,]
PCs <- 1:dim(X_cdf)[2]
X_cdf <- cbind(PCs, t(X_cdf))
write.table(X_cdf,file="coordinates_cdf.txt", quote = FALSE, sep="\t" ,row.names=F)
 
#---------------------------------CDr---------------------------------------------------#
cdr <- read.delim("~/Desktop/ThesisData/progettoR/HS_CDr/Sokol_16S_taxa_HS_CDr_commsamp_training.txt")[-1]
names_cdr <- as.data.frame(colnames(cdr))
index <- which(names_cdr[,1]%in%coord_names[,1])
 
#delete the coordinates that are not in the dataset 
row_del <-which(coord_names[,1]%in%names_cdr[,1]==F)
X_cdr <- X[-row_del,]
PCs <- 1:dim(X_cdr)[2]
X_cdr <- cbind(PCs, t(X_cdr))
write.table(X_cdr,file="coordinates_cdr.txt", quote = FALSE, sep="\t" ,row.names=F)
 
 #----------------------------------iCDf----------------------------------------#
icdf <- read.delim("~/Desktop/ThesisData/progettoR/HS_iCDf/Sokol_16S_taxa_HS_iCDflare_commsamp_training.txt")[-1]
names_icdf <- as.data.frame(colnames(icdf))
index <- which(names_icdf[,1]%in%coord_names[,1])

#delete the coordinates that are not in the dataset 
row_del <-which(coord_names[,1]%in%names_icdf[,1]==F)
X_icdf <- X[-row_del,]
PCs <- 1:dim(X_icdf)[2]
X_icdf <- cbind(PCs, t(X_icdf))
write.table(X_icdf,file="coordinates_icdf.txt", quote = FALSE, sep="\t" ,row.names=F)

#---------------------------------iCDr---------------------------------------------------#
icdr <- read.delim("~/Desktop/ThesisData/progettoR/HS_iCDr/Sokol_16S_taxa_HS_iCDr_commsamp_training.txt")[-1]
names_icdr <- as.data.frame(colnames(icdr))
index <- which(names_icdr[,1]%in%coord_names[,1])

#delete the coordinates that are not in the dataset 
row_del <-which(coord_names[,1]%in%names_icdr[,1]==F)
X_icdr <- X[-row_del,]
PCs <- 1:dim(X_icdr)[2]
X_icdr<- cbind(PCs, t(X_icdr))
write.table(X_icdr,file="coordinates_icdr.txt", quote = FALSE, sep="\t" ,row.names=F)
 