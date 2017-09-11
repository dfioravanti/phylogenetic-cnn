# Perform a sMDS analysis of bacteria  dataset
 
suppressPackageStartupMessages(library(argparse))

# load functions re-implementing multiple cointertia

parser <- ArgumentParser(description="Perform sparse MDS")

parser$add_argument("--X", type="character", help = "First dataset [samples X features]")
parser$add_argument("--num", type="integer",  help = "number of components")
parser$add_argument("--opt_p", type="character", choices = c("soft_thr", "hard_thr", "SCAD"), default = "SCAD", help = "type of penalty for regularized SVD")
#parser$add_argument("--opt_loo", type="logical", default = FALSE, help = "if loo has to be used in tun_j")
parser$add_argument("--K_cv", type="integer", default = 5, help = "K-CV")
parser$add_argument("--Nfolds", type="integer", default = 10, help = "Number of repetion for 5-CV")
#parser$add_argument("--sDir", type="character", help = "Directory with sMDS.R script. ")
parser$add_argument("--outf", type="character", help = "Output file [basename only]")

args <- parser$parse_args()

# Read input parameters
dataFile <- args$X
num <- args$num
outFile <- args$outf
type <- args$opt_p
ntimes <- args$Nfolds
nfold <- args$K_cv


# Use of rSVD to solve the main eigen-system
source("smds_last.R")
#sink('output_realdata.txt')
# Read input data
dataTable <- read.table(dataFile, sep='\t', header=TRUE, check.names=FALSE)

nsamples <- dim(dataTable)[1]
# MDS
mds_result<- sMDS(X=as.matrix(dataTable), num =num, type= type, ntimes=ntimes, nfold=nfold, verbose = T)
#sink()
#sink('output_realdata.txt', append=TRUE)
save.image(file='workspace_smds_true_data.RData')
# 
s_XFile<- paste(outFile, "_sparse_coordinates.txt", sep="")
write.table(mds_result$X_sp, file=s_XFile, sep='\t', row.names = FALSE, col.names = TRUE, quote=FALSE)
# 
s_UFile<- paste(outFile, "_sparse_u.txt", sep="")
write.table(mds_result$U, file=s_UFile, sep='\t', row.names = FALSE, col.names = TRUE, quote=FALSE)
# 
s_VFile<- paste(outFile, "_sparse_v.txt", sep="")
write.table(mds_result$V, file=s_VFile, sep='\t', row.names = FALSE, col.names = TRUE, quote=FALSE)
# 
# 
s_jFile<- paste(outFile, "_sparse_j.txt", sep="")
write.table(mds_result$j_opt, file=s_jFile, sep='\t', row.names = FALSE, col.names = TRUE, quote=FALSE)
s_cvFile<- paste(outFile, "_sparse_cv.txt", sep="")
write.table(mds_result$cv_score, file=s_cvFile, sep='\t', row.names = FALSE, col.names = TRUE, quote=FALSE)

#Check and order
X_sp <-mds_result$X_sp
index_zero <- rep(0, dim(X_sp)[1])
for(i in 1:dim(X_sp)[1]){
  if (all(X_sp[i,]==0))
    index_zero[i] <- i
}   
variables <- which(index_zero==0)
var_exp <- order(mds_result$lambda)#order the components based on lambda
X_red <- X_sp[variables,var_exp]
dimnames(X_red) <- list(colnames(X_red), paste0("PC", 1:length(var_exp)))
#rownames(X_red) <- rownames(dataTable1)[variables]
names <-  rownames(dataTable1)[variables]
X_red <- cbind(names,X_red)

write.table(t(X_red), file="sparse_X_11.txt", sep='\t', row.names = T, col.names = F, quote=FALSE)
  