# This script generates synthetic data  starting from the HS and CDflare datasets  

library(compositions)
library(mvtnorm)
set.seed(1234)

coordinates_cdf <- read.delim("~/Desktop/ThesisData/progettoR/coordinates_cdf.txt")
var_names <- colnames(coordinates_cdf)

table_tr <- read.delim("~/Desktop/ThesisData/progettoR/HS_CDf/Sokol_16S_taxa_HS_CDflare_commsamp_training.txt")
table_ts <- read.delim("~/Desktop/ThesisData/progettoR/HS_CDf/Sokol_16S_taxa_HS_CDflare_commsamp_validation.txt")

#HS
hs_tr <-read.delim("~/Desktop/ThesisData/progettoR/HS_CDf/Sokol_16S_taxa_HS_CDflare_commsamp_training.txt")[1:27,-1]
hs_ts <-read.delim("~/Desktop/ThesisData/progettoR/HS_CDf/Sokol_16S_taxa_HS_CDflare_commsamp_validation.txt")[1:11,-1]
hs <- rbind(hs_tr,hs_ts)

# View data as compositional data
hs <- rcomp(hs)
#Apply log-ratio trasformation 
transf_hs <- ilr(hs) 

#Compute mean and variance
vect_mean_hs <- mean(transf_hs)
cov_matrix_hs <- cov(transf_hs)

#CDf
cdf_tr <-read.delim("~/Desktop/ThesisData/progettoR/HS_CDf/Sokol_16S_taxa_HS_CDflare_commsamp_training.txt")[28:dim(table_tr)[1],-1]
cdf_ts <-read.delim("~/Desktop/ThesisData/progettoR/HS_CDf/Sokol_16S_taxa_HS_CDflare_commsamp_validation.txt")[12:dim(table_ts)[1],-1]
cdf <- rbind(cdf_tr, cdf_ts)

# View data as compositional data
cdf <- rcomp(cdf) 
#Apply log-ratio trasformation 
transf_cdf <- ilr(cdf)

#Compute mean and variance
vect_mean_cdf <- mean(transf_cdf)
cov_matrix_cdf <- cov(transf_cdf)

#compute projections
AC <- transf_hs-vect_mean_hs
AB <- vect_mean_cdf-vect_mean_hs
project_hs <- AC%*%AB
AD <- transf_cdf-vect_mean_hs
project_cdf <- AD%*%AB

#compute sigma
m1 <- 0
m2 <- sum((vect_mean_cdf-vect_mean_hs)^2)
mu <- (vect_mean_hs+vect_mean_cdf)/2
sigma1 <- sqrt((sum(project_hs^2)+sum(project_cdf-m2))/(dim(hs)[1]+dim(cdf)[1]))

# Dataset 0
alpha <- 0
mean1 <- mu + alpha*sigma1*vect_mean_hs/norm(vect_mean_hs,'2')
mean2 <- mu + alpha*sigma1*vect_mean_cdf/norm(vect_mean_cdf,'2')

multivariate_hs <- rmvnorm(dim(hs)[1], mean = mean1 , sigma = cov_matrix_hs, method="svd")
multivariate_cdf <- rmvnorm(dim(cdf)[1],mean = mean2, sigma = cov_matrix_cdf, method="svd")
#multivariate_hs <- rmvnorm(dim(hs)[1], mean = , sigma = cov_matrix_hs, method="svd")

synthetic_data_hs <- ilrInv(multivariate_hs)
synthetic_data_cdf <- ilrInv(multivariate_cdf)

#training and validation
id_hs_ts <- sample(1:38,11)
id_hs_tr <- setdiff(1:38, id_hs_ts)
id_cdf_ts <- sample(1:60,18)
id_cdf_tr <- setdiff(1:60, id_cdf_ts)

synthetic_training <- rbind(synthetic_data_hs[id_hs_tr,],synthetic_data_cdf[id_cdf_tr,])
synthetic_validation <- rbind(synthetic_data_hs[id_hs_ts,],synthetic_data_cdf[id_cdf_ts,])
names_tr <- 1:dim(synthetic_training)[1]
names_ts <- 1:dim(synthetic_validation)[1]
synthetic_training <- cbind(names_tr,synthetic_training)
synthetic_validation <-cbind(names_ts,synthetic_validation)
colnames(synthetic_training) <- var_names
colnames(synthetic_validation) <- var_names
write.table(synthetic_training,file="~/Desktop/ThesisData/progettoR/synthetic_data/Sokol_16S_taxa_HS_CDflare_commsamp_training_0.txt", quote = FALSE, sep="\t" ,row.names=F)
write.table(synthetic_validation,file="~/Desktop/ThesisData/progettoR/synthetic_data/Sokol_16S_taxa_HS_CDflare_commsamp_validation_0.txt", quote = FALSE, sep="\t" ,row.names=F)


#Dataset 1
alpha <- 1
mean1 <- mu + alpha*sigma1*vect_mean_hs/norm(vect_mean_hs,'2')
mean2 <- mu + alpha*sigma1*vect_mean_cdf/norm(vect_mean_cdf,'2')

multivariate_hs <- rmvnorm(dim(hs)[1], mean = mean1 , sigma = cov_matrix_hs, method="svd")
multivariate_cdf <- rmvnorm(dim(cdf)[1],mean = mean2, sigma = cov_matrix_cdf, method="svd")

synthetic_data_hs <- ilrInv(multivariate_hs)
synthetic_data_cdf <- ilrInv(multivariate_cdf)


id_hs_ts <- sample(1:38,11)
id_hs_tr <- setdiff(1:38, id_hs_ts)
id_cdf_ts <- sample(1:60,18)
id_cdf_tr <- setdiff(1:60, id_cdf_ts)
synthetic_training <- rbind(synthetic_data_hs[id_hs_tr,],synthetic_data_cdf[id_cdf_tr,])
synthetic_validation <- rbind(synthetic_data_hs[id_hs_ts,],synthetic_data_cdf[id_cdf_ts,])
names_tr <- 1:dim(synthetic_training)[1]
names_ts <- 1:dim(synthetic_validation)[1]
synthetic_training <- cbind(names_tr,synthetic_training)
synthetic_validation <-cbind(names_ts,synthetic_validation)
colnames(synthetic_training) <- var_names
colnames(synthetic_validation) <- var_names
write.table(synthetic_training,file="~/Desktop/ThesisData/progettoR/synthetic_data/Sokol_16S_taxa_HS_CDflare_commsamp_training_1.txt", quote = FALSE, sep="\t" ,row.names=F)
write.table(synthetic_validation,file="~/Desktop/ThesisData/progettoR/synthetic_data/Sokol_16S_taxa_HS_CDflare_commsamp_validation_1.txt", quote = FALSE, sep="\t" ,row.names=F)

#Dataset 2
alpha <- 2
mean1 <- mu + alpha*sigma1*vect_mean_hs/norm(vect_mean_hs,'2')
mean2 <- mu + alpha*sigma1*vect_mean_cdf/norm(vect_mean_cdf,'2')


multivariate_hs <- rmvnorm(dim(hs)[1], mean = mean1 , sigma = cov_matrix_hs, method="svd")
multivariate_cdf <- rmvnorm(dim(cdf)[1],mean = mean2, sigma = cov_matrix_cdf, method="svd")

synthetic_data_hs <- ilrInv(multivariate_hs)
synthetic_data_cdf <- ilrInv(multivariate_cdf)

sum(synthetic_data_cdf[1,])
mean(synthetic_data_cdf)
plot(density(synthetic_data_cdf[1,]))
norm(mean2,mean1,'2')
id_hs_ts <- sample(1:38,11)
id_hs_tr <- setdiff(1:38, id_hs_ts)
id_cdf_ts <- sample(1:60,18)
id_cdf_tr <- setdiff(1:60, id_cdf_ts)
synthetic_training <- rbind(synthetic_data_hs[id_hs_tr,],synthetic_data_cdf[id_cdf_tr,])
synthetic_validation <- rbind(synthetic_data_hs[id_hs_ts,],synthetic_data_cdf[id_cdf_ts,])
names_tr <- 1:dim(synthetic_training)[1]
names_ts <- 1:dim(synthetic_validation)[1]
synthetic_training <- cbind(names_tr,synthetic_training)
synthetic_validation <-cbind(names_ts,synthetic_validation)
colnames(synthetic_training) <- var_names
colnames(synthetic_validation) <- var_names
write.table(synthetic_training,file="~/Desktop/ThesisData/progettoR/synthetic_data/Sokol_16S_taxa_HS_CDflare_commsamp_training_2.txt", quote = FALSE, sep="\t" ,row.names=F)
write.table(synthetic_validation,file="~/Desktop/ThesisData/progettoR/synthetic_data/Sokol_16S_taxa_HS_CDflare_commsamp_validation_2.txt", quote = FALSE, sep="\t" ,row.names=F)

#Dataset 3
alpha <- 3
mean1 <- mu + alpha*sigma1*vect_mean_hs/norm(vect_mean_hs,'2')
mean2 <- mu + alpha*sigma1*vect_mean_cdf/norm(vect_mean_cdf,'2')

multivariate_hs <- rmvnorm(dim(hs)[1], mean = mean1 , sigma = cov_matrix_hs, method="svd")
multivariate_cdf <- rmvnorm(dim(cdf)[1],mean = mean2, sigma = cov_matrix_cdf, method="svd")

synthetic_data_hs <- ilrInv(multivariate_hs)
synthetic_data_cdf <- ilrInv(multivariate_cdf)

id_hs_ts <- sample(1:38,11)
id_hs_tr <- setdiff(1:38, id_hs_ts)
id_cdf_ts <- sample(1:60,18)
id_cdf_tr <- setdiff(1:60, id_cdf_ts)
synthetic_training <- rbind(synthetic_data_hs[id_hs_tr,],synthetic_data_cdf[id_cdf_tr,])
synthetic_validation <- rbind(synthetic_data_hs[id_hs_ts,],synthetic_data_cdf[id_cdf_ts,])
names_tr <- 1:dim(synthetic_training)[1]
names_ts <- 1:dim(synthetic_validation)[1]
synthetic_training <- cbind(names_tr,synthetic_training)
synthetic_validation <-cbind(names_ts,synthetic_validation)
colnames(synthetic_training) <- var_names
colnames(synthetic_validation) <- var_names
write.table(synthetic_training,file="~/Desktop/ThesisData/progettoR/synthetic_data/Sokol_16S_taxa_HS_CDflare_commsamp_training_3.txt", quote = FALSE, sep="\t" ,row.names=F)
write.table(synthetic_validation,file="~/Desktop/ThesisData/progettoR/synthetic_data/Sokol_16S_taxa_HS_CDflare_commsamp_validation_3.txt", quote = FALSE, sep="\t" ,row.names=F)



