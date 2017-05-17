# This script generates synthetic data  starting from the HS and CDflare datasets  

library(compositions)
library(mvtnorm)
set.seed(1234)

coordinates_cdf <- read.delim("datasets/coordinates/coordinates_cdf.txt")
var_names <- colnames(coordinates_cdf)

table_tr <- read.delim("datasets/true_data/HS_CDf/Sokol_16S_taxa_HS_CDflare_commsamp_training.txt")
table_ts <- read.delim("datasets/true_data/HS_CDf/Sokol_16S_taxa_HS_CDflare_commsamp_validation.txt")

#HS
hs_tr <-read.delim("datasets/true_data/HS_CDf/Sokol_16S_taxa_HS_CDflare_commsamp_training.txt")[1:27,-1]
hs_ts <-read.delim("datasets/true_data/HS_CDf/Sokol_16S_taxa_HS_CDflare_commsamp_validation.txt")[1:11,-1]
hs <- rbind(hs_tr,hs_ts)

# View data as compositional data
hs_cmp <- rcomp(hs)
#Apply log-ratio trasformation 
transf_hs <- ilr(hs_cmp) 

#Compute mean and variance
vect_mean_hs <- mean(transf_hs)
cov_matrix_hs <- cov(transf_hs)

#CDf
cdf_tr <-read.delim("datasets/true_data/HS_CDf/Sokol_16S_taxa_HS_CDflare_commsamp_training.txt")[28:dim(table_tr)[1],-1]
cdf_ts <-read.delim("datasets/true_data/HS_CDf/Sokol_16S_taxa_HS_CDflare_commsamp_validation.txt")[12:dim(table_ts)[1],-1]
cdf <- rbind(cdf_tr, cdf_ts)

# View data as compositional data
cdf_cmp <- rcomp(cdf) 
#Apply log-ratio trasformation 
transf_cdf <- ilr(cdf_cmp)

#Compute mean and variance
vect_mean_cdf <- mean(transf_cdf)
cov_matrix_cdf <- cov(transf_cdf)

# compute means difference
AB <- vect_mean_cdf-vect_mean_hs

#compute projections
AC <- transf_hs-vect_mean_hs
project_hs <- AC%*%AB
AD <- transf_cdf-vect_mean_hs
project_cdf <- AD%*%AB

#compute sigma
m1 <- 0
m2 <- sum((vect_mean_cdf-vect_mean_hs)^2)
mu <- (vect_mean_hs+vect_mean_cdf)/2
sigma1 <- sqrt((sum(project_hs^2)+sum(project_cdf-m2))/(dim(hs)[1]+dim(cdf)[1]))

alpha <- 1
mean1 <- mu + alpha*sigma1*vect_mean_hs/norm(vect_mean_hs,'2')
mean2 <- mu + alpha*sigma1*vect_mean_cdf/norm(vect_mean_cdf,'2')

#number of samples
nb_total <- 20000
prp_sick <- (dim(cdf)/(dim(cdf) + dim(hs)))[1] # Mantaining the original proportion between sick and healty
nb_sick <- trunc(nb_total * prp_sick)
nb_healty <- nb_total - nb_sick

multivariate_healty <- rmvnorm(nb_healty, mean=mean1 , sigma=cov_matrix_hs, method="svd")
multivariate_sick <- rmvnorm(nb_sick, mean=mean2, sigma=cov_matrix_cdf, method="svd")

synthetic_data_healty <- ilrInv(multivariate_healty)
synthetic_data_sick <- ilrInv(multivariate_sick)

#training and validation
prp_test = 0.20
nb_healty_ts = floor(prp_test * nb_healty)
nb_sick_ts = floor(prp_test * nb_sick)
index_healty_ts <- sample(1:nb_healty, nb_healty_ts)
index_healty_tr <- setdiff(1:nb_healty, index_healty_ts)
index_sick_ts <- sample(1:nb_sick, nb_sick_ts)
index_sick_tr <- setdiff(1:nb_sick, index_sick_ts)

synthetic_training <- rbind(synthetic_data_healty[index_healty_tr,],
                            synthetic_data_sick[index_sick_tr,])
names_tr <- 1:dim(synthetic_training)[1]
synthetic_training <- cbind(names_tr, synthetic_training)
colnames(synthetic_training) <- var_names
write.table(synthetic_training,
            file="datasets/synthetic_data/Sokol_16S_taxa_HS_CDflare_commsamp_training.txt",
            quote = FALSE, sep="\t" ,row.names=F)

training_labels <- matrix(c(rep(0, length(index_healty_tr)),
                            rep(1, length(index_sick_tr))), 
                          ncol = 1)
write.table(training_labels,
            file="datasets/synthetic_data/Sokol_16S_taxa_HS_CDflare_commsamp_training_lab.txt",
            quote = FALSE, sep="\t" ,
            row.names = FALSE, col.names = FALSE)

synthetic_test <- rbind(synthetic_data_healty[index_healty_ts,],
                        synthetic_data_sick[index_sick_ts,])
names_ts <- 1:dim(synthetic_test)[1]
synthetic_test <- cbind(names_ts, synthetic_test)
colnames(synthetic_test) <- var_names
write.table(synthetic_test,
            file="datasets/synthetic_data/Sokol_16S_taxa_HS_CDflare_commsamp_test.txt",
            quote = FALSE, sep="\t" ,row.names=F)

test_labels <- matrix(c(rep(0, length(index_healty_ts)),
                        rep(1, length(index_sick_ts))), 
                      ncol = 1)
write.table(test_labels,
            file="datasets/synthetic_data/Sokol_16S_taxa_HS_CDflare_commsamp_test_lab.txt",
            quote = FALSE, sep="\t" ,
            row.names = FALSE, col.names = FALSE)

