# This script generates synthetic data  starting from the real datasets

library(compositions)
library(mvtnorm)
set.seed(1234)

# Total number of sample to be generated
nb_total_output_samples <- 200

# Proportion of test data to be generated
proportion_test = 0.20

# Path configuration

desease <- "CDf"

datasets_path <- "../datasets/"

coordinates_path <- paste(datasets_path, "coordinates/coordinates_", tolower(desease), ".txt", sep="")
training_samples_path <- paste(datasets_path,"true_data/HS_", desease, "/Sokol_16S_taxa_HS_",
                                desease, "_commsamp_training.txt", sep="")
test_samples_path <- paste(datasets_path, "true_data/HS_", desease, "/Sokol_16S_taxa_HS_",
                           desease, "_commsamp_validation.txt", sep="")
training_target_path <- paste(datasets_path, "true_data/HS_", desease, "/Sokol_16S_taxa_HS_",
                               desease, "_commsamp_training_lab.txt", sep="")
test_target_path <- paste(datasets_path, "true_data/HS_", desease, "/Sokol_16S_taxa_HS_",
                           desease, "_commsamp_validation_lab.txt", sep="")

output_dir_path <- paste(datasets_path, "synthetic_data/HS_", desease,
                         "/", nb_total_output_samples, sep="")
dir.create(output_dir_path, recursive = TRUE)

training_samples_output_path <- paste(output_dir_path, "/Sokol_16S_taxa_HS_",
                                      desease, "_commsamp_training.txt", sep="")
training_targets_output_path <- paste(output_dir_path, "/Sokol_16S_taxa_HS_",
                                     desease, "_commsamp_training_lab.txt", sep="")
test_samples_output_path <- paste(output_dir_path, "/Sokol_16S_taxa_HS_",
                                  desease, "_commsamp_test.txt", sep="")
test_targets_output_path <- paste(output_dir_path, "/Sokol_16S_taxa_HS_",
                                  desease, "_commsamp_test_lab.txt", sep="")


# Compute the number of healty and sick sample for both training and test
training_target <- read.delim(training_target_path, header = FALSE)
nb_helthy_training = sum(training_target == 0)
nb_sample_training = dim(training_target)[1]

test_target <- read.delim(test_target_path, header = FALSE)
nb_helthy_test = sum(test_target == 0)
nb_sample_test = dim(test_target)[1]

# Load samples
features_names <- colnames(read.delim(coordinates_path))
training_samples <- read.delim(training_samples_path)
test_samples <- read.delim(test_samples_path)

#HS
healty_training <- training_samples[1:nb_helthy_training, -1]
healty_test <- test_samples[1:nb_helthy_test, -1]
original_healty <- rbind(healty_training, healty_test)

# Apply log-ratio trasformation, since ilr need the data as compositional data
# we apply the rcomp before we call ilr
original_healty_ilr <- ilr(rcomp(original_healty))

#Compute mean and variance
means_healty <- mean(original_healty_ilr)
covariance_healty <- cov(original_healty_ilr)

#CDf
sick_training <- training_samples[(nb_helthy_training + 1):nb_sample_training, -1]
sick_test <- test_samples[(nb_helthy_test + 1):nb_sample_test, -1]
original_sick <- rbind(sick_training, sick_test)

# Apply log-ratio trasformation, since ilr need the data as compositional data
# we apply the rcomp before we call ilr
original_sick_ilr <- ilr(rcomp(original_sick))

#Compute mean and variance
means_sick <- mean(original_sick_ilr)
covariance_sick <- cov(original_sick_ilr)

# compute means difference
AB <- means_sick-means_healty

#compute projections
AC <- original_healty_ilr - means_healty
project_hs <- AC%*%AB
AD <- original_sick_ilr - means_healty
project_cdf <- AD%*%AB

#compute sigma
m1 <- 0
m2 <- sum((means_sick-means_healty)^2)
mu <- (means_healty+means_sick)/2
sigma1 <- sqrt((sum(project_hs^2)+sum(project_cdf-m2))/(dim(original_healty)[1]+dim(original_sick)[1]))

alpha <- 1
mean1 <- mu + alpha*sigma1*means_healty/norm(means_healty,'2')
mean2 <- mu + alpha*sigma1*means_sick/norm(means_sick,'2')

# Mantaining the original proportion between sick and healty
prp_sick <- (dim(original_sick)/(dim(original_sick) + dim(original_healty)))[1]
nb_sick <- trunc(nb_total_output_samples * prp_sick)
nb_healty <- nb_total_output_samples - nb_sick

multivariate_healty <- rmvnorm(nb_healty, mean=mean1 , sigma=covariance_healty, method="svd")
multivariate_sick <- rmvnorm(nb_sick, mean=mean2, sigma=covariance_sick, method="svd")

synthetic_data_healty <- ilrInv(multivariate_healty)
synthetic_data_sick <- ilrInv(multivariate_sick)

#training and validation
nb_healty_ts = floor(proportion_test * nb_healty)
nb_sick_ts = floor(proportion_test * nb_sick)
index_healty_ts <- sample(1:nb_healty, nb_healty_ts)
index_healty_tr <- setdiff(1:nb_healty, index_healty_ts)
index_sick_ts <- sample(1:nb_sick, nb_sick_ts)
index_sick_tr <- setdiff(1:nb_sick, index_sick_ts)

synthetic_training <- rbind(synthetic_data_healty[index_healty_tr,],
                            synthetic_data_sick[index_sick_tr,])
names_tr <- 1:dim(synthetic_training)[1]
synthetic_training_samples <- cbind(names_tr, synthetic_training)
colnames(synthetic_training_samples) <- features_names

# Save synthetic training data
write.table(synthetic_training_samples,
            file=training_samples_output_path,
            quote = FALSE, sep="\t" ,row.names=F)

training_labels <- matrix(c(rep(0, length(index_healty_tr)),
                            rep(1, length(index_sick_tr))), 
                          ncol = 1)
write.table(training_labels,
            file=training_targets_output_path,
            quote = FALSE, sep="\t" ,
            row.names = FALSE, col.names = FALSE)

# Save synthetic test data
synthetic_test <- rbind(synthetic_data_healty[index_healty_ts,],
                        synthetic_data_sick[index_sick_ts,])
names_ts <- 1:dim(synthetic_test)[1]
synthetic_test_samples <- cbind(names_ts, synthetic_test)
colnames(synthetic_test_samples) <- features_names
write.table(synthetic_test_samples,
            file=test_samples_output_path,
            quote = FALSE, sep="\t" ,row.names=F)

test_labels <- matrix(c(rep(0, length(index_healty_ts)),
                        rep(1, length(index_sick_ts))), 
                      ncol = 1)
write.table(test_labels,
            file=test_targets_output_path,
            quote = FALSE, sep="\t" ,
            row.names = FALSE, col.names = FALSE)
