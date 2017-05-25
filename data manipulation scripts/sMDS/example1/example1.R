################################################################################
# Aims : creation of synthetic data set to validate the sPCA-rSVD method 
# fix the first 2 eigenvector, then construction of the covariance matrix. Compare the eigenvector computed 
# using sPCA-rSVD with the true one
# selection of the degree of sparsity using the paramenter tuning
#-------------------------------------------------------------------------------


library("pracma")
library("cvTools")
library("circular")
library("TunePareto")
library("ggplot2")


sPCA_rSVD <- function(decomp, X, j_par, type=c("SCAD", "soft_thr", "hard_thr"), tol=1e-13, max.iter=100) {
  type <- match.arg(type)  
  #decomp <- svd(X)
  v_old <- decomp$v[,1] 
  u_old <-decomp$d[1]*decomp$u[,1]
  #cat("#### v_old ", v_old, " ####\n")
  if (j_par==0) {
    lambda_par <- 0
  } else {
    
    Xvold <- sort(abs(X%*%v_old))
    lambda_par <- (Xvold[j_par]+Xvold[(j_par+1)])/2
  }
  
  h_lambda <- switch(type,
                     soft_thr = mcia:::h_lambda_soft,
                     hard_thr = mcia:::h_lambda_hard,
                     SCAD = mcia:::h_lambda_SCAD)
  
  #convergence in norm
  norm_diffv <- norm_diffu <- tol+1
  iter <- 0
  while ((norm_diffv >= tol || norm_diffu >=tol) && iter < max.iter) {
    iter <- iter + 1
    
    u_new <- h_lambda(X%*%v_old, lambda=lambda_par)
    v_new <- drop(t(X)%*%u_new)
    v_new <- v_new/sqrt(crossprod(v_new,v_new))  
    norm_diffv <- sqrt(crossprod((v_new-v_old),(v_new-v_old)))
    norm_diffu <- sqrt(crossprod((u_new-u_old),(u_new-u_old)))
    v_old <- v_new
    u_old <- u_new
  }
  
  u <- u_new/sqrt(crossprod(u_new,u_new))
  res <- list(v=drop(v_new), u=drop(u), convergence=iter <= max.iter)
  return(res)
}

#-------------------------------------------------------------------------------
#tuning_j_par  K-fold CV Tuning j_par parameter selection 
#   input:
#             X ------ matrix to be decomposed using SVD
#             ntimes ------ ntimes number of repetition on CV
#             nfold ------ nfold cross validation 
#   opt_penalty ------ type of penalty to select sparsity 
#
#   output: type numeric
#         j_opt ------ optimal degree of sparsity      
#        scores ------ matrix of CV scores (nrow=p,ncol=N)
#-------------------------------------------------------------------------------

tuning_j_par <- function(X, ntimes, nfold, type, fun=median, verbose=FALSE, ...) {
  nr <- nrow(X) #n
  nc <- ncol(X) #p 
  #CV_score_fold <- rep(0, ntimes)
  j_opt_cv <- rep(0, ntimes)
  
  #matrix saving CV_score for each j in each fold
  CV_score <- c()#matrix(0, nrow=nr, ncol=ntimes) 
  set.seed(1234)
  cv_folds <- generateCVRuns(1:nc, ntimes=ntimes, nfold=nfold, stratified=TRUE)
  CV_temp <- rep(0, ntimes)
  for (i in 1:ntimes) {
    if (verbose)
      cat("Repetion: ",  i, "\n", sep = "")
    cv_folds_current <- cv_folds[[i]]
    
    #array with CV score_j for each l-fold
    CV_score_j <- matrix(0, nrow=nr, ncol=nfold) 
    
    for (l in 1:nfold) {
      X_whitout_l <- X[,-cv_folds_current[[l]],drop=FALSE]
      X_l <- X[,cv_folds_current[[l]],drop=FALSE]
      ncj <- NCOL(X_l)
      decomp <- svd(X_whitout_l)
      for (j in 0:(nr-1)) {
        if (verbose)
          cat("Number of variables", j, "\n")
        sPCA <- sPCA_rSVD(decomp, X=X_whitout_l, type=type, j_par=j, ...)
        u_j <- sPCA$u
        v_j <- drop(t(X_l)%*%u_j)
        #compute single CV score,depend on j and the l-fold
        CV_score_j[j+1, l] <- (1/(ncj*nr))*sum((X_l- outer(u_j,v_j))^2)
        #cat("#### CV_score_j ",CV_score_j , " ####\n")
      }
    }
    CV_score<-cbind(CV_score, apply(CV_score_j[,1:nfold], 1, sum))
    #cat("#### CV_score ",CV_score , " ####\n")
  }
  # save j responsible for min of CV score
  goals <- apply(CV_score, 1, fun)
  j_opt_cv <- which.min(goals) - 1
  if(j_opt_cv==nr)
    cat("the optimal degree is exactly the length of the vector")
  res <- list(j=j_opt_cv, scores=CV_score)
  return(res)
}



# function to create data X from eigenvector and eigenvalues
create_X <- function(v1, v2, C, n=6, tol=1e-13, m = 10){
  p <- length(v1)
  # normalize eigenvector
  v1 <- v1/ sqrt(sum(v1 * v1))
  v2 <- v2/ sqrt(sum(v2 * v2))
  
  V <- matrix(0,ncol=p,nrow=p)
  V[,1:2]<-c(v1,v2)
  
  # creation of the other 8 eigenvector
  for (j in 3:ncol(V)){
    set.seed(j+m)  #fissare la V?
    V[,j] <- runif(p)
  }
  # if V is not full rank, recompute it
  if (det(V) < tol){
    for (j in 3:ncol(V)){
      set.seed(j+1234+m)
      V[,j] <- runif(p)
    }
  }
  GS_orth <- gramSchmidt(V, tol = tol)
  # V = Q %*% R
  
  V <- GS_orth$Q #orthogonal matrix
  
  # Z ~ N(0,I_P)
  # set.seed(m)
  # Z <- matrix(rnorm(n*p), nrow = p, ncol = n)
  
  Z <- matrix(0, nrow = p, ncol = n)
  for (j in 1:n){
    set.seed(j+m)
    Z[,j] <- rnorm(p)
  }
  
  # generation of the data
  X <- V %*% diag(sqrt(C)) %*% Z #p x n
  return(X)
}


# test sPCA with known degree of sparsity
# generation of l = 100 datasets
p <- 300
v1 <- c(rep(1,10),rep(0,290))
v2 <- c(rep(0,10),rep(1,10),rep(0,280))
C <- c(400,300,rep(1,298)) #eigeinvalues
opt_penalty <- "SCAD"
l <- 100



sPCA_oracle_method <- function(v1, v2, C, l=100, opt_penalty){
  p <- length(v1)
  j1_known <- length(which(v1==0))
  j2_known <- length(which(v2==0))
  j3_known <- 0
  j4_known <- 0
  j5_known <- 0
  j6_known <- 0
  
  v1_spca <- list()
  v2_spca <- list()
  v3_spca <- list()
  v4_spca <- list()
  v5_spca <- list()
  v6_spca <- list()
  
  for (m in 1:l){
    X_temp <- create_X(v1, v2, C, m = m) #each matrix is p x n
    X <-X_temp
    V <- c()
    #component 1
    cat("##### component: ",  1, " #####\n")
    decomp <- svd(X_temp)
    sPCA_1 <- sPCA_rSVD(decomp,X = X_temp, j_par = j1_known, type = opt_penalty)
    v1_spca[[m]] <- sPCA_1$u 
    V <- cbind(V,sPCA_1$v)
    X_1 <- X%*%tcrossprod(sPCA_1$v,sPCA_1$v)
    #cumulative percentage of variance (CPEV)
    cum_perc_1 <- sum(diag(tcrossprod(X_1,X_1)))/sum(diag(tcrossprod(X,X)))
    if(max(cum_perc_1) > 1){
      cat("one component is sufficient")
      next
    }
    lambda_1 <- cum_perc_1
    
    cat("##### : cum_perc_1 ",  cum_perc_1, " #####\n")
    cat("##### : lambda_1 ",  lambda_1, " #####\n")
    
    #component 2
    cat("##### component: ",  2, " #####\n")
    X_temp <- X_temp - sqrt(lambda_1 * sum(diag(t(X)%*%X))) * sPCA_1$u %*% t(sPCA_1$v)  
    decomp <- svd(X_temp)
    sPCA_2 <-sPCA_rSVD(decomp, X = X_temp, j_par = j2_known, type = opt_penalty)
    v2_spca[[m]] <- sPCA_2$u
    if (qr(cbind(V,sPCA_2$v))$rank==min(nrow(V),ncol(V)+1)) {
      
      V <- cbind(V,sPCA_2$v)
    } else {
      
      cat("#solve(crossprod(V,V))=inf#\n")
      next
    }
    X_2 <- X %*% V %*% solve(crossprod(V,V), tol=1e-30) %*% t(V) # proiezione
    cum_perc_2 <- sum(diag(tcrossprod(X_2,X_2)))/sum(diag(tcrossprod(X,X)))
    lambda_2 <- cum_perc_2- cum_perc_1
    if(max(cum_perc_2) > 1){
      cat("2 components are sufficient")
      next
    }
    
    cat("##### : cum_perc_2 ",  cum_perc_2, " #####\n")
    cat("##### : lambda_2 ",  lambda_2, " #####\n")
    
    
    #component 3
    cat("##### component: ",  3, " #####\n")
    X_temp <- X_temp - sqrt(lambda_2 * sum(diag(t(X)%*%X))) * sPCA_2$u %*% t(sPCA_2$v)
    #j3_opt<-tuning_j_par(X_new_temp, ntimes=10, nfold=5, type=opt_penalty, fun=median, verbose=FALSE)$j
    decomp <- svd(X_temp)
    sPCA_3 <- sPCA_rSVD(decomp, X = X_temp, j_par = j3_known, type = opt_penalty)
    v3_spca[[m]] <- sPCA_3$u
    if (qr(cbind(V,sPCA_3$v))$rank==min(nrow(V),ncol(V)+1)) {
      
      V <- cbind(V,sPCA_3$v)
    } else {
      
      cat("#solve(crossprod(V,V))=inf#\n")
      next
    }
    X_3 <- X %*% V %*% solve(crossprod(V,V),tol=1e-30) %*% t(V)
    cum_perc_3 <- sum(diag(tcrossprod(X_3,X_3)))/sum(diag(tcrossprod(X,X)))
    if(max(cum_perc_3) > 1){
      cat("3 components are sufficient\n")
      next
    }
    lambda_3 <- cum_perc_3- cum_perc_2
    cat("##### : cum_perc_3 ",  cum_perc_3, " #####\n")
    cat("##### : lambda_3 ",  lambda_3, " #####\n")
    
    
    
    #component 4
    cat("##### component: ",  4, " #####\n")
    X_temp <- X_temp - sqrt(lambda_3 * sum(diag(t(X)%*%X))) * sPCA_3$u %*% t(sPCA_3$v)
    #j4_opt<-tuning_j_par(X_new_temp, ntimes=10, nfold=5, type=opt_penalty, fun=median, verbose=FALSE)$j
    decomp <- svd(X_temp)
    sPCA_4 <- sPCA_rSVD(decomp, X = X_temp, j_par = j4_known, type = opt_penalty)
    v4_spca[[m]] <- sPCA_4$u
    if (qr(cbind(V,sPCA_4$v))$rank==min(nrow(V),ncol(V)+1)) {
      
      V <- cbind(V,sPCA_4$v)
    } else {
      
      cat("#solve(crossprod(V,V))=inf#\n")
      next
    }
    X_4 <- X %*% V %*% solve(crossprod(V,V),tol= 1e-30) %*% t(V)
    cum_perc_4 <- sum(diag(tcrossprod(X_4,X_4)))/sum(diag(tcrossprod(X,X)))
    if(max(cum_perc_4) > 1){
      cat("4 components are sufficient")
      next
    }
    lambda_4 <- cum_perc_4 - cum_perc_3
    cat("##### : cum_perc_4 ",  cum_perc_4, " #####\n")
    cat("##### : lambda_4 ",  lambda_4, " #####\n")
    
    
    #component 5
    cat("##### component: ",  5, " #####\n")
    X_temp <- X_temp - sqrt(lambda_4 * sum(diag(t(X)%*%X))) * sPCA_4$u %*% t(sPCA_4$v)
    #j5_opt<-tuning_j_par(X_new_temp, ntimes=10, nfold=5, type=opt_penalty, fun=median, verbose=FALSE)$j
    decomp <- svd(X_temp)
    sPCA_5 <- sPCA_rSVD(decomp, X = X_temp, j_par = j5_known, type = opt_penalty)
    v5_spca[[m]] <- sPCA_5$u
    if (qr(cbind(V,sPCA_5$v))$rank==min(nrow(V),ncol(V)+1)) {
      
      V <- cbind(V,sPCA_5$v)
    } else {
      
      cat("#solve(crossprod(V,V))=inf#\n")
      next
    }
   
    X_5 <- X %*% V %*% solve(crossprod(V,V), tol=1e-30) %*% t(V)
    cum_perc_5 <- sum(diag(tcrossprod(X_5,X_5)))/sum(diag(tcrossprod(X,X)))
    if(max(cum_perc_5) > 1){
      cat("2 components are sufficient")
      next
    }
    lambda_5 <- cum_perc_5 - cum_perc_4
    cat("##### : cum_perc_5 ",  cum_perc_5, " #####\n")
    cat("##### : lambda_5 ",  lambda_5, " #####\n")
    
    
    #component 6
    cat("##### component: ",  6, " #####\n")
    X_temp <- X_temp - sqrt(lambda_5 * sum(diag(t(X)%*%X))) * sPCA_5$u %*% t(sPCA_5$v)
    #j6_opt<-tuning_j_par(X_new_temp, ntimes=10, nfold=5, type=opt_penalty, fun=median, verbose=FALSE)$j
    decomp <- svd(X_temp)
    sPCA_6 <- sPCA_rSVD(decomp, X = X_temp, j_par = j6_known, type = opt_penalty)
    v6_spca[[m]] <- sPCA_6$u
    cat(m,"\n")
  }
  return(list(v1=v1_spca,v2=v2_spca,v3=v3_spca,v4=v4_spca,v5=v5_spca,v6=v6_spca))
}

sPCA_CV_method <- function(v1, v2, C, l=100, opt_penalty,N){
  p <- length(v1)

  
  v1_spca <- list()
  v2_spca <- list()
  v3_spca <- list()
  v4_spca <- list()
  v5_spca <- list()
  v6_spca <- list()
  
  for (m in 1:l){
    X_temp <- create_X(v1, v2, C, m = m+2) #each matrix is p x n
    X <-X_temp
    V <- c()
    #component 1
    cat("##### component: ",  1, " #####\n")
    j1_opt<-tuning_j_par(X_temp, ntimes=10, nfold=5, type=opt_penalty, fun=median, verbose=FALSE)$j
    decomp <- svd(X_temp)
    sPCA_1 <- sPCA_rSVD(decomp,X = X_temp, j_par = j1_opt, type = opt_penalty)
    v1_spca[[m]] <- sPCA_1$u 
    V <- cbind(V,sPCA_1$v)
    X_1 <- X%*%tcrossprod(sPCA_1$v,sPCA_1$v)
    #cumulative percentage of variance (CPEV)
    cum_perc_1 <- sum(diag(tcrossprod(X_1,X_1)))/sum(diag(tcrossprod(X,X)))
    if(max(cum_perc_1) > 1){
      cat("one component is sufficient")
      next
    }
    lambda_1 <- cum_perc_1
    
    cat("##### : cum_perc_1 ",  cum_perc_1, " #####\n")
    cat("##### : lambda_1 ",  lambda_1, " #####\n")
    
    #component 2
    cat("##### component: ",  2, " #####\n")
    X_temp <- X_temp - sqrt(lambda_1 * sum(diag(t(X)%*%X))) * sPCA_1$u %*% t(sPCA_1$v)  
    j2_opt<-tuning_j_par(X_temp, ntimes=10, nfold=5, type=opt_penalty, fun=median, verbose=FALSE)$j
    decomp <- svd(X_temp)
    sPCA_2 <-sPCA_rSVD(decomp, X = X_temp, j_par = j2_opt, type = opt_penalty)
    v2_spca[[m]] <- sPCA_2$u
    if (qr(cbind(V,sPCA_2$v))$rank==min(nrow(V),ncol(V)+1)) {
      
      V <- cbind(V,sPCA_2$v)
    } else {
      
      cat("#solve(crossprod(V,V))=inf#\n")
      next
    }
    X_2 <- X %*% V %*% solve(crossprod(V,V), tol=1e-30) %*% t(V) # proiezione
    cum_perc_2 <- sum(diag(tcrossprod(X_2,X_2)))/sum(diag(tcrossprod(X,X)))
    lambda_2 <- cum_perc_2- cum_perc_1
    if(max(cum_perc_2) > 1){
      cat("2 components are sufficient")
      next
    }
    
    cat("##### : cum_perc_2 ",  cum_perc_2, " #####\n")
    cat("##### : lambda_2 ",  lambda_2, " #####\n")
    
    
    #component 3
    cat("##### component: ",  3, " #####\n")
    X_temp <- X_temp - sqrt(lambda_2 * sum(diag(t(X)%*%X))) * sPCA_2$u %*% t(sPCA_2$v)
    j3_opt<-tuning_j_par(X_temp, ntimes=10, nfold=5, type=opt_penalty, fun=median, verbose=FALSE)$j
    decomp <- svd(X_temp)
    sPCA_3 <- sPCA_rSVD(decomp, X = X_temp, j_par = j3_opt, type = opt_penalty)
    v3_spca[[m]] <- sPCA_3$u
    if (qr(cbind(V,sPCA_3$v))$rank==min(nrow(V),ncol(V)+1)) {
      
      V <- cbind(V,sPCA_3$v)
    } else {
      
      cat("#solve(crossprod(V,V))=inf#\n")
      next
    }
    X_3 <- X %*% V %*% solve(crossprod(V,V),tol=1e-30) %*% t(V)
    cum_perc_3 <- sum(diag(tcrossprod(X_3,X_3)))/sum(diag(tcrossprod(X,X)))
    if(max(cum_perc_3) > 1){
      cat("3 components are sufficient\n")
      next
    }
    lambda_3 <- cum_perc_3- cum_perc_2
    cat("##### : cum_perc_3 ",  cum_perc_3, " #####\n")
    cat("##### : lambda_3 ",  lambda_3, " #####\n")
    
    
    
    #component 4
    cat("##### component: ",  4, " #####\n")
    X_temp <- X_temp - sqrt(lambda_3 * sum(diag(t(X)%*%X))) * sPCA_3$u %*% t(sPCA_3$v)
    j4_opt<-tuning_j_par(X_temp, ntimes=10, nfold=5, type=opt_penalty, fun=median, verbose=FALSE)$j
    decomp <- svd(X_temp)
    sPCA_4 <- sPCA_rSVD(decomp, X = X_temp, j_par = j4_opt, type = opt_penalty)
    v4_spca[[m]] <- sPCA_4$u
    if (qr(cbind(V,sPCA_4$v))$rank==min(nrow(V),ncol(V)+1)) {
      
      V <- cbind(V,sPCA_4$v)
    } else {
      
      cat("#solve(crossprod(V,V))=inf#\n")
      next
    }
    X_4 <- X %*% V %*% solve(crossprod(V,V),tol= 1e-30) %*% t(V)
    cum_perc_4 <- sum(diag(tcrossprod(X_4,X_4)))/sum(diag(tcrossprod(X,X)))
    if(max(cum_perc_4) > 1){
      cat("4 components are sufficient")
      next
    }
    lambda_4 <- cum_perc_4 - cum_perc_3
    cat("##### : cum_perc_4 ",  cum_perc_4, " #####\n")
    cat("##### : lambda_4 ",  lambda_4, " #####\n")
    
    
    #component 5
    cat("##### component: ",  5, " #####\n")
    X_temp <- X_temp - sqrt(lambda_4 * sum(diag(t(X)%*%X))) * sPCA_4$u %*% t(sPCA_4$v)
    j5_opt<-tuning_j_par(X_temp, ntimes=10, nfold=5, type=opt_penalty, fun=median, verbose=FALSE)$j
    decomp <- svd(X_temp)
    sPCA_5 <- sPCA_rSVD(decomp, X = X_temp, j_par = j5_opt, type = opt_penalty)
    v5_spca[[m]] <- sPCA_5$u
    if (qr(cbind(V,sPCA_5$v))$rank==min(nrow(V),ncol(V)+1)) {
      
      V <- cbind(V,sPCA_5$v)
    } else {
      
      cat("#solve(crossprod(V,V))=inf#\n")
      next
    }
    
    X_5 <- X %*% V %*% solve(crossprod(V,V), tol=1e-30) %*% t(V)
    cum_perc_5 <- sum(diag(tcrossprod(X_5,X_5)))/sum(diag(tcrossprod(X,X)))
    if(max(cum_perc_5) > 1){
      cat("2 components are sufficient")
      next
    }
    lambda_5 <- cum_perc_5 - cum_perc_4
    cat("##### : cum_perc_5 ",  cum_perc_5, " #####\n")
    cat("##### : lambda_5 ",  lambda_5, " #####\n")
    
    
    #component 6
    cat("##### component: ",  6, " #####\n")
    X_temp <- X_temp - sqrt(lambda_5 * sum(diag(t(X)%*%X))) * sPCA_5$u %*% t(sPCA_5$v)
    j6_opt<-tuning_j_par(X_temp, ntimes=10, nfold=5, type=opt_penalty, fun=median, verbose=FALSE)$j
    decomp <- svd(X_temp)
    sPCA_6 <- sPCA_rSVD(decomp, X = X_temp, j_par = j6_opt, type = opt_penalty)
    v6_spca[[m]] <- sPCA_6$u
    cat(m,"\n")
  }
  return(list(v1=v1_spca,v2=v2_spca,v3=v3_spca,v4=v4_spca,v5=v5_spca,v6=v6_spca))
}


#compute the median angle
# v_true: vector of the true eigenvector
# v_spca: list of vector, one vector for each simulation (100)
compare_eig <- function(v_true,v_spca){
  
  v_true <- v_true/ sqrt(sum(v_true^2))
  M <- length(v_spca)
  
  # sgn_true <- sign(v_true)
  # create list v_spca with signs according to v_true sign
  # v_spca_sgn <- list()
  
  # for (h in 1:M){
  #   sgn_h <- sign(v_spca[[h]])
  #   if (any(abs(sgn_true-sgn_h)==2)){
  #     v_spca_sgn[[h]] <- v_spca[[h]]*(-1)
  #   }
  #   else v_spca_sgn[[h]] <- v_spca[[h]] 
  # }
  # 
  # # angle between the 2 vectors
  # angle <- vector(length = M)
  # for (h in 1:M)
  #   angle[h] <- acos(sum(v_true * t(v_spca_sgn[[h]])) / (sqrt(sum(v_true^2)) * sqrt(sum(v_spca_sgn[[h]]^2))))
  
  # angle between the 2 vectors
  angle <- vector(length = M)
  for (h in 1:M){
    true_angle <- acos(sum(v_true * t(v_spca[[h]])) / (sqrt(sum(v_true^2)) * sqrt(sum(v_spca[[h]]^2))))
    angle[h] <- min(abs(pi-true_angle),abs(true_angle))  
  }
  median_angle <- median(circular(x = angle))
  
  
  #percentage of correctly identified zero loadings
  correct  <- vector(length = M)
  correct_perc <- vector()
  #percentage of incorrectly identified zero loadings
  incorrect  <- vector(length = M)
  incorrect_perc <- vector()
  
  #find zero loadings in v_true
  zero_true <- which(v_true==0)
  n_zero_true <- length(zero_true)
  not_zero_true <- which(v_true!=0)
  n_not_zero_true <- length(not_zero_true)
  
  for (h in 1:M) {
    zero_spca_temp <- which(v_spca[[h]]==0)
    not_zero_spca_temp <- which(v_spca[[h]]!=0)
    
    #number of zero loadings in v_true found also in v_spca
    correct[h] <- length(intersect(zero_true,zero_spca_temp))/n_zero_true 
    correct_perc[h] <- 100 * correct[h]
    
    #number of loadings set to zero in v_spca but not zero in v_true
    #incorrect[h] <- length(intersect(not_zero_true,zero_spca_temp))/n_not_zero_true 
    incorrect[h] <- (length(intersect(not_zero_true,zero_spca_temp))+length(intersect(zero_true,not_zero_spca_temp)))/length(v_true) 
    incorrect_perc[h] <- 100 * incorrect[h]
  }
  return(list(angle=angle,median_angle=median_angle,correct=correct,correct_perc=correct_perc,incorrect=incorrect,incorrect_perc=incorrect_perc))
}
########################RUN###############################################
# generate n=100 data in R^300 (similar to the IBD data)

p <- 300
v1 <- c(rep(1,10),rep(0,290))
v2 <- c(rep(0,10),rep(1,10),rep(0,280))
C <- c(400,300,rep(1,298)) #eigeinvalues
opt_penalty <- "SCAD"
l <- 100

# j known
spca_or <- sPCA_oracle_method(v1,v2,C,opt_penalty = opt_penalty)
v1_spca_or <- spca_or$v1
v2_spca_or <- spca_or$v2
comp_v1_or <- compare_eig(v1,v1_spca_or)
comp_v2_or <- compare_eig(v2,v2_spca_or)
save.image(file='global_en_example1_300x6.RData')

# j by tuning
spca_cv <- sPCA_CV_method(v1=v1,v2=v2,C=C,opt_penalty = opt_penalty,l = 100,N = 10)


v1_spca_cv <- spca_cv$v1
v2_spca_cv <- spca_cv$v2
comp_v1_cv <- compare_eig(v1,v1_spca_cv)
comp_v2_cv <- compare_eig(v2,v2_spca_cv)
save.image(file='global_en_example1_300x6.RData')

# boxplot correct
Method <- c(rep("oracle", length(v1)), rep("CV", length(v1)),rep("oracle", length(v1)),rep("CV", length(v1)))
V <- c(comp_v1_or$correct_perc,comp_v1_cv$correct_perc,comp_v2_or$correct_perc,comp_v2_cv$correct_perc)
eig <- c(rep("u1",2*length(v1)),rep("u2",2*length(v1)))
data <- data.frame(Method, V, eig)
p <- ggplot(data, aes(x=Method, y=V, fill=Method),dotPosition="center") + 
  geom_boxplot(outlier.size = 0.001)+ geom_jitter(width = 0.05)+facet_wrap(~eig)+ labs(y = "percentage")
p
#+scale_fill_manual(values=alpha(c( "blue", "red"), 0.7))
# boxplot incorrect
Method <- c(rep("oracle", length(v1)), rep("CV", length(v1)),rep("oracle", length(v1)),rep("CV", length(v1)))
V <- c(comp_v1_or$incorrect_perc,comp_v1_cv$incorrect_perc,comp_v2_or$incorrect_perc,comp_v2_cv$incorrect_perc)
eig <- c(rep("u1",2*length(v1)),rep("u2",2*length(v1)))
data <- data.frame(Method, V, eig)
p <- ggplot(data, aes(x=Method, y=V, fill=Method), dotPosition="center") +
  geom_boxplot(outlier.size = 0.001)+geom_jitter(width = 0.05)+facet_wrap(~eig) + labs(y = "percentage")
p


# boxplot angle
Method <- c(rep("oracle", length(v1)), rep("CV", length(v1)),rep("oracle", length(v1)),rep("CV", length(v1)))
rad <- c(comp_v1_or$angle,comp_v1_cv$angle,comp_v2_or$angle,comp_v2_cv$angle)
eig <- c(rep("u1",2*length(v1)),rep("u2",2*length(v1)))
data <- data.frame(Method, rad, eig)
p <- ggplot(data, aes(x=Method, y=rad, fill=Method)) + 
  geom_boxplot(outlier.size = 0.95)+scale_fill_manual(values=c( "#E69F00", "#CC0000"))+facet_wrap(~eig)
plot(p)
