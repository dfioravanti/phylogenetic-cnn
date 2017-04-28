################################################################################
# Aims : Given a distance/dissimilarities matrix, the aim is the creation of a matrix of sparse coordinates by
# using MDS system (via low rank regularized SVD), 
#-------------------------------------------------------------------------------


library("cvTools")
library("ggplot2")
library("gridExtra")
library("TunePareto")
library("energy")
library("smacof")
require("caret")
library("matrixcalc")
library("corpcor")


#-------------------------------------------------------------------------------
#sPCA         paper Shen and Huang, algorithm 1
#   input:
#             X ------ n x p matrix to be decomposed using SVD, 
#           
#
#   output: type numeric
#            u ------- n   vector
#            v ------- p vector
# 
#-------------------------------------------------------------------------------


sPCA_rSVD <- function(decomp, X, j_par, type=c("SCAD", "soft_thr", "hard_thr"), tol=1e-13, max.iter=100) {
  type <- match.arg(type)  
  #decomp <- svd(X)
  
  v_old <- decomp$v[,1] 
  u_old <-decomp$d[1]*decomp$u[,1]
  
  
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
    
    
    v_new <- t(X)%*%u_new
    
    v_new <- v_new/sqrt(sum(v_new*v_new))  
    norm_diffv <- sqrt(sum((v_new-v_old)*(v_new-v_old)))
    
    norm_diffu <- sqrt(sum((u_new-u_old)*(u_new-u_old)))
   
    v_old <- v_new
    u_old <- u_new
    
    
  }
  u <- u_new/sqrt(sum(u_new*u_new))
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
  if(j_opt_cv==dim(X)[1])
    cat("the optimal degree is exactly the length of the vector")
  res <- list(j=j_opt_cv, scores=CV_score)
  return(res)
}


sMDS <- function(X, num=2, type, ntimes, nfold, tol=1e-13, verbose=FALSE, ...){
  
  X <- sweep(X, 2, colMeans(X)) # centered the matrix
  
  #initialize
  U <- c()
  V <- c()
  j_opt <- c()
  CV <- c()
  X_r <- X
  cum_perc <- vector(length=num)
  lambda <- vector(length=num)
  
  #Component 1
  print(paste("##### component: 1  #####", sep = ""))
  tun_j_res <-tuning_j_par(X_r, ntimes, nfold, type, fun=median, verbose=FALSE)
  j_opt[1] <- tun_j_res$j
  cat("#### j_opt ", j_opt[1], " ####\n")
  decomp <- svd(X_r)
  s_pca <- sPCA_rSVD(decomp, X=X_r, j_par=j_opt[1], type=type) 
  U <- cbind(U,s_pca$u)
  V <- cbind(V,s_pca$v)
  
  #projection
  X_1 <- X %*% as.matrix(V) %*% solve(t(as.matrix(V)) %*% as.matrix(V), 1e-30) %*% t(as.matrix(V))
  #cumulative percentage of variance (CPEV)
  cum_perc[1] <- sum(diag((X_1)%*%t(X_1)))/sum(diag((X) %*% t(X)))
  lambda[1] <- cum_perc[1] 
  X_r <- X_r - sqrt(lambda[1] * sum(diag((X)%*%t(X)))) * U %*% t(V)
  cat("#### cum ", cum_perc, " ####\n")
  cat("#### lambda ", lambda, " ####\n")
 
  
  for(i in 2:num){
    
    print(paste("##### component: ",  i, " #####", sep = ""))
    #cat("#### X_r ", X_r, " ####\n")
    
    tun_j_res <-tuning_j_par(X_r, ntimes, nfold, type, fun=median, verbose=FALSE) 
    j_opt[i] <- tun_j_res$j
    cat("#### j_opt ", j_opt[i], " ####\n")
    decomp <- svd(X_r)
    s_pca <- sPCA_rSVD(decomp, X=X_r, j_par=j_opt[i], type=type) 
    U <- cbind(U,s_pca$u)
    V <- cbind(V,s_pca$v)
    
    
    #cumulative percentage of variance
    X_i <- X_i <- X %*% V %*% solve(crossprod(V,V)) %*% t(V)
    #X_i <- X %*% as.matrix(V) %*% solve(t(as.matrix(V)) %*% as.matrix(V),1e-30) %*% t(as.matrix(V))
    #X_i <- X %*% t(X) %*%U %*% solve(t(U) %*% X %*% t(X) %*% U,tol=1e-20) %*% t(U)%*%X
    cum_perc[i] <- sum(diag((X_i)%*%t(X_i)))/sum(diag((X) %*% t(X)))
    cat("#### cum ", cum_perc, " ####\n")
    lambda[i] <- cum_perc[i] - sum(lambda)
    cat("#### lambda ", lambda[i], " ####\n")
    
    #Residual matrix
    X_r <- X_r - sqrt(lambda[i] * sum(diag((X)%*%t(X)))) * U %*% t(V)
    cat("#### j_opt ", j_opt, " ####\n")
  }
  
  d <- svd(X)$d[1:num]
  X_sp <- U%*%diag(d)
  return(list(X_sp=X_sp,U=U,V=V,lambda=lambda, cum_perc=cum_perc, j_opt=j_opt))
}


