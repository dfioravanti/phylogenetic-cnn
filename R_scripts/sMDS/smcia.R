h_lambda_soft <- function(y, lambda) {
  sign(y) * pmax((abs(y) - lambda), 0)
}
  
h_lambda_hard <- function(y, lambda) {
  y * ifelse((abs(y) > lambda), 1, 0)
}
  
h_lambda_SCAD <- function(y, lambda, a=3.7) {
  ifelse(abs(y) <= (2*lambda), sign(y)*pmax((abs(y)-lambda),0),
  ifelse(abs(y)<= (a*lambda), ((a-1)*y - sign(y)*a*lambda)/(a-2), y))
}

#-------------------------------------------------------------------------------
#sPCA_rSVD  sparse PCA via regularized low rank matrix approximation
#           penalty imposed on u vector
#   input:
#       X ------ matrix to be decomposed using SVD, will be Q^1/2 X^t D (p x n matrix)
#   j_par ------ degree of sparsity, choosen through tuning parameter selection 
#    type ------ type of penalty to select sparsity
#
#   output: a list structure
#        v ------ principal componenet vector (first order solution) nx1 dimension
#        u ------ sparse loadings vector (first order solution) px1 dimension
#          
#-------------------------------------------------------------------------------

sPCA_rSVD <- function(X, j_par, type=c("SCAD", "soft_thr", "hard_thr"), tol=1e-13, max.iter=100) {
  type <- match.arg(type)  
  decomp <- svd(X)
  v_old <- decomp$v[,1] 
  u_old <- decomp$d[1] * decomp$u[,1]
  
  if (j_par==0) {
    lambda_par <- 0
  } else {
    Xvold <- sort(abs(X%*%v_old))
    lambda_par <- (Xvold[j_par]+Xvold[(j_par+1)])/2
  }
  
  h_lambda <- switch(type,
    soft_thr = h_lambda_soft,
    hard_thr = h_lambda_hard,
    SCAD = h_lambda_SCAD)

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
#             X ------ matrix to be decomposed using SVD, will be Q^1/2 X^t D p x n
#             ntimes ------ ntimes number of repetition on CV
#             nfold ------ nfold cross validation 
#   opt_penalty ------ type of penalty to select sparsity 
#
#   output: type numeric
#         j_opt ------ optimal degree of sparsity      
#        scores ------ matrix of CV scores (nrow=p,ncol=N)
#-------------------------------------------------------------------------------

tuning_j_par <- function(X, ntimes, nfold, type, fun=median, verbose=FALSE, ...) {
  nr <- nrow(X) #p
  nc <- ncol(X) #n  
  CV_score_fold <- rep(0, ntimes)
  j_opt_cv <- rep(0, ntimes)
  
  #matrix saving CV_score for each j in each fold
  CV_score <- matrix(0, nrow=nr, ncol=ntimes) 
  cv_folds <- generateCVRuns(1:nc, ntimes=ntimes, nfold=nfold, stratified=TRUE)
  
  for (j in 0:(nr-1)) {
    if (verbose)
      cat("Number of variables", j, "\n")
    CV_temp <- rep(0, ntimes)
    for (i in 1:ntimes) {
      if (verbose)
        cat("Repetion: ",  i, "\n", sep = "")
      cv_folds_current <- cv_folds[[i]]
      
      #array with CV score_j for each l-fold
      CV_score_j <- rep(0, nfold)
      
      for (l in 1:nfold) {
        X_whitout_l <- X[,-cv_folds_current[[l]],drop=FALSE]
        X_l <- X[,cv_folds_current[[l]],drop=FALSE]
        ncj <- NCOL(X_l)
        sPCA <- sPCA_rSVD(X=X_whitout_l, type=type, j_par=j, ...)
        u_j <- sPCA$u
        v_j <- drop(t(X_l)%*%u_j)
        #compute single CV score,depend on j and the l-fold
        CV_score_j[l] <- (1/(ncj*nr))*sum((X_l- outer(u_j,v_j))^2)
      }
      CV_temp[i] <- sum(CV_score_j)
    }
    # for each repetition of CV, compute min value of CV score over j  
    CV_score[j+1,] <- CV_temp
  }
  # save j responsible for min of CV score
  goals <- apply(CV_score, 1, fun)
  j_opt_cv <- which.min(goals) - 1
  res <- list(j=j_opt_cv, scores=CV_score)
  return(res)
}

#-------------------------------------------------------------------------------
# mcia_syst_rSVD 
#   input:
#      triplets ------ results of the function create_triplets
#          type ------ type of penalty to select sparsity
#       ntimes  ------ ntimes to be used in tuning_j_par_tr  
#         nfold ------ mfold to be used in tuning_j_par_tr
#
#   output: a list structure
#          v ------ first eigenvector of sum covariances (sparse auxiliary variable)
#          u ------ list of features weights, one for each dataset
#      j_opt ------ optimal degree of sparsity      
#   cv_score ------ matrix of CV scores (nrow=p,ncol=N)
#          X ------ transformed input dataset
#      row_w ------ row weights
#      col_w ------ column weights
#-------------------------------------------------------------------------------

smcia_syst_rSVD <- function(triplets, type, ntimes, nfold, fun=median, verbose=FALSE, ...) {
  K <- length(triplets)
  row_w <- triplets[[1]]$row_w
  nr <- length(row_w)
  col_w <- sapply(triplets, function(z) z$col_w)
  dim_Xk <- sapply(col_w, length) #number of columns of each table  

  X <- matrix(0, nrow=nr, ncol=sum(dim_Xk))
  inizio <- 1
  for (k in 1:K) {
    fine <- sum(dim_Xk[1:k])  
    X[,inizio:fine] <- triplets[[k]]$weight_tab*triplets[[k]]$data
    inizio <- fine+1
  }
  
  #Y <- diag(sqrt(col_w)) %*% t(X) %*% D
  Y <- X*row_w
  Y <- t(Y)*sqrt(unlist(col_w))
  
  #tuning parameter j
  tun_j_res <- tuning_j_par(X=Y, type=type, ntimes=ntimes, nfold=nfold, fun=fun, verbose=verbose, ...)
  j_opt <- tun_j_res$j
  #CV best scores
  cv_score_best <- tun_j_res$scores
  
  eig_dec <- sPCA_rSVD(X=Y, j_par=j_opt, type=type, ...) 
  
  v <- eig_dec$v
  
  u_tot <- eig_dec$u #vector length p1+...+pk
  u_tilde <- (1/sqrt(unlist(col_w)))*u_tot
  u <- list() #list containing the u_k
  #divide by block
  # normilization of each uk wrt Qk
  inizio <- 1
  for (k in 1:K) {
    fine <- sum(dim_Xk[1:k])  
    u[[k]] <- u_tilde[inizio:fine]
    if (length(which(u[[k]]!=0)) > 0) {
      u[[k]] <- u[[k]]/sqrt(sum(col_w[[k]]*u[[k]]^2))
    }
    inizio <- fine+1
  }
  # vector v D-normed:
  v <- v / sqrt(sum(v * row_w *v)) 
  res <- list(v=v, u=u, u_tilde=u_tilde, data=X, row_w=row_w, col_w=col_w, j=j_opt, cv_score=cv_score_best)
  return(res)
}

#-------------------------------------------------------------------------------
# mcia_axes
#-------------------------------------------------------------------------------
smcia_axes <- function(triplets, num=2, zero.freq=TRUE, fix.negative=TRUE, type, ntimes, nfold, tol=1e-13, verbose=FALSE, ...) {
  num <- as.integer(num)
  if (num < 1)
      num <- 1
  trips <- list()
  trips[[1]] <- triplets
  K <- length(trips[[1]])
  if (verbose) {
    cat("#### Axis 1 ####\n")
  }
  sol <- list()
  sol[[1]] <- smcia_syst_rSVD(trips[[1]], type=type, ntimes = ntimes, nfold=nfold, verbose=verbose, ...)
  X <- sol[[1]]$data
  col_w <- unlist(sol[[1]]$col_w)
  row_w <- sol[[1]]$row_w
  Q <- diag(col_w)
  D <- diag(row_w)
  u_tilde <- sol[[1]]$u_tilde
  cum_perc <- rep(0,num)
  #construct X1
  X1 <- X %*% Q %*% u_tilde %*% solve(t(u_tilde) %*% Q %*% u_tilde)%*%t(u_tilde)
  cum_perc[1] <- sum(diag(t(X1) %*% D %*% X1 %*% Q))/sum(diag(t(X) %*% D %*% X %*% Q))
  sol[[1]]$lambda <- cum_perc[1]  
  if (num > 1) {
    for (i in 2:num) {
      if (verbose) {
        cat("#### Axis ", i, " ####\n")
      }
      trips[[i]] <- list()      
  #creation of P1_k matrix, Z_k matrix and statistical triplet with Z_k
      P <- list()
      Z_l <- list()
  # pseudo_eig <- list()
      u <- sol[[i-1]]$u  
      for (k in 1:K) {
        t_k <-trips[[i-1]][[k]]
        Q_k <- diag(t_k$col_w)
        X_k <- as.matrix(t_k$data)
    # pseudo_eig[[k]] <- t_k$pseudo_eig
        P[[k]] <- as.matrix(u[[k]]) %*% t(u[[k]]) %*% Q_k
        Z_l[[k]] <- X_k - X_k %*% t(P[[k]])
        trips[[i]][[k]] <- list(data = Z_l[[k]], row_w= t_k$row_w, col_w=t_k$col_w, inertia = t_k$inertia , weight_tab= t_k$weight_tab)
      }
      sol[[i]] <- smcia_syst_rSVD(trips[[i]], type=type, ntimes = ntimes, nfold=nfold, verbose=verbose, ...)
      u_tilde <- cbind(u_tilde,sol[[i]]$u_tilde)
      X2 <- X %*% Q %*% u_tilde %*% solve(t(u_tilde) %*% Q %*% u_tilde) %*% t(u_tilde)
      cum_perc[i] <- sum(diag(t(X2) %*% D %*% X2 %*% Q))/sum(diag(t(X) %*% D %*% X %*% Q))
      sol[[i]]$lambda <- cum_perc[i] - cum_perc[i-1]  
    }
  }
  return(sol)
}

#-------------------------------------------------------------------------------
# smcia
#   input:
#   x ------ list of K datatables: [samples X features] dataframes rownames: sampleID, colnames: feature names
#   output: a list structure
#   sol ------ solutions of sMCIA 
#-------------------------------------------------------------------------------
smcia <- function(x, num=2, zero.freq=TRUE, fix.negative=TRUE, type=c("SCAD", "soft_thr", "hard_thr"), ntimes=10, nfold=5, tol=1e-13, ...) {
  type <- match.arg(type)
  vn <- function(i) {
    if (is.null(nn <- colnames(x[[i]]))) {
      colnames(x[[i]]) <- nn <- paste("T",i,".V", 1:NCOL(x[[i]]), sep="")
    }
    return(nn)
  }
  vnames <- sapply(1:length(x), vn)
  triplets <- create_triplets(x=x, zero.freq=zero.freq, fix.negative=fix.negative, tol=tol)
  mcia_result <- smcia_axes(triplets, num=num, zero.freq=zero.freq, fix.negative=fix.negative, type=type, ntimes=ntimes, nfold=nfold, tol=tol, ...)  
  # Features weights
  weights_ax <- lapply(mcia_result, function(x) unlist(x$u))
  # Weight features from the axes wrt the respective explained percentage variance 
  lambda_ax <- lapply(mcia_result, function(x) x$lambda)
  weights_ax_scaled <- sapply(mcia_result, function(x) unlist(x$u)^2*x$lambda, simplify=TRUE)
  weights <- rowSums(weights_ax_scaled)
  # Index of features ranked by their weights
  feats_ord_idx <- order(weights, decreasing = TRUE)
  # Index of features ranked by their weights, with weights > 0 for at least one axis
  feats_ord_nozero_idx <- feats_ord_idx[1:sum(weights[feats_ord_idx]>0)]
  # Names of features extracted in the previous step
  feats_ord_nozero <- unlist(vnames)[feats_ord_nozero_idx]
  # Weights ordered in decreasing order, leaving out null values
  weights_ord_nozero <- weights[feats_ord_nozero_idx]
  # Dataframe with ranked feats and weights
  ranked_feats_weights <- data.frame(feats=feats_ord_nozero,weights=weights_ord_nozero)
  
  res <- list()
  res$mcia <- mcia_result
  res$lambda <- lambda_ax
  res$rfw <- ranked_feats_weights
  res$vnames <- vnames
  res$data <- x
  class(res) <- c("smcia", "mcia")
  return(res)
}

print.smcia <- function(x, ...) {
  cat("Sparse MCIA object\n")
  cat("Number of axes ", length(x$mcia), "\n")
  cat("Optimal zero variables for each axis\n")
  cat(sapply(x$mcia, function(z) z$j), "\n")
  invisible(x)
}
