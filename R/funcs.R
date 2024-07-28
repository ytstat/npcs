#' @export
predict_mnpo <- function(object, newx) {
  if (object$classifier == "logistic") {
    if (length(object$fit$lev) > 2) {
      posterior.test <- predict(object$fit, newdata = data.frame(x = newx), type = "prob")
    } else {
      pt1 <- predict(object$fit, newdata = data.frame(x = newx), type = "prob")
      posterior.test <- cbind(1-pt1, pt1)
    }

  } else if (object$classifier == "svm") {
    posterior.test <- attr(predict(object$fit, newdata = newx, probability = TRUE), "probabilities")
    posterior.test <- posterior.test[, match(1:length(object$fit$labels), as.numeric(colnames(posterior.test)))]
  } else if (object$classifier == "knn" || object$classifier == "randomforest") {
    posterior.test <- predict(object$fit, newdata = newx, type = "prob")
  } else if (object$classifier == "nb") {
    posterior.test <- predict(object$fit, newdata = newx, type = "raw")
  } else if (object$classifier == "tree") {
    posterior.test <- predict(object$fit, newdata = data.frame(x = newx), type = "prob")
  } else if (object$classifier == "neuralnet") {
    if (length(object$pik) > 2) {
      posterior.test <-  predict(object$fit, newdata = data.frame(x = newx), type = "raw")
    } else {
      pt1 <- predict(object$fit, newdata = data.frame(x = newx), type = "raw")
      posterior.test <- cbind(1-pt1, pt1)
    }

  } else if (object$classifier == "lda" || object$classifier == "qda") {
    posterior.test <- predict(object$fit, newx)$posterior
  } else if (object$classifier == "nnb") {
    if (is.null(colnames(newx))) {
      colnames(newx) <- paste0("x", 1:ncol(newx))
    }
    posterior.test <- predict(object$fit, newdata = newx, type = "prob")
  }

  cost_posterior.test <- t(t(posterior.test)*object$phi)
  pred.test <- as.numeric(apply(cost_posterior.test, 1, which.max))
  return(pred.test)
}



#' @export
mnpo <- function(x, y, classifier, alpha, w, increment = 0.1, ...) {
  n <- length(y)
  K <- length(unique(y))
  if (length(increment) == 1) {
    increment <- rep(increment, K-1)
  }
  phi <- expand.grid(sapply(1:(K-1), function(k){seq(0, 1, increment[k])}, simplify = F))
  phi[, K] <- 1-rowSums(phi)
  phi <- phi[phi[, K] >=0, ]

  ind <- sample(n, floor(n/2)) # index set of test data
  if (classifier == "logistic") {
    fit <- multinom(y~., data = data.frame(x = x[-ind, ], y = y[-ind]) , trace = F, ...)
    if (K> 2) {
      posterior <- predict(fit, newdata = data.frame(x = x[ind, ], y = y[ind]), type = "prob")
    } else {
      pt1 <- predict(fit, newdata = data.frame(x = x[ind, ], y = y[ind]), type = "prob")
      posterior <- cbind(1-pt1, pt1)
    }
  } else if (classifier == "knn") {
    fit <- knn3(x = x[-ind, ], y = factor(y)[-ind], ...)
    posterior <- predict(fit, newdata = x[ind, ], type = "prob")
  } else if (classifier == "randomforest") {
    fit <- randomForest(x = x[-ind, ], y = factor(y)[-ind], ...)
    posterior <- predict(fit, newdata = x[ind, ], type = "prob")
  } else if (classifier == "svm") {
    fit <- svm(x = x[-ind, ], y = factor(y)[-ind], probability = TRUE, ...)
    posterior <- attr(predict(fit, newdata = x[ind, ], probability = TRUE), "probabilities")
    posterior <- posterior[, match(1:K, as.numeric(colnames(posterior)))]
  } else if (classifier == "nb") {
    fit <- naiveBayes(x = x[-ind, ], y = factor(y)[-ind], ...)
    posterior <- predict(fit, newdata = x[ind, ], type = "raw")
  } else if (classifier == "tree") {
    fit <- rpart(y ~ ., data = data.frame(x = x[-ind, ], y = factor(y)[-ind]), ...)
    posterior <- predict(fit, newdata = data.frame(x = x[ind, ]), type = "prob")
  } else if (classifier == "neuralnet") {
    fit <- nnet(y ~ ., data = data.frame(x = x[-ind, ], y = factor(y)[-ind]), trace = FALSE, ...)
    if (K> 2) {
      posterior <- predict(fit, newdata = data.frame(x = x[ind, ]), type = "raw")
    } else {
      pt1 <- predict(fit, newdata = data.frame(x = x[ind, ]), type = "raw")
      posterior <- cbind(1-pt1, pt1)
    }
  } else if (classifier == "lda") {
    fit <- lda(x = x[-ind, ], grouping = factor(y)[-ind])
    posterior <- predict(fit, x[ind, ])$posterior
  } else if (classifier == "qda") {
    fit <- qda(x = x[-ind, ], grouping = factor(y)[-ind])
    posterior <- predict(fit, x[ind, ])$posterior
  } else if (classifier == "nnb") {
    if (is.null(colnames(x))) {
      colnames(x) <- paste0("x", 1:ncol(x))
    }
    fit <- nonparametric_naive_bayes(x = x[-ind, ], y = factor(y)[-ind], ...)
    posterior <- predict(fit, x[ind, ], type = "prob")
  }

  error_phi <- t(sapply(1:nrow(phi), function(i){
    cost_posterior <- t(t(posterior)*as.numeric(phi[i, ]))
    y.pred <- as.numeric(apply(cost_posterior, 1, which.max))
    er <- error_rate(y.pred, y[ind])
    er[K+1] <- sum(w*er)
    er
  }))


  feasible_phi <- apply(error_phi, 1, function(x){all((x[-(K+1)]<=alpha)[!is.na(alpha)])})

  if (sum(feasible_phi) >= 1) {
    feasible_min_ind <- which.min(error_phi[feasible_phi, K+1])
    phi_best <- as.numeric((phi[feasible_phi,])[feasible_min_ind, ])

    return(list(fit = fit, phi = phi_best, classifier = classifier))
  } else {
    warning("infeasible!\n")
    return(NA)
  }

}


#' @export
mnpo_confusion <- function(x, y, classifier, alpha, w, increment = 0.1, ...) {
  n <- length(y)
  K <- length(unique(y))
  if (length(increment) == 1) {
    increment <- rep(increment, K-1)
  }
  phi <- expand.grid(sapply(1:(K-1), function(k){seq(0, 1, increment[k])}, simplify = F))
  phi[, K] <- 1-rowSums(phi)
  phi <- phi[phi[, K] >=0, ]

  ind <- sample(n, floor(n/2)) # index set of test data
  if (classifier == "logistic") {
    fit <- multinom(y~., data = data.frame(x = x[-ind, ], y = y[-ind]) , trace = F, ...)
    if (K> 2) {
      posterior <- predict(fit, newdata = data.frame(x = x[ind, ], y = y[ind]), type = "prob")
    } else {
      pt1 <- predict(fit, newdata = data.frame(x = x[ind, ], y = y[ind]), type = "prob")
      posterior <- cbind(1-pt1, pt1)
    }
  } else if (classifier == "knn") {
    fit <- knn3(x = x[-ind, ], y = factor(y)[-ind], ...)
    posterior <- predict(fit, newdata = x[ind, ], type = "prob")
  } else if (classifier == "randomforest") {
    fit <- randomForest(x = x[-ind, ], y = factor(y)[-ind], ...)
    posterior <- predict(fit, newdata = x[ind, ], type = "prob")
  } else if (classifier == "svm") {
    fit <- svm(x = x[-ind, ], y = factor(y)[-ind], probability = TRUE, ...)
    posterior <- attr(predict(fit, newdata = x[ind, ], probability = TRUE), "probabilities")
    posterior <- posterior[, match(1:K, as.numeric(colnames(posterior)))]
  } else if (classifier == "nb") {
    fit <- naiveBayes(x = x[-ind, ], y = factor(y)[-ind], ...)
    posterior <- predict(fit, newdata = x[ind, ], type = "raw")
  } else if (classifier == "tree") {
    fit <- rpart(y ~ ., data = data.frame(x = x[-ind, ], y = factor(y)[-ind]), ...)
    posterior <- predict(fit, newdata = data.frame(x = x[ind, ]), type = "prob")
  } else if (classifier == "neuralnet") {
    fit <- nnet(y ~ ., data = data.frame(x = x[-ind, ], y = factor(y)[-ind]), trace = FALSE, ...)
    if (K> 2) {
      posterior <- predict(fit, newdata = data.frame(x = x[ind, ]), type = "raw")
    } else {
      pt1 <- predict(fit, newdata = data.frame(x = x[ind, ]), type = "raw")
      posterior <- cbind(1-pt1, pt1)
    }
  } else if (classifier == "lda") {
    fit <- lda(x = x[-ind, ], grouping = factor(y)[-ind])
    posterior <- predict(fit, x[ind, ])$posterior
  } else if (classifier == "qda") {
    fit <- qda(x = x[-ind, ], grouping = factor(y)[-ind])
    posterior <- predict(fit, x[ind, ])$posterior
  } else if (classifier == "nnb") {
    if (is.null(colnames(x))) {
      colnames(x) <- paste0("x", 1:ncol(x))
    }
    fit <- nonparametric_naive_bayes(x = x[-ind, ], y = factor(y)[-ind], ...)
    posterior <- predict(fit, x[ind, ], type = "prob")
  }

  error_phi <- t(sapply(1:nrow(phi), function(i){
    cost_posterior <- t(t(posterior)*as.numeric(phi[i, ]))
    y.pred <- as.numeric(apply(cost_posterior, 1, which.max))
    er <- confusion_matrix(y.pred, y[ind])
    obj_value <- sum(w*er)
    c(er, obj_value)
  }))


  feasible_phi <- apply(error_phi, 1, function(x){all((x[-length(x)]<=as.numeric(alpha))[!is.na(as.numeric(alpha))])})

  if (sum(feasible_phi) >= 1) {
    feasible_min_ind <- which.min(error_phi[feasible_phi, ncol(error_phi)])
    phi_best <- as.numeric((phi[feasible_phi,])[feasible_min_ind, ])

    return(list(fit = fit, phi = phi_best, classifier = classifier))
  } else {
    warning("infeasible!\n")
    return(NA)
  }

}


#' @export
confusion_matrix <- function(y.pred, y, class.names = NULL) {
  if (is.null(class.names)) {
    class.names <- as.numeric(levels(as.factor(y)))
  }


  error_rate <- t(sapply(1:length(class.names),function(i){ # actual class
    sapply(1:length(class.names), function(j){ # predicted class
      length(y.pred[y == class.names[i] & y.pred == class.names[j]])/length(y[y == class.names[i]])
    })
  }))
  # diag(error_rate) <- 0
  rownames(error_rate) <- class.names
  colnames(error_rate) <- class.names

  return(error_rate)
}

obj.CX <- function(lambda, w, pik, alpha, posterior, index) {
  lambda.full <- rep(0, length(w))
  lambda.full[index] <- lambda
  ck <- (lambda.full+w)/pik

  cost_posterior <- t(t(posterior)*ck)

  cost_posterior_pred <- as.numeric(apply(cost_posterior, 1, max))
  - mean(cost_posterior_pred) + 1 + sum(lambda*(1-alpha[index]))
}

obj.CX.min <- function(lambda, w, pik, alpha, posterior, index) {
  lambda.full <- rep(0, length(w))
  lambda.full[index] <- lambda
  ck <- (lambda.full+w)/pik

  cost_posterior <- t(t(posterior)*ck)

  cost_posterior_pred <- as.numeric(apply(cost_posterior, 1, max))
  -(- mean(cost_posterior_pred) + 1 + sum(lambda*(1-alpha[index])))
}

obj.CX.confusion <- function(lambda, w, pik, alpha, posterior, index) {
  lambda.full <- matrix(0, nrow = nrow(w), ncol = ncol(w))
  lambda.full[index] <- lambda
  ckr <- (lambda.full+w)/pik
  diag(ckr) <- 0
  cost_posterior <- sapply(1:nrow(w), function(r){
    posterior_k_matrix <- t(posterior)*ckr[, r]
    colSums(posterior_k_matrix)
  })

  cost_posterior_pred <- as.numeric(apply(cost_posterior, 1, min))
  mean(cost_posterior_pred) - sum(lambda*(alpha[index]))
}

obj.ER.confusion <- function(lambda, w, pik, alpha, posterior, index, y) {
  lambda.full <- matrix(0, nrow = nrow(w), ncol = ncol(w))
  lambda.full[index] <- lambda
  ckr <- (lambda.full+w)/pik
  diag(ckr) <- 0
  cost_posterior <- sapply(1:nrow(w), function(r){
    posterior_k_matrix <- t(posterior)*ckr[, r]
    colSums(posterior_k_matrix)
  })

  cost_posterior_pred <- as.numeric(apply(cost_posterior, 1, which.min))

  er.cur <- confusion_matrix(cost_posterior_pred, y)
  diag(er.cur) <- 0
  sum((lambda.full+w)*er.cur) - sum(lambda*(alpha[index]))
}

obj.ER <- function(lambda, w, pik, alpha, posterior, index, y) {
  lambda.full <- rep(0, length(w))
  lambda.full[index] <- lambda
  ck <- (lambda.full+w)/pik

  cost_posterior <- t(t(posterior)*ck)

  cost_posterior_pred <- as.numeric(apply(cost_posterior, 1, which.max))
  er.cur <- 1-error_rate(cost_posterior_pred, y)
  - sum((lambda.full+w)*er.cur) + 1 + sum(lambda*(1-alpha[index]))
}


obj.ER.min <- function(lambda, w, pik, alpha, posterior, index, y) {
  lambda.full <- rep(0, length(w))
  lambda.full[index] <- lambda
  ck <- (lambda.full+w)/pik

  cost_posterior <- t(t(posterior)*ck)

  cost_posterior_pred <- as.numeric(apply(cost_posterior, 1, which.max))
  er.cur <- 1-error_rate(cost_posterior_pred, y)
  -(- sum((lambda.full+w)*er.cur) + 1 + sum(lambda*(1-alpha[index])))
}

hjkb1 <- function(par, fn, lower = -Inf, upper = Inf, control = list(), ...) {
  if (!is.numeric(par))
    stop("Argument 'par' must be a numeric vector.", call. = FALSE)
  n <- length(par)
  # if (n == 1)
  #   stop("For univariate functions use some different method.", call. = FALSE)

  if(!is.numeric(lower) || !is.numeric(upper))
    stop("Lower and upper limits must be numeric.", call. = FALSE)
  if (length(lower) == 1) lower <- rep(lower, n)
  if (length(upper) == 1) upper <- rep(upper, n)
  if (!all(lower <= upper))
    stop("All lower limits must be smaller than upper limits.", call. = FALSE)
  if (!all(lower <= par) || !all(par <= upper))
    stop("Infeasible starting values -- check limits.", call. = FALSE)


  #-- Control list handling ----------
  cntrl <- list(tol      = 1.e-06,
                maxfeval = Inf,       # set to Inf if no limit wanted
                maximize = FALSE,     # set to TRUE for maximization
                target   = Inf,       # set to Inf for no restriction
                info     = FALSE)     # for printing interim information
  nmsCo <- match.arg(names(control), choices = names(cntrl), several.ok = TRUE)
  if (!is.null(names(control))) cntrl[nmsCo] <- control

  tol      <- cntrl$tol;
  maxfeval <- cntrl$maxfeval
  maximize <- cntrl$maximize
  target   <- cntrl$target
  info     <- cntrl$info

  scale <- if (maximize) -1 else 1
  fun <- match.fun(fn)
  f <- function(x) scale * fun(x, ...)

  #-- Setting steps and stepsize -----
  nsteps <- floor(log2(1/tol))        # number of steps
  steps  <- 2^c(-(0:(nsteps-1)))      # decreasing step size
  dir <- diag(1, n, n)                # orthogonal directions

  x <- par                            # start point
  fx <- fbest <- f(x)                 # smallest value so far
  fcount <- 1                         # counts number of function calls

  if (info) cat("step\tnofc\tfmin\txpar\n")   # info header

  #-- Start the main loop ------------
  ns <- 0
  while (ns < nsteps && fcount < maxfeval && abs(fx) < target) {
    ns <- ns + 1
    hjs    <- .hjbsearch(x, f, lower, upper,
                         steps[ns], dir, fcount, maxfeval, target)
    x      <- hjs$x
    fx     <- hjs$fx
    sf     <- hjs$sf
    fcount <- fcount + hjs$finc

    if (info)
      cat(ns, "\t",  fcount, "\t", fx/scale, "\t", x[1], "...\n")
  }

  if (fcount > maxfeval) {
    warning("Function evaluation limit exceeded -- may not converge.")
    conv <- 1
  } else if (abs(fx) > target) {
    warning("Function exceeds min/max value -- may not converge.")
    conv <- 1
  } else {
    conv <- 0
  }

  fx <- fx / scale                    # undo scaling
  return(list(par = x, value = fx,
              convergence = conv, feval = fcount, niter = ns))
}

##  Search with a single scale -----------------------------
.hjbsearch <- function(xb, f, lo, up, h, dir, fcount, maxfeval, target) {
  x  <- xb
  xc <- x
  sf <- 0
  finc <- 0
  hje  <- .hjbexplore(xb, xc, f, lo, up, h, dir)
  x    <- hje$x
  fx   <- hje$fx
  sf   <- hje$sf
  finc <- finc + hje$numf

  # Pattern move
  while (sf == 1) {
    d  <- x-xb
    xb <- x
    xc <- x+d
    xc <- pmax(pmin(xc, up), lo)
    fb <- fx
    hje  <- .hjbexplore(xb, xc, f, lo, up, h, dir, fb)
    x    <- hje$x
    fx   <- hje$fx
    sf   <- hje$sf
    finc <- finc + hje$numf

    if (sf == 0) {  # pattern move failed
      hje  <- .hjbexplore(xb, xb, f, lo, up, h, dir, fb)
      x    <- hje$x
      fx   <- hje$fx
      sf   <- hje$sf
      finc <- finc + hje$numf
    }
    if (fcount + finc > maxfeval || abs(fx) > target) break
  }

  return(list(x = x, fx = fx, sf = sf, finc = finc))
}

##  Exploratory move ---------------------------------------
.hjbexplore <- function(xb, xc, f, lo, up, h, dir, fbold) {
  n <- length(xb)
  x <- xb

  if (missing(fbold)) {
    fb <- f(x)
    numf <- 1
  } else {
    fb <- fbold
    numf <- 0
  }

  fx <- fb
  xt <- xc
  sf <- 0                             # do we find a better point ?
  dirh <- h * dir
  fbold <- fx
  for (k in sample.int(n, n)) {       # resample orthogonal directions
    p1 <- xt + dirh[, k]
    if ( p1[k] <= up[k] ) {
      ft1 <- f(p1)
      numf <- numf + 1
    } else {
      ft1 <- fb
    }

    p2 <- xt - dirh[, k]
    if ( lo[k] <= p2[k] ) {
      ft2 <- f(p2)
      numf <- numf + 1
    } else {
      ft2 <- fb
    }

    if (min(ft1, ft2) < fb) {
      sf <- 1
      if (ft1 < ft2) {
        xt <- p1
        fb <- ft1
      } else {
        xt <- p2
        fb <- ft2
      }
    }
  }

  if (sf == 1) {
    x  <- xt
    fx <- fb
  }

  return(list(x = x, fx = fx, sf = sf, numf = numf))
}


nmkb1 <- function (par, fn, lower = -Inf, upper = Inf, control = list(), ...)
{
  ctrl <- list(tol = 1e-06, maxfeval = min(5000, max(1500,
                                                     20 * length(par)^2)), regsimp = TRUE, maximize = FALSE,
               restarts.max = 3, trace = FALSE)
  namc <- match.arg(names(control), choices = names(ctrl),
                    several.ok = TRUE)
  if (!all(namc %in% names(ctrl)))
    stop("unknown names in control: ", namc[!(namc %in% names(ctrl))])
  if (!is.null(names(control)))
    ctrl[namc] <- control
  ftol <- ctrl$tol
  maxfeval <- ctrl$maxfeval
  regsimp <- ctrl$regsimp
  restarts.max <- ctrl$restarts.max
  maximize <- ctrl$maximize
  trace <- ctrl$trace
  n <- length(par)

  g <- function(x) {
    gx <- x
    gx[c1] <- atanh(2 * (x[c1] - lower[c1]) / (upper[c1] - lower[c1]) - 1)
    gx[c3] <- log(x[c3] - lower[c3])
    gx[c4] <- log(upper[c4] - x[c4])
    gx
  }

  ginv <- function(x) {
    gix <- x
    gix[c1] <- lower[c1] + (upper[c1] - lower[c1])/2 * (1 + tanh(x[c1]))
    gix[c3] <- lower[c3] + exp(x[c3])
    gix[c4] <- upper[c4] - exp(x[c4])
    gix
  }

  if (length(lower) == 1) lower <- rep(lower, n)
  if (length(upper) == 1) upper <- rep(upper, n)

  if (any(c(par < lower, upper < par))) stop("Infeasible starting values!", call.=FALSE)

  low.finite <- is.finite(lower)
  upp.finite <- is.finite(upper)
  c1 <- low.finite & upp.finite  # both lower and upper bounds are finite
  c2 <- !(low.finite | upp.finite) # both lower and upper bounds are infinite
  c3 <- !(c1 | c2) & low.finite # finite lower bound, but infinite upper bound
  c4 <- !(c1 | c2) & upp.finite  # finite upper bound, but infinite lower bound

  if (all(c2)) stop("Use `nmk()' for unconstrained optimization!", call.=FALSE)

  if (maximize)
    fnmb <- function(par) -fn(ginv(par), ...)
  else fnmb <- function(par) fn(ginv(par), ...)

  x0 <- g(par)
  # if (n == 1)
  #   stop(call. = FALSE, "Use `optimize' for univariate optimization")
  if (n > 30)
    warning("Nelder-Mead should not be used for high-dimensional optimization")
  V <- cbind(rep(0, n), diag(n))
  f <- rep(0, n + 1)
  f[1] <- fnmb(x0)
  V[, 1] <- x0
  scale <- max(1, sqrt(sum(x0^2)))
  if (regsimp) {
    alpha <- scale/(n * sqrt(2)) * c(sqrt(n + 1) + n - 1,
                                     sqrt(n + 1) - 1)
    V[, -1] <- (x0 + alpha[2])
    diag(V[, -1]) <- x0[1:n] + alpha[1]
    for (j in 2:ncol(V)) f[j] <- fnmb(V[, j])
  }
  else {
    V[, -1] <- x0 + scale * V[, -1]
    for (j in 2:ncol(V)) f[j] <- fnmb(V[, j])
  }
  f[is.nan(f)] <- Inf
  nf <- n + 1
  ord <- order(f)
  f <- f[ord]
  V <- V[, ord]
  rho <- 1
  gamma <- 0.5
  chi <- 2
  sigma <- 0.5
  conv <- 1
  oshrink <- 1
  restarts <- 0
  orth <- 0
  dist <- f[n + 1] - f[1]
  v <- V[, -1] - V[, 1]
  delf <- f[-1] - f[1]
  diam <- sqrt(colSums(v^2))
  #    sgrad <- c(solve(t(v), delf))
  sgrad <- c(crossprod(t(v), delf))
  alpha <- 1e-04 * max(diam)/sqrt(sum(sgrad^2))
  simplex.size <- sum(abs(V[, -1] - V[, 1]))/max(1, sum(abs(V[,
                                                              1])))
  itc <- 0
  conv <- 0
  message <- "Succesful convergence"
  while (nf < maxfeval & restarts < restarts.max & dist > ftol &
         simplex.size > 1e-06) {
    fbc <- mean(f)
    happy <- 0
    itc <- itc + 1
    xbar <- rowMeans(V[, 1:n])
    xr <- (1 + rho) * xbar - rho * V[, n + 1]
    fr <- fnmb(xr)
    nf <- nf + 1
    if (is.nan(fr))
      fr <- Inf
    if (fr >= f[1] & fr < f[n]) {
      happy <- 1
      xnew <- xr
      fnew <- fr
    }
    else if (fr < f[1]) {
      xe <- (1 + rho * chi) * xbar - rho * chi * V[, n +
                                                     1]
      fe <- fnmb(xe)
      if (is.nan(fe))
        fe <- Inf
      nf <- nf + 1
      if (fe < fr) {
        xnew <- xe
        fnew <- fe
        happy <- 1
      }
      else {
        xnew <- xr
        fnew <- fr
        happy <- 1
      }
    }
    else if (fr >= f[n] & fr < f[n + 1]) {
      xc <- (1 + rho * gamma) * xbar - rho * gamma * V[,
                                                       n + 1]
      fc <- fnmb(xc)
      if (is.nan(fc))
        fc <- Inf
      nf <- nf + 1
      if (fc <= fr) {
        xnew <- xc
        fnew <- fc
        happy <- 1
      }
    }
    else if (fr >= f[n + 1]) {
      xc <- (1 - gamma) * xbar + gamma * V[, n + 1]
      fc <- fnmb(xc)
      if (is.nan(fc))
        fc <- Inf
      nf <- nf + 1
      if (fc < f[n + 1]) {
        xnew <- xc
        fnew <- fc
        happy <- 1
      }
    }
    if (happy == 1 & oshrink == 1) {
      fbt <- mean(c(f[1:n], fnew))
      delfb <- fbt - fbc
      armtst <- alpha * sum(sgrad^2)
      if (delfb > -armtst/n) {
        if (trace)
          cat("Trouble - restarting: \n")
        restarts <- restarts + 1
        orth <- 1
        diams <- min(diam)
        sx <- sign(0.5 * sign(sgrad))
        happy <- 0
        V[, -1] <- V[, 1]
        diag(V[, -1]) <- diag(V[, -1]) - diams * sx[1:n]
      }
    }
    if (happy == 1) {
      V[, n + 1] <- xnew
      f[n + 1] <- fnew
      ord <- order(f)
      V <- V[, ord]
      f <- f[ord]
    }
    else if (happy == 0 & restarts < restarts.max) {
      if (orth == 0)
        orth <- 1
      V[, -1] <- V[, 1] - sigma * (V[, -1] - V[, 1])
      for (j in 2:ncol(V)) f[j] <- fnmb(V[, j])
      nf <- nf + n
      ord <- order(f)
      V <- V[, ord]
      f <- f[ord]
    }
    v <- V[, -1] - V[, 1]
    delf <- f[-1] - f[1]
    diam <- sqrt(colSums(v^2))
    simplex.size <- sum(abs(v))/max(1, sum(abs(V[, 1])))
    f[is.nan(f)] <- Inf
    dist <- f[n + 1] - f[1]
    #        sgrad <- c(solve(t(v), delf))
    sgrad <- c(crossprod(t(v), delf))
    if (trace & !(itc%%2))
      cat("iter: ", itc, "\n", "value: ", f[1], "\n")
  }
  if (dist <= ftol | simplex.size <= 1e-06) {
    conv <- 0
    message <- "Successful convergence"
  }
  else if (nf >= maxfeval) {
    conv <- 1
    message <- "Maximum number of fevals exceeded"
  }
  else if (restarts >= restarts.max) {
    conv <- 2
    message <- "Stagnation in Nelder-Mead"
  }
  return(list(par = ginv(V[, 1]), value = f[1] * (-1)^maximize, feval = nf,
              restarts = restarts, convergence = conv, message = message))
}


knearest <- function (D, P, n_clust) {
  knD <- knnx.index(D, P, k = (n_clust + 1), algo = "kd_tree")
  knD = knD * (knD != row(knD))
  que = which(knD[, 1] > 0)
  for (i in que) {
    knD[i, which(knD[i, ] == 0)] = knD[i, 1]
    knD[i, 1] = 0
  }
  return(knD[, 2:(n_clust + 1)])
}
