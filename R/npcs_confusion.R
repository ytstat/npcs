#' Fit a multi-class Neyman-Pearson classifier with error controls via cost-sensitive learning.
#' @export

npcs_confusion <- function(x, y, algorithm = c("CX", "ER"), classifier = c("logistic", "knn", "randomforest", "tree", "neuralnet", "svm", "lda", "qda", "nb", "nnb", "logistic-l1"),
                    w, alpha, split.ratio = 0.5, split.mode = c("by-class", "merged"), tol = 1e-6, refit = TRUE, protect = TRUE, opt.alg = c("Hooke-Jeeves", "Nelder-Mead"), limit = 200, fitted.model = NULL, inf.threshold = 1, ...) {

  algorithm <- match.arg(algorithm)
  classifier <- match.arg(classifier)
  split.mode <- match.arg(split.mode)
  opt.alg <- match.arg(opt.alg)
  n <- length(y)
  K <- length(unique(y))
  w.ori <- w
  w <- w/sum(w)
  index <- which(!is.na(alpha), arr.ind = TRUE)


  if (algorithm == "CX") {
    pik <- as.numeric(table(y)/n)
    if (is.null(fitted.model)) {
      if (classifier == "logistic") {
        fit <- multinom(y ~ ., data = data.frame(x = x, y = y), trace = FALSE, ...)
        if (K> 2) {
          posterior <- predict(fit, newdata = data.frame(x = x, y = y), type = "prob")
        } else {
          pt1 <- predict(fit, newdata = data.frame(x = x, y = y) , type = "prob")
          posterior <- cbind(1-pt1, pt1)
        }
      } else if (classifier == "knn") {
        fit <- knn3(x = x, y = factor(y), ...)
        posterior <- predict(fit, newdata = x, type = "prob")
      } else if (classifier == "randomforest") {
        fit <- randomForest(x = x, y = factor(y), ...)
        posterior <- predict(fit, newdata = x, type = "prob")
      } else if (classifier == "svm") {
        fit <- svm(x = x, y = factor(y), probability = TRUE, ...)
        posterior <- attr(predict(fit, newdata = x, probability = TRUE), "probabilities")
        posterior <- posterior[, match(1:K, as.numeric(colnames(posterior)))]
      } else if (classifier == "nb") {
        fit <- naiveBayes(x = x, y = factor(y), ...)
        posterior <- predict(fit, newdata = x, type = "raw")
      } else if (classifier == "tree") {
        fit <- rpart(y ~ ., data = data.frame(x = x, y = factor(y)), ...)
        posterior <- predict(fit, newdata = data.frame(x = x, y = factor(y)), type = "prob")
      } else if (classifier == "neuralnet") {
        fit <- nnet(y ~ ., data = data.frame(x = x, y = factor(y)), trace = FALSE, ...)
        if (K> 2) {
          posterior <- predict(fit, newdata = data.frame(x = x, y = factor(y)) , type = "raw")
        } else {
          pt1 <- predict(fit, newdata = data.frame(x = x, y = factor(y)), type = "raw")
          posterior <- cbind(1-pt1, pt1)
        }
      } else if (classifier == "lda") {
        fit <- lda(x = x, grouping = factor(y), ...)
        posterior <- predict(fit, x)$posterior
      } else if (classifier == "qda") {
        fit <- qda(x = x, grouping = factor(y), ...)
        posterior <- predict(fit, x)$posterior
      } else if (classifier == "nnb") {
        if (is.null(colnames(x))) {
          colnames(x) <- paste0("x", 1:ncol(x))
        }
        fit <- nonparametric_naive_bayes(x = x, y = factor(y), ...)
        posterior <- predict(fit, x, type = "prob")
      } else if (classifier == "logistic-l1") {
        D <- data.frame(x = x, y = y)
        x.model <- model.matrix(y~.-1, D)
        fit <- cv.glmnet(x = as.matrix(x.model), y = y, family = "multinomial", nfolds = 5)
        posterior <- as.matrix(predict(fit, as.matrix(x.model[ind, ]), type = "response")[ , , 1])
      }
    } else { # the base model has been fitted and input
      fit <- fitted.model
      if (classifier == "logistic") {
        if (K> 2) {
          posterior <- predict(fit, newdata = data.frame(x = x), type = "prob")
        } else {
          pt1 <- predict(fit, newdata = data.frame(x = x), type = "prob")
          posterior <- cbind(1-pt1, pt1)
        }
      } else if (classifier == "knn") {
        posterior <- predict(fit, newdata = x, type = "prob")
      } else if (classifier == "randomforest") {
        posterior <- predict(fit, newdata = x, type = "prob")
      } else if (classifier == "svm") {
        posterior <- attr(predict(fit, newdata = x, probability = TRUE), "probabilities")
        posterior <- posterior[, match(1:K, as.numeric(colnames(posterior)))]
      } else if (classifier == "nb") {
        posterior <- predict(fit, newdata = x, type = "raw")
      } else if (classifier == "tree") {
        posterior <- predict(fit, newdata = data.frame(x = x, y = factor(y)), type = "prob")
      } else if (classifier == "neuralnet") {
        if (K> 2) {
          posterior <- predict(fit, newdata = data.frame(x = x), type = "raw")
        } else {
          pt1 <- predict(fit, newdata = data.frame(x = x), type = "raw")
          posterior <- cbind(1-pt1, pt1)
        }
      } else if (classifier == "lda") {
        posterior <- predict(fit, x)$posterior
      } else if (classifier == "qda") {
        posterior <- predict(fit, x)$posterior
      } else if (classifier == "nnb") {
        if (is.null(colnames(x))) {
          colnames(x) <- paste0("x", 1:ncol(x))
        }
        posterior <- predict(fit, x, type = "prob")
      } else if (classifier == "logistic-l1") {
        D <- data.frame(x = x, y = y)
        x.model <- model.matrix(y~.-1, D)
        posterior <- as.matrix(predict(fit, as.matrix(x.model), type = "response")[ , , 1])
      }
    }


    if (nrow(index) == 1) {
      if (opt.alg == "Nelder-Mead"){
        lambda <- optimize(f = obj.CX.confusion, lower = 0, maximum = T, upper = limit, alpha = alpha, pik = pik, posterior = posterior, w = w, index = index, tol = tol)
      } else if (opt.alg == "Hooke-Jeeves") {
        lambda <- hjkb1(par = rep(0, nrow(index)), fn = obj.CX.confusion, upper = rep(limit, nrow(index)), lower = rep(0, nrow(index)), control = c(maximize = TRUE, tol = tol), alpha = alpha, pik = pik, posterior = posterior, w = w, index = index)
      }
    } else if (nrow(index) > 1) {
      if (opt.alg == "Nelder-Mead"){
        lambda <- nmkb(par = rep(0.0001, nrow(index)), fn = obj.CX.confusion, upper = rep(limit, nrow(index)), lower = rep(0, nrow(index)), control = c(maximize = TRUE, tol = tol), alpha = alpha, pik = pik, posterior = posterior, w = w, index = index)
      } else if (opt.alg == "Hooke-Jeeves") {
        lambda <- hjkb(par = rep(0, nrow(index)), fn = obj.CX.confusion, upper = rep(limit, nrow(index)), lower = rep(0, nrow(index)), control = c(maximize = TRUE, tol = tol), alpha = alpha, pik = pik, posterior = posterior, w = w, index = index)
      }
    }

  }


  if (algorithm == "ER") {
    if (is.null(fitted.model)) {
      if (split.mode == "merged") {
        ind <- sample(n, floor(n*split.ratio)) # the indices of samples used to estimate lambda
        pik <- as.numeric(table(y[-ind])/length(y[-ind]))
      } else {
        ind <- Reduce("c", sapply(1:K, function(k){
          ind.k <- which(y == k)
          sample(ind.k, floor(length(ind.k)*split.ratio))
        }, simplify = F))
        pik <- as.numeric(table(y[-ind])/length(y[-ind]))
      }

      if (classifier == "logistic") {
        fit <- multinom(y~., data = data.frame(x = x[-ind, ], y = y[-ind]) , trace = F, ...)
        if (K> 2) {
          posterior <- predict(fit, newdata = data.frame(x = x[ind, ], y = y[ind]), type = "prob")
        } else {
          pt1 <- predict(fit, newdata = data.frame(x = x[ind, ], y = y[ind]), type = "prob")
          posterior <- cbind(1-pt1, pt1)
        }
        if (refit) {
          fit <- multinom(y~., data = data.frame(x = x, y = y), trace = F, ...)
          pik <- as.numeric(table(y)/length(y))
        }
      } else if (classifier == "knn") {
        fit <- knn3(x = x[-ind, ], y = factor(y)[-ind], ...)
        posterior <- predict(fit, newdata = x[ind, ], type = "prob")
        if (refit) {
          fit <- knn3(x = x, y = factor(y), ...)
          pik <- as.numeric(table(y)/length(y))
        }
      } else if (classifier == "randomforest") {
        fit <- randomForest(x = x[-ind, ], y = factor(y)[-ind], ...)
        posterior <- predict(fit, newdata = x[ind, ], type = "prob")
        if (refit) {
          fit <- randomForest(x = x, y = factor(y), ...)
          pik <- as.numeric(table(y)/length(y))
        }
      } else if (classifier == "svm") {
        fit <- svm(x = x[-ind, ], y = factor(y)[-ind], probability = TRUE, ...)
        posterior <- attr(predict(fit, newdata = x[ind, ], probability = TRUE), "probabilities")
        posterior <- posterior[, match(1:K, as.numeric(colnames(posterior)))]
        if (refit) {
          fit <- svm(x = x, y = factor(y), probability = TRUE, ...)
          pik <- as.numeric(table(y)/length(y))
        }
      } else if (classifier == "nb") {
        fit <- naiveBayes(x = x[-ind, ], y = factor(y)[-ind], ...)
        posterior <- predict(fit, newdata = x[ind, ], type = "raw")
        if (refit) {
          fit <- naiveBayes(x = x, y = factor(y), ...)
          pik <- as.numeric(table(y)/length(y))
        }
      } else if (classifier == "tree") {
        fit <- rpart(y ~ ., data = data.frame(x = x[-ind, ], y = factor(y)[-ind]), ...)
        posterior <- predict(fit, newdata = data.frame(x = x[ind, ]), type = "prob")
        if (refit) {
          fit <- rpart(y ~ ., data = data.frame(x = x, y = factor(y)), ...)
          pik <- as.numeric(table(y)/length(y))
        }
      } else if (classifier == "neuralnet") {
        fit <- nnet(y ~ ., data = data.frame(x = x[-ind, ], y = factor(y)[-ind]), trace = FALSE, ...)
        if (K> 2) {
          posterior <- predict(fit, newdata = data.frame(x = x[ind, ]), type = "raw")
        } else {
          pt1 <- predict(fit, newdata = data.frame(x = x[ind, ]), type = "raw")
          posterior <- cbind(1-pt1, pt1)
        }
        if (refit) {
          fit <- nnet(y ~ ., data = data.frame(x = x, y = factor(y)), trace = FALSE, ...)
          pik <- as.numeric(table(y)/length(y))
        }
      } else if (classifier == "lda") {
        fit <- lda(x = x[-ind, ], grouping = factor(y)[-ind])
        posterior <- predict(fit, x[ind, ])$posterior
        if (refit) {
          fit <- lda(x = x, grouping = factor(y), ...)
          pik <- as.numeric(table(y)/length(y))
        }
      } else if (classifier == "qda") {
        fit <- qda(x = x[-ind, ], grouping = factor(y)[-ind])
        posterior <- predict(fit, x[ind, ])$posterior
        if (refit) {
          fit <- qda(x = x, grouping = factor(y), ...)
          pik <- as.numeric(table(y)/length(y))
        }
      } else if (classifier == "nnb") {
        if (is.null(colnames(x))) {
          colnames(x) <- paste0("x", 1:ncol(x))
        }
        fit <- nonparametric_naive_bayes(x = x[-ind, ], y = factor(y)[-ind], ...)
        posterior <- predict(fit, x[ind, ], type = "prob")
        if (refit) {
          fit <- nonparametric_naive_bayes(x = x, y = factor(y), ...)
          pik <- as.numeric(table(y)/length(y))
        }
      } else if (classifier == "logistic-l1") {
        D <- data.frame(x = x, y = y)
        x.model <- model.matrix(y~.-1, D)
        fit <- cv.glmnet(x = as.matrix(x.model)[-ind, ], y = y[-ind], family = "multinomial", nfolds = 5)
        if (K> 2) {
          posterior <- as.matrix(predict(fit, as.matrix(x.model[ind, ]), type = "response")[ , , 1])
        } else { # this case may need further check!!!
          pt1 <- as.matrix(predict(fit, newdata = as.matrix(x.model[ind, ]), type = "response")[ , , 1])
          posterior <- cbind(1-pt1, pt1)
        }
        if (refit) {
          fit <- cv.glmnet(x = as.matrix(x.model), y = y, family = "multinomial", nfolds = 5)
          pik <- as.numeric(table(y)/length(y))
        }
      }
    } else { # if the model has been fitted and input --- needs to be a list where the 1st element is the model fitted on training data, the 2nd element is the model fitted on the whole dataset, and the 3rd element is the indices of training samples
      pik <- as.numeric(table(y[-ind])/length(y[-ind]))
      ind <- fitted.model[[3]]
      if (classifier == "logistic") {
        fit <- fitted.model[[1]]
        if (K> 2) {
          posterior <- predict(fit, newdata = data.frame(x = x[ind, ], y = y[ind]), type = "prob")
        } else {
          pt1 <- predict(fit, newdata = data.frame(x = x[ind, ], y = y[ind]), type = "prob")
          posterior <- cbind(1-pt1, pt1)
        }
        if (refit) {
          fit <- fitted.model[[2]]
          pik <- as.numeric(table(y)/length(y))
        }
      } else if (classifier == "knn") {
        fit <- fitted.model[[1]]
        posterior <- predict(fit, newdata = x[ind, ], type = "prob")
        if (refit) {
          fit <- fitted.model[[2]]
          pik <- as.numeric(table(y)/length(y))
        }
      } else if (classifier == "randomforest") {
        fit <- fitted.model[[1]]
        posterior <- predict(fit, newdata = x[ind, ], type = "prob")
        if (refit) {
          fit <- fitted.model[[2]]
          pik <- as.numeric(table(y)/length(y))
        }
      } else if (classifier == "svm") {
        fit <- fitted.model[[1]]
        posterior <- attr(predict(fit, newdata = x[ind, ], probability = TRUE), "probabilities")
        posterior <- posterior[, match(1:K, as.numeric(colnames(posterior)))]
        if (refit) {
          fit <- fitted.model[[2]]
          pik <- as.numeric(table(y)/length(y))
        }
      } else if (classifier == "nb") {
        fit <- fitted.model[[1]]
        posterior <- predict(fit, newdata = x[ind, ], type = "raw")
        if (refit) {
          fit <- fitted.model[[2]]
          pik <- as.numeric(table(y)/length(y))
        }
      } else if (classifier == "tree") {
        fit <- fitted.model[[1]]
        posterior <- predict(fit, newdata = data.frame(x = x[ind, ]), type = "prob")
        if (refit) {
          fit <- fitted.model[[2]]
          pik <- as.numeric(table(y)/length(y))
        }
      } else if (classifier == "neuralnet") {
        fit <- fitted.model[[1]]
        if (K> 2) {
          posterior <- predict(fit, newdata = data.frame(x = x[ind, ]), type = "raw")
        } else {
          pt1 <- predict(fit, newdata = data.frame(x = x[ind, ]), type = "raw")
          posterior <- cbind(1-pt1, pt1)
        }
        if (refit) {
          fit <- fitted.model[[2]]
          pik <- as.numeric(table(y)/length(y))
        }
      } else if (classifier == "lda") {
        fit <- fitted.model[[1]]
        posterior <- predict(fit, x[ind, ])$posterior
        if (refit) {
          fit <- fitted.model[[2]]
          pik <- as.numeric(table(y)/length(y))
        }
      } else if (classifier == "qda") {
        fit <- fitted.model[[1]]
        posterior <- predict(fit, x[ind, ])$posterior
        if (refit) {
          fit <- fitted.model[[2]]
          pik <- as.numeric(table(y)/length(y))
        }
      } else if (classifier == "nnb") {
        if (is.null(colnames(x))) {
          colnames(x) <- paste0("x", 1:ncol(x))
        }
        fit <- fitted.model[[1]]
        posterior <- predict(fit, x[ind, ], type = "prob")
        if (refit) {
          fit <- fitted.model[[2]]
          pik <- as.numeric(table(y)/length(y))
        }
      } else if (classifier == "logistic-l1") {
        D <- data.frame(x = x, y = y)
        x.model <- model.matrix(y~.-1, D)
        fit <- fitted.model[[1]]
        if (K> 2) {
          posterior <- as.matrix(predict(fit, as.matrix(x.model[ind, ]), type = "response")[ , , 1])
        } else { # this case may need further check!!!
          pt1 <- as.matrix(predict(fit, newdata = as.matrix(x.model[ind, ]), type = "response")[ , , 1])
          posterior <- cbind(1-pt1, pt1)
        }
        if (refit) {
          fit <- fitted.model[[1]]
          pik <- as.numeric(table(y)/length(y))
        }
      }
    }


    if (nrow(index) == 1) {
      if (opt.alg == "Nelder-Mead"){
        lambda <- optimize(f = obj.ER.confusion, lower = 0, maximum = T, upper = limit, alpha = alpha, pik = pik, posterior = posterior, w = w, index = index, y = y[ind], tol = tol)
      } else if (opt.alg == "Hooke-Jeeves") {
        lambda <- hjkb1(par = rep(0, nrow(index)), fn = obj.ER.confusion, upper = rep(limit, nrow(index)), lower = rep(0, nrow(index)), control = c(maximize = TRUE, tol = tol), alpha = alpha, pik = pik, posterior = posterior, w = w, index = index, y = y[ind])
      }
    } else if (nrow(index) > 1) {
      if (opt.alg == "Nelder-Mead"){
        lambda <- nmkb(par = rep(0.0001, nrow(index)), fn = obj.ER.confusion, upper = rep(limit, nrow(index)), lower = rep(0, nrow(index)), control = c(maximize = TRUE, tol = tol), alpha = alpha, pik = pik, posterior = posterior, w = w, index = index, y = y[ind])
      } else if (opt.alg == "Hooke-Jeeves") {
        lambda <- hjkb(par = rep(0, nrow(index)), fn = obj.ER.confusion, upper = rep(limit, nrow(index)), lower = rep(0, nrow(index)), control = c(maximize = TRUE, tol = tol), alpha = alpha, pik = pik, posterior = posterior, w = w, index = index, y = y[ind])
      }
    }
  }

  if (nrow(index) == 1 && opt.alg == "Nelder-Mead") {
    obj.value <- lambda$objective
    lambda <- lambda$maximum
  } else {
    obj.value <- lambda$value
    lambda <- lambda$par
  }

  if (obj.value > inf.threshold) {
    warning("The NP problem is infeasible!")
    # stop("The NP problem is infeasible!")
  }

  if (algorithm == "CX") {
    D1.index <- NA
  } else {
    D1.index <- ind
  }

  if (protect) {
    if (is.numeric(protect)) {
      lambda[lambda <= protect] <- protect
    } else {
      lambda[lambda <= 1e-3] <- 1e-3
    }
  }

  lambda.full <- matrix(NA, nrow = nrow(alpha), ncol = ncol(alpha))
  lambda.full[index] <- lambda

  L <- list(lambda = lambda.full*sum(w.ori), fit = fit, classifier = classifier, algorithm = algorithm, alpha = alpha, w = w.ori, pik = pik,
            obj.value = obj.value*sum(w.ori), D1.index = D1.index)
  class(L) <- "npcs_confusion"


  return(L)
}
