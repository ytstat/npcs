#' Fit a multi-class Neyman-Pearson classifier with error controls via cost-sensitive learning.
#'
#' Fit a multi-class Neyman-Pearson classifier with error controls via cost-sensitive learning. This function implements two algorithms proposed in Tian, Y. & Feng, Y. (2021). The problem is minimize a linear combination of P(hat(Y)(X) != k| Y=k) for some classes k while controlling P(hat(Y)(X) != k| Y=k) for some classes k. See Tian, Y. & Feng, Y. (2021) for more details.
#' @export
#' @importFrom stats predict
#' @importFrom stats runif
#' @importFrom stats optimize
#' @importFrom stats rnorm
#' @importFrom dfoptim nmkb
#' @importFrom dfoptim hjkb
#' @importFrom nnet multinom
#' @importFrom nnet nnet
#' @importFrom magrittr %>%
#' @importFrom caret knn3
#' @importFrom e1071 svm
#' @importFrom e1071 naiveBayes
#' @importFrom randomForest randomForest
#' @importFrom naivebayes nonparametric_naive_bayes
#' @importFrom FNN knnx.index
#' @importFrom MASS lda
#' @importFrom MASS qda
#' @importFrom rpart rpart
#' @importFrom foreach foreach
#' @importFrom foreach %do%
#' @importFrom formatR tidy_eval
#' @importFrom xgboost xgboost
#' @param x the predictor matrix of training data, where each row and column represents an observation and predictor, respectively.
#' @param y the response vector of training data. Must be integers from 1 to K for some K >= 2. Can be either a numerical or factor vector.
#' @param algorithm the NPMC algorithm to use. String only. Can be either "CX" or "ER", which implements NPMC-CX or NPMC-ER in Tian, Y. & Feng, Y. (2021).
#' @param classifier which model to use for estimating the posterior distribution P(Y|X = x). String only.
#' \itemize{
#' \item logistic: multinomial logistic regression, which is implemented via \code{\link[nnet]{multinom}} in package \code{nnet}.
#' \item knn: k-nearest neighbor, which is implemented via \code{\link[caret]{knn3}} in package \code{caret}. An addition parameter (number of nearest neighbors) "k" is needed, which is 5 in default.
#' \item randomforest: random forests, which is implemented via \code{\link[randomForest]{randomForest}} in package \code{randomForest}.
#' \item tree: decition trees, which is implemented via \code{\link[rpart]{rpart}} in package \code{rpart}.
#' \item neuralnet: single-layer neural networks, which is implemented via \code{\link[nnet]{nnet}} in package \code{nnet}.
#' \item svm: support vector machines, which is implemented via \code{\link[e1071]{svm}} in package \code{e1071}.
#' \item lda: linear discriminant analysis, which is implemented via \code{\link[MASS]{lda}} in package \code{MASS}.
#' \item qda: quadratic discriminant analysis, which is implemented via \code{\link[MASS]{qda}} in package \code{MASS}.
#' \item nb: naive Bayes classifier with Gaussian marginals, which is implemented via \code{\link[e1071]{naiveBayes}} in package \code{e1071}.
#' \item nnb: naive Bayes classifier with non-parametric-estimated marginals (kernel-based), which is implemented via \code{\link[naivebayes]{nonparametric_naive_bayes}} in package \code{naivebayes}. The default kernel is the Gaussian kernel. Check the documentation of function \code{\link[naivebayes]{nonparametric_naive_bayes}} to see how to change the estimation settings.
#' }
#' @param w the weights in objective function. Should be a vector of length K, where K is the number of classes.
#' @param alpha the levels we want to control for error rates of each class. Should be a vector of length K, where K is the number of classes. Use NA if if no error control is imposed for specific classes.
#' @param split.ratio the proportion of data to be used in searching lambda (cost parameters). Should be between 0 and 1. Default = 0.5. Only useful when \code{algorithm} = "ER".
#' @param split.mode two different modes to split the data for NPMC-ER. String only. Can be either "per-class" or "merged". Default = "per-class". Only useful when \code{algorithm} = "ER".
#' \itemize{
#' \item per-class: split the data by class.
#' \item merged: split the data as a whole.
#' }
#' @param tol the convergence tolerance. Default = 1e-06. Used in the lambda-searching step. The optimization is terminated when the step length of the main loop becomes smaller than \code{tol}. See pages of \code{\link[dfoptim]{hjkb}} and \code{\link[dfoptim]{nmkb}} for more details.
#' @param refit whether to refit the classifier using all data after finding lambda or not. Boolean value. Default = TRUE. Only useful when \code{algorithm} = "ER".
#' @param protect whether to threshold the close-zero lambda or not. Boolean value. Default = TRUE. This parameter is set to avoid extreme cases that some lambdas are set equal to zero due to computation accuracy limit. When \code{protect} = TRUE, all lambdas smaller than 1e-03 will be set equal to 1e-03.
#' @param opt.alg optimization method to use when searching lambdas. String only. Can be either "Hooke-Jeeves" or "Nelder-Mead". Default = "Hooke-Jeeves".
#' @param ... additional arguments. Will be passed to the function which fits the model indicated in \code{classifier}. For example, when \code{classifier} = "knn", the number of nearest neightbors \code{k} should be inputed. When \code{classifier} = "neuralnets"
#' @return An object with S3 class \code{"npcs"}.
#' \item{lambda}{the estimated lambda vector, which consists of Lagrangian multipliers. It is related to the cost. See Section 2 of Tian, Y. & Feng, Y. (2021) for details.}
#' \item{fit}{the fitted classifier.}
#' \item{classifier}{which classifier to use for estimating the posterior distribution P(Y|X = x).}
#' \item{algorithm}{the NPMC algorithm to use.}
#' \item{alpha}{the levels we want to control for error rates of each class.}
#' \item{w}{the weights in objective function.}
#' \item{pik}{the estimated marginal probability for each class.}
#' @seealso \code{\link{predict.npcs}}, \code{\link{error_rate}}, \code{\link{generate_data}}, \code{\link{gamma_smote}}.
#' @references
#' Tian, Y., & Feng, Y. (2021). Neyman-Pearson Multi-class Classification via Cost-sensitive Learning. Submitted. Available soon on arXiv.
#'
#' @examples
#' # data generation: case 1 in Tian, Y., & Feng, Y. (2021) with n = 1000
#' set.seed(123, kind = "L'Ecuyer-CMRG")
#' train.set <- generate_data(n = 1000, model.no = 1)
#' x <- train.set$x
#' y <- train.set$y
#'
#' test.set <- generate_data(n = 1000, model.no = 1)
#' x.test <- test.set$x
#' y.test <- test.set$y
#'
#' # contruct the multi-class NP problem: case 1 in Tian, Y., & Feng, Y. (2021)
#' alpha <- c(0.05, NA, 0.01)
#' w <- c(0, 1, 0)
#'
#' # try NPMC-CX, NPMC-ER, and vanilla multinomial logistic regression
#' fit.npmc.CX <- try(npcs(x, y, algorithm = "CX", classifier = "logistic", w = w, alpha = alpha))
#' fit.npmc.ER <- try(npcs(x, y, algorithm = "ER", classifier = "logistic", w = w, alpha = alpha,
#' refit = TRUE))
#' fit.vanilla <- nnet::multinom(y~., data = data.frame(x = x, y = factor(y)), trace = FALSE)
#'
#' # test error of NPMC-CX
#' y.pred.CX <- predict(fit.npmc.CX, x.test)
#' error_rate(y.pred.CX, y.test)
#'
#' # test error of NPMC-ER
#' y.pred.ER <- predict(fit.npmc.ER, x.test)
#' error_rate(y.pred.ER, y.test)
#'
#' # test error of vanilla multinomial logistic regression
#' y.pred.vanilla <- predict(fit.vanilla, newdata = data.frame(x = x.test))
#' error_rate(y.pred.vanilla, y.test)



npcs <- function(x, y, algorithm = c("CX", "ER", "CX-ER"), classifier = c("logistic", "knn", "randomforest", "tree", "neuralnet", "svm", "lda", "qda", "nb", "nnb", "logistic-l1", "xgboost", "logistic-l1-contrast"),
                 w, alpha, split.ratio = 0.5, split.mode = c("by-class", "merged"), tol = 1e-6, refit = TRUE, protect = TRUE, opt.alg = c("Hooke-Jeeves", "Nelder-Mead", "multi-Hooke-Jeeves", "BFGS", "CG", "L-BFGS-B"), limit = 1000, fitted.model = NULL,
                 inf.threshold = 1, delta = NULL, B = 20, ...) {

  algorithm <- match.arg(algorithm)
  classifier <- match.arg(classifier)
  split.mode <- match.arg(split.mode)
  opt.alg <- match.arg(opt.alg)
  n <- length(y)
  K <- length(unique(y))
  w.ori <- w
  w <- w/sum(w)
  index <- which(!is.na(alpha))

  # if (classifier %in% c("logistic", "neuralnet", "qda", "nb")) { # parametric classifier: delta = 0.1
  #   delta <- 0.1
  # } else { # non-parametric classifier: delta = 0.2
  #   delta <- 0.2
  # }

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
        posterior <- as.matrix(predict(fit, as.matrix(x.model), type = "response")[ , , 1])
      } else if (classifier == "xgboost") {
        D <- data.frame(x = x, y = y)
        x.model <- model.matrix(y~.-1, D)
        fit <- xgboost(data = x.model, label = as.numeric(D$y)-1, objective = "multi:softprob", nrounds = 50, num_class = K, verbose = 0, ...)
        posterior <- predict(fit, x.model, reshape = TRUE)
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
      } else if (classifier == "logistic-l1-contrast") {
        D <- data.frame(x = x, y = y)
        x.model <- model.matrix(y~.-1, D)
        posterior <- as.matrix(predict_fit(fit$beta.min, as.matrix(x.model), type = "prob"))
      } else if (classifier == "logistic-l1") {
        D <- data.frame(x = x, y = y)
        x.model <- model.matrix(y~.-1, D)
        posterior <- as.matrix(predict(fit, as.matrix(x.model), type = "response")[ , , 1])
      } else if (classifier == "xgboost") {
        D <- data.frame(x = x, y = y)
        x.model <- model.matrix(y~.-1, D)
        posterior <- predict(fit, x.model, reshape = TRUE)
      }
    }


    if (length(index) == 1) {
      if (opt.alg == "Nelder-Mead"){
        lambda <- optimize(f = obj.CX, lower = 0, maximum = T, upper = limit, alpha = alpha, pik = pik, posterior = posterior, w = w, index = index, tol = tol)
        # lambda <- nmkb1(par = rep(0.0001, length(index)), fn = obj.CX, upper = rep(200, length(index)), lower = rep(0, length(index)), control = c(maximize = TRUE, tol = tol), alpha = alpha, pik = pik, posterior = posterior, w = w, index = index)
      } else if (opt.alg == "Hooke-Jeeves") {
        lambda <- hjkb1(par = rep(0, length(index)), fn = obj.CX, upper = rep(limit, length(index)), lower = rep(0, length(index)), control = c(maximize = TRUE, tol = tol), alpha = alpha, pik = pik, posterior = posterior, w = w, index = index)
      }
    } else if (length(index) > 1) {
      if (opt.alg == "Nelder-Mead"){
        lambda <- nmkb(par = rep(0.0001, length(index)), fn = obj.CX, upper = rep(limit, length(index)), lower = rep(0, length(index)), control = c(maximize = TRUE, tol = tol), alpha = alpha, pik = pik, posterior = posterior, w = w, index = index)
      } else if (opt.alg == "Hooke-Jeeves") {
        lambda <- hjkb(par = rep(0, length(index)), fn = obj.CX, upper = rep(limit, length(index)), lower = rep(0, length(index)), control = c(maximize = TRUE, tol = tol), alpha = alpha, pik = pik, posterior = posterior, w = w, index = index)
      }
    }

  }


  if (algorithm %in% c("ER", "CX-ER")) {
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
      } else if (classifier == "xgboost") {
        D <- data.frame(x = x, y = y)
        x.model <- model.matrix(y~.-1, D)
        fit <- xgboost(data = x.model[-ind, ], label = (as.numeric(D$y)-1)[-ind], objective = "multi:softprob", nrounds = 50, num_class = K, verbose = 0, ...)
        if (K> 2) {
          posterior <- predict(fit, x.model[ind, ], reshape = TRUE)
        } else { # this case may need further check!!!
          # no idea --- needs to figure out
        }
        if (refit) {
          fit <- xgboost(data = x.model, label = as.numeric(D$y)-1, objective = "multi:softprob", nrounds = 50, num_class = K, verbose = 0, ...)
          pik <- as.numeric(table(y)/length(y))
        }
      }
    } else { # if the model has been fitted and input --- needs to be a list where the 1st element is the model fitted on training data, the 2nd element is the model fitted on the whole dataset, and the 3rd element is the indices of test samples
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
          fit <- fitted.model[[2]]
          pik <- as.numeric(table(y)/length(y))
        }
      } else if (classifier == "xgboost") {
        D <- data.frame(x = x, y = y)
        x.model <- model.matrix(y~.-1, D)
        fit <- fitted.model[[1]]
        if (K> 2) {
          posterior <- predict(fit, x.model[ind, ], reshape = TRUE)
        } else { # this case may need further check!!!
          # no idea --- needs to figure out
        }
        if (refit) {
          fit <- fitted.model[[2]]
          pik <- as.numeric(table(y)/length(y))
        }
      }
    }

    if (algorithm == "ER") {
      if (length(index) == 1) {
        if (opt.alg == "Nelder-Mead"){
          lambda <- optimize(f = obj.ER, lower = 0, maximum = T, upper = limit, alpha = alpha, pik = pik, posterior = posterior, w = w, index = index, y = y[ind], tol = tol)
        } else if (opt.alg == "Hooke-Jeeves") {
          lambda <- hjkb1(par = rep(0, length(index)), fn = obj.ER, upper = rep(limit, length(index)), lower = rep(0, length(index)), control = c(maximize = TRUE, tol = tol), alpha = alpha, pik = pik, posterior = posterior, w = w, index = index, y = y[ind])
        }
      } else if (length(index) > 1) {
        if (opt.alg == "Nelder-Mead"){
          lambda <- nmkb(par = rep(0.001, length(index)), fn = obj.ER, upper = rep(limit, length(index)), lower = rep(0, length(index)), control = c(maximize = TRUE, tol = tol), alpha = alpha, pik = pik, posterior = posterior, w = w, index = index, y = y[ind])
          # lambda <- nmkb(par = rep(0.0001, length(index)), fn = obj.ER, upper = rep(limit, length(index)), lower = rep(0, length(index)), control = c(maximize = TRUE, tol = tol), alpha = alpha, pik = pik, posterior = posterior, w = w, index = index, y = y[ind])
        } else if (opt.alg == "Hooke-Jeeves") {
          lambda <- hjkb(par = rep(0, length(index)), fn = obj.ER, upper = rep(limit, length(index)), lower = rep(0, length(index)), control = c(maximize = TRUE, tol = tol), alpha = alpha, pik = pik, posterior = posterior, w = w, index = index, y = y[ind])
        } else if (opt.alg == "multi-Hooke-Jeeves") {
          initial_value_list <- matrix(runif(length(index)*B, min = 0, max = 1), nrow = B)
          obj_value_list <- sapply(1:nrow(initial_value_list), function(i){
            lambda <- hjkb(par = initial_value_list[i, ], fn = obj.ER, upper = rep(limit, length(index)), lower = rep(0, length(index)), control = c(maximize = TRUE, tol = tol), alpha = alpha, pik = pik, posterior = posterior, w = w, index = index, y = y[ind])
            lambda$value
          })
          lambda <- hjkb(par = initial_value_list[which.max(obj_value_list), ], fn = obj.ER, upper = rep(limit, length(index)), lower = rep(0, length(index)), control = c(maximize = TRUE, tol = tol), alpha = alpha, pik = pik, posterior = posterior, w = w, index = index, y = y[ind])
        } else if (opt.alg == "BFGS") {
          lambda <- optim(par = rep(0.001, length(index)), fn = obj.ER.min,  method = "BFGS", alpha = alpha, pik = pik, posterior = posterior, w = w, index = index, y = y[ind])
        } else if (opt.alg == "CG") {
          lambda <- optim(par = rep(0.001, length(index)), fn = obj.ER.min,  method = "CG", alpha = alpha, pik = pik, posterior = posterior, w = w, index = index, y = y[ind])
        } else if (opt.alg == "L-BFGS-B") {
          lambda <- optim(par = rep(0.001, length(index)), fn = obj.ER.min, upper = rep(limit, length(index)), lower = rep(0, length(index)), method = "L-BFGS-B", alpha = alpha, pik = pik, posterior = posterior, w = w, index = index, y = y[ind])
        }
      }
    } else if (algorithm == "CX-ER") {
      if (length(index) == 1) {
        if (opt.alg == "Nelder-Mead"){
          lambda <- optimize(f = obj.CX, lower = 0, maximum = T, upper = limit, alpha = alpha, pik = pik, posterior = posterior, w = w, index = index, tol = tol)
        } else if (opt.alg == "Hooke-Jeeves") {
          lambda <- hjkb1(par = rep(0, length(index)), fn = obj.CX, upper = rep(limit, length(index)), lower = rep(0, length(index)), control = c(maximize = TRUE, tol = tol), alpha = alpha, pik = pik, posterior = posterior, w = w, index = index)
        }
      } else if (length(index) > 1) {
        if (opt.alg == "Nelder-Mead"){
          lambda <- nmkb(par = rep(0.001, length(index)), fn = obj.CX, upper = rep(limit, length(index)), lower = rep(0, length(index)), control = c(maximize = TRUE, tol = tol), alpha = alpha, pik = pik, posterior = posterior, w = w, index = index)
          # lambda <- nmkb(par = rep(0.0001, length(index)), fn = obj.ER, upper = rep(limit, length(index)), lower = rep(0, length(index)), control = c(maximize = TRUE, tol = tol), alpha = alpha, pik = pik, posterior = posterior, w = w, index = index, y = y[ind])
        } else if (opt.alg == "Hooke-Jeeves") {
          lambda <- hjkb(par = rep(0, length(index)), fn = obj.CX, upper = rep(limit, length(index)), lower = rep(0, length(index)), control = c(maximize = TRUE, tol = tol), alpha = alpha, pik = pik, posterior = posterior, w = w, index = index)
        } else if (opt.alg == "multi-Hooke-Jeeves") {
          initial_value_list <- matrix(runif(length(index)*B, min = 0, max = 1), nrow = B)
          obj_value_list <- sapply(1:nrow(initial_value_list), function(i){
            lambda <- hjkb(par = initial_value_list[i, ], fn = obj.CX, upper = rep(limit, length(index)), lower = rep(0, length(index)), control = c(maximize = TRUE, tol = tol), alpha = alpha, pik = pik, posterior = posterior, w = w, index = index)
            lambda$value
          })
          lambda <- hjkb(par = initial_value_list[which.max(obj_value_list), ], fn = obj.CX, upper = rep(limit, length(index)), lower = rep(0, length(index)), control = c(maximize = TRUE, tol = tol), alpha = alpha, pik = pik, posterior = posterior, w = w, index = index)
        } else if (opt.alg == "BFGS") {
          lambda <- optim(par = rep(0.001, length(index)), fn = obj.CX.min,  method = "BFGS", alpha = alpha, pik = pik, posterior = posterior, w = w, index = index)
        } else if (opt.alg == "CG") {
          lambda <- optim(par = rep(0.001, length(index)), fn = obj.CX.min,  method = "CG", alpha = alpha, pik = pik, posterior = posterior, w = w, index = index)
        } else if (opt.alg == "L-BFGS-B") {
          lambda <- optim(par = rep(0.001, length(index)), fn = obj.CX.min, upper = rep(limit, length(index)), lower = rep(0, length(index)), method = "L-BFGS-B", alpha = alpha, pik = pik, posterior = posterior, w = w, index = index)
        }
      }
    }

  }

  if (length(index) == 1 && opt.alg == "Nelder-Mead") {
    obj.value <- lambda$objective
    lambda <- lambda$maximum
  } else if (opt.alg %in% c("Hooke-Jeeves", "multi-Hooke-Jeeves")){
    obj.value <- lambda$value
    lambda <- lambda$par
  } else {
    obj.value <- -lambda$value
    lambda <- lambda$par
  }

  if (obj.value > inf.threshold) {
    warning("The NP problem is infeasible!")
    # stop("The NP problem is infeasible!")
  }



  if (protect) {
    if (is.numeric(protect)) {
      lambda[lambda <= protect] <- protect
    } else {
      lambda[lambda <= 1e-3] <- 1e-3
    }
  }

  lambda.full <- rep(NA, length(alpha))
  lambda.full[index] <- lambda

  if (algorithm == "CX") {
    D1.index <- NA
  } else {
    D1.index <- ind
  }

  L <- list(lambda = lambda.full*sum(w.ori), fit = fit, classifier = classifier, algorithm = algorithm, alpha = alpha, w = w.ori, pik = pik,
            obj.value = obj.value*sum(w.ori), D1.index = D1.index)
  class(L) <- "npcs"


  return(L)
}
