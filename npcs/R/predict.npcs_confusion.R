#' Predict new labels from new data based on the fitted NPMC classifier.
#'
#' Predict new labels from new data based on the fitted NPMC classifier, which belongs to S3 class "npcs_confusion".
#' @export
#' @param object the fitted NPMC classifier from function \code{\link{npcs}}, which is an object of S3 class "npcs".
#' @param newx the new observations. Should be a matrix or a data frame, where each row and column represents an observation and predictor, respectively.
#' @param ... additional arguments.
#' @return the predicted labels.
#' @seealso \code{\link{npcs}}, \code{\link{error_rate}}, \code{\link{generate_data}}, \code{\link{gamma_smote}}.
#' @references
#' Tian, Y., & Feng, Y. (2021). Neyman-Pearson Multi-class Classification via Cost-sensitive Learning. Submitted. Available soon on arXiv.
#'
predict.npcs_confusion <- function(object, newx, ...) {
  if (object$classifier == "logistic") {
    if (length(object$pik) > 2) {
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
  } else if (object$classifier == "logistic-l1") {
    D <- data.frame(x = newx, y = 1)
    x.model.new <- model.matrix(y~.-1, D)
    posterior.test <- as.matrix(predict(object$fit, as.matrix(x.model.new), type = "response")[ , , 1])
  }
  object$lambda[is.na(object$lambda)] <- 0

  ckr <- (object$lambda + object$w)/object$pik

  cost_posterior.test <- sapply(1:nrow(object$w), function(r){
    posterior_k_matrix <- t(posterior.test)*ckr[, r]
    colSums(posterior_k_matrix)
  })


  pred.test <- as.numeric(apply(cost_posterior.test, 1, which.min))
  return(pred.test)
}
