#' @export
duality_check <- function(fit, x, y, delta = 0.1, R.G = 10, pattern = c("multiplicative", "additive")) {
  pattern <- match.arg(pattern)
  if (fit$algorithm == "CX") {
    y.pred <- predict(fit, x)
    er <- error_rate(y.pred, y)
  } else if (fit$algorithm %in% c("ER", "CX-ER")) {
    y.pred <- predict(fit, x[fit$D1.index,])
    er <- error_rate(y.pred, y[fit$D1.index])
  }

  if (pattern == "multiplicative") {
    feasibility_vec <- (er <= alpha*(1+delta))
  } else if (pattern == "additive") {
    feasibility_vec <- (er <= alpha + delta)
  }

  # if (all(feasibility_vec[!is.na(feasibility_vec)]) && fit$obj.value <= 1+delta) { #
  #   cat("Strong duality holds, the NPMC problem is feasible\n")
  #   s <- 1
  #   f <- 1
  # } else if (fit$obj.value <= 1+delta) {
  #   cat("Strong duality fails, the NPMC problem is feasible\n")
  #   s <- 0
  #   f <- 1
  # } else if (fit$obj.value > R.G) {
  #   cat("Strong duality holds, the NPMC problem is infeasible\n")
  #   s <- 1
  #   f <- 0
  # } else {
  #   cat("Strong duality fails, the NPMC problem is infeasible\n")
  #   s <- 0
  #   f <- 0
  # }

  if (fit$obj.value > R.G) {
    cat("Strong duality holds, the NPMC problem is infeasible\n")
    s <- 1
    f <- 0
  } else if (all(feasibility_vec[!is.na(feasibility_vec)]) && fit$obj.value <= 1+delta) { #
    cat("Strong duality holds, the NPMC problem is feasible\n")
    s <- 1
    f <- 1
  } else if (all(feasibility_vec[!is.na(feasibility_vec)]) || fit$obj.value <= 1+delta) {
    cat("Strong duality fails, the NPMC problem is feasible\n")
    s <- 0
    f <- 1
  } else {
    cat("Strong duality fails, the NPMC problem is infeasible\n")
    s <- 0
    f <- 0
  }

  indicator <- c(s, f)
  names(indicator) <- c("s", "f")
  return(indicator)
}



#' @export
duality_check_confusion <- function(fit, x, y, delta = 0.1, R.G = 10, pattern = c("multiplicative", "additive")) {
  pattern <- match.arg(pattern)
  if (fit$algorithm == "CX") {
    y.pred <- predict(fit, x)
    er <- confusion_matrix(y.pred, y)
  } else if (fit$algorithm == "ER") {
    y.pred <- predict(fit, x[fit$D1.index,])
    er <- confusion_matrix(y.pred, y[fit$D1.index])
  }

  if (pattern == "multiplicative") {
    feasibility_matrix <- (er <= alpha*(1+delta))
  } else if (pattern == "additive") {
    feasibility_matrix <- (er <= alpha + delta)
  }

  # if (all(feasibility_matrix[!is.na(feasibility_matrix)]) && fit$obj.value <= 1+delta) { # && fit$obj.value <= 1+delta
  #   cat("Strong duality holds, the NPMC problem is feasible\n")
  #   s <- 1
  #   f <- 1
  # } else if (fit$obj.value <= 1+delta) {
  #   cat("Strong duality fails, the NPMC problem is feasible\n")
  #   s <- 0
  #   f <- 1
  # } else if (fit$obj.value > R.G) {
  #   cat("Strong duality holds, the NPMC problem is infeasible\n")
  #   s <- 1
  #   f <- 0
  # } else {
  #   cat("Strong duality fails, the NPMC problem is infeasible\n")
  #   s <- 0
  #   f <- 0
  # }

  if (fit$obj.value > R.G) {
    cat("Strong duality holds, the NPMC problem is infeasible\n")
    s <- 1
    f <- 0
  } else if (all(feasibility_matrix[!is.na(feasibility_matrix)]) && fit$obj.value <= 1+delta) { #
    cat("Strong duality holds, the NPMC problem is feasible\n")
    s <- 1
    f <- 1
  } else if (all(feasibility_matrix[!is.na(feasibility_matrix)]) || fit$obj.value <= 1+delta) {
    cat("Strong duality fails, the NPMC problem is feasible\n")
    s <- 0
    f <- 1
  } else {
    cat("Strong duality fails, the NPMC problem is infeasible\n")
    s <- 0
    f <- 0
  }

  indicator <- c(s, f)
  names(indicator) <- c("s", "f")
  return(indicator)
}
