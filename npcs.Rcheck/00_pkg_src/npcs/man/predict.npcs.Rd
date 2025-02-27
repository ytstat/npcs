% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/predict.npcs.R
\name{predict.npcs}
\alias{predict.npcs}
\title{Predict new labels from new data based on the fitted NPMC classifier.}
\usage{
\method{predict}{npcs}(object, newx, ...)
}
\arguments{
\item{object}{the fitted NPMC classifier from function \code{\link{npcs}}, which is an object of S3 class "npcs".}

\item{newx}{the new observations. Should be a matrix or a data frame, where each row and column represents an observation and predictor, respectively.}

\item{...}{additional arguments.}
}
\value{
the predicted labels.
}
\description{
Predict new labels from new data based on the fitted NPMC classifier, which belongs to S3 class "npcs".
}
\references{
Tian, Y., & Feng, Y. (2021). Neyman-Pearson Multi-class Classification via Cost-sensitive Learning. Submitted. Available soon on arXiv.
}
\seealso{
\code{\link{npcs}}, \code{\link{error_rate}}, \code{\link{generate_data}}, \code{\link{gamma_smote}}.
}
