pkgname <- "npcs"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
library('npcs')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("error_rate")
### * error_rate

flush(stderr()); flush(stdout())

### Name: error_rate
### Title: Calculate the error rates for each class.
### Aliases: error_rate

### ** Examples

# data generation: case 1 in Tian, Y., & Feng, Y. (2021) with p = 1000
set.seed(123, kind = "L'Ecuyer-CMRG")
train.set <- generate_data(n = 1000, model.no = 1)
x <- train.set$x
y <- train.set$y

test.set <- generate_data(n = 1000, model.no = 1)
x.test <- test.set$x
y.test <- test.set$y

# contruct the multi-class NP problem: case 1 in Tian, Y., & Feng, Y. (2021)
alpha <- c(0.05, NA, 0.01)
w <- c(0, 1, 0)

# try NPMC-CX, NPMC-ER with multinomial logistic regression, and vanilla multinomial
## logistic regression
fit.npmc.CX <- try(npcs(x, y, algorithm = "CX", classifier = "logistic", w = w, alpha = alpha))
fit.npmc.ER <- try(npcs(x, y, algorithm = "ER", classifier = "logistic", w = w, alpha = alpha,
refit = TRUE))
fit.vanilla <- nnet::multinom(y~., data = data.frame(x = x, y = factor(y)), trace = FALSE)

# test error of NPMC-CX
y.pred.CX <- predict(fit.npmc.CX, x.test)
error_rate(y.pred.CX, y.test)

# test error of NPMC-ER
y.pred.ER <- predict(fit.npmc.ER, x.test)
error_rate(y.pred.ER, y.test)

# test error of vanilla multinomial logistic regression
y.pred.vanilla <- predict(fit.vanilla, newdata = data.frame(x = x.test))
error_rate(y.pred.vanilla, y.test)



cleanEx()
nameEx("gamma_smote")
### * gamma_smote

flush(stderr()); flush(stdout())

### Name: gamma_smote
### Title: Gamma-synthetic minority over-sampling technique (gamma-SMOTE).
### Aliases: gamma_smote

### ** Examples

## Not run: 
##D set.seed(123, kind = "L'Ecuyer-CMRG")
##D train.set <- generate_data(n = 200, model.no = 1)
##D x <- train.set$x
##D y <- train.set$y
##D 
##D test.set <- generate_data(n = 1000, model.no = 1)
##D x.test <- test.set$x
##D y.test <- test.set$y
##D 
##D # contruct the multi-class NP problem: case 1 in Tian, Y., & Feng, Y. (2021)
##D alpha <- c(0.05, NA, 0.01)
##D w <- c(0, 1, 0)
##D 
##D ## try NPMC-CX, NPMC-ER based on multinomial logistic regression, and vanilla multinomial
##D ## logistic regression without SMOTE. NPMC-ER outputs the infeasibility error information.
##D fit.npmc.CX <- try(npcs(x, y, algorithm = "CX", classifier = "logistic", w = w, alpha = alpha))
##D fit.npmc.ER <- try(npcs(x, y, algorithm = "ER", classifier = "logistic", w = w, alpha = alpha,
##D refit = TRUE))
##D fit.vanilla <- nnet::multinom(y~., data = data.frame(x = x, y = factor(y)), trace = FALSE)
##D 
##D # test error of NPMC-CX based on multinomial logistic regression without SMOTE
##D y.pred.CX <- predict(fit.npmc.CX, x.test)
##D error_rate(y.pred.CX, y.test)
##D 
##D # test error of vanilla multinomial logistic regression without SMOTE
##D y.pred.vanilla <- predict(fit.vanilla, newdata = data.frame(x = x.test))
##D error_rate(y.pred.vanilla, y.test)
##D 
##D 
##D ## create synthetic data by 0.5-SMOTE
##D D.syn <- gamma_smote(x, y, dup_rate = 1, gamma = 0.5, k = 5)
##D x <- D.syn$x
##D y <- D.syn$y
##D 
##D ## try NPMC-CX, NPMC-ER based on multinomial logistic regression, and vanilla multinomial logistic
##D ## regression with SMOTE. NPMC-ER can successfully find a solution after SMOTE.
##D fit.npmc.CX <- try(npcs(x, y, algorithm = "CX", classifier = "logistic", w = w, alpha = alpha))
##D fit.npmc.ER <- try(npcs(x, y, algorithm = "ER", classifier = "logistic", w = w, alpha = alpha,
##D refit = TRUE))
##D fit.vanilla <- nnet::multinom(y~., data = data.frame(x = x, y = factor(y)), trace = FALSE)
##D 
##D # test error of NPMC-CX based on multinomial logistic regression with SMOTE
##D y.pred.CX <- predict(fit.npmc.CX, x.test)
##D error_rate(y.pred.CX, y.test)
##D 
##D # test error of NPMC-ER based on multinomial logistic regression with SMOTE
##D y.pred.ER <- predict(fit.npmc.ER, x.test)
##D error_rate(y.pred.ER, y.test)
##D 
##D # test error of vanilla multinomial logistic regression wit SMOTE
##D y.pred.vanilla <- predict(fit.vanilla, newdata = data.frame(x = x.test))
##D error_rate(y.pred.vanilla, y.test)
## End(Not run)



cleanEx()
nameEx("generate_data")
### * generate_data

flush(stderr()); flush(stdout())

### Name: generate_data
### Title: Generate the data.
### Aliases: generate_data

### ** Examples

set.seed(123, kind = "L'Ecuyer-CMRG")
train.set <- generate_data(n = 1000, model.no = 1)
x <- train.set$x
y <- train.set$y




cleanEx()
nameEx("npcs")
### * npcs

flush(stderr()); flush(stdout())

### Name: npcs
### Title: Fit a multi-class Neyman-Pearson classifier with error controls
###   via cost-sensitive learning.
### Aliases: npcs

### ** Examples

# data generation: case 1 in Tian, Y., & Feng, Y. (2021) with n = 1000
set.seed(123, kind = "L'Ecuyer-CMRG")
train.set <- generate_data(n = 1000, model.no = 1)
x <- train.set$x
y <- train.set$y

test.set <- generate_data(n = 1000, model.no = 1)
x.test <- test.set$x
y.test <- test.set$y

# contruct the multi-class NP problem: case 1 in Tian, Y., & Feng, Y. (2021)
alpha <- c(0.05, NA, 0.01)
w <- c(0, 1, 0)

# try NPMC-CX, NPMC-ER, and vanilla multinomial logistic regression
fit.npmc.CX <- try(npcs(x, y, algorithm = "CX", classifier = "logistic", w = w, alpha = alpha))
fit.npmc.ER <- try(npcs(x, y, algorithm = "ER", classifier = "logistic", w = w, alpha = alpha,
refit = TRUE))
fit.vanilla <- nnet::multinom(y~., data = data.frame(x = x, y = factor(y)), trace = FALSE)

# test error of NPMC-CX
y.pred.CX <- predict(fit.npmc.CX, x.test)
error_rate(y.pred.CX, y.test)

# test error of NPMC-ER
y.pred.ER <- predict(fit.npmc.ER, x.test)
error_rate(y.pred.ER, y.test)

# test error of vanilla multinomial logistic regression
y.pred.vanilla <- predict(fit.vanilla, newdata = data.frame(x = x.test))
error_rate(y.pred.vanilla, y.test)



### * <FOOTER>
###
cleanEx()
options(digits = 7L)
base::cat("Time elapsed: ", proc.time() - base::get("ptime", pos = 'CheckExEnv'),"\n")
grDevices::dev.off()
###
### Local variables: ***
### mode: outline-minor ***
### outline-regexp: "\\(> \\)?### [*]+" ***
### End: ***
quit('no')
