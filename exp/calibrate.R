suppressWarnings(suppressMessages(library(L0Learn)))

fit_path_l0learn <- function(x, A, y) {
  c <- L0Learn.cvfit(A, y, loss="SquaredError", penalty="L0", intercept=FALSE)
  r <- c$fit$suppSize[[1]] != sum(x != 0)
  c$cvMeans[[1]][r] = Inf         # remove solutions with the wrong support size
  i <- which.min(c$cvMeans[[1]])  # index with the least CV error
  x <- c$fit$beta[[1]][,i]        # solution with the least CV error
  l <- c$fit$lambda[[1]][i]       # lambda with the least CV error
  return(list(x=x, l=l))
}
