# Reference values for the Bayesian part of the checkmodel test suite.
#
# Run from the repository root (needs rstanarm, loo, posterior, bayestestR):
#
#     Rscript tests/r_reference/generate_bayes_reference.R
#
# MCMC draws differ between Stan and PyMC, so the comparison is made on the
# *same* draws: the models are fitted once here, their posterior draws are
# written to tests/data/bayes_<model>.csv.gz, and every statistic R computes
# from them goes to tests/r_reference/bayes_reference.json. The Python tests
# rebuild an ArviZ DataTree from those draws and must reproduce the numbers.

suppressPackageStartupMessages({
  library(rstanarm)
  library(performance)
  library(loo)
  library(posterior)
  library(jsonlite)
})

out_dir <- file.path("tests", "r_reference")
data_dir <- file.path("tests", "data")

hash_u <- function(i) {
  v <- sin(i * 12.9898) * 43758.5453
  v - floor(v)
}
hash_e <- function(i) qnorm(pmin(pmax(hash_u(i), 1e-6), 1 - 1e-6))

mixed_small <- local({
  i <- 1:60
  g <- (i - 1) %% 10 + 1
  x1 <- hash_e(i + 13000)
  data.frame(g = factor(g), x1 = x1, y = 1 + 0.5 * x1 + 1.5 * hash_e(g + 13500) + hash_e(i + 14000))
})

fit_args <- list(seed = 20261021, chains = 2, iter = 600, refresh = 0)
models <- list(
  bayes_lm = do.call(stan_glm, c(list(mpg ~ wt + hp, data = mtcars), fit_args)),
  bayes_logit = do.call(
    stan_glm, c(list(vs ~ mpg + wt, data = mtcars, family = binomial()), fit_args)
  ),
  bayes_mixed = do.call(stan_lmer, c(list(y ~ x1 + (1 | g), data = mixed_small), fit_args))
)

num <- function(x) unname(as.numeric(x))

hdi <- function(x, ci = 0.95) {
  h <- bayestestR::hdi(x, ci = ci)
  c(h$CI_low, h$CI_high)
}

summarise_r2 <- function(draws) {
  list(median = median(draws), mad = mad(draws), hdi = hdi(draws), draws_mean = mean(draws))
}

describe <- function(name, m) {
  mixed <- inherits(m, "lmerMod")
  y <- num(insight::get_response(m))
  if (is.factor(insight::get_response(m))) y <- as.numeric(y) - 1
  params <- as.array(m) # iterations x chains x parameters
  draws <- as.matrix(m)
  chain <- rep(seq_len(dim(params)[2]), each = dim(params)[1])
  draw <- rep(seq_len(dim(params)[1]), times = dim(params)[2])
  # as.matrix() stacks chain 1 then chain 2, matching `chain` and `draw` above.
  mu <- posterior_epred(m)
  set.seed(1)
  yrep <- posterior_predict(m)
  ll <- log_lik(m)
  colnames(mu) <- paste0("mu_", seq_len(ncol(mu)))
  colnames(yrep) <- paste0("yrep_", seq_len(ncol(yrep)))
  colnames(ll) <- paste0("ll_", seq_len(ncol(ll)))
  par_cols <- draws[, !grepl("^(b\\[|Sigma\\[)", colnames(draws)), drop = FALSE]
  colnames(par_cols) <- make.names(colnames(par_cols))
  if (mixed) {
    # Only what the R2 needs, to keep the file small.
    mum <- posterior_epred(m, re.form = NA)
    colnames(mum) <- paste0("mum_", seq_len(ncol(mum)))
    table <- data.frame(chain = chain, draw = draw, par_cols, mu, mum, check.names = FALSE)
  } else {
    table <- data.frame(chain = chain, draw = draw, par_cols, mu, yrep, ll, check.names = FALSE)
  }
  path <- file.path(data_dir, paste0(name, ".csv.gz"))
  con <- gzfile(path, "w")
  write.csv(signif(as.matrix(table), 10), con, row.names = FALSE)
  close(con)
  # Diagnostics and LOO are rank- and tail-sensitive, so compute them from the
  # rounded draws the Python tests will read, not from Stan's full precision.
  saved <- read.csv(gzfile(path))
  as_chains <- function(v) matrix(v, ncol = length(unique(saved$chain)))

  r2 <- r2_bayes(m)
  r2_draws <- r2_posterior(m)
  ll_saved <- as.matrix(saved[, grep("^ll_", names(saved)), drop = FALSE])
  if (ncol(ll_saved) == 0) ll_saved <- ll # mixed model: log-lik not exported
  r_eff <- relative_eff(exp(ll_saved), chain_id = saved$chain)
  loo_res <- loo(ll_saved) # r_eff = 1, which is what loo(<stanreg>) uses
  loo_reff <- loo(ll_saved, r_eff = r_eff)
  waic_res <- suppressWarnings(waic(ll_saved))
  diag <- lapply(colnames(par_cols), function(p) {
    arr <- as_chains(saved[[p]])
    list(rhat = rhat(arr), ess_bulk = ess_bulk(arr), ess_tail = ess_tail(arr))
  })
  names(diag) <- colnames(par_cols)
  ref <- list(
    family = family(m)$family,
    mixed = mixed,
    y = y,
    n_draws = nrow(draws),
    r2 = summarise_r2(r2_draws$R2_Bayes),
    r2_performance = list(R2 = num(r2$R2_Bayes), SE = num(attr(r2, "SE")$R2_Bayes),
                          CI = num(unlist(attr(r2, "CI")$R2_Bayes[c("CI_low", "CI_high")]))),
    loo = as.list(setNames(num(loo_res$estimates), c(
      "elpd_loo", "p_loo", "looic", "elpd_loo_se", "p_loo_se", "looic_se"
    ))),
    pareto_k = num(loo_res$diagnostics$pareto_k),
    loo_reff = as.list(setNames(num(loo_reff$estimates), c(
      "elpd_loo", "p_loo", "looic", "elpd_loo_se", "p_loo_se", "looic_se"
    ))),
    pareto_k_reff = num(loo_reff$diagnostics$pareto_k),
    r_eff = num(r_eff),
    waic = as.list(setNames(num(waic_res$estimates), c(
      "elpd_waic", "p_waic", "waic", "elpd_waic_se", "p_waic_se", "waic_se"
    ))),
    diagnostics = diag,
    sigma_median = if ("sigma" %in% colnames(draws)) median(draws[, "sigma"]) else NULL,
    rmse_median_mu = sqrt(mean((y - apply(mu, 2, median))^2)),
    performance = as.list(as.data.frame(suppressWarnings(model_performance(m))))
  )
  if (mixed) ref$r2_marginal <- summarise_r2(r2_draws$R2_Bayes_marginal)
  ref
}

reference <- list(
  meta = list(
    R = R.version.string,
    rstanarm = as.character(packageVersion("rstanarm")),
    performance = as.character(packageVersion("performance")),
    loo = as.character(packageVersion("loo")),
    posterior = as.character(packageVersion("posterior")),
    generated = format(Sys.Date())
  ),
  synthetic_checksums = list(mixed_small = as.list(colSums(within(mixed_small, g <- as.numeric(g))))),
  models = Map(describe, names(models), models)
)

write_json(reference, file.path(out_dir, "bayes_reference.json"),
           auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
cat("Wrote", file.path(out_dir, "bayes_reference.json"), "\n")
