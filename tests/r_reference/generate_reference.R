# Reference values for the checkmodel test suite, computed with R.
#
# Run from the repository root:
#
#     Rscript tests/r_reference/generate_reference.R
#
# It writes the datasets the tests use to tests/data/ and every statistic the
# tests compare against to tests/r_reference/reference.json, so the Python test
# suite never needs R. Re-run it after changing a model or a dataset here, and
# keep tests/synthetic.py in sync with the synthetic data section below.

suppressPackageStartupMessages({
  library(performance)
  library(lmtest)
  library(jsonlite)
  library(lme4)
  library(MASS)
})

out_dir <- file.path("tests", "r_reference")
data_dir <- file.path("tests", "data")
dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------- datasets
write.csv(mtcars, file.path(data_dir, "mtcars.csv"))
write.csv(iris, file.path(data_dir, "iris.csv"), row.names = FALSE)
write.csv(warpbreaks, file.path(data_dir, "warpbreaks.csv"), row.names = FALSE)
write.csv(sleepstudy, file.path(data_dir, "sleepstudy.csv"), row.names = FALSE)
data("bioChemists", package = "pscl")
write.csv(bioChemists, file.path(data_dir, "biochemists.csv"), row.names = FALSE)

# Deterministic pseudo-random data, reproduced bit-for-bit-ish in tests/synthetic.py.
hash_u <- function(i) {
  v <- sin(i * 12.9898) * 43758.5453
  v - floor(v)
}
hash_e <- function(i) qnorm(pmin(pmax(hash_u(i), 1e-6), 1 - 1e-6))

synthetic <- list(
  good = local({
    i <- 1:60
    x1 <- i / 6
    x2 <- hash_u(i + 1000) * 10
    data.frame(x1 = x1, x2 = x2, y = 1 + 0.5 * x1 - 0.3 * x2 + hash_e(i))
  }),
  hetero = local({
    i <- 1:200
    x <- i / 20
    data.frame(x = x, y = 1 + 2 * x + (0.2 + x) * hash_e(i))
  }),
  skewed = local({
    i <- 1:200
    x <- i / 20
    data.frame(x = x, y = 1 + 2 * x + exp(1.5 * hash_e(i)))
  }),
  autocorr = local({
    i <- 1:80
    x <- i / 8
    data.frame(x = x, y = 1 + 0.5 * x + 2 * sin(i / 5) + 0.3 * hash_e(i))
  }),
  autocorr_large = local({
    i <- 1:300
    x <- i / 30
    data.frame(x = x, y = 1 + 0.5 * x + 2 * sin(i / 8) + 0.5 * hash_e(i))
  }),
  large = local({
    i <- 1:6000
    x <- hash_u(i + 5000) * 4
    data.frame(x = x, y = 2 + x + hash_e(i))
  }),
  collinear = local({
    i <- 1:100
    x1 <- hash_e(i)
    x2 <- x1 + 0.1 * hash_e(i + 2000)
    x3 <- hash_e(i + 3000)
    data.frame(x1 = x1, x2 = x2, x3 = x3, y = x1 + x2 + x3 + hash_e(i + 4000))
  }),
  logit = local({
    i <- 1:400
    x <- hash_e(i + 6000)
    z <- hash_e(i + 8000)
    p <- plogis(-0.5 + 1.2 * x - 0.6 * z)
    data.frame(x = x, z = z, y = as.numeric(hash_u(i + 7000) < p))
  }),
  pois = local({
    i <- 1:300
    x <- hash_e(i + 12000)
    data.frame(x = x, y = qpois(hash_u(i + 12500), exp(0.3 + 0.5 * x)))
  }),
  mixed = local({
    i <- 1:200
    g <- (i - 1) %% 20 + 1
    x1 <- hash_e(i + 9000)
    x2 <- 0.5 * x1 + hash_e(i + 9500)
    data.frame(g = g, x1 = x1, x2 = x2,
               y = 1 + 0.5 * x1 - 0.3 * x2 + 2 * hash_e(g + 9900) + hash_e(i + 10000))
  }),
  # No between-group variance at all: the noise is centred within each group.
  mixed_singular = local({
    i <- 1:200
    g <- (i - 1) %% 20 + 1
    x1 <- hash_e(i + 9000)
    e <- hash_e(i + 11000)
    data.frame(g = g, x1 = x1, y = 1 + 0.5 * x1 + e - ave(e, g))
  })
)

# ------------------------------------------------------------------ models
tight <- glm.control(epsilon = 1e-14, maxit = 100)
models <- list(
  lm_mtcars = lm(mpg ~ wt + hp, data = mtcars),
  lm_mtcars_cyl = lm(mpg ~ wt + hp + factor(cyl), data = mtcars),
  lm_mtcars_wls = lm(mpg ~ wt + hp, data = mtcars, weights = disp / 100),
  lm_iris = lm(Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width, data = iris),
  lm_good = lm(y ~ x1 + x2, data = synthetic$good),
  lm_hetero = lm(y ~ x, data = synthetic$hetero),
  lm_skewed = lm(y ~ x, data = synthetic$skewed),
  lm_autocorr = lm(y ~ x, data = synthetic$autocorr),
  lm_autocorr_large = lm(y ~ x, data = synthetic$autocorr_large),
  lm_large = lm(y ~ x, data = synthetic$large),
  lm_collinear = lm(y ~ x1 + x2 + x3, data = synthetic$collinear),
  glm_gauss = glm(mpg ~ wt + hp, family = gaussian, data = mtcars),
  glm_logit = glm(vs ~ mpg + wt, family = binomial, data = mtcars),
  glm_logit_synth = glm(y ~ x + z, family = binomial, data = synthetic$logit),
  glm_pois = glm(breaks ~ wool + tension, family = poisson, data = warpbreaks),
  glm_pois_bio = glm(art ~ fem + mar + kid5 + phd + ment, family = poisson, data = bioChemists),
  glm_pois_synth = glm(y ~ x, family = poisson, data = synthetic$pois),
  glm_negbin = glm.nb(art ~ fem + mar + kid5 + phd + ment, data = bioChemists),
  # Non-canonical links: R's default IRLS stopping rule (relative deviance change
  # < 1e-8) leaves the estimates short of the optimum by ~1e-5, so converge fully.
  glm_probit = glm(vs ~ mpg + wt, family = binomial(link = "probit"), data = mtcars,
                   control = tight),
  glm_cloglog = glm(vs ~ mpg + wt, family = binomial(link = "cloglog"), data = mtcars,
                    control = tight),
  glm_gamma = glm(mpg ~ wt + hp, family = Gamma(link = "log"), data = mtcars, control = tight),
  glm_invgauss = glm(mpg ~ wt + hp, family = inverse.gaussian(link = "log"), data = mtcars,
                     control = tight)
)

safe <- function(expr) tryCatch(expr, error = function(e) NULL)

# NULL when lmtest's exact algorithm fails and silently falls back to the
# normal approximation (it warns "exact p value cannot be computed").
exact_dw_p <- function(m) {
  tryCatch(
    dwtest(m, alternative = "two.sided", exact = TRUE)$p.value,
    warning = function(w) NULL
  )
}
num <- function(x) if (is.null(x)) NULL else unname(as.numeric(x))

# Per-observation vectors: in full up to 400 observations, else head and sum.
obs <- function(x) {
  x <- num(x)
  if (length(x) <= 400) x else list(head = head(x, 50), sum = sum(x))
}

describe <- function(m) {
  info <- insight::model_info(m)
  # Residual checks are referenced for lm fits only: performance refuses them
  # for glm objects, even Gaussian ones (the Python suite checks that a
  # Gaussian GLM gives the same results as the equivalent OLS fit instead).
  linear <- isTRUE(info$is_linear) && !inherits(m, "glm")
  ref <- list(
    n = nobs(m),
    coef = num(coef(m)),
    fitted = obs(fitted(m)),
    hat = obs(hatvalues(m)),
    rstandard = obs(rstandard(m)),
    cooks = obs(cooks.distance(m)),
    loglik = as.numeric(logLik(m)),
    loglik_df = attr(logLik(m), "df")
  )
  if (linear) {
    ref$normality <- if (length(residuals(m)) >= 5000) {
      ks <- suppressWarnings(ks.test(rstandard(m), "pnorm"))
      list(method = "ks", statistic = num(ks$statistic), p = ks$p.value)
    } else {
      sw <- shapiro.test(rstandard(m))
      list(method = "shapiro", statistic = num(sw$statistic), p = sw$p.value)
    }
    ref$normality$performance_p <- num(check_normality(m))
    ncv <- car::ncvTest(m)
    ref$heteroscedasticity <- list(
      p = num(check_heteroscedasticity(m)),
      statistic = ncv$ChiSquare,
      p_car = ncv$p
    )
    if (is.null(weights(m))) {
      ref$dw <- list(
        statistic = num(dwtest(m)$statistic),
        p_auto = dwtest(m, alternative = "two.sided")$p.value,
        p_exact = if (nobs(m) <= 400) exact_dw_p(m),
        p_normal = dwtest(m, alternative = "two.sided", exact = FALSE)$p.value
      )
    }
  }
  if (!is.null(weights(m)) && linear) {
    w <- weights(m)
    e <- sqrt(w) * residuals(m)
    ref$dw <- list(statistic = sum(diff(e)^2) / sum(e^2))
  }
  vif <- safe(check_collinearity(m, verbose = FALSE))
  if (!is.null(vif)) ref$vif <- as.list(as.data.frame(vif))
  out <- check_outliers(m, method = "cook")
  ref$outliers <- list(
    threshold = attr(out, "threshold")$cook,
    flags = if (length(out) <= 400) as.logical(out) else which(as.logical(out))
  )
  r <- safe(r2(m))
  if (!is.null(r)) ref$r2 <- lapply(r, num)
  ref$performance <- as.list(as.data.frame(model_performance(m)))
  if (isTRUE(info$is_binomial)) {
    br <- binned_residuals(m, verbose = FALSE)
    ref$binned <- c(as.list(as.data.frame(br)), list(resid_ok = attr(br, "resid_ok")))
    ref$pseudo_r2 <- list(
      tjur = num(r2_tjur(m)),
      coxsnell = num(r2_coxsnell(m)),
      nagelkerke = num(r2_nagelkerke(m)),
      mcfadden = num(r2_mcfadden(m)$R2),
      mcfadden_adjusted = num(r2_mcfadden(m)$R2_adjusted)
    )
  }
  if (isTRUE(info$is_count)) {
    ref$pseudo_r2 <- list(nagelkerke = num(r2_nagelkerke(m)), coxsnell = num(r2_coxsnell(m)))
  }
  # Negative binomial models use simulated residuals in R: random, so not referenced.
  if (isTRUE(info$is_count) && !isTRUE(info$is_negbin)) {
    od <- check_overdispersion(m)
    ref$overdispersion <- list(
      chisq = od$chisq_statistic, ratio = od$dispersion_ratio,
      df = od$residual_df, p = od$p_value
    )
    zi <- suppressMessages(check_zeroinflation(m))
    if (!is.null(zi)) {
      ref$zeroinflation <- list(
        predicted = zi$predicted.zeros, observed = zi$observed.zeros, ratio = zi$ratio
      )
    }
  }
  ref
}

reference <- list(
  meta = list(
    R = R.version.string,
    performance = as.character(packageVersion("performance")),
    insight = as.character(packageVersion("insight")),
    lmtest = as.character(packageVersion("lmtest")),
    generated = format(Sys.Date())
  ),
  synthetic_checksums = lapply(synthetic, function(d) as.list(colSums(d))),
  models = lapply(models, describe)
)

# ------------------------------------------------------------ mixed models
mixed_data <- within(synthetic$mixed, g <- factor(g))
singular_data <- within(synthetic$mixed_singular, g <- factor(g))
mixed_models <- list(
  mixed_sleep = lmer(Reaction ~ Days + (1 | Subject), data = sleepstudy),
  mixed_sleep_slope = lmer(Reaction ~ Days + (Days | Subject), data = sleepstudy),
  mixed_sleep_ml = lmer(Reaction ~ Days + (1 | Subject), data = sleepstudy, REML = FALSE),
  mixed_synth = lmer(y ~ x1 + x2 + (1 | g), data = mixed_data),
  mixed_singular = suppressMessages(lmer(y ~ x1 + (1 | g), data = singular_data))
)

# performance::check_heteroscedasticity() takes sigma from insight::get_deviance(),
# which for lmer fits is a likelihood deviance rather than the residual sum of
# squares, so its p-value is driven by the scale of the response. checkmodel uses
# sigma(m) instead; this is that computation, done in R.
het_with_sigma <- function(m) {
  r <- residuals(m)
  s_sq <- df.residual(m) * sigma(m)^2 / length(r)
  ss <- anova(lm(I(r^2 / s_sq) ~ fitted(m)))$"Sum Sq"
  chisq <- (sum(ss) - ss[length(ss)]) / 2
  list(statistic = chisq, p = pchisq(chisq, 1, lower.tail = FALSE))
}

describe_mixed <- function(m) {
  vc <- lapply(VarCorr(m), function(v) unname(as.matrix(v)[, , drop = FALSE]))
  ref <- list(
    n = nobs(m),
    fixef = num(fixef(m)),
    varcorr = lapply(vc, function(v) as.list(as.data.frame(v))),
    sigma = sigma(m),
    loglik = as.numeric(logLik(m)),
    loglik_df = attr(logLik(m), "df"),
    df_residual = df.residual(m),
    fitted = obs(fitted(m)),
    hat = obs(hatvalues(m)),
    rstudent = obs(rstudent(m)),
    cooks = obs(cooks.distance(m)),
    singular = check_singularity(m),
    converged = isTRUE(check_convergence(m)),
    normality = list(
      fixed = num(check_normality(m)),
      random = num(safe(check_normality(m, effects = "random")))
    ),
    heteroscedasticity = c(
      list(performance_p = num(check_heteroscedasticity(m))),
      het_with_sigma(m)
    ),
    performance = as.list(as.data.frame(suppressWarnings(model_performance(m))))
  )
  ic <- suppressWarnings(icc(m))
  if (!all(is.na(ic))) ref$icc <- list(adjusted = ic$ICC_adjusted, unadjusted = ic$ICC_unadjusted)
  rr <- suppressWarnings(r2_nakagawa(m))
  ref$r2 <- list(conditional = num(rr$R2_conditional), marginal = num(rr$R2_marginal))
  vif <- safe(check_collinearity(m, verbose = FALSE))
  if (!is.null(vif)) ref$vif <- as.list(as.data.frame(vif))
  out <- check_outliers(m, method = "cook")
  ref$outliers <- list(threshold = attr(out, "threshold")$cook, flags = as.logical(out))
  ref
}
reference$mixed <- lapply(mixed_models, describe_mixed)

# --------------------------------------------------------------- comparisons
small <- lm(mpg ~ wt, data = mtcars)
medium <- lm(mpg ~ wt + hp, data = mtcars)
big <- lm(mpg ~ wt + hp + qsec, data = mtcars)
logit_1 <- glm(vs ~ mpg, family = binomial, data = mtcars)
logit_2 <- glm(vs ~ mpg + wt, family = binomial, data = mtcars)

as_table <- function(x) as.list(as.data.frame(x))
reference$compare <- list(
  lm = as_table(compare_performance(small, medium, big, verbose = FALSE)),
  lm_rank = as_table(compare_performance(small, medium, big, rank = TRUE, verbose = FALSE)),
  logit = as_table(compare_performance(logit_1, logit_2, verbose = FALSE))
)

write_json(
  reference,
  file.path(out_dir, "reference.json"),
  auto_unbox = TRUE,
  digits = NA,
  pretty = TRUE,
  na = "null"
)
cat("Wrote", file.path(out_dir, "reference.json"), "\n")
