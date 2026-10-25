Consumer Loan Delinquency Forecasting: Exploratory Analysis and Feature
Engineering
================
Farida Issifou
2026-04-26

- [Setup](#setup)
- [1. Data Construction and
  Description](#1-data-construction-and-description)
- [2. Exploratory Analysis and Feature
  Engineering](#2-exploratory-analysis-and-feature-engineering)
  - [2.1 Stationarity Testing](#21-stationarity-testing)
  - [2.2 Transformations](#22-transformations)
  - [2.3 Seasonal Adjustment](#23-seasonal-adjustment)
  - [2.4 Pre-Whitening and Cross-Correlation
    Analysis](#24-pre-whitening-and-cross-correlation-analysis)
  - [2.5 Final Dataset Assembly](#25-final-dataset-assembly)

------------------------------------------------------------------------

## Setup

``` r
library(forecast)  # auto.arima, nsdiffs
library(xts)       # time series objects
library(ggplot2)   # plots
library(readr)     # read_csv
library(dplyr)     # rename
library(tseries)   # adf.test
```

``` r
# Simple line plot for a single xts series
plot_series <- function(x, title, subtitle = NULL) {
  df <- data.frame(date = index(x), value = as.numeric(x))
  ggplot(df, aes(date, value)) +
    geom_line(colour = "#2c3e50") +
    labs(title = title, subtitle = subtitle, x = NULL, y = NULL) +
    theme_minimal()
}
```

------------------------------------------------------------------------

## 1. Data Construction and Description

This section documents the construction of the dataset that will be used
throughout the analysis. It covers the period **1987 Q1 to 2025 Q4**
(156 quarterly observations) and includes five variables: the consumer
loan delinquency rate (here our target variable) and four macroeconomic
predictors: the yield spread, the unemployment rate, the federal funds
rate, and consumer confidence.

``` r
# Quarterly dataset built from FRED by src/build_dataset.py
data <- read_csv("../data/data.csv", show_col_types = FALSE)

# Rename variables
data <- data %>%
  rename(
    dlq    = delinquency,
    yspread = yield_spread,
    unemp  = unemployment,
    ffr    = federal_fund_rate,
    conf   = consumer_confidence
  )
```

``` r
# Convert to xts for time series operations
data_xts <- xts(data[, -1], order.by = as.Date(data$date))
```

------------------------------------------------------------------------

## 2. Exploratory Analysis and Feature Engineering

### 2.1 Stationarity Testing

A time series is stationary when its mean, variance, and autocorrelation
structure do not change over time. This is a prerequisite for time
series models, since a non-stationary series will produce spurious
relationships and therefore, unreliable forecasts.

Here, we use the Augmented Dickey-Fuller (ADF) test to assess each
series. The null hypothesis of the ADF test is that the series contains
a unit root (meaning, it is non-stationary). A small p-value (\< 0.05)
leads us to reject the null hypothesis and conclude that the series is
stationary.

``` r
# Helper to print ADF results 
run_adf <- function(series, label) {
  test <- adf.test(na.omit(series))
  cat(label, "\n")
  cat("  ADF statistic:", round(test$statistic, 3), "\n")
  cat("  p-value:      ", round(test$p.value, 3), "\n")
  cat("  Conclusion:   ",
      ifelse(test$p.value < 0.05, "Stationary", "Non-stationary"), "\n\n")
}
```

------------------------------------------------------------------------

#### Delinquency Rate (dlq)

``` r
plot_series(data_xts[, "dlq"],
        title    = "Consumer Loan Delinquency Rate",
        subtitle = "Quarterly, 1987 Q1 – 2025 Q4")
```

![](/home/claude/consumer-delinquency-forecasting/reports/02_feature_engineering_files/figure-gfm/plot-dlq-1.png)<!-- -->

As previously stated, the delinquency rate is our target variable.
Visually, we can see that it rises sharply during recessions and falls
back during recoveries, with the 2008 financial crisis producing the
most severe spike in the sample. We also notice that it does not drift
permanently in one direction.

``` r
acf(na.omit(data_xts[, "dlq"]),  main = "ACF — Delinquency Rate (level)")
```

![](/home/claude/consumer-delinquency-forecasting/reports/02_feature_engineering_files/figure-gfm/acf-dlq-1.png)<!-- -->

This suggests that the series is stationary, and the ADF test confirms
it: with a p-value of 0.01 we reject the null of a unit root. The
delinquency rate is stationary in levels and thus requires no
transformation.

``` r
run_adf(data_xts[, "dlq"], "Delinquency Rate (dlq)")
```

    ## Delinquency Rate (dlq) 
    ##   ADF statistic: -4.196 
    ##   p-value:       0.01 
    ##   Conclusion:    Stationary

------------------------------------------------------------------------

#### Unemployment Rate (unemp)

``` r
plot_series(data_xts[, "unemp"],
        title    = "Unemployment Rate",
        subtitle = "Quarterly, 1987 Q1 – 2025 Q4")
```

![](/home/claude/consumer-delinquency-forecasting/reports/02_feature_engineering_files/figure-gfm/plot-unemp-1.png)<!-- -->

The unemployment rate tends to rise sharply during periods of crisis and
recover slowly. As observed here, each cycle tends to start from a
different level, and there is no clear tendency to return to a fixed
long-run level. This kind of persistent, drifting behavior is
characteristic of a non-stationary series.

``` r
acf(na.omit(data_xts[, "unemp"]),  main = "ACF — Unemployment Rate (level)")
```

![](/home/claude/consumer-delinquency-forecasting/reports/02_feature_engineering_files/figure-gfm/acf-unemp-1.png)<!-- -->

The ADF test confirms this: with a p-value of 0.21, we cannot reject the
presence of a unit root. The unemployment rate will need to be
first-differenced before entering the model.

``` r
run_adf(data_xts[, "unemp"], "Unemployment Rate (unemp)")
```

    ## Unemployment Rate (unemp) 
    ##   ADF statistic: -2.88 
    ##   p-value:       0.21 
    ##   Conclusion:    Non-stationary

------------------------------------------------------------------------

#### Federal Funds Rate (ffr)

``` r
plot_series(data_xts[, "ffr"],
        title    = "Federal Funds Rate",
        subtitle = "Quarterly, 1987 Q1 – 2025 Q4")
```

![](/home/claude/consumer-delinquency-forecasting/reports/02_feature_engineering_files/figure-gfm/plot-ffr-1.png)<!-- -->

As seen on the graph above, the federal funds rate has been on a
downward path since the late 1980s, punctuated by sharp policy cycles.
Unlike the delinquency rate, it never goes back to a stable center; it
just drifts lower over time.

``` r
acf(na.omit(data_xts[, "ffr"]),  main = "ACF — Federal Funds Rate (level)")
```

![](/home/claude/consumer-delinquency-forecasting/reports/02_feature_engineering_files/figure-gfm/acf-ffr-1.png)<!-- -->

The ADF test returns a p-value of 0.159, confirming non-stationarity in
levels. First-differencing will transform it into the quarter-on-quarter
change in the policy rate, which is stationary.

``` r
run_adf(data_xts[, "ffr"], "Federal Funds Rate (ffr)")
```

    ## Federal Funds Rate (ffr) 
    ##   ADF statistic: -3.001 
    ##   p-value:       0.159 
    ##   Conclusion:    Non-stationary

------------------------------------------------------------------------

#### Yield Spread (yspread)

``` r
plot_series(data_xts[, "yspread"],
        title    = "Yield Spread (10Y - 2Y)",
        subtitle = "Quarterly, 1987 Q1 – 2025 Q4")
```

![](/home/claude/consumer-delinquency-forecasting/reports/02_feature_engineering_files/figure-gfm/plot-yspread-1.png)<!-- -->

The yield spread, or the difference between the 10-year and the 2-year
Treasury yields, is a stationary series by construction. It swings
between positive and negative values as the shape of the yield curve
shifts across the business cycle, but it does not trend persistently in
one direction.

``` r
acf(na.omit(data_xts[, "yspread"]),  main = "ACF — Yield Spread (level)")
```

![](/home/claude/consumer-delinquency-forecasting/reports/02_feature_engineering_files/figure-gfm/acf-yspread-1.png)<!-- -->

The ADF test confirms the stationarity of this variable, with a p-value
of 0.048. No transformation is needed in this case.

``` r
run_adf(data_xts[, "yspread"], "Yield Spread (yspread)")
```

    ## Yield Spread (yspread) 
    ##   ADF statistic: -3.469 
    ##   p-value:       0.048 
    ##   Conclusion:    Stationary

------------------------------------------------------------------------

#### Consumer Confidence (conf)

``` r
plot_series(data_xts[, "conf"],
        title    = "Consumer Confidence Index",
        subtitle = "Quarterly, 1987 Q1 – 2025 Q4")
```

![](/home/claude/consumer-delinquency-forecasting/reports/02_feature_engineering_files/figure-gfm/plot-conf-1.png)<!-- -->

Consumer confidence measures household sentiment over time. As observed
above, it rises during expansions and drops sharply in crises. The
COVID-19 collapse in 2020 marks the sharpest drop in the sample and is
more representative of an abrupt structural break rather than a gradual
cyclical shift.

``` r
acf(na.omit(data_xts[, "conf"]),  main = "ACF — Consumer Confidence (level)")
```

![](/home/claude/consumer-delinquency-forecasting/reports/02_feature_engineering_files/figure-gfm/acf-conf-1.png)<!-- -->

Despite the absence of a clear long-run trend, the ADF test returns a
p-value of 0.567, meaning we cannot reject non-stationarity. The series
will be first-differenced. It is worth noting that the structural break
introduced by COVID may be distorting the test result. The ADF is
sensitive to sharp level shifts of this kind, and a stationary series
with a break can appear non-stationary to the test. We difference
regardless to be conservative, and we do flag this as something to note
in the interpretation.

``` r
run_adf(data_xts[, "conf"], "Consumer Confidence (conf)")
```

    ## Consumer Confidence (conf) 
    ##   ADF statistic: -2.022 
    ##   p-value:       0.567 
    ##   Conclusion:    Non-stationary

------------------------------------------------------------------------

### 2.2 Transformations

Based on the ADF results above, we apply first differences to the
unemployment rate, the federal funds rate, and the Consumer Confidence
Index, which were found to be non-stationary. First-differencing removes
stochastic trends by computing period-to-period changes, converting a
level series into a change series.

``` r
# First differences for non-stationary series

d_unemp  <- diff(data_xts[, "unemp"])   # change in unemployment rate
d_ffr    <- diff(data_xts[, "ffr"])     # change in federal funds rate
d_conf   <- diff(data_xts[, "conf"])    # change in consumer confidence
# Retain in levels because stationary
dlq     <- data_xts[, "dlq"]
yspread <- data_xts[, "yspread"]
```

``` r
# Re-run ADF on transformed series to confirm stationarity
run_adf(d_unemp, "Δ Unemployment Rate (d_unemp)")
```

    ## Δ Unemployment Rate (d_unemp) 
    ##   ADF statistic: -5.219 
    ##   p-value:       0.01 
    ##   Conclusion:    Stationary

``` r
run_adf(d_ffr,   "Δ Federal Funds Rate (d_ffr)")
```

    ## Δ Federal Funds Rate (d_ffr) 
    ##   ADF statistic: -4.639 
    ##   p-value:       0.01 
    ##   Conclusion:    Stationary

``` r
run_adf(d_conf,   "Δ Consumer Confidence Index (d_conf)")
```

    ## Δ Consumer Confidence Index (d_conf) 
    ##   ADF statistic: -5.317 
    ##   p-value:       0.01 
    ##   Conclusion:    Stationary

``` r
# Visual check of differenced series
plot_series(d_unemp,
        title    = "Change in Unemployment Rate",
        subtitle = "First difference")
```

![](/home/claude/consumer-delinquency-forecasting/reports/02_feature_engineering_files/figure-gfm/plot-transformed-1.png)<!-- -->

``` r
plot_series(d_ffr,
        title    = "Change in Federal Funds Rate",
        subtitle = "First difference")
```

![](/home/claude/consumer-delinquency-forecasting/reports/02_feature_engineering_files/figure-gfm/plot-transformed-2.png)<!-- -->

``` r
plot_series(d_conf,
        title    = "Change in Consumer Confidence Index",
        subtitle = "First difference")
```

![](/home/claude/consumer-delinquency-forecasting/reports/02_feature_engineering_files/figure-gfm/plot-transformed-3.png)<!-- -->

------------------------------------------------------------------------

### 2.3 Seasonal Adjustment

Quarterly data can sometimes carry mechanical seasonal patterns, which
are fluctuations that repeat at fixed calendar intervals and reflect
regularities shaped by institutional or behavioral conditions rather
than underlying economic dynamics. Including those unadjusted seasonal
variations in a forecasting model risks mistaking those very different
underlying factors. We test each series for seasonality using the
Canova-Hansen test.

``` r
# Canova-Hansen test for seasonality
# Convert each xts series to ts object first (required by nsdiffs)
to_ts <- function(x) ts(as.numeric(x), frequency = 4)

ch_dlq     <- nsdiffs(to_ts(dlq),     test = "ch")
ch_unemp   <- nsdiffs(to_ts(d_unemp), test = "ch")
ch_ffr     <- nsdiffs(to_ts(d_ffr),   test = "ch")
ch_yspread <- nsdiffs(to_ts(yspread), test = "ch")
ch_conf    <- nsdiffs(to_ts(d_conf),    test = "ch")

cat("Seasonal differences required (0 = no seasonality detected):\n")
```

    ## Seasonal differences required (0 = no seasonality detected):

``` r
cat("  dlq:     ", ch_dlq,     "\n")
```

    ##   dlq:      0

``` r
cat("  d_unemp: ", ch_unemp,   "\n")
```

    ##   d_unemp:  0

``` r
cat("  d_ffr:   ", ch_ffr,     "\n")
```

    ##   d_ffr:    0

``` r
cat("  yspread: ", ch_yspread, "\n")
```

    ##   yspread:  0

``` r
cat("  d_conf:  ", ch_conf,    "\n")
```

    ##   d_conf:   0

The result is 0 for all our variables, meaning that no seasonal
differencing is required here.

### 2.4 Pre-Whitening and Cross-Correlation Analysis

Before examining the relationship between each predictor and the
delinquency rate, we must account for the autocorrelation structure
within each series. In practice raw cross-correlations between two
persistent series will be inflated and potentially misleading. The
correlation we observe could simply reflect the fact that both series
are smooth and slowly-moving, or in other words accidental, and not that
one genuinely leads the other.

Pre-whitening addresses this by fitting an ARIMA model to each
predictor, extracting the residuals (which are approximately white noise
by construction), applying the same filter to the delinquency rate, and
then computing the cross-correlation function (CCF) on the filtered
series. Any significant spikes in the CCF after pre-whitening should
therefore reflect a genuine lead-lag relationship between our variables.

This step is particularly important because delinquency is a lagging
indicator, meaning that borrowers do not default immediately when
conditions deteriorate. So the variable tends to move with a lag
compared to the business cycle. The CCF will tell us how many quarters
each predictor leads delinquency, which directly informs our lag
specification in the ARMA and extended models.

Two implementation details matter here. First, the predictor and the
delinquency rate are aligned on their dates before filtering, since the
differenced series start one quarter later than the series in levels.
Second, the CCF is computed as `ccf(dlq, x)`, so that the bar at lag *k*
is the correlation between the predictor at quarter *t* and delinquency
at quarter *t + k*. **Positive lags therefore mean the predictor leads
delinquency**, while negative lags mean delinquency leads the predictor.
The dashed lines are the approximate 95% significance bounds (±2/√n).

``` r
# Pre-whitening function:
# 0. Align predictor x and target y on common dates
# 1. Fit auto.arima to predictor x
# 2. Extract residuals from x (whitened predictor)
# 3. Apply same filter to dlq (whitened target)
# 4. Compute and plot CCF on whitened pair

prewhiten_ccf <- function(x, y, xname, max_lag = 12) {
  
  # Step 0: keep only quarters where both series exist, so they stay aligned
  xy   <- na.omit(merge(x, y))
  x_ts <- to_ts(xy[, 1])
  y_ts <- to_ts(xy[, 2])
  
  # Step 1: fit ARIMA to predictor
  fit_x <- auto.arima(x_ts)
  
  # Step 2: residuals of predictor (whitened x)
  resid_x <- residuals(fit_x)
  
  # Step 3: filter dlq with same ARIMA coefficients
  resid_y <- residuals(Arima(y_ts, model = fit_x))
  
  # Step 4: CCF on whitened pair
  # ccf(a, b) at lag k estimates cor(a[t+k], b[t]); with a = dlq and b = x,
  # positive k means x leads dlq by k quarters.
  # as.numeric() makes the lag axis count quarters (a ts object would show years)
  ccf(as.numeric(resid_y), as.numeric(resid_x),
      lag.max = max_lag,
      main    = paste0("Pre-whitened CCF: ", xname, " → dlq"),
      ylab    = "Cross-correlation",
      xlab    = "Lag in quarters (positive = predictor leads)")
  
  cat("ARIMA model fitted to", xname, ":", "\n")
  print(fit_x)
  cat("\n")
}
```

``` r
# Unemployment → delinquency
# We expect unemployment to lead delinquency by several quarters:
# job loss precedes financial distress and eventual default
prewhiten_ccf(d_unemp, dlq, xname = "Δ Unemployment")
```

![](/home/claude/consumer-delinquency-forecasting/reports/02_feature_engineering_files/figure-gfm/ccf-unemp-1.png)<!-- -->

    ## ARIMA model fitted to Δ Unemployment : 
    ## Series: x_ts 
    ## ARIMA(0,0,1) with zero mean 
    ## 
    ## Coefficients:
    ##           ma1
    ##       -0.1886
    ## s.e.   0.0848
    ## 
    ## sigma^2 = 0.7421:  log likelihood = -196.34
    ## AIC=396.68   AICc=396.76   BIC=402.77

Our first figure shows significant positive correlations from lag 0 up
to lag 3 (lag −1 is also significant, reflecting the strong co-movement
around the same quarter), fading out after about a year. Rising
unemployment is therefore followed by higher delinquency both within the
same quarter and up to three quarters later, which is consistent with
job losses gradually translating into missed payments. Based on that, we
include d_unemp at lags 0 to 3.

``` r
# Federal funds rate → delinquency
# Rate changes transmit to borrowers through variable-rate debt repricing,
# but the pass-through takes time — we expect a positive lag of 2–4 quarters
prewhiten_ccf(d_ffr, dlq, xname = "Δ Federal Funds Rate")
```

![](/home/claude/consumer-delinquency-forecasting/reports/02_feature_engineering_files/figure-gfm/ccf-ffr-1.png)<!-- -->

    ## ARIMA model fitted to Δ Federal Funds Rate : 
    ## Series: x_ts 
    ## ARIMA(2,0,0)(2,0,0)[4] with zero mean 
    ## 
    ## Coefficients:
    ##          ar1     ar2     sar1     sar2
    ##       0.6867  0.0065  -0.0697  -0.0590
    ## s.e.  0.0809  0.0821   0.0848   0.0834
    ## 
    ## sigma^2 = 0.1114:  log likelihood = -48.13
    ## AIC=106.27   AICc=106.67   BIC=121.49

Here, the correlations are mostly negative but almost all of them stay
within the significance bounds. The only clear spike is at lag −1:
rising delinquency tends to be followed by rate cuts one quarter later,
which reflects the central bank reacting to credit stress rather than
rates driving delinquency. Lag 1 is borderline. We therefore keep d_ffr
at lag 1 as a candidate, to be confirmed or dropped at the modeling
stage.

``` r
# Yield spread → delinquency
# The yield spread is a forward-looking indicator — an inverted curve
# signals tightening credit conditions before they materialize in defaults
prewhiten_ccf(yspread, dlq, xname = "Yield Spread")
```

![](/home/claude/consumer-delinquency-forecasting/reports/02_feature_engineering_files/figure-gfm/ccf-yspread-1.png)<!-- -->

    ## ARIMA model fitted to Yield Spread : 
    ## Series: x_ts 
    ## ARIMA(2,0,1)(0,0,1)[4] with non-zero mean 
    ## 
    ## Coefficients:
    ##          ar1      ar2      ma1     sma1    mean
    ##       1.7807  -0.8199  -0.4832  -0.1395  0.9722
    ## s.e.  0.0775   0.0737   0.1189   0.0845  0.1890
    ## 
    ## sigma^2 = 0.04572:  log likelihood = 20.2
    ## AIC=-28.4   AICc=-27.84   BIC=-10.1

Contrary to our expectation, no positive lag is significant within the
12-quarter window. The one significant spike is at lag −4: higher
delinquency is followed by a steeper yield curve about a year later,
consistent with monetary easing after periods of credit stress. The
yield spread is known to anticipate recessions over long horizons, but
it does not show a clear short-run lead on delinquency here. We keep
yspread at lag 1 as a theory-motivated candidate and let the models
decide whether it adds predictive power.

``` r
# Change in consumer confidence → delinquency
# Confidence captures household sentiment around financial stability;

prewhiten_ccf(d_conf, dlq, xname = "Consumer Confidence")
```

![](/home/claude/consumer-delinquency-forecasting/reports/02_feature_engineering_files/figure-gfm/ccf-conf-1.png)<!-- -->

    ## ARIMA model fitted to Consumer Confidence : 
    ## Series: x_ts 
    ## ARIMA(0,0,0) with zero mean 
    ## 
    ## sigma^2 = 28:  log likelihood = -478.17
    ## AIC=958.35   AICc=958.37   BIC=961.39

As shown by the graph above, the CCF shows no clear signal in this case.
Therefore, we only include d_conf at lag 0.

**Summary of the lag specification**

| Predictor | Lags retained | Evidence from the CCF                                      |
|-----------|---------------|------------------------------------------------------------|
| d_unemp   | 0, 1, 2, 3    | Significant at lags 0–3                                    |
| d_ffr     | 1             | Borderline at lag 1; delinquency leads rate cuts (lag −1)  |
| yspread   | 1             | No significant lead; delinquency leads the spread (lag −4) |
| d_conf    | 0             | No significant correlation                                 |

Note: these lags serve as an initial specification choice that will be
refined later on, based on coefficient significance in the next stages
of our analysis. Moreover, those lag structures will be applied within
each model specification in the next section of our work, which will
allow us to maintain some flexibility across our various models.

------------------------------------------------------------------------

### 2.5 Final Dataset Assembly

Having verified stationarity, assessed seasonality, and identified the
lead-lag structure between predictors and the target variable, we now
assemble the final dataset for modeling. Series that required
differencing are replaced with their differenced versions. The one
observation lost through differencing is handled by `na.omit()` at the
merge stage.

``` r
# Align all series — differenced variables lose one observation
# model_data starts from 1987 Q2 to preserve alignment

model_data <- na.omit(merge(
  dlq,
  yspread,
  d_unemp,
  d_ffr,
  d_conf
))

colnames(model_data) <- c("dlq", "yspread", "d_unemp", "d_ffr", "d_conf")

# Verify dimensions and date range
cat("Observations:", nrow(model_data), "\n")
```

    ## Observations: 155

``` r
cat("Period:      ", format(min(index(model_data))), "to",
                     format(max(index(model_data))), "\n")
```

    ## Period:       1987-06-30 to 2025-12-31

``` r
cat("Variables:   ", ncol(model_data), "\n")
```

    ## Variables:    5

``` r
# Final visual check of all modeling series in their stationary form
plot_series(model_data[, "dlq"],     title = "Delinquency Rate",              subtitle = "Levels")
```

![](/home/claude/consumer-delinquency-forecasting/reports/02_feature_engineering_files/figure-gfm/plot-final-series-1.png)<!-- -->

``` r
plot_series(model_data[, "yspread"], title = "Yield Spread",                  subtitle = "Levels")
```

![](/home/claude/consumer-delinquency-forecasting/reports/02_feature_engineering_files/figure-gfm/plot-final-series-2.png)<!-- -->

``` r
plot_series(model_data[, "d_unemp"], title = "Change in Unemployment Rate",   subtitle = "First difference")
```

![](/home/claude/consumer-delinquency-forecasting/reports/02_feature_engineering_files/figure-gfm/plot-final-series-3.png)<!-- -->

``` r
plot_series(model_data[, "d_ffr"],   title = "Change in Federal Funds Rate",  subtitle = "First difference")
```

![](/home/claude/consumer-delinquency-forecasting/reports/02_feature_engineering_files/figure-gfm/plot-final-series-4.png)<!-- -->

``` r
plot_series(model_data[, "d_conf"],    title = "Change in Consumer Confidence Index",     subtitle = "First Difference")
```

![](/home/claude/consumer-delinquency-forecasting/reports/02_feature_engineering_files/figure-gfm/plot-final-series-5.png)<!-- -->

``` r
# Convert xts to dataframe and save as CSV for use in Python
# CSV is the most portable format across Python libraries
# (pandas, statsmodels, sklearn all read it natively)

model_df <- data.frame(
  date = index(model_data),
  coredata(model_data)
)

write.csv(model_df, "../data/model_data.csv", row.names = FALSE)

cat("Dataset saved as data/model_data.csv\n")
```

    ## Dataset saved as data/model_data.csv

``` r
cat("Columns:", paste(colnames(model_df), collapse = ", "), "\n")
```

    ## Columns: date, dlq, yspread, d_unemp, d_ffr, d_conf
