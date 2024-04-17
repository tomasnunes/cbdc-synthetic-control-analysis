# Install packages
#install.packages("Synth")
#install.packages("ggplot2")  # For enhanced plotting capabilities
#install.packages("tidyverse")
#install.packages("zoo")

# Load packages
library(Synth)
library(ggplot2)
library(tidyverse)
library(zoo)

# Read the dataset
data <- read.csv('data/synthetic-control-dataset.csv')

# Prepare the data using the Synth package
dataprep.out <- dataprep(
  foo = data,
  # YEARLY DATA
  #predictors = c("NGDP_XDC_R_CH", "PCPI_IX_CH", "ENDE_XDC_USD_RATE", "FITB_PA_R", "FILR_PA_R", "FISR_PA_R", "FPOLM_PA"),
  # QUARTERLY DATA
  #predictors = c("NGDP_NSA_XDC_R_CH", "PCPI_IX_CH", "ENDE_XDC_USD_RATE", "FITB_PA_R", "FILR_PA_R", "FISR_PA_R", "FPOLM_PA"),
  predictors = c("NGDP_NSA_XDC_R_CH_DM", "PCPI_IX_CH_DM", "ENDE_XDC_USD_RATE", "FITB_PA_R_DM", "FILR_PA_R_DM", "FISR_PA_R_DM", "FPOLM_PA_DM"),
  predictors.op = "mean",
  time.variable = "year",
  #dependent = "FIDR_PA_R",
  dependent = "FIDR_PA_R_DM",
  unit.variable = "code",
  treatment.identifier = 1,
  controls.identifier = c(2:18),
  # YEARLY DATA
  #time.predictors.prior = seq(min(data$year), 2020, by = 1),
  #time.optimize.ssr = seq(min(data$year), 2020, by = 1),
  #time.plot = seq(min(data$year), max(data$year), by = 1)
  # QUARTERLY DATA
  time.predictors.prior = seq(2011.0, 2020.75, by = 0.25),
  time.optimize.ssr = seq(2011.0, 2020.75, by = 0.25),
  time.plot = seq(2011.0, max(data$year), by = 0.25)
)

# Build the synthetic control model
synth.out <- synth(dataprep.out, optimxmethod='All')

# Plot the paths of actual vs synthetic
path.plot(synth.res = synth.out,
          dataprep.res = dataprep.out,
          tr.intake = 2020.75,
          Ylab = "Interest Rate on Deposits",
          Xlab = "Year",
          Legend = c("The Bahamas", "Synthetic The Bahamas"),
          Main = "The Bahamas vs Synthetic The Bahamas")

# Plot the gaps
gaps.plot(synth.res = synth.out,
          dataprep.res = dataprep.out,
          tr.intake = 2020.75,
          Ylab = "Effect",
          Xlab = "Year",
          Main = "Gap between the Interest Rate on Deposits in The Bahamas and its synthetic version")
