# Install packages
install.packages("Synth")
install.packages("ggplot2")  # For enhanced plotting capabilities

# Load packages
library(Synth)
library(ggplot2)

# Read the dataset
data <- read.csv('data/synthetic-control-dataset.csv')

# Prepare the data using the Synth package
dataprep.out <- dataprep(
  foo = data,
  # YEARLY DATA
  predictors = c("NGDP_XDC", "PCPI_IX", "ENDE_XDC_USD_RATE", "FITB_PA", "FILR_PA", "FPOLM_PA"),
  # QUARTERLY DATA
  #predictors = c("NGDP_NSA_XDC", "PCPI_IX", "ENDE_XDC_USD_RATE", "FITB_PA", "FILR_PA", "FPOLM_PA"),
  predictors.op = "mean",
  time.variable = "year",
  dependent = "FIDR_PA",
  unit.variable = "code",
  treatment.identifier = 1,
  controls.identifier = c(2:9),
  # YEARLY DATA
  time.predictors.prior = seq(min(data$year), 2020, by = 1),
  time.optimize.ssr = seq(min(data$year), 2020, by = 1),
  time.plot = seq(min(data$year), max(data$year), by = 1)
  # QUARTERLY DATA
  #time.predictors.prior = seq(min(data$year), 2020.75, by = 0.25),
  #time.optimize.ssr = seq(min(data$year), 2020.75, by = 0.25),
  #time.plot = seq(min(data$year), max(data$year), by = 0.25)
)

# Build the synthetic control model
synth.out <- synth(dataprep.out, optimxmethod='All')

# Plot the paths of actual vs synthetic
path.plot(synth.res = synth.out,
          dataprep.res = dataprep.out,
          tr.intake = 2020,
          Ylab = "Interest Rate on Deposits",
          Xlab = "Year",
          Legend = c("The Bahamas", "Synthetic The Bahamas"),
          Main = "The Bahamas vs Synthetic The Bahamas")

# Plot the gaps
gaps.plot(synth.res = synth.out,
          dataprep.res = dataprep.out,
          tr.intake = 2020,
          Ylab = "Effect",
          Xlab = "Year",
          Main = "Gap between the Interest Rate on Deposits in The Bahamas and its synthetic version")
