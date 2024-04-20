# Install packages
#install.packages("Synth")
#install.packages("ggplot2")  # For enhanced plotting capabilities
#install.packages("tidyverse")
#install.packages("zoo")
#install.packages("dplyr")

# Load packages
library(Synth)
library(ggplot2)
library(tidyverse)
library(zoo)
#library(dplyr)

# Read the dataset
data <- read.csv('data/synthetic-control-dataset.csv')

# Define predictors and time window
inputs <- c("PCPI_IX_CH_DM", "FILR_PA_DM", "FISR_PA_DM", "IR_SPREAD_DM", "FITB_PA_DM", "BANK_PREM_RATE_DM", "FM2_CH_DM", "FM2_NGDP_DM", "FM2_RES_DM")
#inputs <- c("NGDP_NSA_XDC_R_CH","PCPI_IX_CH_DM", "ENDE_XDC_USD_RATE", "FILR_PA_DM", "FISR_PA_DM", "IR_SPREAD_DM", "FITB_PA_DM", "BANK_PREMIUM_RATE_DM")
#inputs_demeaned <- c("DM_FIDR_PA", "DM_PCPI_IX_CH", "DM_FILR_PA", "DM_FISR_PA", "DM_IR_SPREAD", "DM_FITB_PA_DM", "DM_BANK_PREMIUM_RATE")

output <- c("FIDR_PA_DM")
#output <- c("FIDR_PA")
#output_demeaned <- c("DM_FIDR_PA")

treatment_unit = 1
control_units <- c(2:16)
start_year <- min(data$year)
event_year <- 2020
end_year <- max(data$year)
increment <- 1 # 1 for yearly, 1/4 for quarterly, and 1/12 for monthly

# Generate the sequence from start_year to end_year and round it to 3 decimal places
predict_time_seq <- round(seq(start_year, event_year, by = increment), 2)
full_time_seq <- round(seq(start_year, end_year, by = increment), 2)

# Calculate the mean only for the specific time window but apply the subtraction to all years
#data <- data %>%
#  group_by(code) %>%
#  mutate(
#    across(all_of(c(output,inputs)),
#           ~ . - mean(.[year >= start_year & year <= end_year], na.rm = TRUE),
#           .names = "DM_{.col}")
#  ) %>%
#  ungroup()

# Prepare the data using the Synth package
dataprep.out <- dataprep(
  foo = data,
  predictors = inputs,
  predictors.op = "mean",
  time.variable = "year",
  dependent = output,
  unit.variable = "code",
  treatment.identifier = treatment_unit,
  controls.identifier = control_units,
  time.predictors.prior = predict_time_seq,
  time.optimize.ssr = predict_time_seq,
  time.plot = full_time_seq
)

# Build the synthetic control model
synth.out <- synth(dataprep.out, optimxmethod='All')

# Plot the paths of actual vs synthetic
path.plot(synth.res = synth.out,
          dataprep.res = dataprep.out,
          tr.intake = event_year,
          Ylab = "Interest Rate on Deposits",
          Xlab = "Year",
          Ylim = c(-2,2),
          Legend = c("The Bahamas", "Synthetic The Bahamas"),
          Main = "The Bahamas vs Synthetic The Bahamas")

# Plot the gaps
gaps.plot(synth.res = synth.out,
          dataprep.res = dataprep.out,
          tr.intake = event_year,
          Ylab = "Effect",
          Xlab = "Year",
          Ylim = c(-1,1),
          Main = "Gap between the Interest Rate on Deposits in The Bahamas and its synthetic version")
