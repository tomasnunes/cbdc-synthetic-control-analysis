# Install packages
#install.packages("Synth")
#install.packages("tidyverse")
#install.packages("zoo")
#install.packages("ggpubr")
#install.packages("grid")
#install.packages("gridExtra")
#install.packages("gridGraphics")
#install.packages("ggplot2")
#install.packages("SCtools")

# Load packages
library(Synth)
library(tidyverse)
library(zoo)
library(grid)
library(gridExtra)
library(gridGraphics)
library(ggpubr)
library(ggplot2)
library(SCtools)

# Read the dataset
data <- read.csv('data/synthetic-control-dataset-bahamas-yearly.csv')

# Define predictors and time window

# The Bahamas
inputs <- c("NGDP_XDC_CH_DM", "PCPI_IX_CH_DM", "FILR_PA_DM", "FISR_PA_DM", "FITB_PA_DM", "IR_SPREAD_DM", "CREDIT_PREM_RATE_DM", "FM2_CH_DM", "FM2_NGDP_DM", "FM2_RES_DM", "ENDE_XDC_USD_RATE_CH_DM")
# Nigeria
#inputs <- c("NGDP_XDC_CH_DM", "PCPI_IX_CH_DM", "ENDE_XDC_USD_RATE_CH_DM", "FILR_PA_DM", "IR_SPREAD_DM", "FITB_PA_DM", "BANK_PREM_RATE_DM", "FM2_CH_DM", "FM2_NGDP_DM", "FM2_RES_DM", "FPOLM_PA_DM")

output <- c("FIDR_PA_DM")

treatment_unit = 1

# The Bahamas
control_units <- c(2,3,4,5,6,7,8,9,10,11,12,15,16)
# Nigeria
#control_units <- c(2:26)

# The Bahamas
event_year <- 2020
# Nigeria
#event_year <- 2021

start_year <- min(data$year)
end_year <- max(data$year)
increment <- 1 # 1 for yearly, 1/4 for quarterly, and 1/12 for monthly

# Generate the sequence from start_year to end_year and round it to 3 decimal places
predict_time_seq <- round(seq(start_year, event_year, by = increment), 2)
optimize_time_seq <- predict_time_seq
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
  time.optimize.ssr = optimize_time_seq,
  time.plot = full_time_seq
)

# Build the synthetic control model
synth.out <- synth(dataprep.out, optimxmethod='All')
#synth.out <- synth(dataprep.out, Sigf.ipop=5)

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
          Ylim = c(-2,2),
          Main = "Gap between The Bahamas and its synthetic version")

## run the generate.placebos command to reassign treatment status
## to each unit listed as control, one at a time, and generate their
## synthetic versions. Sigf.ipop = 2 for faster computing time.
## Increase to the default of 5 for better estimates.
tdf <- generate.placebos(dataprep.out,synth.out, Sigf.ipop = 5, strategy='multicore')

## Plot the gaps in outcome values over time of each unit --
## treated and placebos -- to their synthetic controls

p <- plot_placebos(tdf, discard.extreme=TRUE, mspe.limit=10, xlab='Year')
print(p)


# Create empty lists to store the path and gap plots
path_plots <- list()
gap_plots <- list()

# Perform placebo tests for each control unit
for (placebo_unit in control_units) {
  # Prepare the data using the Synth package
  dataprep.out <- dataprep(
    foo = data,
    predictors = inputs,
    predictors.op = "mean",
    time.variable = "year",
    dependent = output,
    unit.variable = "code",
    treatment.identifier = placebo_unit,
    controls.identifier = setdiff(c(treatment_unit, control_units), placebo_unit),
    time.predictors.prior = predict_time_seq,
    time.optimize.ssr = optimize_time_seq,
    time.plot = full_time_seq
  )

  # Build the synthetic control model
  synth.out <- synth(dataprep.out, optimxmethod = 'All')

  # Generate the path plot and convert it to a grob
  path_plot <- path.plot(synth.res = synth.out,
                         dataprep.res = dataprep.out,
                         tr.intake = event_year,
                         Ylab = "Interest Rate on Deposits",
                         Xlab = "Year",
                         Legend = c(paste("Control Unit", placebo_unit), paste("Synthetic Control Unit", placebo_unit)),
                         Main = paste("Control Unit", placebo_unit, "vs Synthetic Control Unit", placebo_unit))
  path_plots[[length(path_plots) + 1]] <- ggplotGrob(path_plot)

  # Generate the gap plot and convert it to a grob
  gap_plot <- gaps.plot(synth.res = synth.out,
                        dataprep.res = dataprep.out,
                        tr.intake = event_year,
                        Ylab = "Effect",
                        Xlab = "Year",
                        Main = paste("Gap between Control Unit", placebo_unit, "and its synthetic version"))
  gap_plots[[length(gap_plots) + 1]] <- ggplotGrob(gap_plot)
}

# Arrange the path plots in a grid
path_grid <- arrangeGrob(grobs = path_plots, ncol = 4)

# Arrange the gap plots in a grid
gap_grid <- arrangeGrob(grobs = gap_plots, ncol = 4)

# Display the path plot grid
grid.draw(path_grid)

# Display the gap plot grid
grid.draw(gap_grid)