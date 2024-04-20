# Install required packages
# Make sure that the scipy version is 1.4.1 for the SyntheticControlMethods package
# pip install requests pandas numpy matplotlib seaborn cvxpy openpyxl scikit-learn SyntheticControlMethods scipy==1.4.1 pycountry

import requests
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import pycountry
#from SyntheticControlMethods import Synth

### GLOBAL CONSTANTS ###

FREQUENCY = 'A' # 'M' for monthly, 'Q' for quarterly, 'A' for annual

# IMF Data API Indicator Codes
GDP = 'NGDP_XDC' if FREQUENCY == 'A' else 'NGDP_NSA_XDC'
CPI = 'PCPI_IX'
POLICY_RATE = 'FPOLM_PA'
SAVINGS_RATE = 'FISR_PA'
LENDING_RATE = 'FILR_PA'
DEPOSITS_RATE = 'FIDR_PA'
TREASURY_BILL_RATE = 'FITB_PA'
EXCHANGE_RATE = 'ENDE_XDC_USD_RATE'

# Other Indicators
M2_GROWTH_RATE = 'FM2_CH'
M2_TO_GDP_RATIO = 'FM2_NGDP'
M2_TO_RESERVES_RATIO = 'FM2_RES'
INTEREST_RATE_SPREAD = 'IR_SPREAD' # Lending Rate - Deposits Rate
BANK_PREMIUM_RATE = 'BANK_PREM_RATE' # Lending Rate - Treasury Bill Rate

REAL_GDP = GDP + '_R'
REAL_SAVINGS_RATE = SAVINGS_RATE + '_R'
REAL_LENDING_RATE = LENDING_RATE + '_R'
REAL_DEPOSITS_RATE = DEPOSITS_RATE + '_R'
REAL_TREASURY_BILL_RATE = TREASURY_BILL_RATE + '_R'

INFLATION_RATE = CPI + '_CH'
GDP_GROWTH_RATE = GDP + '_CH'
REAL_GDP_GROWTH_RATE = REAL_GDP + '_CH'
EXCHANGE_RATE_CHANGE = EXCHANGE_RATE + '_CH'

DEMEAN_POLICY_RATE = POLICY_RATE + '_DM'
DEMEAN_SAVINGS_RATE = SAVINGS_RATE + '_DM'
DEMEAN_LENDING_RATE = LENDING_RATE + '_DM'
DEMEAN_DEPOSITS_RATE = DEPOSITS_RATE + '_DM'
DEMEAN_TREASURY_BILL_RATE = TREASURY_BILL_RATE + '_DM'
DEMEAN_M2_GROWTH_RATE = M2_GROWTH_RATE + '_DM'
DEMEAN_M2_TO_GDP_RATIO = M2_TO_GDP_RATIO + '_DM'
DEMEAN_M2_TO_RESERVES_RATIO = M2_TO_RESERVES_RATIO + '_DM'
DEMEAN_INFLATION_RATE = INFLATION_RATE + '_DM'
DEMEAN_GDP_GROWTH_RATE = GDP_GROWTH_RATE + '_DM'
DEMEAN_EXCHANGE_RATE_CHANGE = EXCHANGE_RATE_CHANGE + '_DM'
DEMEAN_INTEREST_RATE_SPREAD = INTEREST_RATE_SPREAD + '_DM'
DEMEAN_BANK_PREMIUM_RATE = BANK_PREMIUM_RATE + '_DM'

DEMEAN_REAL_SAVINGS_RATE = REAL_SAVINGS_RATE + '_DM'
DEMEAN_REAL_LENDING_RATE = REAL_LENDING_RATE + '_DM'
DEMEAN_REAL_DEPOSITS_RATE = REAL_DEPOSITS_RATE + '_DM'
DEMEAN_REAL_TREASURY_BILL_RATE = REAL_TREASURY_BILL_RATE + '_DM'
DEMEAN_REAL_GDP_GROWTH_RATE = REAL_GDP_GROWTH_RATE + '_DM'

INDICATORS_TO_DEMEAN = [
    DEPOSITS_RATE, GDP_GROWTH_RATE, INFLATION_RATE, TREASURY_BILL_RATE, LENDING_RATE,
    SAVINGS_RATE, POLICY_RATE, EXCHANGE_RATE_CHANGE, INTEREST_RATE_SPREAD, BANK_PREMIUM_RATE,
    M2_GROWTH_RATE, M2_TO_GDP_RATIO, M2_TO_RESERVES_RATIO, REAL_DEPOSITS_RATE,
    REAL_GDP_GROWTH_RATE, REAL_TREASURY_BILL_RATE, REAL_LENDING_RATE, REAL_SAVINGS_RATE
]

INDICATORS_TO_REAL = [
    DEPOSITS_RATE, GDP, TREASURY_BILL_RATE, LENDING_RATE, SAVINGS_RATE
]

INDICATORS_TO_GROWTH = [
    GDP, REAL_GDP, EXCHANGE_RATE
]

INDICATORS_IN_FILES = {
    M2_GROWTH_RATE: 'data/broad-money-growth-dataset.csv',
    M2_TO_GDP_RATIO: 'data/broad-money-to-gdp-dataset.csv.csv',
    M2_TO_RESERVES_RATIO: 'data/broad-money-to-reserves-dataset.csv'
}

### FUNCTIONS ###

def get_country_name(iso2_code):
    country = pycountry.countries.get(alpha_2=iso2_code)
    return country.name if country else None

def get_country_iso3_code(iso2_code):
    country = pycountry.countries.get(alpha_2=iso2_code)
    return country.alpha_3 if country else None

def get_country_iso2_code(iso3_code):
    country = pycountry.countries.get(alpha_3=iso3_code)
    return country.alpha_2 if country else None

def calculate_growth_rate(values):
    '''Calculate the year-over-year growth rate, handling NaN values.
    The first year will have a NaN growth rate since there is no previous year to compare to.

    :param values: List of values
    :return: List of growth rates, with NaN for missing or invalid values
    '''
    growth_rates = [np.nan]  # Initialize with NaN since the first year does not have a previous year to compare to.
    for i in range(1, len(values)):
        # Calculate growth rate only if both current and previous values are not NaN.
        if not np.isnan(values[i]) and not np.isnan(values[i-1]) and values[i-1] != 0:
            growth_rate = ((values[i] - values[i-1]) / values[i-1]) * 100
        else:
            growth_rate = np.nan
        growth_rates.append(growth_rate)
    return growth_rates

def convert_dates_to_number(dates):
    '''
    Convert a date string in the format 'YYYY', 'YYYY-QX', or 'YYYY-XX' to a number.
    E.g., '2010-Q1' will be converted to 2010.0, '2010-Q2' to 2010.25, etc.

    :param dates: Pandas Series containing date strings in the format 'YYYY', 'YYYY-QX', or 'YYYY-XX'
    :param FREQUENCY: A character indicating the FREQUENCY ('A' for annual, 'Q' for quarterly, 'M' for monthly)
    :return: Pandas Series of numbers representing the dates
    '''
    if FREQUENCY == 'A':
        # Directly convert year strings to integers
        return dates.astype(int)
    elif FREQUENCY == 'Q':
        # Split the year and quarter, then calculate the decimal year
        year_quarter = dates.str.split('-Q', expand=True)
        return year_quarter[0].astype(int) + (year_quarter[1].astype(int) - 1) / 4.0
    elif FREQUENCY == 'M':
        # Split the year and month, then calculate the decimal year
        year_month = dates.str.split('-', expand=True)
        return year_month[0].astype(int) + round((year_month[1].astype(int) - 1) / 12.0, 2)

def calculate_moving_average(values, window=4):
    '''
    Calculate the moving average for a given list of values.

    :param values: List of tuples (time, value)
    :param window: The number of periods over which to calculate the moving average
    :return: List of tuples (time, moving_average)
    '''
    ma_values = []
    # Convert list of tuples to a dictionary for easier manipulation
    value_dict = dict(values)
    times = [time for time, _ in values]

    for i in range(window - 1, len(times)):
        # Calculate average of the window
        window_values = [value_dict[times[j]] for j in range(i-window+1, i+1) if times[j] in value_dict]
        if len(window_values) == window:
            average = sum(window_values) / window
        else:
            average = np.nan  # Assign NaN if the window is not full (e.g., beginning periods)
        ma_values.append((times[i], average))

    return ma_values

def convert_nominal_to_real(indicator, nominal_values, inflation_rates, base_index=100):
    '''
    Converts nominal values to real values using the moving average of the Consumer Price Index (CPI).

    :param indicator: The indicator for which to convert nominal values to real values (e.g., NGDP_XDC, FITB_PA)
                      Different indicators have different formulas for calculating real values.
    :param nominal_values: List of tuples (time, nominal_value)
    :param inflation_rates: List of tuples (time, inflation_rate)
    :param base_index: The index value of the base year (usually 100)
    :return: List of tuples (time, real_value)
    '''
    # First, calculate the moving average for the CPI
    ma_inflation_rates = calculate_moving_average(inflation_rates, window=4)
    inflation_rates_dict = dict(ma_inflation_rates)  # Convert list of tuples to a dictionary

    real_values = []
    for time, nominal_value in nominal_values:
        if time in inflation_rates_dict and inflation_rates_dict[time] != 0:
            inflation_rate = inflation_rates_dict[time]
            if indicator == GDP:
                real_value = (nominal_value / (1 + inflation_rate/100)) * base_index
            elif indicator in [TREASURY_BILL_RATE, LENDING_RATE, SAVINGS_RATE, DEPOSITS_RATE]:
                real_value = nominal_value - inflation_rate
            else:
                print(f'ERROR: Indicator {indicator} not supported for real value calculation.')
                real_value = np.nan
        else:
            real_value = np.nan  # Assign NaN if no corresponding CPI value or CPI is zero
        real_values.append((time, real_value))

    return real_values

def calculate_rolling_volatility(values, window=8):
    '''
    Calculate rolling volatility using a window of the specified number of periods.

    :param values: List of tuples (time, value)
    :param window: The number of periods over which to calculate the rolling standard deviation
    :return: List of tuples (time, rolling_volatility)
    '''
    # Convert values to DataFrame for easier manipulation
    df = pd.DataFrame(values, columns=['time', 'value'])
    df.set_index('time', inplace=True)

    # Calculate rolling standard deviation
    rolling_std = df['value'].rolling(window=window, min_periods=1).std()

    # Return the result as a list of tuples
    return list(rolling_std.reset_index().itertuples(index=False, name=None))


### CONFIGURATION ###

start_period = '2010'
event_period = '2020'
end_period = '2023'
if FREQUENCY == 'Q':
    start_period += '-Q1'
    event_period += '-Q3'
    end_period += '-Q1'
elif FREQUENCY == 'M':
    start_period += '-01'
    event_period += '-10'
    end_period += '-01'

inputs = [
    POLICY_RATE,
    SAVINGS_RATE,
    LENDING_RATE,
    TREASURY_BILL_RATE,
    M2_GROWTH_RATE,
    M2_TO_GDP_RATIO,
    M2_TO_RESERVES_RATIO,
    INFLATION_RATE,
    GDP_GROWTH_RATE,
    EXCHANGE_RATE_CHANGE,
    INTEREST_RATE_SPREAD,
    BANK_PREMIUM_RATE
]
demeaned_inputs = [
    DEMEAN_POLICY_RATE,
    DEMEAN_SAVINGS_RATE,
    DEMEAN_LENDING_RATE,
    DEMEAN_TREASURY_BILL_RATE,
    DEMEAN_M2_GROWTH_RATE,
    DEMEAN_M2_TO_GDP_RATIO,
    DEMEAN_M2_TO_RESERVES_RATIO,
    DEMEAN_INFLATION_RATE,
    DEMEAN_GDP_GROWTH_RATE,
    DEMEAN_EXCHANGE_RATE_CHANGE,
    DEMEAN_INTEREST_RATE_SPREAD,
    DEMEAN_BANK_PREMIUM_RATE
]
output = DEPOSITS_RATE
demeaned_output = DEMEAN_DEPOSITS_RATE

# The Bahamas
treatment_unit = 'BS'
control_units = [
    # Fixed Monetary Policy
    'AG', 'BB', 'BZ', 'DM', 'GD',
    'KN', 'VC', 'LC', 'MS', 'AI',
    # Flexible Monetary Policy
    'TT', 'JM', 'FJ', 'MU', 'DO'
]

# # Nigeria
# treatment_unit = 'NG'

# # Without interest rates
# control_units = [
#     'KE', 'GH', 'BD', 'ID', 'RW',
#     'MZ', 'EG', 'ZA', 'DZ', 'MA',
#     'AO', 'CD', 'GN', 'BF', 'BW',
#     'BJ', 'MG', 'MU', 'NA', 'SL',
#     'CV', 'SC', 'KM'
# ]

# With interest rates
# control_units = [
#     'KE', 'GH', 'BD', 'EG', 'ZA',
#     'AO', 'MU', 'SL', 'CV'
# ]

# # With interest rates, no Central Bank
# control_units = [
#     'KE', 'GH', 'BD', 'RW',
#     'MZ', 'EG', 'ZA', 'DZ',
#     'AO', 'MG', 'MU', 'NA',
#     'SL', 'CV', 'SC'
# ]

# Quarterly - With interest rates, no Central Bank
# control_units = [
#     'KE', 'GH', 'RW',
#     'EG', 'ZA', 'DZ',
#     'MU', 'NA', 'SC'
# ]

countries = [treatment_unit] + control_units

country_code = {country: index for index, country in enumerate(countries, start=1)}

### GET DATA FROM IMF DATA API ###

imf_indicators = [
    GDP,
    CPI,
    POLICY_RATE,
    SAVINGS_RATE,
    LENDING_RATE,
    DEPOSITS_RATE,
    TREASURY_BILL_RATE,
    EXCHANGE_RATE
]

imf_indicators_key = '+'.join(imf_indicators)
imf_countries_key = '+'.join(countries)

imf_base_url = 'http://dataservices.imf.org/REST/SDMX_JSON.svc/'
imf_request_key = f'CompactData/IFS/{frequency}.{imf_countries_key}.{imf_indicators_key}?startPeriod={start_period}&endPeriod={end_period}'
imf_request_url = f'{imf_base_url}{imf_request_key}'

try:
    response = requests.get(imf_request_url)
except Exception as e:
    print('Failed to retrieve data using the IMF Data API; Error:', e)
    sys.exit()
if response.status_code != 200:
    print(f'Failed to retrieve data using the IMF Data API; Response code: {response.status_code}')
    sys.exit()

print('Successfully retrieved data using the IMF Data API')

data = {}
for country in countries:
    data[country] = {}

try:
    series = response.json()['CompactData']['DataSet']['Series']

    for s in series:
        country = s['@REF_AREA']
        indicator = s['@INDICATOR']

        if 'Obs' not in s:
            print(f'No data found for {country}.{indicator}')
            continue

        print(f'{country}.{indicator} has {len(s["Obs"])} observations')

        observations = s['Obs']
        times = [obs.get('@TIME_PERIOD') for obs in observations]
        values = [float(obs.get('@OBS_VALUE')) if obs.get('@OBS_VALUE') else np.nan for obs in observations]

        data[country][indicator] = list(zip(times, values))

except Exception as e:
    print('Failed to process data; Error:', e)
    sys.exit()

# Add missing data manually (if needed)
if treatment_unit == 'BS':
    if frequency == 'A':
        if 'BB' in data:
            # Add latest value, 2023, manually
            data['BB']['FIDR_PA'].append(('2023', 0.15))
    elif frequency == 'Q':
        if 'BB' in data:
            data['BB']['FIDR_PA'].append(('2023-Q1', 0.15))
            data['BB']['FIDR_PA'].append(('2023-Q2', 0.15))
            data['BB']['FIDR_PA'].append(('2023-Q3', 0.15))
        if 'TT' in data:
            data['TT']['FIDR_PA'].append(('2022-Q1', 1.5))
            data['TT']['FIDR_PA'].append(('2022-Q2', 1.5))
            data['TT']['FIDR_PA'].append(('2022-Q3', 1.5))

# if treatment_unit == 'NG':
#     # Add manual data for Ghana Lending Rates from https://www.focus-economics.com/country-indicator/dominica/interest-rate/
#     if frequency == 'A':
#         data['GH']['FILR_PA'] = [
#             ('2010', 31), ('2011', 27), ('2012', 26), ('2013', 28),
#             ('2014', 31), ('2015', 36.5), ('2016', 40), ('2017', 37),
#             ('2018', 35)
#         ]
#     else:
#         data['GH']['FILR_PA'] = [
#             ('2010-Q1', 31), ('2010-Q2', 31), ('2010-Q3', 31), ('2010-Q4', 31),
#             ('2011-Q1', 27), ('2011-Q2', 27), ('2011-Q3', 27), ('2011-Q4', 27),
#             ('2012-Q1', 26), ('2012-Q2', 26), ('2012-Q3', 26), ('2012-Q4', 26),
#             ('2013-Q1', 28), ('2013-Q2', 28), ('2013-Q3', 28), ('2013-Q4', 28),
#             ('2014-Q1', 31), ('2014-Q2', 31), ('2014-Q3', 31), ('2014-Q4', 31),
#             ('2015-Q1', 36.5), ('2015-Q2', 36.5), ('2015-Q3', 36.5), ('2015-Q4', 36.5),
#             ('2016-Q1', 40), ('2016-Q2', 40), ('2016-Q3', 40), ('2016-Q4', 40),
#             ('2017-Q1', 37), ('2017-Q2', 37), ('2017-Q3', 37), ('2017-Q4', 37),
#             ('2018-Q1', 35), ('2018-Q2', 35), ('2018-Q3', 35), ('2018-Q4', 35)
#         ]

# Get indicators not available through the API from CSV files
for indicator, file_path in INDICATORS_IN_FILES.items():
    # Load the CSV file, skip initial rows
    file_data = pd.read_csv(file_path, skiprows=2)

    # Filter the data for the relevant countries
    countries_iso3 = [get_country_iso3_code(code) for code in countries]
    file_data = file_data[file_data['Country Code'].isin(countries_iso3)]

    # Filter the data for the relevant years
    years = [str(year) for year in range(int(start_period), int(end_period) + 1)]
    filtered_data = file_data[['Country Code'] + years]

    for index, row in filtered_data.iterrows():
        country_iso3 = row['Country Code']
        country_iso2 = get_country_iso2_code(country_iso3)
        country_data = []
        for year in years:
            country_data.append((year, row[year]))

        data[country_iso2][indicator] = country_data

        print(f'{country_iso2}.{indicator} has {len(data[country_iso2][indicator])} observations')


### REFINE DATA ###

for country in countries:
    # Calculate the inflation rate
    if CPI in data[country]:
        times = [obs[0] for obs in data[country][CPI]]
        values = calculate_growth_rate([obs[1] for obs in data[country][CPI]])
        data[country][INFLATION_RATE] = list(zip(times, values))
    # Calculate the interest rate spread -> Lending Rate - Deposits Rate
    if LENDING_RATE in data[country] and DEPOSITS_RATE in data[country]:
        lending_rates = dict(data[country][LENDING_RATE])
        deposits_rates = dict(data[country][DEPOSITS_RATE])
        interest_rate_spread = [(time, lending_rates[time] - deposits_rates.get(time, np.nan)) for time in lending_rates]
        data[country][INTEREST_RATE_SPREAD] = interest_rate_spread
    # Calculate the bank premium rate -> Lending Rate - Treasury Bill Rate
    if TREASURY_BILL_RATE in data[country] and LENDING_RATE in data[country]:
        treasury_bill_rates = dict(data[country][TREASURY_BILL_RATE])
        lending_rates = dict(data[country][LENDING_RATE])
        bank_premium_rate = [(time, lending_rates[time] - treasury_bill_rates.get(time, np.nan)) for time in lending_rates]
        data[country][BANK_PREMIUM_RATE] = bank_premium_rate

# Calculate the real values for GDP, Treasury Bills, Savings Rate, Lending Rate, Deposits Rate
# This is an approximation using the moving average of the inflation rate
for indicator in INDICATORS_TO_REAL:
    for country in countries:
        if indicator in data[country] and INFLATION_RATE in data[country]:
            data[country][indicator + '_R'] = convert_nominal_to_real(indicator, data[country][indicator], data[country][INFLATION_RATE])

# Calculate the change rates for GDP, Real GDP, and Exchange Rate
for indicator in INDICATORS_TO_GROWTH:
    for country in countries:
        print(f'Calculating growth rate for {country}.{indicator}')
        if indicator in data[country]:
            times = [obs[0] for obs in data[country][indicator]]
            values = calculate_growth_rate([obs[1] for obs in data[country][indicator]])
            data[country][indicator + '_CH'] = list(zip(times, values))
            print(f'{country}.{indicator} has {len(data[country][indicator])} observations')

# # Calculate rolling volatility for the Interest Rate on Deposits
# for country in countries:
#     if REAL_DEPOSITS_RATE in data[country]:
#         # Extract the real deposit rates and times
#         values = data[country][REAL_DEPOSITS_RATE]
#         # Calculate and store the rolling volatility
#         values = calculate_rolling_volatility(values)
#         data[country][REAL_DEPOSITS_RATE + '_VOL'] = values
#     else:
#         real_deposit_rate_rolling_volatility[country] = []  # No data for this indicator

# Demean the data (only based on the prediction and optimization period)
for country in countries:
    for indicator in INDICATORS_TO_DEMEAN:
        print(f'Calculating demean for {country}.{indicator}')

        if indicator in data[country]:
            # Filter values within the specified time window, between the start date and the event date, and calculate the mean
            window_values = [value for year, value in data[country][indicator] if start_period <= year <= event_period]
            mean_value = np.nanmean(window_values)

            # Demean all values using the calculated mean from the window
            demeaned_values = [(year, value - mean_value) for year, value in data[country][indicator]]

            # Update the data structure with demeaned values
            data[country][indicator + '_DM'] = demeaned_values
            print(f'{country}.{indicator} has {len(data[country][indicator])} observations')

# Demean the data (based on the entire period)
# for country in countries:
#     for indicator in INDICATORS_TO_DEMEAN:
#         print(f'Calculating demean for {country}.{indicator}')

#         if indicator in data[country]:
#             values = [obs[1] for obs in data[country][indicator]]
#             mean_value = np.nanmean(values)
#             demeaned_values = [(obs[0], obs[1] - mean_value) for obs in data[country][indicator]]
#             data[country][indicator + '_DM'] = demeaned_values
#             print(f'{country}.{indicator} has {len(data[country][indicator])} observations')


### PLOT DATA ###

# Create a separate figure for each indicator
for indicator in [demeaned_output] + demeaned_inputs:
    plotted_data = False
    plt.figure(figsize=(10, 6))  # Larger figure for better visibility

    for country in countries:
        country_name = get_country_name(country)

        # Check if the indicator exists for the country
        if indicator in data[country]:
            plotted_data = True

            # Prepare the data, transforming the date format
            times = [pd.to_datetime(obs[0]) for obs in data[country][indicator]]
            values = [float(obs[1]) if obs[1] else np.nan for obs in data[country][indicator]]

            # Plot the data for the country
            plt.plot(times, values, linewidth=2, markersize=8, label=country_name)

    if plotted_data:
        plt.title(f'{indicator}')
        plt.xlabel('Year')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
    else:
        plt.close()  # Close the plot if no data is present to avoid empty figures

    # Uncomment the next line if you want to save the figure to a file
    # plt.savefig(f"{indicator}_plot.png")

    plt.show()

# # Plot indicators into a tiled grid
# # Determine the layout of the subplot grid
# num_indicators = len([demeaned_output] + demeaned_inputs)
# num_columns = 2  # Or any other number you wish
# num_rows = (num_indicators + num_columns - 1) // num_columns  # This ensures enough rows

# plt.figure(figsize=(5 * num_columns, 4 * num_rows))

# i = 1
# # Create a subplot for each indicator
# for indicator in [demeaned_output] + demeaned_inputs:
#     ax = plt.subplot(num_rows, num_columns, i)
#     i += 1

#     # Track if we have added any data to plot (for cases where an indicator may not be present)
#     plotted_data = False

#     for country in countries:
#         # Check if the indicator exists for the country
#         if indicator in data[country]:
#             plotted_data = True

#             # Prepare the data, transforming the date format
#             times = [pd.to_datetime(obs[0]) for obs in data[country][indicator]]
#             values = [float(obs[1]) if obs[1] else np.nan for obs in data[country][indicator]]

#             # Plot the data for the country
#             ax.plot(times, values, marker='o', label=COUNTRY_NAME.get(country, country))

#     # Customize the subplot
#     if plotted_data:
#         ax.set_title(indicator)
#         ax.set_xlabel('Year')
#         ax.set_ylabel('Value')
#         ax.grid(True)
#         # ax.legend()
#     else:
#         plt.delaxes(ax)  # Remove the subplot if no data is present for the indicator

# # Adjust the layout to prevent overlapping
# plt.tight_layout()
# plt.show()


### FORMAT DATA AND EXPORT TO CSV

rows_list = []

# Iterate over each country and its corresponding data
for country, indicators in data.items():
    # Prepare a nested dictionary for each indicator by year
    year_data = {}
    for indicator, observations in indicators.items():
        for obs in observations:
            year, value = obs
            value = float(value) if value else np.nan  # Ensure value is float or NaN
            if year not in year_data:
                year_data[year] = {}
            year_data[year][indicator] = value

    # Now, flatten the year_data dictionary into rows for our DataFrame
    for year, indicators in year_data.items():
        row = {
            'code': country_code[country],
            'country': country,
            'year': year,
        }
        # Add each indicator's value to the row; if an indicator is missing, the value will be NaN
        row.update(indicators)
        rows_list.append(row)

# Convert the list of rows into a DataFrame
dataframe = pd.DataFrame(rows_list)

# Ensure all indicators are present as columns, filling missing ones with NaN
final_columns = ['code', 'country', 'year'] + [output] + inputs + [demeaned_output] + demeaned_inputs
for col in final_columns:
    if col not in dataframe.columns:
        dataframe[col] = np.nan

# Reorder the DataFrame columns as needed
dataframe = dataframe[final_columns]

# Transform date to int or float format according to frequency
dataframe['year'] = convert_dates_to_number(dataframe['year'])

# Optionally, sort the DataFrame by country and year for better readability
dataframe.sort_values(by=['code', 'year'], inplace=True)

# Save the DataFrame to a CSV file
dataframe.to_csv('data/synthetic-control-dataset.csv', index=False)

print("CSV file created successfully.")


### SYNTHETIC CONTROL

# #Import data
# file_path = 'data/synthetic-control-dataset.csv'
# data = pd.read_csv(file_path)
# data = data.drop(columns="code", axis=1)

# #Fit classic Synthetic Control
# sc = Synth(data, "FIDR_PA", "country", "year", 2020.75, "BS", pen=0)
# # dsc = DiffSynth(data, "FIDR_PA", "country", "year", 2020, "BS", not_diff_cols=["NGDP_XDC","PCPI_IX","FITB_PA","FISR_PA","FILR_PA"], pen="auto")

# print("\nSynthetic Control Wieghts")
# print(sc.original_data.weight_df)

# #Visualize synthetic control
# sc.plot(["original", "pointwise", "cumulative"], treated_label="The Bahamas", synth_label="Synthetic The Bahamas", treatment_label="Introduction of CBDC")
