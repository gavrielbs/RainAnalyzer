# Final Project - Rain Analyzer

One of the questions raised in the lab lately is how patterns of rain affect ecosystem productivity.
Rain plays a key role and its effects on plants ecosystems can be seen in the following year and even in the year after.  
In order to understand how the rain affects the productivity of plants, it's yearly seasonality characteristics are needed to be extracted.
In this project I will analyze time series of rain, downloaded from the Israel Meteorological Service's API.

Input data:
A .csv file in the 'input' directory. 
The file should have the columns 'date' and 'Rain' (in mm). 
The 'date' should be in the format '%Y-%m-%d'.

Output data:
'yearly_rain_data_df.csv' file in the 'output' directory, containing rain seasonality data for each year presented in the input file.
The extracted characteristics are: 
- hydro_year : the hydrological year, starting on September 1st of the corresponding year
- rain_pulse_peak: yearly cumulative rain (mm)
- rain_pulse_peak_1_years_before: yearly cumulative rain of the previous year (mm)
- rain_pulse_peak_2_years_before: yearly cumulative rain of the 2 years before (mm)
- rain_left_deriv: rate of accumulation (mm\day)
- rain_season_duration: season duration (days)
- rain_season_start_from_hydro: number of days from the beginning of the hydrological year (September 1st) to the beginning of the rain season (days)
- rain_season_end_from_hydro: number of days from the beginning of the hydrological year (September 1st) to the ending of the rain season (days)
- rain_consecutive_dry_days: consecutive dry days between rain seasons (days)
- avg_rain_per_rainy_day: average rain per rainy day (mm\day)
- rain_dispersion: average rain per day in the rain season (mm\day)
- rain_cdd_within_season: longest consecutive dry days within rain seasons (days)

Plots:
- heatmap of pearson correlation matrix between all the rain parameters
- linear regression of each rain parameter over time

Step 1: Install all required packages:
> pip install os
> pip install pandas
> pip install numpy
> pip install datetime
> pip install seaborn
> pip install matplotlib
> pip install sklearn
> pip install statsmodels

 Gabriel Bar-Sella: [https://gavrielbs.github.io/](https://gavrielbs.github.io/)
 
