import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from rain_modules import (
    load_rain_data,
    calc_cumulative_rain,
    calc_rate_of_rain_accumulation,
    calc_start_end_rain_season,
    calc_rain_season_duration,
    calc_rain_season_start_end_from_hydro_year,
    calc_cdd,
    calc_avg_rain_per_rainy_day,
    calc_rain_dispersion,
    calc_cdd_within_season,
    plot_the_data,
    plot_correlation_matrix,
    run_regressions
    )

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(SCRIPT_DIR, 'input/')
OUTPUT_PATH = 'output/'
OUTPUT_FILE = 'yearly_rain_data_df'
RAIN_BASE_VALUE = 1

def main():
    for filename in os.listdir(INPUT_PATH):
        rain_data_file = os.path.join(INPUT_PATH, filename) 

        ## load and filter IMS data file
        rain_data = load_rain_data(rain_data_file)

        ## initialize dataframe for yaerly rain data
        yearly_rain_data_df = pd.DataFrame()
        hydro_years = [year for year in rain_data['hydro_year'].unique()]
        yearly_rain_data_df['hydro_year'] = hydro_years

        ## calculate peak rain - cumulative rain in each hydrological year
        print('calculate peak rain')
        rain_data, yearly_rain_data_df = calc_cumulative_rain(rain_data, yearly_rain_data_df)

        ## calculate left deriv of cumulative precipitation
        print('calculate rain left deriv')
        yearly_rain_data_df = calc_rate_of_rain_accumulation(rain_data, yearly_rain_data_df)

        ## extract start and end date of rain season
        print('extract start and end date of rain season')
        yearly_rain_data_df = calc_start_end_rain_season(rain_data, RAIN_BASE_VALUE, yearly_rain_data_df)

        ## calculate the rain season duration
        print('calculate the rain season duration')
        yearly_rain_data_df = calc_rain_season_duration(yearly_rain_data_df)

        ## number of days from start of hydro year to the start of rain season adn to the end of rain season
        print('number of days from start of hydro year')
        yearly_rain_data_df = calc_rain_season_start_end_from_hydro_year(yearly_rain_data_df)

        ## calc CDD - consecutive dry days  from end of last rain season to the start of the current
        print('consecutive dry days')
        yearly_rain_data_df = calc_cdd(yearly_rain_data_df)

        ## calc SDII - total precipitation/number of rainy days - average rain per rainy day
        print('average rain per rainy day')
        yearly_rain_data_df = calc_avg_rain_per_rainy_day(rain_data, yearly_rain_data_df)

        ## rain dispersion - number of rainy days / season duration, where rainy day means rain > 0
        print('rain dispersion')
        yearly_rain_data_df = calc_rain_dispersion(rain_data, yearly_rain_data_df)

        ## longest period of consecutive dry days within the reain season
        print('longest period of cdd within the reain season')
        yearly_rain_data_df = calc_cdd_within_season(rain_data, yearly_rain_data_df)

        # plot the rain over time
        plot_the_data(rain_data, yearly_rain_data_df, SCRIPT_DIR)

        # plot the correlation matrixof all the features
        plot_correlation_matrix(yearly_rain_data_df, SCRIPT_DIR)

        #plot the regression analysis of each feature over time
        run_regressions(yearly_rain_data_df, SCRIPT_DIR)

        yearly_rain_data_df = yearly_rain_data_df.drop(columns=['rain_season_start', 'rain_season_end'])
        
        output_file_path = os.path.join(INPUT_PATH, OUTPUT_FILE) 

        yearly_rain_data_df.to_csv(output_file_path, index=False)

        print('saved')

if __name__ == "__main__":
    main()
