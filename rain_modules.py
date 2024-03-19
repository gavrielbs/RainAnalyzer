import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

from tests import (
    check_date_column_existance,
    check_rain_column_existance,
    check_rain_column_format,
    check_date_column_format
)

def calc_hydrological_year(original_date, format):
    ## function to calculate hydrological year based on date
    
    if type(original_date) == str:
        formated_date = datetime.strptime(original_date, format)

        if datetime(formated_date.year, 9, 1) <= formated_date <= datetime(formated_date.year, 12, 31):
            return formated_date.year + 1

        elif datetime(formated_date.year, 1, 1) <= formated_date <= datetime(formated_date.year, 8, 31):
            return formated_date.year
    else:
        formated_date = original_date
        if datetime(formated_date.year, 9, 1) <= formated_date <= datetime(formated_date.year, 12, 31):
            return formated_date.year + 1

        elif datetime(formated_date.year, 1, 1) <= formated_date <= datetime(formated_date.year, 8, 31):
            return formated_date.year

def load_rain_data(rain_data_file_path):
    ## loads the ims meterorological data, keeps only daily rain data starting from 2015 to 2023 including
    df = pd.read_csv(rain_data_file_path)
    # test formats
    check_date_column_existance(df)
    check_rain_column_existance(df)
    check_rain_column_format(df)
    check_date_column_format(df)
    
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df=df.dropna()
    df.loc[:, 'hydro_year'] = df['date'].apply(calc_hydrological_year ,format="%Y-%m-%d")
    # df = df.loc[(df['hydro_year'] >= 2000) & (df['hydro_year']<2024)]
    df.reset_index(inplace=True)
    df = df[['date', 'hydro_year', 'Rain']]
    df = df.groupby(by=['date','hydro_year'])['Rain'].sum().reset_index()
    ## convert the date to julian day (numerical instead of date) for regression amalysis
    df['julian_day'] = pd.DatetimeIndex(df['date']).to_julian_date()
    print('rain data loaded\n')
    return df

def calc_cumulative_rain(rain_data_df, yearly_rain_data_df):
    ## function to calculate the cumulative rain of each hydrological year and store it in yearly_rain_data_df.
    ## the function also calculate peak values of the previous year and two years before the current
    hydro_years = rain_data_df['hydro_year'].unique()
    print(hydro_years)

    yearly_peak_rain = []
    for ind, year in enumerate(hydro_years):
        cumu_rain = rain_data_df.loc[rain_data_df['hydro_year'] == year,'Rain'].cumsum()
        rain_data_df.loc[rain_data_df['hydro_year'] == year, 'cumulative'] = cumu_rain
        pulse_peak = round(cumu_rain.values[-1],1)
        yearly_peak_rain.append(pulse_peak)
        yearly_rain_data_df.loc[yearly_rain_data_df['hydro_year'] == year,'rain_pulse_peak'] = pulse_peak
        print(f'{year} - pulse peak {pulse_peak}')

         # Identify the point where the rain stops
        cumu_rain_rounded = np.round(cumu_rain, 1)
        rain_stop_index = np.argmax(np.array(cumu_rain_rounded) >= pulse_peak)  # Adjust 0.8 based on your criteria
        cumu_rain.iloc[rain_stop_index+1:] = np.nan
        rain_data_df.loc[rain_data_df['hydro_year'] == year, 'cumulative_with_stop'] = cumu_rain

        ## calc previous years peack values
        if year >= hydro_years[2]:
            pulse_peak_one_year_before = yearly_peak_rain[ind-1]
            pulse_peak_two_year_before = yearly_peak_rain[ind-2]
            yearly_rain_data_df.loc[yearly_rain_data_df['hydro_year'] == year,'rain_pulse_peak_1_years_before'] = pulse_peak_one_year_before
            yearly_rain_data_df.loc[yearly_rain_data_df['hydro_year'] == year,'rain_pulse_peak_2_years_before'] = pulse_peak_two_year_before
            print(f'pulse peak 1 years before - {pulse_peak_one_year_before}')
            print(f'pulse peak 2 years before - {pulse_peak_two_year_before}')

    rain_data_df.fillna(0, inplace=True)
    print('\n')
    return rain_data_df, yearly_rain_data_df

def calc_rate_of_rain_accumulation(rain_data_df, yearly_rain_data_df):
    ## calculate the left derivative of the cumulative rain, between the values of 20% of peak to 80% of peak 
    ## and stores in yearly_rain_data_df

    for year in rain_data_df['hydro_year'].unique():
        max_val = rain_data_df[rain_data_df['hydro_year'] == year]['cumulative'].max()
        values_mask = (rain_data_df['cumulative'] >= max_val*0.2) & (rain_data_df['cumulative'] <= max_val*0.8)
        rain_data_filtered = rain_data_df.loc[(rain_data_df['hydro_year'] == year) & values_mask]
        Y = np.asarray(rain_data_filtered['cumulative'])
        X = np.asarray(rain_data_filtered['julian_day'])
        regr = LinearRegression().fit(X.reshape(-1, 1), Y.reshape(-1, 1))
        coef_ = round(regr.coef_[0][0],3)
        yearly_rain_data_df.loc[yearly_rain_data_df['hydro_year'] == year,'rain_left_deriv'] = coef_

        # ax = sns.regplot(x=X, y=Y, ci=95, label=f'R={r}\nR^2={r2}\npv={pv}\nlinear eqaution: {eqaution}')
        # xdata = ax.get_lines()[0].get_xdata()
        # ydata = ax.get_lines()[0].get_ydata()
        # slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=xdata,y=ydata)

        print(year, 'slope -', coef_)
    print('\n')
    return yearly_rain_data_df

def calc_start_end_rain_season(rain_data_df, rain_base_value, yearly_rain_data_df):
    ## get the first and last day of rain season, based on rain_base_value, assuming the 
    ## season has started after the rain_base_value and ended when daily rain was lower
    for year in rain_data_df['hydro_year'].unique():
        rain_season_start = rain_data_df.loc[(rain_data_df['hydro_year']==year) & (rain_data_df['Rain']>rain_base_value), 'date'].values[0]
        rain_season_end = rain_data_df.loc[(rain_data_df['hydro_year']==year) & (rain_data_df['Rain']>rain_base_value), 'date'].values[-1]
        yearly_rain_data_df.loc[yearly_rain_data_df['hydro_year'] == year,'rain_season_start'] = rain_season_start
        yearly_rain_data_df.loc[yearly_rain_data_df['hydro_year'] == year,'rain_season_end'] = rain_season_end
        print(f'{year} : season start date - {pd.to_datetime(rain_season_start)}, season end date - {pd.to_datetime(rain_season_end)}')
    print('\n')
    return yearly_rain_data_df

def calc_rain_season_duration(yearly_rain_data_df):
    ## calculates the season duration from rain_season_start to rain_season_end
    ## and stores in yearly_rain_data_df
    for year in yearly_rain_data_df['hydro_year'].unique():
        rain_season_start = yearly_rain_data_df.loc[(yearly_rain_data_df['hydro_year']==year) , 'rain_season_start'].values[0]
        rain_season_start =  pd.to_datetime(rain_season_start)
        rain_season_end = yearly_rain_data_df.loc[(yearly_rain_data_df['hydro_year']==year) , 'rain_season_end'].values[0]
        rain_season_end =  pd.to_datetime(rain_season_end)
        duration = (rain_season_end - rain_season_start).days
        yearly_rain_data_df.loc[yearly_rain_data_df['hydro_year'] == year,'rain_season_duration'] = duration   
        print(f'{year} - {duration} days')
    print('\n')
    return yearly_rain_data_df

def calc_days_from_start_of_hydro_year(original_date, format):
    ## calculate the number of days from the begining of the hydro year till the given original_date
    if type(original_date) == str:
        formated_date = datetime.strptime(original_date, format)
        first_hydro_date = datetime(formated_date.year, 9, 1)
        days_diff = (formated_date - first_hydro_date).days
        return days_diff
     
    elif type(original_date) == np.datetime64:
        formated_date = pd.to_datetime(original_date)
        first_hydro_date = datetime(formated_date.year, 9, 1)
        days_diff = (formated_date - first_hydro_date).days
        return days_diff
        
    else:
        first_hydro_date = datetime(original_date.year, 9, 1)
        days_diff = (original_date - first_hydro_date).days
        return days_diff

def calc_rain_season_start_end_from_hydro_year(yearly_rain_data_df):
    ## Calulates the number of days from start of hydro year to the start of rain season
    ## and calculates the season duration from start of hydro year to the end of rain season
    for year in yearly_rain_data_df['hydro_year'].unique():
        rain_season_start = yearly_rain_data_df.loc[(yearly_rain_data_df['hydro_year']==year) , 'rain_season_start'].values[0]
        rain_season_end = yearly_rain_data_df.loc[(yearly_rain_data_df['hydro_year']==year) , 'rain_season_end'].values[0]
        start_days = calc_days_from_start_of_hydro_year(rain_season_start, format="%m/%d/%Y")
        end_days = calc_days_from_start_of_hydro_year(rain_season_end, format="%m/%d/%Y")
        yearly_rain_data_df.loc[(yearly_rain_data_df['hydro_year']==year) , 'rain_season_start_from_hydro'] = start_days
        yearly_rain_data_df.loc[(yearly_rain_data_df['hydro_year']==year) , 'rain_season_end_from_hydro'] = end_days
        yearly_rain_data_df['rain_season_start_from_hydro'] = [365+i if i<0 else i for i in yearly_rain_data_df['rain_season_start_from_hydro'].to_list()]
        yearly_rain_data_df['rain_season_end_from_hydro'] = [365+i if i<0 else i for i in yearly_rain_data_df['rain_season_end_from_hydro'].to_list()]
        if start_days < 0:
            print(f'{year} - start from hydro: {start_days+365} days')
            print(f'{year} - end from hydro: {end_days+365} days')
        else:
            print(f'{year} - start from hydro: {start_days} days')
            print(f'{year} - end from hydro: {end_days+365} days')
    print('\n')
    return yearly_rain_data_df

def calc_cdd(yearly_rain_data_df):
    ## calc CDD - consecutive dry days (where rain < 1 mm) - number of days from end of last rain season to 
    ## the start of the current season
    for year in yearly_rain_data_df['hydro_year'].unique()[1:]:
        rain_season_start = yearly_rain_data_df.loc[(yearly_rain_data_df['hydro_year']==year) , 'rain_season_start'].values[0]
        rain_season_start =  pd.to_datetime(rain_season_start)    
        previous_year = year-1
        previous_rain_season_end = yearly_rain_data_df.loc[(yearly_rain_data_df['hydro_year']==previous_year) , 'rain_season_end'].values[0]
        previous_rain_season_end =  pd.to_datetime(previous_rain_season_end)
        duration = (rain_season_start - previous_rain_season_end).days
        yearly_rain_data_df.loc[yearly_rain_data_df['hydro_year'] == year,'rain_consecutive_dry_days'] = duration   
        print(f'previous_year {previous_year} end {previous_rain_season_end} , current year {year} start {rain_season_start}, duration {duration} days')
    print('\n')
    return yearly_rain_data_df

def calc_avg_rain_per_rainy_day(rain_data_df, yearly_rain_data_df):
    ## Calc SDII - total precipitation/number of rainy days, where rain >0,  within the season duration.
    ## The total precipitation conputed in this function will be lower then the peak value calculated before, 
    ## as we neglect the rainy days outside the season
    for year in rain_data_df['hydro_year'].unique():
        season_start_date = yearly_rain_data_df.loc[yearly_rain_data_df['hydro_year']==year, 'rain_season_start'].values[0]
        season_end_date = yearly_rain_data_df.loc[yearly_rain_data_df['hydro_year']==year, 'rain_season_end'].values[0]
        percipitation = round(rain_data_df[(rain_data_df['hydro_year']==year) & (rain_data_df['date']>=season_start_date) & (rain_data_df['date']<=season_end_date)]['Rain'].sum(),2)
        rainy_days = len(rain_data_df[(rain_data_df['hydro_year']==year) & (rain_data_df['date']>=season_start_date) & (rain_data_df['date']<=season_end_date) & (rain_data_df['Rain']>0)]['Rain'])
        avg_rain_per_rainy_day = round((percipitation / rainy_days),2)
        yearly_rain_data_df.loc[yearly_rain_data_df['hydro_year'] == year,'avg_rain_per_rainy_day'] = avg_rain_per_rainy_day
        print(f'{year} - percipitation {percipitation} rainy_days {rainy_days}, avg_rain_per_rainy_day {avg_rain_per_rainy_day}')
    print('\n')
    return yearly_rain_data_df

def calc_rain_dispersion(rain_data_df, yearly_rain_data_df):
    ## calculates the dispersion of the rain over the season duration as the number of rainy days / season duration.
    for year in rain_data_df['hydro_year'].unique():
        season_start_date = yearly_rain_data_df.loc[yearly_rain_data_df['hydro_year']==year, 'rain_season_start'].values[0]
        season_end_date = yearly_rain_data_df.loc[yearly_rain_data_df['hydro_year']==year, 'rain_season_end'].values[0]
        season_duration = yearly_rain_data_df.loc[yearly_rain_data_df['hydro_year']==year, 'rain_season_duration'].values[0]
        rainy_days = len(rain_data_df[(rain_data_df['hydro_year']==year) & (rain_data_df['date']>=season_start_date) & (rain_data_df['date']<=season_end_date) & (rain_data_df['Rain']>0)]['Rain'])
        rain_dispersion = round((rainy_days / season_duration),3)
        yearly_rain_data_df.loc[yearly_rain_data_df['hydro_year'] == year,'rain_dispersion'] = rain_dispersion
        print(f'{year} - rainy_days {rainy_days}, season_duration {season_duration}, rain_dispersion {rain_dispersion}')
    print('\n')
    return yearly_rain_data_df


def calc_cdd_within_season(rain_data_df, yearly_rain_data_df):
    ## calculated the longest period of consecutive dry days within the season duration. 
    ## A dry day is a day with 0 mm rain.
    for year in rain_data_df['hydro_year'].unique():
        season_start_date = yearly_rain_data_df.loc[yearly_rain_data_df['hydro_year']==year, 'rain_season_start'].values[0]
        season_end_date = yearly_rain_data_df.loc[yearly_rain_data_df['hydro_year']==year, 'rain_season_end'].values[0]
        rain_filtered = rain_data_df.loc[(rain_data_df['hydro_year']==year) & (rain_data_df['date']>=season_start_date) & (rain_data_df['date']<=season_end_date) ,'cumulative'].to_list()

        delta_list = np.empty([1])
        cdds = []
        for i in range(len(rain_filtered[1:])):
            if rain_filtered[i+1] - rain_filtered[i] == 0:
                delta_list = np.append(delta_list,0)
            else:
                delta_list = np.append(delta_list,1)

        splitted = np.split(delta_list[1:], np.where(delta_list == 1)[0])
        # print(splitted, '\n')

        for i in splitted:
            if len(i) == 1:
                pass
            else:
                a = np.split(i, np.where(i == 1)[0])
        #         print(a)
                for z in a:
                    if len(z) == 1:
                        pass
                    else:
#                         print(len(z))
                        cdds.append(len(z))

        longest_cdd = max(cdds)
        print(f'{year} longest cdd {longest_cdd}')
        yearly_rain_data_df.loc[yearly_rain_data_df['hydro_year'] == year,'rain_cdd_within_season'] = longest_cdd
    print('\n')
    return yearly_rain_data_df

def plot_the_data(rain_data, yearly_rain_data_df):
    hydro_years = [ i for i in yearly_rain_data_df['hydro_year']]
    hydro_year_dates = [datetime.strptime(f"01/01/{year}", '%m/%d/%Y') for year in hydro_years]

    # hydro_year_dates = [datetime.strptime(str(year), '%Y') for year in hydro_years]

    begin_of_year = [ '09/01/' + str(i) for i in yearly_rain_data_df['hydro_year']]
    begin_of_year_dates = [datetime.strptime(date_str, '%m/%d/%Y') for date_str in begin_of_year]

    # cumu_data = rain_data.loc[(rain_data['date']>='09/01/2015') & (rain_data['date']<'01/01/2024')]
    cumu_data = rain_data.loc[ (rain_data['date']<'01/01/2024')]
    total_rain = [ round(i) for i in cumu_data.groupby('hydro_year')['cumulative'].max()]
    
    # fig, ax1 = plt.subplots(figsize=(12,4))
    fig, axes = plt.subplots(2, 1, figsize=(18, 6))
    # color = 'tab:green'
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Cumulative Rain (mm)')#, color=color)
    axes[0].set_xticks(hydro_year_dates)
    axes[0].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))  # Display only year
    
    for i in begin_of_year_dates:
        axes[0].axvline(x=i, c='black', axes=axes[0])

    sns.lineplot(data=cumu_data, x='date', y='cumulative_with_stop', alpha=0.3, ax=axes[0])
    axes[0].fill_between(cumu_data['date'], cumu_data['cumulative_with_stop'], alpha=0.2)

    axes[0].tick_params(axis='y')#, labelcolor=color)
    for i in begin_of_year_dates:
        axes[1].axvline(x=i, c='black')

    # axes[1] = ax1.twinx()  
    # color = 'tab:blue'
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Rain (mm)')#, color=color)
    axes[1].set_xticks(hydro_year_dates)
    axes[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))  # Display only year
    
    # ax2.set_ylabel('Rain', color=color) 

    sns.lineplot(data=cumu_data, x='date', y='Rain', ax=axes[1])
    for year, rain in zip(cumu_data['hydro_year'].unique(), total_rain):
        # for the x coordination march 1 i choshen 
        date_coord = f"01/01/{year}"
        x_coord = datetime.strptime(date_coord, '%m/%d/%Y') 
        # x_coord = cumu_data.loc[cumu_data['date'] == date_coord, 'date'].values[0]
        # for the t coordination 0.9 of graph hight is chosen
        y_coord =   0.9 * cumu_data['cumulative_with_stop'].max()
        axes[0].text(x_coord, y_coord, f'{rain}\nmm', ha='left', va='center', color='blue', fontsize=10)
    plt.savefig("plots/rain_graph.png")
    plt.show()
    
def plot_correlation_matrix(yearly_rain_data_df):
    yearly_rain_data_for_corr = yearly_rain_data_df.drop(columns=['rain_season_start', 'rain_season_end', 'hydro_year'])
    correlation_matrix = yearly_rain_data_for_corr.corr()

    plt.figure(figsize = (10,8))
    sns.heatmap(correlation_matrix, cmap = 'coolwarm', vmin = -1, vmax = 1, center = 0, annot=True, fmt=".2f", square=True, linewidths=.5)
    plt.tight_layout()
    plt.savefig("plots/correlation_matrix.png")
    plt.show()


def run_regressions(yearly_rain_data_df):
    yearly_rain_data_df = yearly_rain_data_df.drop(columns=['rain_season_start', 'rain_season_end', 'rain_pulse_peak_1_years_before', 'rain_pulse_peak_2_years_before'])
    yearly_rain_data_df['time'] = np.arange(len(yearly_rain_data_df.index))
    X = yearly_rain_data_df['time']  # Keep time for modeling
    y = yearly_rain_data_df.drop(columns=['hydro_year', 'time'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for target in y:
        sm_X_train = sm.add_constant(X_train)
        sm_X_test = sm.add_constant(X_test)
        model_sm = sm.OLS(y_train[target], sm_X_train).fit()
        y_pred = model_sm.predict(sm_X_test)
        model_sm_params = model_sm.params
        r2 = model_sm.rsquared
        model_pvalue = round(model_sm.f_pvalue, 3)

        equation = f"y = {model_sm_params.const:.2f} {model_sm_params['time']:.2f}x"
        r2_text = f"{r2:.2f}"
        pvalue_text = f"{model_pvalue:.2f}"

        plt.figure(figsize=(7, 8))
        plt.scatter(X, y[target], color='blue')
        plt.plot(X_test, y_pred, color='red', label=f'$R^2$: {r2_text}\np-value: {pvalue_text}\nLinear Equation: {equation}')

        hydro_years_list = yearly_rain_data_df['hydro_year'].tolist()
        num_ticks = len(hydro_years_list)
        tick_positions = np.linspace(0, len(hydro_years_list) - 1, num_ticks)
        plt.xticks(tick_positions, hydro_years_list)
        plt.xticks(rotation=45, ha='right')

        plt.xlabel('Hydro Year')
        plt.ylabel(target)
        plt.title(f"{target} over time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/{target} over time.png")
        plt.show()