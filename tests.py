import pandas as pd

def check_date_column_existance(df):
    assert 'date' in df.columns

def check_rain_column_existance(df):
    assert 'Rain' in df.columns

def check_rain_column_format(df):
    assert (df['Rain'].dtypes == 'float64' or df['Rain'].dtypes == 'int64')

def check_date_column_format(df):
  try:
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    assert df['date'].notna().all()
  except (ValueError, pd.errors.ParserError):
    raise AssertionError("Dates in 'date' column are not in the format '%Y-%m-%d'")
