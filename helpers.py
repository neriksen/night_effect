import os
import time
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from functools import lru_cache


def find_latest_business_day():
    today = dt.date.today()

    # If today is a weekend (Saturday or Sunday), adjust the date accordingly
    if today.weekday() == 5:  # Saturday (5)
        days_to_subtract = 1
    elif today.weekday() == 6:  # Sunday (6)
        days_to_subtract = 2
    else:
        days_to_subtract = 0

    latest_business_day = today - dt.timedelta(days=days_to_subtract)
    return latest_business_day


def find_data(ticker, last_business_day):
    file_name = f"data/{ticker}_data.csv"

    # If data is cached (exists in a local CSV), read from it
    if os.path.exists(file_name):
        data = pd.read_csv(file_name, index_col=0, parse_dates=True)
        try:
            is_old_data = data.index[-1].date() < last_business_day
        except IndexError:
            raise ValueError
        if not is_old_data:
            return data
    # Otherwise, download the data and save to a local CSV
    data = yf.download(ticker, start='2000-01-01', progress=False)
    data.to_csv(file_name)
    return data


def get_tickers_with_market_cap_limit(lower_percentile, upper_percentile):
    assert 0 <= lower_percentile <= 1
    assert 0 <= upper_percentile <= 1
    assert lower_percentile < upper_percentile
    tickers = pd.read_csv("tickers.csv", sep=";", index_col=0)
    tickers['Volume'] = tickers['Volume'].apply(parse_millions_and_billions)
    tickers['Avg Vol (3 month)'] = tickers['Avg Vol (3 month)'].apply(parse_millions_and_billions)
    tickers['Market Cap'] = tickers['Market Cap'].apply(parse_millions_and_billions)

    # Calculate the percentiles
    lower_percentile = tickers['Market Cap'].quantile(lower_percentile)
    upper_percentile = tickers['Market Cap'].quantile(upper_percentile)

    # Filter the DataFrame based on the percentile range
    tickers = tickers[(tickers['Market Cap'] >= lower_percentile) & (tickers['Market Cap'] <= upper_percentile)]
    return tuple(tickers.index.values)


def get_all_tickers_from_dir():
    # Directory containing the files
    directory_path = 'data'  # current directory as an example

    # Create an empty list to store filenames
    filenames_without_extension = []

    # Loop through each file in the directory
    for filename in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, filename)):  # Ensure it's a file and not a directory or a symlink
            name, extension = os.path.splitext(filename)
            filenames_without_extension.append(name.split("_")[0])
    return tuple(filenames_without_extension)


def parse_millions_and_billions(value):
    value = str(value)
    if 'M' in value:
        return float(value.replace('M', '')) * 1e6
    elif 'B' in value:
        return float(value.replace('B', '')) * 1e9
    else:
        return float(value)


@lru_cache(1000)
def calculate_night_effect_of_tickers(tickers):
    night_effects = []
    last_business_day = find_latest_business_day()
    for i, ticker in enumerate(tickers):
        try:
            data = find_data(ticker, last_business_day).loc['2009-01-01':]
        except ValueError:
            night_effects.append(pd.Series([np.nan], index=['2023-10-04']))
            continue
        night = nightsession_return(data)
        night_effects.append(night)
        print(f"Fetching data: {i+1}/{len(tickers)}, {ticker}", end='\r')  # Print progress

    print("\nCleaning up")
    df = pd.concat(night_effects, axis=1)
    df.columns = tickers
    df = df.dropna(axis=0, how="all")  # Drop rows if all NaN's

    df = df.sort_index()

    # Find index of the first non-NaN value for each column
    first_valid_idx = df.apply(lambda col: col.first_valid_index())

    # Replace NaNs with 1s below the first non-NaN value for each column
    for col in df.columns:
        idx = first_valid_idx[col]
        if idx is not None:  # if there's at least one non-NaN value in the column
            df.loc[idx:, col] = df.loc[idx:, col].fillna(1)
    print("Done")
    return df


def nightsession_return(df):
    # Convention: The return on say Tuesday Nov 3rd 2023, is the return
    #             you get from buying at closing auction on Nov 2nd and
    #             selling again at open on Nov 3rd. First row will therefore be NaN
    gross_night_return = df['Open']/df['Close'].shift()
    return gross_night_return


def daysession_return(df):
    daysession = df['Close']/df['Open']
    return daysession


class OverNightStrategy:
    def __init__(self, tickers, signal_sample_period_days=10, skew_factor=10, fee_pr_day=0.001):
        self.tickers = tickers
        self.overnight_df = calculate_night_effect_of_tickers(tickers=self.tickers)
        self.signal_df = self.compute_signal(signal_sample_period_days)
        self.skew_factor = skew_factor
        self.fee_pr_day = fee_pr_day


    def compute_signal(self, sample_period=1000):
        # Remember the return on say Tuesday Nov 3rd 2023, is the return
        # you get from buying at closing auction on Nov 2nd and
        # selling again at open on Nov 3rd. So, date X in this df will only
        # contain information from date 0 to X and crucially not 0 to X+1
        return self.overnight_df.rolling(sample_period, min_periods=sample_period).mean().dropna(axis=1,
                                                                                                           how="all").dropna(
            axis=0, how="all")

    def compute_portfolio(self, number_of_stocks_in_portfolio, portfolio_weight_type=None):
        self.number_of_stocks_in_portfolio = number_of_stocks_in_portfolio
        # The idea is, given a "signal", to sort the by the best looking signal from the day before
        # and choose those best-looking stocks to be in the portfolio for tomorrow

        stocks_chosen = []

        portfolio_returns = np.empty((len(self.signal_df)-1, 1))
        assert number_of_stocks_in_portfolio <= len(self.signal_df.columns)

        # First we limit the overnight_df to start at the same time as the portfolio
        assert len(self.overnight_df) >= len(self.signal_df)
        limited_overnight_df = self.overnight_df.reindex(self.signal_df.index)
        overnight_arr = limited_overnight_df.values

        for i, (_, row) in enumerate(self.signal_df.iterrows()):
            if i == len(self.signal_df) - 1:
                break

            # Drop NaNs
            assert len(row.dropna()) >= number_of_stocks_in_portfolio    # Make sure stocks with NA returns are not included

            # Sort indices while keeping track of the column names
            indices_without_nan = np.where(~np.isnan(row))[0]
            sorted_indices = indices_without_nan[np.argsort(row.values[indices_without_nan])][::-1]

            sorted_columns = row.index[sorted_indices]

            # Append the top N stock names
            stocks_chosen.append(list(sorted_columns[:number_of_stocks_in_portfolio].values))
            this_period_stock_chosen_idx = sorted_indices[:number_of_stocks_in_portfolio]

            # Determine portfolio weights
            if portfolio_weight_type == "skewed":
                # We boost the signal by a factor to give some more conviction.. :)
                signal_this_period = (row.values[this_period_stock_chosen_idx])**self.skew_factor
                # A simple weighting that gives more weight to stocks with a higher signal
                weights = signal_this_period/signal_this_period.sum()
                assert weights.sum() != 0
            else:
                weights = None

            # Calculate the portfolio return
            next_period_returns = np.average(overnight_arr[i+1, this_period_stock_chosen_idx] - self.fee_pr_day, weights=weights)
            portfolio_returns[i] = next_period_returns

        # The index begins one period later, as we dont have a return when buying the portfolio the first day
        cum_returns = pd.DataFrame(portfolio_returns, index=self.signal_df.iloc[1:].index).cumprod()
        cum_returns = cum_returns / cum_returns.iloc[0, :]

        self.cum_returns = cum_returns
        self.portfolio = stocks_chosen


    def plot_performance(self, compare_to_spy=True, log=True):
        if not hasattr(self, "cum_returns"):
            print("No portfolio calculated yet. Call .compute_portfolio()")
            return

        plt.plot(self.cum_returns)
        if compare_to_spy:
            spx_compare = yf.download('SPY', start=self.cum_returns.iloc[1].name, progress=False)
            plt.plot(spx_compare['Close'] / spx_compare['Close'].iloc[0])
        plt.title(f"Total afkast, natteeffekt, {self.number_of_stocks_in_portfolio} aktier")
        if log:
            plt.yscale('log')
        plt.grid(True)
        plt.show()

    def compute_portfolio_stats(self, print_stats=False):
        if not hasattr(self, "cum_returns"):
            print("No portfolio calculated yet. Call .compute_portfolio()")
            return

        final_return = self.cum_returns.iloc[-1]
        num_years = len(self.cum_returns) / 260

        yearly_return = (final_return ** (1 / num_years) - 1)[0]
        yearly_std = (self.cum_returns.pct_change().std() * (250 ** 0.5))[0]
        sharpe = yearly_return / yearly_std

        if print_stats:
            x = self.cum_returns.pct_change().iloc[1:].values.flatten()
            spx_compare = yf.download('SPY', start=self.cum_returns.iloc[0].name, progress=False)
            y = spx_compare['Close'].reindex(self.cum_returns.index).pct_change().fillna(0).iloc[1:].values.flatten()
            correlation = np.corrcoef(x, y)[0, 1]

            print(f"Ann. return: {round(yearly_return*100, 1)}%, Ann. std: {round(yearly_std*100, 1)}%, Sharpe: {round(sharpe, 2)}, Beta: {round(correlation, 2)}")
        return yearly_return, yearly_std, sharpe


if __name__ == '__main__':
    tickers = get_tickers_with_market_cap_limit(0.95, 1)
    strat1 = OverNightStrategy(tickers)
    start = time.time()
    strat1.compute_portfolio(3, portfolio_weight_type=None)
    end = time.time()
    print(f"Time took: {end-start}")

