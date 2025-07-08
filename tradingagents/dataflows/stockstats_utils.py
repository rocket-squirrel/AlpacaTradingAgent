import pandas as pd
from stockstats import wrap
from typing import Annotated
import os
from .config import get_config
from .alpaca_utils import AlpacaUtils


class StockstatsUtils:
    @staticmethod
    def get_stock_stats(
        symbol: Annotated[str, "ticker symbol for the company"],
        indicator: Annotated[
            str, "quantitative indicators based off of the stock data for the company"
        ],
        curr_date: Annotated[
            str, "curr date for retrieving stock price data, YYYY-mm-dd"
        ],
        data_dir: Annotated[
            str,
            "directory where the stock data is stored.",
        ],
        online: Annotated[
            bool,
            "whether to use online tools to fetch data or offline tools. If True, will use online tools.",
        ] = False,
    ):
        df = None
        data = None
        
        # Sanitize symbol for filename (replace / with _)
        safe_symbol = symbol.replace('/', '_')

        if not online:
            try:
                data = pd.read_csv(
                    os.path.join(
                        data_dir,
                        f"{safe_symbol}-Alpaca-data-2015-01-01-2025-03-25.csv",
                    )
                )
                df = wrap(data)
            except FileNotFoundError:
                raise Exception("Stockstats fail: Alpaca data not fetched yet!")
        else:
            # Parse the current date
            curr_date_dt = pd.to_datetime(curr_date)
            
            # Get more historical data to ensure proper technical indicator calculations
            # Technical indicators like 50 SMA need at least 50+ days of data
            end_date = pd.Timestamp.today()
            start_date = end_date - pd.DateOffset(days=365)  # Get 1 year of data for reliable indicators
            
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")

            # Get config and ensure cache directory exists
            config = get_config()
            os.makedirs(config["data_cache_dir"], exist_ok=True)

            data_file = os.path.join(
                config["data_cache_dir"],
                f"{safe_symbol}-Alpaca-data-{start_date_str}-{end_date_str}.csv",
            )

            try:
                if os.path.exists(data_file):
                    # Load cached data
                    data = pd.read_csv(data_file)
                    if 'Date' in data.columns:
                        data["Date"] = pd.to_datetime(data["Date"])
                else:
                    # Fetch fresh data from Alpaca
                    data = AlpacaUtils.get_stock_data(
                        symbol=symbol,  # Use original symbol for API call
                        start_date=start_date_str,
                        end_date=end_date_str,
                        timeframe="1Day"
                    )
                    
                    # Ensure we have data
                    if data.empty:
                        return f"N/A: No data available for {symbol}"
                    
                    # Standardize column names for stockstats
                    if 'timestamp' in data.columns:
                        data = data.rename(columns={
                            'timestamp': 'Date',
                            'open': 'Open',
                            'high': 'High', 
                            'low': 'Low',
                            'close': 'Close',
                            'volume': 'Volume'
                        })
                    
                    # Ensure Date column is datetime
                    if 'Date' in data.columns:
                        data["Date"] = pd.to_datetime(data["Date"])
                    
                    # Sort by date to ensure proper chronological order for indicators
                    data = data.sort_values('Date').reset_index(drop=True)
                    
                    # Save to cache
                    data.to_csv(data_file, index=False)

                # Ensure we have sufficient data for technical indicators
                if len(data) < 100:
                    return f"N/A: Insufficient data for {indicator} calculation (need at least 100 days, got {len(data)})"

                # Wrap with stockstats for technical indicator calculations
                df = wrap(data)
                
                # Trigger the indicator calculation
                try:
                    indicator_series = df[indicator]
                except KeyError:
                    return f"N/A: Invalid indicator '{indicator}'"
                except Exception as e:
                    return f"N/A: Error calculating {indicator}: {str(e)}"

                # Convert date column to string for matching
                df["date_str"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
                curr_date_str = curr_date_dt.strftime("%Y-%m-%d")

                # Find the most recent trading day on or before the requested date
                available_dates = df["date_str"].tolist()
                matching_rows = df[df["date_str"] == curr_date_str]
                
                if not matching_rows.empty:
                    indicator_value = matching_rows[indicator].iloc[-1]  # Get the last (most recent) value
                    # Handle NaN values
                    if pd.isna(indicator_value):
                        return f"N/A: {indicator} not calculable for {curr_date_str}"
                    return float(indicator_value)
                else:
                    # If exact date not found, try to find the most recent trading day before the requested date
                    df_filtered = df[df["date_str"] <= curr_date_str]
                    if not df_filtered.empty:
                        most_recent = df_filtered.iloc[-1]
                        indicator_value = most_recent[indicator]
                        if pd.isna(indicator_value):
                            return f"N/A: {indicator} not calculable for most recent trading day"
                        actual_date = most_recent["date_str"]
                        return f"{float(indicator_value)} (as of {actual_date})"
                    else:
                        return f"N/A: No trading data available on or before {curr_date_str}"

            except Exception as e:
                return f"N/A: Error processing data for {symbol}: {str(e)}"
