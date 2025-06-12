import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import chardet
import traceback
from scipy import stats

# Set page configuration
st.set_page_config(page_title="Future Trend Forecasting Tool", layout="wide", initial_sidebar_state="expanded")

# Cache key for new uploads
if 'upload_key' not in st.session_state:
    st.session_state.upload_key = 0

def clear_cache():
    st.cache_data.clear()
    st.session_state.upload_key += 1

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(file, key):
    try:
        file_size = file.size / (1024 * 1024)
        if file_size > 10:
            st.warning(f"File size: {file_size:.2f} MB. Files >10 MB may fail on Streamlit Cloud.")
        if file_size > 200:
            st.error(f"File size: {file_size:.2f} MB exceeds limit (200 MB).")
            return None
        raw_data = file.read()
        encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
        st.write(f"Detected encoding: {encoding}")
        file.seek(0)
        delimiters = [',', ';', '\t', '|', ' ', ':']
        for delimiter in delimiters:
            try:
                file.seek(0)
                df = pd.read_csv(file, delimiter=delimiter, encoding=encoding, on_bad_lines='skip', low_memory=False)
                if len(df.columns) >= 2:
                    st.info(f"Loaded CSV with delimiter '{delimiter}'")
                    st.write(f"First 5 rows:\n{df.head().to_string()}")
                    return df
            except Exception as e:
                st.warning(f"Failed with delimiter '{delimiter}': {str(e)}")
        st.error("Could not parse CSV with common delimiters. Check file format.")
        file.seek(0)
        sample = file.read(1000).decode(encoding, errors='ignore')
        st.write(f"Sample content (first 1000 bytes):\n{sample}")
        return None
    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}\n{traceback.format_exc()}")
        return None

def is_date_column(series, sample_size=100):
    try:
        sample = series.dropna().head(sample_size).astype(str).str.strip()
        if len(sample) < 5:
            return False
        date_formats = [
            '%Y-%m-%d', '%Y%m%d', '%Y/%m/%d', '%d/%m/%Y', '%d-%m-%Y',
            '%Y-%m', '%Y%m', '%m/%Y', '%b %Y', '%Y', '%d %b %Y', '%m-%d-%Y',
            '%Y/%m', '%d.%m.%Y', '%m-%d-%Y', '%Y.%m.%d', '%b-%d-%Y',
            '%Y-%b-%d', '%d/%b/%Y', '%Y/%b/%d', '%m.%d.%Y'
        ]
        if sample.str.match(r'^\d{6}$').sum() >= len(sample) * 0.8:
            return True
        for fmt in date_formats:
            try:
                parsed = pd.to_datetime(sample, format=fmt, errors='coerce')
                if parsed.notna().sum() >= len(sample) * 0.8:
                    st.write(f"Detected date format: {fmt}")
                    return True
            except:
                continue
        try:
            parsed = pd.to_datetime(sample, errors='coerce')
            if parsed.notna().sum() >= len(sample) * 0.8:
                st.write("Detected date via auto-parsing")
                return True
        except:
            pass
        st.warning(f"Column '{series.name}' not recognized as date. Sample:\n{sample.head().to_string()}")
        return False
    except Exception as e:
        st.warning(f"Date detection failed: {str(e)}")
        return False

def preprocess_time_series(df, date_col, target_col, freq):
    try:
        st.write(f"Preprocessing: Date = {date_col}, Target = {target_col}, Freq = {freq}")
        st.write(f"Initial rows: {len(df)}")
        st.write(f"Raw {date_col} sample:\n{df[date_col].head().to_string()}")
        st.write(f"Unique {date_col} values (first 10):\n{df[date_col].unique()[:10]}")
        st.write(f"Total unique {date_col} values: {len(df[date_col].unique())}")

        if date_col == target_col:
            st.error("Date and target columns must be different.")
            return None

        df = df.copy()
        df[date_col] = df[date_col].astype(str).str.strip()

        if df[date_col].str.match(r'^\d{6}$').any():
            df[date_col] = pd.to_datetime(df[date_col], format='%Y%m', errors='coerce')
            st.info("Parsed dates as YYYYMM")
        else:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        for fmt in [
            '%Y-%m-%d', '%Y%m%d', '%Y/%m/%d', '%d/%m/%Y', '%d-%m-%Y',
            '%Y-%m', '%Y%m', '%m/%Y', '%b %Y', '%Y', '%d %b %Y', '%m-%d-%Y',
            '%Y/%m', '%d.%m.%Y', '%m-%d-%Y', '%Y.%m.%d', '%b-%d-%Y',
            '%Y-%b-%d', '%d/%b/%Y', '%Y/%b/%d', '%m.%d.%Y'
        ]:
            if df[date_col].isna().sum() > len(df) * 0.5:
                try:
                    df[date_col] = pd.to_datetime(df[date_col], format=fmt, errors='coerce')
                    if df[date_col].notna().sum() > 0:
                        st.info(f"Parsed dates using format: {fmt}")
                        break
                except:
                    continue

        if df[date_col].isna().any():
            invalid_count = df[date_col].isna().sum()
            st.warning(f"Dropped {invalid_count} rows with invalid dates.")
            df = df.dropna(subset=[date_col])

        if df.empty:
            st.error("No valid dates remain.")
            return None

        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        if df[target_col].isna().all():
            st.error(f"Target '{target_col}' contains no valid numeric values.")
            return None

        # Aggregate to specified frequency
        df = df[[date_col, target_col]].set_index(date_col)
        df = df.resample(freq).mean().reset_index()
        if df[target_col].isna().any():
            st.warning("Filling NaN values in target with mean.")
            df[target_col] = df[target_col].fillna(df[target_col].mean())

        df = df.rename(columns={date_col: 'ds', target_col: 'y'})
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds')

        unique_dates = len(df['ds'].unique())
        st.write(f"Unique dates after resampling: {unique_dates}")
        if unique_dates < 2:
            st.error(f"Only {unique_dates} unique date(s) found. Need at least 2 for forecasting.")
            return None

        return df
    except Exception as e:
        st.error(f"Preprocessing failed: {str(e)}\n{traceback.format_exc()}")
        return None

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    return rmse, mae, mape

def generate_forecast(df, date_col, target_col, horizon, period, five_year_forecast=False):
    try:
        freq_map = {'Monthly': 'M', 'Quarterly': 'Q', 'Half-Yearly': '6M', 'Yearly': 'Y'}
        freq = freq_map[period]
        processed_df = preprocess_time_series(df, date_col, target_col, freq)
        if processed_df is None:
            return
        st.write(f"Processed data rows: {len(processed_df)}")
        train_size = int(len(processed_df) * 0.8)
        if train_size < 2:
            st.error("Need at least 2 rows for training.")
            return
        train = processed_df[:train_size]
        test = processed_df[train_size:]
        st.write(f"Train size: {len(train)}, Test size: {len(test)}")
        st.write(f"Test date range: {test['ds'].min()} to {test['ds'].max()}")

        model = Prophet(yearly_seasonality=len(processed_df) >= 12 if freq != 'Y' else False, weekly_seasonality=False, daily_seasonality=False)
        model.fit(train)
        horizon = 60 if five_year_forecast and period == 'Monthly' else 20 if five_year_forecast and period == 'Quarterly' else 10 if five_year_forecast and period == 'Half-Yearly' else 5 if five_year_forecast and period == 'Yearly' else horizon
        future = model.make_future_dataframe(periods=horizon, freq=freq)
        forecast = model.predict(future)
        st.write(f"Forecast date range: {forecast['ds'].min()} to {forecast['ds'].max()}")
        st.write(f"Forecast rows: {len(forecast)}")

        # Align test and forecast
        test_df = test[['ds', 'y']].merge(forecast[['ds', 'yhat']], on='ds', how='left')
        y_pred = test_df['yhat'].values
        y_true = test_df['y'].values
        st.write(f"y_pred length: {len(y_pred)}, y_true length: {len(y_true)}")

        if len(y_pred) == 0 or np.any(np.isnan(y_pred)):
            st.warning("Unable to compute metrics due to invalid predictions. Showing plots only.")
            rmse, mae, mape = None, None, None
        else:
            rmse, mae, mape = calculate_metrics(y_true, y_pred)

        # Future Forecast Line Chart
        st.subheader(f"Future {period} Trend Forecast {'(5 Years)' if five_year_forecast else ''}")
        future_forecast = forecast[forecast['ds'] > processed_df['ds'].max()]
        if future_forecast.empty:
            st.error("No future forecast data generated. Check data range or horizon.")
            return
        st.write(f"Future forecast rows: {len(future_forecast)}")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=processed_df['ds'], y=processed_df['y'], mode='lines', name='Historical', line=dict(color='blue')))
        fig1.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], mode='lines', name='Future Forecast', line=dict(color='red')))
        fig1.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(255,0,0,0.2)', name='Upper CI'))
        fig1.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(255,0,0,0.2)', name='Lower CI'))
        fig1.update_layout(
            title=f"Future {period} Trend Forecast for Next {horizon} Periods {'(5 Years)' if five_year_forecast else ''}",
            xaxis_title="Date",
            yaxis_title=target_col,
            template="plotly_white"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Future Forecast Bar Chart
        st.subheader(f"Future {period} Forecast Bar Chart {'(5 Years)' if five_year_forecast else ''}")
        bar_data = future_forecast.groupby(future_forecast['ds'].dt.to_period(freq))['yhat'].mean().reset_index()
        if bar_data.empty:
            st.error("No data for bar chart. Check forecast output.")
            return
        bar_data['ds'] = bar_data['ds'].astype(str)
        st.write(f"Bar chart data rows: {len(bar_data)}")
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=bar_data['ds'], y=bar_data['yhat'], name='Forecast Mean'))
        fig_bar.update_layout(
            title=f"Mean Forecast per {period} Period {'(5 Years)' if five_year_forecast else ''}",
            xaxis_title="Period",
            yaxis_title=target_col,
            template="plotly_white"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Full Forecast Plot
        st.subheader(f"Full {period} Forecast with Historical Data")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=train['ds'], y=train['y'], mode='lines', name='Train'))
        fig2.add_trace(go.Scatter(x=test['ds'], y=test['y'], mode='lines', name='Test'))
        fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
        fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,100,80,0.2)', name='Upper CI'))
        fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,100,80,0.2)', name='Lower CI'))
        fig2.update_layout(
            title=f"Full {period} Forecast (RMSE: {rmse if rmse else 'N/A'}, MAE: {mae if mae else 'N/A'}, MAPE: {mape if mape else 'N/A'}%)",
            xaxis_title="Date",
            yaxis_title=target_col,
            template="plotly_white"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Trend Components Plot
        st.subheader("Trend Components")
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], mode='lines', name='Trend'))
        if len(processed_df) >= 12 and freq != 'Y':
            fig4.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yearly'], mode='lines', name='Yearly Seasonality'))
        fig4.update_layout(
            title="Trend and Seasonality Components",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_white"
        )
        st.plotly_chart(fig4, use_container_width=True)

        # Future Forecast Report
        future_forecast_df = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={'ds': 'Date', 'yhat': 'Forecast', 'yhat_lower': 'Lower CI', 'yhat_upper': 'Upper CI'})
        st.write(f"Future {period} Forecast (all predicted periods):")
        st.dataframe(future_forecast_df)
        st.download_button(
            f"Download Future {period} Forecast {'(5 Years)' if five_year_forecast else ''}",
            future_forecast_df.to_csv(index=False).encode('utf-8'),
            file_name=f"future_{period.lower()}_forecast{'_5years' if five_year_forecast else ''}.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Forecasting failed: {str(e)}\n{traceback.format_exc()}")

def generate_insights(df, date_col, target_col, horizon, period, five_year_forecast=False):
    try:
        freq_map = {'Monthly': 'M', 'Quarterly': 'Q', 'Half-Yearly': '6M', 'Yearly': 'Y'}
        freq = freq_map[period]
        processed_df = preprocess_time_series(df, date_col, target_col, freq)
        if processed_df is None:
            return

        # Generate future predictions
        model = Prophet(yearly_seasonality=len(processed_df) >= 12 if freq != 'Y' else False, weekly_seasonality=False, daily_seasonality=False)
        horizon = 60 if five_year_forecast and period == 'Monthly' else 20 if five_year_forecast and period == 'Quarterly' else 10 if five_year_forecast and period == 'Half-Yearly' else 5 if five_year_forecast and period == 'Yearly' else horizon
        future = model.make_future_dataframe(periods=horizon, freq=freq)
        forecast = model.predict(future)
        future_forecast = forecast[forecast['ds'] > processed_df['ds'].max()]
        if future_forecast.empty:
            st.error("No future forecast data for insights. Check data range or horizon.")
            return

        st.subheader(f"Future {period} Trend Insights {'(5 Years)' if five_year_forecast else ''}")

        # Future Trend Statistics
        stats_df = pd.Series(future_forecast['yhat']).describe()
        stats_df['skewness'] = future_forecast['yhat'].skew()
        stats_df['kurtosis'] = future_forecast['yhat'].kurtosis()
        st.write(f"**Future {period} Trend Statistics**")
        st.dataframe(stats_df, use_container_width=True)

        # Future Trend Analysis
        trend = "Increasing" if future_forecast['yhat'].iloc[-1] > future_forecast['yhat'].iloc[0] else "Decreasing"
        st.write(f"**Future {period} Trend**: {trend} over the next {horizon} periods.")

        # Future Outlier Detection
        z_scores = np.abs(stats.zscore(future_forecast['yhat']))
        outliers = future_forecast[z_scores > 3]
        if not outliers.empty:
            st.write(f"**Future {period} Outliers Detected** ({len(outliers)})")
            st.dataframe(outliers[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Forecast'}), use_container_width=True)
        else:
            st.write(f"**Future {period} Outliers**: None detected.")

        # Future Trend Line Plot
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], mode='lines', name='Forecast'))
        fig_line.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,100,80,0.2)', name='Upper CI'))
        fig_line.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,0,0,0.2)', name='Lower CI'))
        fig_line.update_layout(
            title=f"Future {period} Trend for Next {horizon} Periods {'(5 Years)' if five_year_forecast else ''}",
            xaxis_title="Date",
            yaxis_title=target_col,
            template="plotly_white"
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # Future Insights Report
        insights_df = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={'ds': 'Date', 'yhat': 'Forecast', 'yhat_lower': 'Lower CI', 'yhat_upper': 'Upper CI'})
        insights_df['Trend'] = 'Increasing' if trend == 'Increasing' else 'Decreasing'
        st.write(f"**Future {period} Trend Insights** (all predicted periods)")
        st.dataframe(insights_df)
        st.download_button(
            f"Download Future {period} Trend Insights {'(5 Years)' if five_year_forecast else ''}",
            insights_df.to_csv(index=False).encode('utf-8'),
            file_name=f"future_{period.lower()}_insights{'_5years' if five_year_forecast else ''}.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Insights failed: {str(e)}\n{traceback.format_exc()}")

def main():
    st.title("Future Trend Forecasting & Insights Tool")
    st.markdown("Upload a CSV with a date column (e.g., 2022-09-01) and a numeric column to forecast future trends on a monthly, quarterly, half-yearly, or yearly basis.")

    with st.sidebar:
        st.header("Settings")
        period = st.selectbox("Forecast Period", ["Monthly", "Quarterly", "Half-Yearly", "Yearly"], index=0)
        max_horizon = 12 if period == "Monthly" else 4 if period == "Quarterly" else 10 if period == "Half-Yearly" else 2
        horizon = st.number_input(f"Forecast Horizon ({period.lower()} periods)", min_value=1, max_value=max_horizon, value=3 if period in ["Monthly", "Half-Yearly"] else 2)
        five_year_forecast = st.checkbox("Show 5-Year Forecast", value=False)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], help="Files >10 MB may fail on Streamlit Cloud.", on_change=clear_cache)

    if uploaded_file is not None:
        df = load_data(uploaded_file, key=st.session_state.upload_key)
        if df is not None and not df.empty:
            st.success(f"Loaded {len(df)} rows, {len(df.columns)} columns!")
            st.write(f"Columns: {df.columns.tolist()}")

            with st.sidebar:
                date_cols = [col for col in df.columns if is_date_column(df[col])]
                if not date_cols:
                    st.error("No date column found (e.g., '2022-09-01'). Check date format.")
                    return
                date_col = st.selectbox("Date Column", date_cols, index=0)

                target_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                target_cols = [col for col in target_cols if col != date_col]
                if not target_cols:
                    st.error("No numeric columns found.")
                    return
                target_col = st.selectbox("Target Column", target_cols, index=0)

            tab1, tab2 = st.tabs(["Future Forecasting", "Future Insights"])
            with tab1:
                st.header(f"Future {period} Trend Forecasting")
                if st.button("Generate Forecast"):
                    with st.spinner("Generating forecast..."):
                        generate_forecast(df, date_col, target_col, horizon, period, five_year_forecast)
            with tab2:
                st.header(f"Future {period} Trend Insights")
                if st.button("Generate Insights"):
                    with st.spinner("Generating insights..."):
                        generate_insights(df, date_col, target_col, horizon, period, five_year_forecast)
        else:
            st.error("Failed to load CSV. Check format or share sample content.")

if __name__ == "__main__":
    main()