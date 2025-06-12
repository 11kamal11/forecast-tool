import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import chardet
import traceback

# Set page configuration
st.set_page_config(page_title="Universal Forecasting Tool", layout="wide", initial_sidebar_state="expanded")

# Cache key for new uploads
if 'upload_key' not in st.session_state:
    st.session_state.upload_key = 0

def clear_cache():
    st.cache_data.clear()
    st.session_state.upload_key += 1

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(file, key):
    try:
        file_size = file.size / (1024 * 1024)  # Size in MB
        if file_size > 10:
            st.warning(f"File size: {file_size:.2f} MB. Files >10 MB may fail on Streamlit Cloud. Try compressing (e.g., zip).")
        if file_size > 200:
            st.error(f"File size: {file_size:.2f} MB exceeds local limit (200 MB). Compress or split the file.")
            return None
        raw_data = file.read()
        encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
        st.write(f"Detected encoding: {encoding}")
        file.seek(0)
        for delimiter in [',', ';', '\t', '|']:
            try:
                file.seek(0)
                df = pd.read_csv(file, delimiter=delimiter, encoding=encoding, on_bad_lines='skip', low_memory=False)
                if len(df.columns) >= 2:
                    st.info(f"Loaded CSV with delimiter '{delimiter}'")
                    return df
            except Exception as e:
                st.warning(f"Failed with delimiter '{delimiter}': {str(e)}")
        st.error("Could not parse CSV with common delimiters.")
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
            '%Y-%m', '%Y%m', '%m/%Y', '%b %Y', '%Y', '%d %b %Y', '%m-%d-%Y', '%Y/%m'
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
        return False
    except Exception as e:
        st.warning(f"Date detection failed: {str(e)}")
        return False

def preprocess_time_series(df, date_col, target_col):
    try:
        st.write(f"Preprocessing: Date = {date_col}, Target = {target_col}")
        st.write(f"Initial rows: {len(df)}")
        st.write(f"Raw {date_col} sample:\n{df[date_col].head().to_string()}")
        st.write(f"Unique {date_col} values (first 10):\n{df[date_col].unique()[:10]}")
        st.write(f"Total unique {date_col} values: {len(df[date_col].unique())}")

        if date_col == target_col:
            st.error("Date column and target column must be different. Please select a numeric column for the target.")
            return None

        df = df.copy()
        df[date_col] = df[date_col].astype(str).str.strip()

        if df[date_col].str.match(r'^\d{6}$').any():
            df[date_col] = pd.to_datetime(df[date_col], format='%Y%m', errors='coerce')
            st.info("Parsed dates as YYYYMM")
        else:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        for fmt in ['%Y-%m-%d', '%Y%m%d', '%Y/%m/%d', '%d/%m/%Y', '%d-%m-%Y', '%Y-%m', '%Y%m', '%m/%Y', '%b %Y', '%Y', '%d %b %Y', '%m-%d-%Y', '%Y/%m']:
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

        df = df[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
        if 'ds' not in df.columns or 'y' not in df.columns:
            st.error("Failed to create 'ds' or 'y' columns. Ensure date and target columns are valid and distinct.")
            return None

        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds')

        if df['y'].isna().any():
            nan_count = df['y'].isna().sum()
            st.warning(f"Filling {nan_count} NaN values in target with mean.")
            df['y'] = df['y'].fillna(df['y'].mean())

        unique_dates = len(df['ds'].unique())
        st.write(f"Unique dates: {unique_dates}")
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

def generate_forecast(df, date_col, target_col, horizon):
    try:
        processed_df = preprocess_time_series(df, date_col, target_col)
        if processed_df is None:
            return
        st.write(f"Processed data rows: {len(processed_df)}")
        train_size = int(len(processed_df) * 0.8)
        if train_size < 2:
            st.error("Insufficient data for training (need at least 2 rows).")
            return
        train = processed_df[:train_size]
        test = processed_df[train_size:]
        st.write(f"Train size: {len(train)}, Test size: {len(test)}")
        st.write(f"Test date range: {test['ds'].min()} to {test['ds'].max()}")

        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
        model.fit(train)
        future = model.make_future_dataframe(periods=horizon + len(test), freq='M')
        forecast = model.predict(future)
        st.write(f"Forecast date range: {forecast['ds'].min()} to {forecast['ds'].max()}")
        st.write(f"Forecast rows: {len(forecast)}")

        # Align test and forecast
        test_dates = test['ds'].values
        forecast_subset = forecast[forecast['ds'].isin(test_dates)]['yhat']
        y_pred = forecast_subset.values
        st.write(f"y_pred length: {len(y_pred)}, test['y'] length: {len(test['y'])}")

        if len(y_pred) == 0 or len(y_pred) != len(test['y']):
            st.warning("Unable to compute metrics due to mismatched or empty predictions. Displaying forecast plot only.")
            rmse, mae, mape = None, None, None
        else:
            rmse, mae, mape = calculate_metrics(test['y'], y_pred)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train['ds'], y=train['y'], mode='lines', name='Train'))
        fig.add_trace(go.Scatter(x=test['ds'], y=test['y'], mode='lines', name='Test'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
        fig.update_layout(
            title=f"Forecast (RMSE: {rmse if rmse else 'N/A'}, MAE: {mae if mae else 'N/A'}, MAPE: {mape if mape else 'N/A'}%)",
            xaxis_title="Date",
            yaxis_title=target_col,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        forecast_df = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Forecast'})
        st.write("Forecast sample (last 10 rows):")
        st.dataframe(forecast_df.tail(10))
        st.download_button(
            "Download Forecast",
            forecast_df.to_csv(index=False).encode('utf-8'),
            file_name="forecast.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Forecasting failed: {str(e)}\n{traceback.format_exc()}")

def generate_insights(df, date_col, target_col, granularity):
    try:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        df = df.dropna(subset=[target_col])

        if df.empty:
            st.error("No valid data for insights.")
            return

        resampling_map = {'Weekly': 'W', 'Monthly': 'M', 'Half-Yearly': '6M', 'Yearly': 'Y'}
        if granularity not in resampling_map:
            st.error("Invalid granularity specified.")
            return

        df.set_index(date_col, inplace=True)
        df_resampled = df[target_col].resample(resampling_map[granularity]).mean().reset_index()
        df_resampled = df_resampled.dropna(subset=[target_col])

        if len(df_resampled) < 1:
            st.error(f"No data available for {granularity} granularity.")
            return

        st.subheader(f"{granularity} Insights")
        stats = df_resampled[target_col].describe()
        st.write("**Summary Statistics**")
        st.dataframe(stats, use_container_width=True)

        trend = "Increasing" if df_resampled[target_col].iloc[-1] > df_resampled[target_col].iloc[0] else "Decreasing"
        st.write(f"**Trend**: {trend} over the period.")

        fig_line = px.line(df_resampled, x=date_col, y=target_col, title=f"{granularity} Trend of {target_col}")
        st.plotly_chart(fig_line, use_container_width=True)

        fig_box = px.box(df_resampled, y=target_col, title=f"{granularity} Distribution of {target_col}")
        st.plotly_chart(fig_box, use_container_width=True)

        insights_df = df_resampled.rename(columns={date_col: 'Date', target_col: 'Value'})
        st.write(f"**{granularity} Data Sample**")
        st.dataframe(insights_df.tail(10))
        st.download_button(
            f"Download {granularity} Report",
            insights_df.to_csv(index=False).encode('utf-8'),
            file_name=f"{granularity.lower()}_report.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Insights generation failed: {str(e)}\n{traceback.format_exc()}")

def main():
    st.title("Universal Forecasting & Insights Tool")
    st.markdown("Upload any CSV with a date column (e.g., 2022-09-01, 202209) and a numeric column to forecast trends and generate insights.")

    with st.sidebar:
        st.header("Settings")
        horizon = st.number_input("Forecast Horizon (months)", min_value=1, max_value=12, value=5)
        granularity = st.selectbox("Report Granularity", ["Weekly", "Monthly", "Half-Yearly", "Yearly"], index=1)

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], help="Ensure CSV has a date and numeric column. Files >10 MB may fail on Streamlit Cloud.", on_change=clear_cache)

    if uploaded_file is not None:
        df = load_data(uploaded_file, key=st.session_state.upload_key)
        if df is not None and not df.empty:
            st.success(f"Loaded {len(df)} rows and {len(df.columns)} columns!")
            st.write(f"Columns: {df.columns.tolist()}")

            with st.sidebar:
                date_cols = [col for col in df.columns if is_date_column(df[col])]
                if not date_cols:
                    st.error("No date-like columns found. CSV must have a date column (e.g., '2022-09-01' or '202209').")
                    return
                date_col = st.selectbox("Date Column", date_cols, index=0)

                target_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if not target_cols:
                    st.error("No numeric columns found. CSV must have a numeric column (e.g., sales, rates).")
                    return
                target_cols = [col for col in target_cols if col != date_col]
                if not target_cols:
                    st.error(f"No numeric columns available after excluding '{date_col}'. Select a different date column or ensure a numeric column exists.")
                    return
                target_col = st.selectbox("Target Column", target_cols, index=0)

            tab1, tab2 = st.tabs(["Forecasting", "Insights"])
            with tab1:
                st.header("Forecasting")
                if st.button("Generate Forecast"):
                    with st.spinner("Generating forecast..."):
                        generate_forecast(df, date_col, target_col, horizon)
            with tab2:
                st.header("Insights")
                if st.button("Generate Insights"):
                    with st.spinner("Generating insights..."):
                        generate_insights(df, date_col, target_col, granularity)
        else:
            st.error("Failed to load CSV. Check format, size (<10 MB for Cloud), or content.")

if __name__ == "__main__":
    main()

    