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
st.set_page_config(page_title="Monthly Forecasting Tool", layout="wide", initial_sidebar_state="expanded")

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
        for delimiter in [',', ';', '\t', '|']:
            try:
                file.seek(0)
                df = pd.read_csv(file, delimiter=delimiter, encoding=encoding, on_bad_lines='skip', low_memory=False)
                if len(df.columns) >= 2:
                    st.info(f"Loaded CSV with delimiter '{delimiter}'")
                    return df
            except Exception as e:
                st.warning(f"Failed with delimiter '{delimiter}': {str(e)}")
        st.error("Could not parse CSV.")
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

def preprocess_time_series(df, date_col, target_col, freq='M'):
    try:
        st.write(f"Preprocessing: Date = {date_col}, Target = {target_col}")
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

        # Aggregate to monthly frequency
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

def generate_forecast(df, date_col, target_col, horizon):
    try:
        processed_df = preprocess_time_series(df, date_col, target_col, freq='M')
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

        model = Prophet(yearly_seasonality=len(processed_df) >= 12, weekly_seasonality=False, daily_seasonality=False)
        model.fit(train)
        future = model.make_future_dataframe(periods=horizon, freq='M')
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

        # Main Forecast Plot
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=train['ds'], y=train['y'], mode='lines', name='Train'))
        fig1.add_trace(go.Scatter(x=test['ds'], y=test['y'], mode='lines', name='Test'))
        fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
        fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,100,80,0.2)', name='Upper CI'))
        fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,100,80,0.2)', name='Lower CI'))
        fig1.update_layout(
            title=f"Monthly Forecast (RMSE: {rmse if rmse else 'N/A'}, MAE: {mae if mae else 'N/A'}, MAPE: {mape if mape else 'N/A'}%)",
            xaxis_title="Date",
            yaxis_title=target_col,
            template="plotly_white"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Actual vs Forecast Plot
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=test['ds'], y=test['y'], mode='lines', name='Actual'))
        fig2.add_trace(go.Scatter(x=test_df['ds'], y=test_df['yhat'], mode='lines', name='Predicted'))
        fig2.update_layout(
            title="Actual vs Predicted (Test Set)",
            xaxis_title="Date",
            yaxis_title=target_col,
            template="plotly_white"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Monthly Forecast Report
        forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={'ds': 'Date', 'yhat': 'Forecast', 'yhat_lower': 'Lower CI', 'yhat_upper': 'Upper CI'})
        st.write("Monthly Forecast (last 10 rows):")
        st.dataframe(forecast_df.tail(10))
        st.download_button(
            "Download Monthly Forecast",
            forecast_df.to_csv(index=False).encode('utf-8'),
            file_name="monthly_forecast.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Forecasting failed: {str(e)}\n{traceback.format_exc()}")

def generate_insights(df, date_col, target_col, granularity='Monthly'):
    try:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        df = df.dropna(subset=[target_col])

        if df.empty:
            st.error("No valid data for insights.")
            return

        df.set_index(date_col, inplace=True)
        df_resampled = df[target_col].resample('M').agg(['mean', 'std', 'count']).reset_index()
        df_resampled = df_resampled.dropna()

        if len(df_resampled) < 1:
            st.error("No data available for monthly insights.")
            return

        st.subheader("Monthly Insights")

        # Summary Statistics
        stats_df = df_resampled['mean'].describe()
        stats_df['skewness'] = df_resampled['mean'].skew()
        stats_df['kurtosis'] = df_resampled['mean'].kurtosis()
        st.write("**Monthly Statistics**")
        st.dataframe(stats_df, use_container_width=True)

        # Trend Analysis
        trend = "Increasing" if df_resampled['mean'].iloc[-1] > df_resampled['mean'].iloc[0] else "Decreasing"
        st.write(f"**Trend**: {trend} over the period.")

        # Outlier Detection
        z_scores = np.abs(stats.zscore(df_resampled['mean']))
        outliers = df_resampled[z_scores > 3]
        if not outliers.empty:
            st.write(f"**Outliers Detected** ({len(outliers)})")
            st.dataframe(outliers[[date_col, 'mean']], use_container_width=True)
        else:
            st.write("**Outliers**: None detected.")

        # Rolling Mean Plot
        df_resampled['rolling_mean'] = df_resampled['mean'].rolling(window=3, min_periods=1).mean()
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=df_resampled[date_col], y=df_resampled['mean'], mode='lines', name='Mean'))
        fig_line.add_trace(go.Scatter(x=df_resampled[date_col], y=df_resampled['rolling_mean'], mode='lines', name='Rolling Mean (3 months)'))
        fig_line.update_layout(
            title="Monthly Trend",
            xaxis_title="Date",
            yaxis_title=target_col,
            template="plotly_white"
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # Distribution Plot
        fig_box = px.box(df_resampled, y='mean', title="Monthly Distribution")
        st.plotly_chart(fig_box, use_container_width=True)

        insights_df = df_resampled.rename(columns={date_col: 'Date', 'mean': 'Mean', 'std': 'Std Dev', 'count': 'Count'})
        st.write("Monthly Insights (last 10 rows):")
        st.dataframe(insights_df.tail(10))
        st.download_button(
            "Download Monthly Insights",
            insights_df.to_csv(index=False).encode('utf-8'),
            file_name="monthly_insights.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Insights failed: {str(e)}\n{traceback.format_exc()}")

def main():
    st.title("Monthly Forecasting & Insights Tool")
    st.markdown("Upload a CSV with a date column (e.g., 2022-09-01) and a numeric column for monthly forecasts and insights.")

    with st.sidebar:
        st.header("Settings")
        horizon = st.number_input("Forecast Horizon (months)", min_value=1, max_value=12, value=3)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], help="Files >10 MB may fail on Streamlit Cloud.", on_change=clear_cache)

    if uploaded_file is not None:
        df = load_data(uploaded_file, key=st.session_state.upload_key)
        if df is not None and not df.empty:
            st.success(f"Loaded {len(df)} rows, {len(df.columns)} columns!")
            st.write(f"Columns: {df.columns.tolist()}")

            with st.sidebar:
                date_cols = [col for col in df.columns if is_date_column(df[col])]
                if not date_cols:
                    st.error("No date column found (e.g., '2022-09-01').")
                    return
                date_col = st.selectbox("Date Column", date_cols, index=0)

                target_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                target_cols = [col for col in target_cols if col != date_col]
                if not target_cols:
                    st.error("No numeric columns found.")
                    return
                target_col = st.selectbox("Target Column", target_cols, index=0)

            tab1, tab2 = st.tabs(["Forecasting", "Insights"])
            with tab1:
                st.header("Monthly Forecasting")
                if st.button("Generate Forecast"):
                    with st.spinner("Generating forecast..."):
                        generate_forecast(df, date_col, target_col, horizon)
            with tab2:
                st.header("Insights")
                if st.button("Generate Insights"):
                    with st.spinner("Generating insights..."):
                        generate_insights(df, date_col, target_col)
        else:
            st.error("Failed to load CSV.")

if __name__ == "__main__":
    main()