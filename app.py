import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
import chardet
import traceback

# Set page configuration
st.set_page_config(page_title="Simple Forecasting Tool", layout="wide", initial_sidebar_state="expanded")

# Cache key to clear on new uploads
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
            st.warning(f"File size: {file_size:.2f} MB. Files >10 MB may fail on Streamlit Cloud. Consider compressing the CSV.")
        if file_size > 200:
            st.error(f"File size: {file_size:.2f} MB exceeds local limit (200 MB). Compress or split the file.")
            return None
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding'] or 'utf-8'
        st.write(f"Detected encoding: {encoding}")
        file.seek(0)
        delimiters = [',', ';', '\t', '|']
        for delimiter in delimiters:
            try:
                file.seek(0)
                df = pd.read_csv(file, delimiter=delimiter, encoding=encoding, on_bad_lines='skip', low_memory=False)
                if len(df.columns) > 1:
                    st.info(f"Loaded CSV with delimiter '{delimiter}'")
                    return df
            except Exception as e:
                st.warning(f"Failed with delimiter '{delimiter}': {str(e)}")
                continue
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
        # Check for YYYYMM pattern (e.g., 202209)
        if sample.str.match(r'^\d{6}$').sum() >= len(sample) * 0.8:
            st.write("Detected YYYYMM pattern (e.g., 202209)")
            return True
        # Try standard date formats
        date_formats = ['%Y-%m', '%Y%m', '%Y/%m', '%m/%Y', '%b %Y', '%Y', '%Y-%m-%d', '%d/%m/%Y']
        for fmt in date_formats:
            try:
                parsed = pd.to_datetime(sample, format=fmt, errors='coerce')
                if parsed.notna().sum() >= len(sample) * 0.8:
                    st.write(f"Detected date format: {fmt}")
                    return True
            except:
                continue
        # Auto-parse as fallback
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
        st.write(f"Raw {date_col} sample (first 5):\n{df[date_col].head().to_string()}")
        st.write(f"Unique {date_col} values (first 10):\n{df[date_col].unique()[:10]}")
        st.write(f"Total unique {date_col} values: {len(df[date_col].unique())}")

        # Convert to string for consistent parsing
        df[date_col] = df[date_col].astype(str)

        # Parse YYYYMM explicitly
        if df[date_col].str.match(r'^\d{6}$').any():
            df[date_col] = pd.to_datetime(df[date_col], format='%Y%m', errors='coerce')
            st.info("Parsed dates as YYYYMM (e.g., 202209 → 2022-09-01)")
        else:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        # Try additional formats
        if df[date_col].isna().all():
            for fmt in ['%Y-%m', '%Y%m', '%Y/%m', '%m/%Y', '%b %Y', '%Y', '%Y-%m-%d', '%d/%m/%Y']:
                try:
                    df[date_col] = pd.to_datetime(df[date_col], format=fmt, errors='coerce')
                    if not df[date_col].isna().all():
                        st.info(f"Parsed dates using format: {fmt}")
                        break
                except:
                    continue

        # Drop invalid dates
        if df[date_col].isna().any():
            invalid_count = df[date_col].isna().sum()
            st.warning(f"Dropped {invalid_count} rows with invalid dates.")
            df = df.dropna(subset=[date_col])

        if df.empty:
            st.error("No valid dates remain.")
            return None

        st.write(f"Rows after date cleaning: {len(df)}")

        # Ensure target is numeric
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        if df[target_col].isna().all():
            st.error(f"Target '{target_col}' contains no valid numeric values.")
            return None

        # Aggregate by date to ensure unique dates
        df = df.groupby(date_col)[target_col].mean().reset_index()
        st.write(f"Rows after aggregation: {len(df)}")
        st.write(f"Sample after aggregation:\n{df.head().to_string()}")

        # Check unique dates
        unique_dates = len(df[date_col].unique())
        st.write(f"Unique dates after aggregation: {unique_dates}")
        if unique_dates < 2:
            st.error(f"Only {unique_dates} unique date(s) found. Need at least 2 for forecasting.")
            return None

        # Prepare for Prophet
        df = df.rename(columns={date_col: 'ds', target_col: 'y'})
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds')
        df['y'] = df['y'].fillna(df['y'].mean())
        st.write(f"Final data (rows: {len(df)}):\n{df.head().to_string()}")

        return df
    except Exception as e:
        st.error(f"Preprocessing failed: {str(e)}\n{traceback.format_exc()}")
        return None

def calculate_metrics(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    return rmse, mae, mape

def forecast_data(df, date_col, target_col, horizon):
    try:
        processed_df = preprocess_time_series(df, date_col, target_col)
        if processed_df is None:
            return
        st.write(f"Processed data rows: {len(processed_df)}")
        train_size = int(len(processed_df) * 0.8)
        if train_size < 1:
            st.error("Insufficient data for training.")
            return
        train = processed_df[:train_size]
        test = processed_df[train_size:]
        st.write(f"Train size: {len(train)}, Test size: {len(test)}")

        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
        model.fit(train)
        future = model.make_future_dataframe(periods=horizon, freq='M')
        forecast = model.predict(future)
        y_pred = forecast['yhat'].iloc[train_size:train_size+len(test)]
        rmse, mae, mape = calculate_metrics(test['y'], y_pred)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train['ds'], y=train['y'], mode='lines', name='Train'))
        fig.add_trace(go.Scatter(x=test['ds'], y=test['y'], mode='lines', name='Test'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
        fig.update_layout(
            title=f"Forecast (RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%)",
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

def main():
    st.title("Simple Forecasting Tool")
    st.markdown("Upload a CSV with a date column (e.g., YYYYMM format) and a numeric column to forecast monthly trends.")

    with st.sidebar:
        st.header("Forecast Settings")
        horizon = st.number_input("Forecast Horizon (months)", min_value=1, max_value=12, value=5, help="Number of months to forecast")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], help="Files >10 MB may fail on Streamlit Cloud.", on_change=clear_cache)

    if uploaded_file is not None:
        df = load_data(uploaded_file, key=st.session_state.upload_key)
        if df is not None and not df.empty:
            st.success(f"Loaded {len(df)} rows and {len(df.columns)} columns!")
            st.write(f"Columns: {df.columns.tolist()}")

            with st.sidebar:
                date_cols = [col for col in df.columns if is_date_column(df[col])]
                if not date_cols:
                    st.error("No date-like columns found. Ensure your CSV has a date column (e.g., '_YearMonth' in YYYYMM format).")
                    return
                default_date = '_YearMonth' if '_YearMonth' in date_cols else date_cols[0]
                date_col = st.selectbox("Date Column", date_cols, index=date_cols.index(default_date), help="E.g., '_YearMonth' (YYYYMM)")

                target_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if not target_cols:
                    st.error("No numeric columns found. Ensure your CSV has a numeric column (e.g., 'MonthlyRate').")
                    return
                default_target = 'MonthlyRate' if 'MonthlyRate' in target_cols else target_cols[0]
                target_col = st.selectbox("Target Column", target_cols, index=target_cols.index(default_target), help="E.g., 'MonthlyRate'")

            if st.button("Generate Forecast"):
                with st.spinner("Generating forecast..."):
                    forecast_data(df, date_col, target_col, horizon)
        else:
            st.error("Failed to load CSV. Check file format, size, or content.")

if __name__ == "__main__":
    main()