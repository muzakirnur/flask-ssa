from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from scipy.linalg import svd

app=Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def perform_ssa(series, window_length):
    n = len(series)
    if window_length >= n:
        raise ValueError("Window length must be smaller than the length of the time series")
    
    # Create the trajectory matrix
    X = np.asarray([series[i:i+window_length] for i in range(n - window_length + 1)])
    
    # Perform Singular Value Decomposition (SVD)
    U, s, Vt = svd(X, full_matrices=False)
    
    # Reconstruct the trend and seasonal components
    trend = np.mean(X, axis=0)
    seasonal = np.mean(X - trend, axis=0)
    
    # Extend the trend and seasonal components to match the original series length
    trend_extended = np.concatenate([np.full(window_length - 1, np.nan), trend])
    seasonal_extended = np.concatenate([np.full(window_length - 1, np.nan), seasonal])
    
    return trend_extended, seasonal_extended

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/perhitungan', methods=['GET', 'POST'])
def perhitungan():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = f"{app.config['UPLOAD_FOLDER']}/{filename}"
            file.save(filepath)

            # read the excel file
            df = pd.read_excel(filepath)

            # Check that the required columns are present
            if 'tanggal' not in df.columns or 'kwh' not in df.columns:
                return "Error: The file must contain 'tanggal' and 'kwh' columns."

            # Convert 'tanggal' to datetime and ensure 'kwh' is numeric
            df['tanggal'] = pd.to_datetime(df['tanggal'])
            df['kwh'] = pd.to_numeric(df['kwh'], errors='coerce')

            # Drop rows with missing values in 'kwh'
            df = df.dropna(subset=['kwh'])

            # Perform SSA on the 'kwh' column
            series = df['kwh'].values
            tanggal = df['tanggal']
            window_length = 12  # Updated window length

            try:
                trend, seasonal = perform_ssa(series, window_length)
            except ValueError as e:
                return str(e)

            # Ensure that trend and seasonal components align with the length of the original series
            extended_tanggal = pd.date_range(start=tanggal.iloc[0], periods=len(trend), freq='D')
            
            # Plot components
            fig, ax = plt.subplots(3, 1, figsize=(12, 15))
            
            # Original Time Series
            ax[0].plot(tanggal, series, label='Original Time Series')
            ax[0].legend(loc='upper right')
            ax[0].set_title('Original Time Series')
            ax[0].set_xlabel('Tanggal')
            ax[0].set_ylabel('kwh')

            # Trend Component
            ax[1].plot(extended_tanggal, trend, label='Trend Component', color='orange')
            ax[1].legend(loc='upper right')
            ax[1].set_title('Trend Component')
            ax[1].set_xlabel('Tanggal')
            ax[1].set_ylabel('Trend')

            # Seasonal Component
            ax[2].plot(extended_tanggal, seasonal, label='Seasonal Component', color='green')
            ax[2].legend(loc='upper right')
            ax[2].set_title('Seasonal Component')
            ax[2].set_xlabel('Tanggal')
            ax[2].set_ylabel('Seasonal')

            plt.tight_layout()
            img = plot_to_base64(fig)
            plt.close(fig)

            return render_template('result.html', plot_url=f"data:image/png;base64,{img}")

    return render_template('perhitungan.html')

app.run(debug=True)