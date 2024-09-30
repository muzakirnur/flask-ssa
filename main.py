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

def Hankelise(Y):
    """
    Hankelises the matrix Y, returning H(Y).
    """
    L, K = Y.shape
    transpose = False
    if L > K:
        Y = Y.T
        L, K = Y, L
        transpose = True

    HY = np.zeros((L,K))
    
    for m in range(L):
        for n in range(K):
            s = m+n
            if 0 <= s <= L-1:
                for l in range(0,s+1):
                    HY[m,n] += 1/(s+1)*Y[l, s-l]    
            elif L <= s <= K-1:
                for l in range(0,L-1):
                    HY[m,n] += 1/(L-1)*Y[l, s-l]
            elif K <= s <= K+L-2:
                for l in range(s-K+1,L):
                    HY[m,n] += 1/(K+L-s-1)*Y[l, s-l]
    if transpose:
        return HY.T
    else:
        return HY

def Y_to_TS(Y_i):
    """Rata-rata diagonal dari matriks dasar tertentu, Y_i, sesuai deret waktu."""
    # ubah urutan ordering
    Y_rev = Y_i[::-1]
    
    # Full credit to Mark Tolonen at https://stackoverflow.com/a/6313414 for this one:
    return np.array([Y_rev.diagonal(i).mean() for i in range(-Y_i.shape[0]+1, Y_i.shape[1])])

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
        window = int(request.form['window_length'])
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = f"{app.config['UPLOAD_FOLDER']}/{filename}"
            file.save(filepath)

            # Read the Excel file
            df = pd.read_excel(filepath)

            # Check that the required columns are present
            if 'tanggal' not in df.columns or 'kwh' not in df.columns:
                raise ValueError("Error: The file must contain 'tanggal' and 'kwh' columns.")

            dataX = np.array(df['tanggal'])
            dataY = np.array(df['kwh'])

            # Create a single figure
            fig, axs = plt.subplots(3, 2, figsize=(24, 18))
            axs = axs.flatten()  # Flatten the 2D array of axes for easy indexing

            # Original Time Series
            axs[0].plot(dataX, dataY)
            axs[0].set_title("Original Time Series")

            # Length Data Frame
            N = len(dataX)

            # Window Length
            L = window

            # Define K
            K = N - L + 1

            Y = np.column_stack([dataY[i:i + L] for i in range(0, K)])
            X = np.column_stack([dataX[i:i + L] for i in range(0, K)])

            # Matrice Lintasan
            im = axs[1].imshow(Y, aspect='auto', cmap='viridis')
            axs[1].set_xlabel("Panjang Lintasan L")
            axs[1].set_ylabel("Panjang Lintasan K")
            fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
            axs[1].set_title("Matriks Lintasan Pada Data Tersebut")

            # Mencari Rank Matriks Y
            d = np.linalg.matrix_rank(Y)

            # SVD Matriks X
            U, Sigma, V = np.linalg.svd(Y)
            V = V.T

            eigen = np.linalg.eig(np.dot(Y, Y.T))
            lambda_i = [eigen[0][i] for i in range(d)]

            Y_elem = np.array([Sigma[i] * np.outer(U[:, i], V[:, i]) for i in range(d)])

            # Components Extraction
            Ftrend = Y_to_TS(Y_elem[[0, 3, 4]].sum(axis=0))
            Fperiodic = Y_to_TS(Y_elem[[1, 2]].sum(axis=0))
            Fnoise = Y_to_TS(Y_elem[5:].sum(axis=0))

            # Plot SSA Components
            components = [("Trend", Ftrend),
                        ("Periodicity", Fperiodic),
                        ("Noise", Fnoise)]

            for n, (name, ssa_comp) in enumerate(components, start=2):  # Start from the 3rd subplot
                axs[n].plot(dataX, ssa_comp)
                axs[n].set_title(name, fontsize=16)
                axs[n].set_xticks([])

            # Data tanpa noise
            lat_without_noise = Y_to_TS(Y_elem[0:5].sum(axis=0))
            axs[5].plot(dataX, dataY, label='Original Data')
            axs[5].plot(dataX, lat_without_noise, label='Data tanpa Noise', color='orange')
            axs[5].set_title("Data tanpa Noise")
            axs[5].legend()

            # Adjust layout
            plt.tight_layout()

            # Save or show the figure
            img = plot_to_base64(fig)
            plt.close(fig)

            return render_template('result.html', plot_url=f"data:image/png;base64,{img}")

    return render_template('perhitungan.html')

app.run(debug=True)