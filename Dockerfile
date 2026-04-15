FROM data-service.inventec.com:1443/pytorch-transformer-gpu-emc:latest

WORKDIR /app

COPY requirements.txt .

# Install dependencies (skip torch — already provided by base image)
RUN pip install --no-cache-dir \
    streamlit>=1.32.0 \
    openpyxl>=3.1.0 \
    pandas>=2.0.0 \
    scikit-learn>=1.4.0 \
    plotly>=5.20.0 \
    sentence-transformers>=3.0.0

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.fileWatcherType=none"]
