# Base image
FROM python:3.10-slim

# Env settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# Install system dependencies (needed for sklearn/xgboost)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install streamlit separately (if not in requirements)
RUN pip install streamlit requests

# Copy project files
COPY . .

# Set PYTHONPATH so src works
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 8000
EXPOSE 8501

# Run both API + Streamlit
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]