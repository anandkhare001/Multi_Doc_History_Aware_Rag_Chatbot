FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# WORKDIR must point to the gui package folder
WORKDIR /app/gui

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy only the gui package contents into WORKDIR
COPY gui/ .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]