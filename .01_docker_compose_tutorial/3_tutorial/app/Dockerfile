FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

RUN python train.py

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]