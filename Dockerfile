FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "handler.py"]
