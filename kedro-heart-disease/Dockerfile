FROM python:3.8.10-slim

WORKDIR /app/

COPY requirements.txt /build/
RUN pip install --no-cache-dir -r /build/requirements.txt

# COPY main.py /app/

COPY . /app/

# CMD ["kedro", "run", "-p", "cm", "&", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["sh", "-c", "kedro run -p cm && uvicorn main:app --host 0.0.0.0 --port 8000"]
