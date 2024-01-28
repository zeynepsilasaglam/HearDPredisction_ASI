FROM python:3.8.10-slim

WORKDIR /app/

COPY requirements.txt /build/
RUN pip install --no-cache-dir -r /build/requirements.txt

COPY main.py /app/

COPY . /app/

ENTRYPOINT ["uvicorn", "--host", "0.0.0.0"]
CMD ["main:app"]