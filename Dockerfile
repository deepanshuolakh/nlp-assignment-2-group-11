FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
COPY code code
CMD ["python", "code/run.py", "all"]
