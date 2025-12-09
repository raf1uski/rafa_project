FROM python:3.11-slim

WORKDIR /app

# Copia arquivos da aplicação
COPY main.py .
COPY requirements.txt .

# Copia pacotes Python já baixados
COPY packages /packages

# Instala pacotes offline usando /packages
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-index --find-links=/packages -r requirements.txt \
    && pip install --no-index --find-links=/packages jinja2 python-multipart httpx

# Variáveis de otimização CPU
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8
ENV NUMEXPR_NUM_THREADS=8
ENV OPENBLAS_NUM_THREADS=8
ENV UVICORN_WORKERS=2

# Expõe porta
EXPOSE 8000

# Inicia backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

