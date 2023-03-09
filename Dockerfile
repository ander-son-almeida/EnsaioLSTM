# Escolha a imagem base
FROM python:3.9-slim-buster

# Copie o código fonte para a imagem
COPY . /app
WORKDIR /app

# Instale as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Execute o código
CMD ["python", "EnsaioLSTM.py"]