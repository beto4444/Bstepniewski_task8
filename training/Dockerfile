FROM python:3.12

ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}

WORKDIR /app

COPY training /app/training/

COPY utils.py /app

COPY ${settings_name} /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN python3 training/train.py