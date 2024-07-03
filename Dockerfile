FROM ubuntu:latest
FROM python:3.9-slim

COPY . /app
WORKDIR /app
EXPOSE 3307
EXPOSE 5000

ENV MYSQL_PASSWORD=x155564py
ENV MYSQL_DATABASE=smashup
ENV MYSQL_USER=root
ENV MYSQL_HOST=188.130.155.181
ENV MYSQL_PORT=3307

RUN python -m pip install -r requirements.txt
CMD ["python", "mashup_recsys.py"]