FROM python:3.8-slim-buster
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
WORKDIR /code
COPY requirements.txt /code
RUN pip install -r requirements.txt
COPY . /code
ENTRYPOINT ["python","demo.py"]