# pull official base image
FROM python:3
# add and run as non-root user.


EXPOSE 8080
# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBUG 0

# set work directory
WORKDIR /app
# install dependencies
COPY ./requirements.txt .
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install numpy
RUN pip install pandas
RUN pip install matplotlib
RUN pip install tensorflow
RUN pip install scikit-learn

RUN pip install -r requirements.txt

# copy project
COPY . .
#run the server
CMD gunicorn dict.wsgi:application --bind 0.0.0.0:$PORT
