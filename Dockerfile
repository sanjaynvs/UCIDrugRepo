# pull python base image
FROM python:3.10-slim

# copy application files
ADD /ucidrugrepo_api /ucidrugrepo_api/

# specify working directory
WORKDIR /ucidrugrepo_api

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

# expose port for application
EXPOSE 5000

# start fastapi application
CMD ["python", "app/app.py"]
