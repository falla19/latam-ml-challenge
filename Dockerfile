# syntax=docker/dockerfile:1.2
FROM python:3.9
# put you docker configuration here
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./challenge /code/challenge
COPY ./data /code/data

CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
