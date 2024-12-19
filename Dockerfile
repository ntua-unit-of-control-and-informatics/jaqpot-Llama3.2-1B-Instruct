## Training stage
#FROM python:3.10 AS trainer
#
#WORKDIR /code
#
#COPY ./requirements.txt /code/requirements.txt
#COPY ./src /code/src
#COPY ./train.py /code/train.py
#COPY ./llama /code/llama
#
#RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
#RUN python train.py

# Deployment stage
FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./src /code/src
COPY ./main.py /code/main.py
COPY /model.pkl /code/model.pkl
COPY /tokenizer.pkl /code/tokenizer.pkl

#COPY --from=trainer /code/model.pkl /code/model.pkl
#COPY --from=trainer /code/tokenizer.pkl /code/tokenizer.pkl

EXPOSE 8005

CMD ["python", "-m", "main", "--port", "8005"]
