FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./src /code/src

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN python train.py

COPY ./main.py ./model.pkl ./tokenizer.pkl /code/

EXPOSE 8002

CMD ["python", "-m", "main", "--port", "8002"]
