FROM python:3.10

WORKDIR /mf-app

COPY ./requirements.txt /mf-app/requirements.txt
COPY ./src /mf-app/src

RUN pip install --no-cache-dir --upgrade -r /mf-app/requirements.txt

CMD [ "python","./src/main.py" ]