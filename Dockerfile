FROM python:3.10

WORKDIR /usr/src/app

COPY ./ ./

COPY ./requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

CMD [ "python3", "./main.py" ]