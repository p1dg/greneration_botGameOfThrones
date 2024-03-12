FROM python:3.10.12

WORKDIR /usr/src/app

COPY ./ ./

COPY ./requirements.txt ./
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install -r requirements.txt

CMD [ "python3", "./main.py" ]