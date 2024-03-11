# retrival_botGameOfThrones

### How to install

1. python3 -m venv .venv
2. source .venv/bin/activate
3. pip3 install --upgrade pip
4. pip3 install -r requirements.txt

### How to start

1. python3 main.py

### OR

docker build --tag 'generation_bot' . 

docker run -it --rm --name generation_bot -p 5000:5000  generation_bot:latest
