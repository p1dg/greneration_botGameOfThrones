from data.data_preparation import parse_dataset
from model.train_llama import train_llama_model
from model.inference_chat_bot import RetrivalBot

from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder="templates")
messages = []
bot = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/send_message", methods=["POST"])
def send_message():
    input = request.json
    user_message = input["message"]
    if len(user_message) != 0:
        messages.append(f"you: {user_message}")
        answer = bot.get_best_rand_reply(query=user_message)
        messages.append(f"bot: {answer[0]}")
    return jsonify({"message": user_message})


@app.route("/get_messages", methods=["GET"])
def get_messages():
    return jsonify({"messages": messages})


if __name__ == "__main__":
    train_data = parse_dataset(
        save_path="./data/datasets/prepaired_data",
        data_path="./data/datasets/prepaired_data",
    )
    model = train_llama_model(train_data,)

    # bot = RetrivalBot(finetuned_ce=cross_encoder_model, data=train_data)
    # app.run(debug=True)
