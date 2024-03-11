from data.data_preparation import parse_dataset
from model.train_llama import train_llama_model
from model.inference_chat_bot import GenerationBot

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
        answer, role = bot.generate_answer(query=user_message)
        messages.append(f"{role}: {answer}")
    return jsonify({"message": user_message})


@app.route("/get_messages", methods=["GET"])
def get_messages():
    return jsonify({"messages": messages})


if __name__ == "__main__":
    train_data = parse_dataset(
        save_path="./data/datasets/prepaired_data",
        # data_path="./data/datasets/prepaired_data",
    )
    model = train_llama_model(train_data,)

    bot = GenerationBot(
        trained_model_dir='./models_storage/checkpoint-700',
        role='daenerys targaryen',
    )
    answer, role = bot.generate_answer('whhat happend to ned stark?')
    app.run(debug=True)
