
import zipfile

import pandas as pd

import re

from tqdm import tqdm

import random

import datasets

from utils.search_engine import SimpleSearchEngine
from utils.warning_catcher import catch_warnings

def remove_between_square_brackets(text):
    return re.sub('', '', text)

def convert2several_words(text):
  new_text = ''
  for word in text.split():
    if not word.isupper():
      word = "".join((" " + w if (w.isupper() and i) else w) for i, w in enumerate(word))
      new_text+= " " + word
    else:
      new_text+=' '+word
  return ' '.join(new_text.split())
    
def make_prompt(df, use_col):
    id = random.randint(0, len(df[use_col]["companions"]))
    query = df['query']
    role = df['companions']
    context = df['context']
    reply = df["close_reply"]
    prompt = f"[INST]"
    prompt += f'Use the given context to guide your an about the query like indicated in your role'
    prompt += f"query: {query}\n\n"
    prompt += f"context: {context}\n\n"
    prompt += f"your role: {role}\n\n"
    prompt += f'answer:{reply}[/INST]'
    return prompt

def clean_text(text):
    text = remove_between_square_brackets(text)
    text = convert2several_words(text)
    return ' '.join(text.split())

def window_back(id, win):
    if (id - win) < 0:
        win = id
    return win


def window(id, win, top):
    if (id + win) > top:
        win = id
    return win


@catch_warnings(log_path="./logs/warnings_data.log")
def parse_dataset(
    archive_path="./data/archive.zip",
    extract_path="./data/datasets",
    replica_min_num=10,
    context_window=5,
    close_reply=10,
    top_k=5,
    save_path=None,
    data_path=None,
):
    if data_path is not None:
        return datasets.load_from_disk(data_path)

    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    raw_data = pd.read_csv(f"{extract_path}/Game_of_Thrones_Script.csv")

    # удаляем не гланых героев

    main_characters = (
        raw_data["Name"]
        .value_counts()[raw_data["Name"].value_counts() > replica_min_num]
        .index
    )
    raw_data = raw_data[raw_data["Name"].isin(main_characters)]

    # удаляем реплики персонажа самому себе
    raw_data["repeat"] = raw_data["Name"].shift(-1)
    raw_data = raw_data[raw_data["Name"] != raw_data["repeat"]]

    # фиксируем строковый формат
    raw_data["Sentence"] = raw_data["Sentence"].apply(str)

    df_result = raw_data[["Season", "Episode"]]
    simple_search_engine = SimpleSearchEngine(raw_data["Sentence"], top_k)

    # speaker - тот кто говорит первый
    df_result["speaker"] = raw_data["Name"].shift(1)

    # Что говорит speaker
    df_result["query"] = raw_data["Sentence"].shift(1)

    # Собеседники
    df_result["companions"] = [
        raw_data.iloc[id + 1 : id + 1 + window(id, close_reply, raw_data.shape[0])][
            "Name"
        ].to_list()
        for id in tqdm(raw_data.index, desc="формируем собеседников")
    ]

    # Ближайшие реплики собеседников
    df_result["close_reply"] = [
        raw_data.iloc[id + 1 : id + 1 + window(id, close_reply, raw_data.shape[0])][
            "Sentence"
        ].to_list()
        for id in tqdm(raw_data.index, desc="формируем реплики собеседников")
    ]

    # реплики подобранные на Tf-Idf
    tqdm.pandas(desc="подбираем best реплики")
    df_result["neutral_reply"] = raw_data["Sentence"].progress_apply(
        lambda query: simple_search_engine.display_relevant_docs(
            query, raw_data["Name"], "best"
        )
    )

    # плохие реплики подобранные на Tf-Idf
    tqdm.pandas(desc="подбираем bad_reply реплики")
    df_result["bad_reply"] = raw_data["Sentence"].progress_apply(
        lambda query: simple_search_engine.display_relevant_docs(
            query, raw_data["Name"], "bad"
        )
    )

    # контекст прошлых реплик разговора
    df_result["context"] = [
        ". ".join(
            raw_data.iloc[id - window_back(id, context_window) : id][
                "Sentence"
            ].to_list()
        )
        for id in tqdm(
            raw_data.index, desc="формируем контекст прошлых реплик разговора"
        )
    ]

    df_result = df_result.dropna()

    data = df_result[~(df_result['companions'].apply(len) == 0)].reset_index(drop=True)
    
    use_col = ['speaker', 'query', 'context', "companions", "close_reply"]

    train_df = data.iloc[:-5][use_col]

    for index, row in tqdm(data.iloc[:-5:,:].iterrows(), total=len(data), desc="формируем train_df"):
        train_df.loc[index,"query"] = clean_text(row["query"])
        train_df.loc[index, "context"] = clean_text(row["context"])
        id = random.randint(0, len(row["companions"]) - 1)
        train_df.loc[index, "companions"] = row["companions"][id]
        train_df.loc[index, "close_reply"] = clean_text(row["close_reply"][id])

    for index, row in tqdm(train_df.iterrows(), total=len(train_df), desc="формируем prompt-ы"):
        train_df.loc[index, "prompt"] = make_prompt(row, use_col=use_col)

    dataset =  datasets.Dataset.from_pandas(train_df[["prompt"]] )

    if save_path is not None:
        dataset.save_to_disk(save_path)

    return dataset
