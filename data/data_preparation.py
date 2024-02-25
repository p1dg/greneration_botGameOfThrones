
import zipfile

import pandas as pd

from tqdm import tqdm

from utils.search_engine import SimpleSearchEngine
from utils.warning_catcher import catch_warnings


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
    replica_min_num=25,
    context_window=5,
    close_reply=4,
    top_k=3,
    save_path=None,
    data_path=None,
):
    if data_path is not None:
        return pd.read_pickle(data_path)

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

    if save_path is not None:
        df_result.to_pickle(save_path)

    return df_result
