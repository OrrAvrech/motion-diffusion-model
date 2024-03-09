import json
from pathlib import Path
from typing import List, Union, Dict


def read_text(text_path: Path) -> Union[List[Dict], Dict]:
    with open(str(text_path), "r") as fp:
        text = json.load(fp)
    return text


def get_gpt_sentences(gpt_path: Path) -> List[str]:
    response = read_text(gpt_path)
    sentences = response["choices"][0]["message"]["content"].split("\n")
    return sentences