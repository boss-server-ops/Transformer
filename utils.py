import os
import logging
from sentencepiece import SentencePieceProcessor as SPP

def load_tokenizer(model_name):
    tokenizer = SPP()
    model_path = f"./tokenizer/{model_name}.model"
    tokenizer.Load(model_path)
    return tokenizer

def chinese_tokenizer_load():
    return load_tokenizer("chn")

def english_tokenizer_load():
    return load_tokenizer("eng")

def set_logger(log_path):
    if os.path.isfile(log_path):
        os.remove(log_path)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)