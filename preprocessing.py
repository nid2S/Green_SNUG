from transformers import TFGPT2LMHeadModel, TFBertModel, GPT2TokenizerFast, BertTokenizerFast
import tensorflow as tf
import pandas as pd
import json

class DatasetGetter:
    def __init__(self):
        self.MODEL_NAME = "skt/kogpt2-base-v2"  # byeongal/Ko-DialoGPT
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.MODEL_NAME)

    def makeDataset(self):
        # emotional conversation
        emotional_t = json.load(open("./data/EmotionalConversation/emotional_final_training.json", "r+", encoding="utf-8"))
        emotional_t = [conv["talk"]["content"] for conv in emotional_t]  # 40827
        emotional_t = pd.DataFrame.from_dict(emotional_t)
        emotional_t.to_csv("./data/Traning_dataset/emotional_t.txt", sep="\t", encoding="utf-8")

        emotional_v = json.load(open("./data/EmotionalConversation/emotional_val.json", "r+", encoding="utf-8"))
        emotional_v = [conv["talk"]["content"] for conv in emotional_v]
        emotional_v2 = json.load(open("data/EmotionalConversation/emotional_final_val.json", "r+", encoding="utf-8"))
        emotional_v2 = [conv["talk"]["content"] for conv in emotional_v2 if conv["talk"]["content"] not in emotional_v]
        emotional_v += emotional_v2  # 9144 + 40007
        emotional_v = pd.DataFrame.from_dict(emotional_v)
        emotional_v.to_csv("./data/Validation_dataset/emotional_v.txt", sep="\t", encoding="utf-8")

        # korean speech summary

