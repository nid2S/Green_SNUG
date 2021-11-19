from transformers import TFGPT2LMHeadModel, TFBertModel, GPT2TokenizerFast, BertTokenizerFast
import tensorflow as tf
import json

class DatasetGetter:
    def __init__(self):
        self.MODEL_NAME = "skt/kogpt2-base-v2"  # byeongal/Ko-DialoGPT
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.MODEL_NAME)

    def makeDataset(self):
        # emotional conversation
        # korean speech summary
        pass
