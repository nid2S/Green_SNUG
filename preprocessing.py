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
        emotional_v2 = json.load(open("./data/EmotionalConversation/emotional_final_val.json", "r+", encoding="utf-8"))
        emotional_v2 = [conv["talk"]["content"] for conv in emotional_v2 if conv["talk"]["content"] not in emotional_v]
        emotional_v += emotional_v2  # 9144 + 40007
        emotional_v = pd.DataFrame.from_dict(emotional_v)
        emotional_v.to_csv("./data/Validation_dataset/emotional_v.txt", sep="\t", encoding="utf-8")

        # korean speech summary
        topics = ["개인및관계", "미용과건강", "상거래(쇼핑)", "시사교육", "식음료", "여가생활", "일과직업", "주거와생활", "행사"]
        # 74024 +
        koreanSpeech_t = ""  # topic | type | turns | participants | dialogue
        koreanSpeech_v = ""
        for topic in topics:
            data_t = json.load(open("./data/KoreanSpeakSummary/Training/"+topic+".json", "r+", encoding="utf-8"))
            data_v = json.load(open("./data/KoreanSpeakSummary/Vaildation/"+topic+".json", "r+", encoding="utf-8"))
            for conv in data_t:
                conv_topic = conv["header"]["topic"]
                conv_type = conv["header"]["type"]
                conv_turn = conv["header"]["numberOfTurns"]
                conv_part = conv["header"]["numberOfParticipants"]

                conv_dialogue = ""
                for dialogue in conv['body']["dialogue"]:
                    conv_dialogue += dialogue["turnID"]+"_"+dialogue["participantID"]+"_"+dialogue["utterance"]+"\t"
                koreanSpeech_t += conv_topic+" | "+conv_type+" | "+conv_turn+" | "+conv_part+" | "+conv_dialogue+"\n"
            koreanSpeech_t += "\n"

            for conv in data_v:
                conv_topic = conv["header"]["topic"]
                conv_type = conv["header"]["type"]
                conv_turn = conv["header"]["numberOfTurns"]
                conv_part = conv["header"]["numberOfParticipants"]

                conv_dialogue = ""
                for dialogue in conv['body']["dialogue"]:
                    conv_dialogue += dialogue["turnID"]+"_"+dialogue["participantID"]+"_"+dialogue["utterance"]+"\t"
                koreanSpeech_v += conv_topic+" | "+conv_type+" | "+conv_turn+" | "+conv_part+" | "+conv_dialogue+"\n"
            koreanSpeech_v += "\n"

        open("./data/Training_dataset/KoreanSpeech_t.txt", "w+", encoding="utf-8").write(koreanSpeech_t)
        open("./data/Vaildation_dataset/KoreanSpeech_v.txt", "w+", encoding="utf-8").write(koreanSpeech_v)
