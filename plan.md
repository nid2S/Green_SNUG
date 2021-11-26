# Task
- 챗봇(대화지향 대화시스템)제작, 공부 | 허깅페이스, 다양한 API, 순수 텐서플로우만 이용해 챗봇 제작.

- 새로운 특별토큰을 추가해 생성되는 대화의 방향을 조작할 수 있게 함 -> 모든 파인튜닝용 데이터에 특별토큰을 추가해야 함.
- 만드려는 챗봇의 의도를 생각해 관련 반응의 대화를 학습시킴과 동시에, 챗봇이기에 일상적인 대화또한 학습시켜야 함.
- 챗봇이 스스로 질문이 가능해야 함 -> 스스로 대화를 시작하거나, 최소 대화에 대해 질문형 응답이 가능해야 함(나 뭐 마시는중 -> 뭐마시는데요? 같이).

- LM의 훈련방법은 다음 단어를 예측시키는 것.
- GPT의 구조는 (?)
- 이루다의 구조 : DialogBERT(NLU모듈, 스캐터랩에서 수집한 카카오톡 대화데이터로 학습된 BERT)를 이용해 텍스트(메세지)를 하나의 벡터로 치환 -> 
  Session/ContentDB(답변 후보들을 저장해놓은 DB, 카카오톡 대화 데이터중 질 높은 대화와 답변을 선정해 저장)에서 여태까지의 대화를 보고 답변 후보 N개 생성
  (문장 벡터를 차원축소해 DB에 저장한 뒤 코사인 거리를 계산) -> 선정된 후보 N개중 하나를 고르기 위해 Re-ranker모델(각 개별 대화데이터를 학습, SSA점수 기반 추가로 레이블링)사용.
- 이루다 구조 장단점 : 모든 답변이 기존 데이터에서 나와 답변이 무척 자연스러우나 수많은 데이터가 필요하고, 실제 글을 그대로 사용해 개인정보 유출의 위험이 있음.

# Data
- [감성대화 말뭉치](https://aihub.or.kr/aidata/7978) 
- [한국어 대화 요약](https://aihub.or.kr/aidata/30714)
- [한국어 대화](https://aihub.or.kr/aidata/85/download)
- [챗봇 데이터](https://github.com/songys/Chatbot_data)
- [한국어 음성](https://aihub.or.kr/aidata/105)

- [트위터에서 수집 및 정제한 대화 시나리오](https://aihub.or.kr/opendata/keti-data/recognition-laguage/KETI-02-008) | 연구용도로만 사용가능
- [한국어 감정 정보가 포함된 연속적 대화 데이터셋](https://aihub.or.kr/opendata/keti-data/recognition-laguage/KETI-02-010) | 연구용도로만 사용가능
- [한국어 대화 데이터셋](https://aihub.or.kr/opendata/keti-data/recognition-laguage/KETI-02-011) | 연구용도로만 사용가능
- [웰니스 대화 스크립트 데이터셋](https://aihub.or.kr/keti_data_board/language_intelligence) | 연구용도로만 사용가능


# HF
- 모델 : [byeongal/Ko-DialoGPT](https://huggingface.co/byeongal/Ko-DialoGPT) 와 [skt/kogpt2-base-v2](https://huggingface.co/skt/kogpt2-base-v2) 를 발견.
  이로 불충분하다면 koGPT를 이용해 lowLevel에서 제작. -> ko-DialoGPT : max_length, num_beams등의 매개변수를 써야 효과가 좋음. 대화모델의 파인튜닝 방법을 찾아봄.
