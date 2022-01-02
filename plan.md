# Task
- 챗봇(대화지향 대화시스템)제작, 공부 | 허깅페이스, 다양한 API, 순수 텐서플로우만 이용해 챗봇 제작.

- 새로운 특별토큰을 추가해 생성되는 대화의 방향을 조작할 수 있게 함 -> 모든 파인튜닝용 데이터에 특별토큰을 추가해야 함.
- 만드려는 챗봇의 의도를 생각해 관련 반응의 대화를 학습시킴과 동시에, 챗봇이기에 일상적인 대화또한 학습시켜야 함.
- 챗봇이 스스로 질문이 가능해야 함 -> 스스로 대화를 시작하거나, 최소 대화에 대해 질문형 응답이 가능해야 함(나 뭐 마시는중 -> 뭐마시는데요? 같이).

- LM의 훈련방법은 다음 단어를 예측시키는 것.

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
- [여기](https://github.com/haven-jeon/KoGPT2-chatbot) 서 훈련 코드를 clone해 사용
