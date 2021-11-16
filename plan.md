# Task
- 챗봇(대화지향 대화시스템)제작, 공부.
- 허깅페이스, 다양한 API, 순수 텐서플로우만 이용해 챗봇 제작.
- 새로운 특별토큰을 추가해 생성되는 대화의 방향을 조작할 수 있게 함 -> 모든 파인튜닝용 데이터에 특별토큰을 추가해야 함.
- 만드려는 챗봇의 의도를 생각해 관련 반응의 대화를 학습시킴과 동시에, 챗봇이기에 일상적인 대화또한 학습시켜야 함.
- 챗봇이 스스로 질문이 가능해야 함 -> 스스로 대화를 시작하거나, 최소 대화에 대해 질문형 응답이 가능해야 함(나 뭐 마시는중 -> 뭐마시는데요? 같이).
- LM의 훈련방법은 다음 단어를 예측시키는 것 -> GPT의 구조를 참고해 훈련 각 대화를 훈련? ==> pretrained된 한국어 모델이 필요. 

# Data
- AI Hub[감성대화 말뭉치](https://aihub.or.kr/aidata/7978) 
- AI Hub[한국어 대화 요약](https://aihub.or.kr/aidata/30714) 
- AI Hub[트위터에서 수집 및 정제한 대화 시나리오](https://aihub.or.kr/opendata/keti-data/recognition-laguage/KETI-02-008) | 상업 이용 불가 -> 미사용

# HF
- 모델 : [Ko-DialoGPT](https://huggingface.co/byeongal/Ko-DialoGPT) 와 [skt/kogpt2-base-v2](https://huggingface.co/skt/kogpt2-base-v2) 를 발견.
  먼저 ko-DialoGPT를 사용(DialoGPT를 알아보는 중)해 챗봇을 만들어보고, 이로 불충분하다면 koGPT를 이용해 lowLevel에서 제작. 이후 tf로의 제작을 위해 구조에 신경.

- 허깅페이스 홈페이지의 예제 코드를 기반으로 영어 챗봇을 제작 > 예제를 기반으로 다른 모델을 사용해 한국어 챗봇을 제작 > 공식 문서를 참고하며 row-level로 챗봇 제작.
