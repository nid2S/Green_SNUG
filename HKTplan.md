# Task
- 사람을 위한 인공지능 -> 챗봇 '나린'.
- 채팅형 대화기반 공감/위로(멘탈웰빙)챗봇

# data
- [챗봇 데이터](https://github.com/songys/Chatbot_data)
- [감성대화 말뭉치](https://aihub.or.kr/aidata/7978) 
- [한국어 대화](https://aihub.or.kr/aidata/85/download)
- [한국어 대화 요약](https://aihub.or.kr/aidata/30714) -> 사용 안함

# 모델
- [byeongal/Ko-DialoGPT](https://huggingface.co/byeongal/Ko-DialoGPT) 파인튜닝해 사용.

# 오류
- ValueError: Dimensions must be equal, but are 170 and 150 for '{{node Equal}} = Equal[T=DT_FLOAT, incompatible_shape_error=true](Cast_14, Cast_15)' with input shapes: [?,170], [?,150].
- 이유 -> 결국 y_pred가 170이여야 하는데 150임. | y_pred를 170으로 만들자!
