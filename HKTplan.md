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
- case 1 : 일반데이터 -> 일반데이터 | case 2 : 데이터</s> -> 데이터 | case 3 : 데이터</s> -> 데이터</s>데이터(</s>)
- 정확한 데이터 형식과 파인튜닝 방법을 안 뒤 파인튜닝.

# 오류
- ValueError: Dimensions must be equal, but are 170 and 150 for '{{node Equal}} = Equal[T=DT_FLOAT, incompatible_shape_error=true](Cast_14, Cast_15)' with input shapes: [?,170], [?,150].
- 이유 : 결국 y_pred가 150으로, y의 170차원과 다름. | 애초부터 입력과 출력의 차원이 같아야 하는건지, 출력의 차원을 결정하는 무언가를 놓친건지 생각 필요.
- 해결시도 : 1. loss함수 변경 -> 데이터 가공 후 진행 | 2. 허깅페이스 모델페이지 재확인 | 3. 허깅페이스 API재확인 -> 크로스어텐션? past? | 4. 깃허브 탐색
- 해결시도2 : 일단 데이터를 가공해 입력은 dict로, 레이블은 원핫 인코딩/평범하게 sparse 로 바꿔 파인튜닝을 진행.
