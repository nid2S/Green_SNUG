# Task
- 사람을 위한 인공지능 -> 챗봇 '나린'.
- 채팅형 대화기반 공감/위로(멘탈웰빙)챗봇

# data
- [챗봇 데이터](https://github.com/songys/Chatbot_data)
- [감성대화 말뭉치](https://aihub.or.kr/aidata/7978) 
- [한국어 대화](https://aihub.or.kr/aidata/85/download)
- [한국어 대화 요약](https://aihub.or.kr/aidata/30714) -> 사용 안함

# 모델
- [byeongal/Ko-DialoGPT](https://huggingface.co/byeongal/Ko-DialoGPT) 파인튜닝해 사용. | cc-by-nc-sa-4.0 License(non-commercial)
- 내 데이터 형식? : x - S1</s>S2</s> | y : <s>R1</s> => x - 그대로(길면 자름?) | y - 1.마스킹<bos>R</s>  2.x+y
  데이터가 길다면 일부를 잘라 사용(x는 뒤에서, y는 앞에서)? 

# 오류
- 1. tensorflow.python.framework.errors_impl.InvalidArgumentError:  logits and labels must have the same first dimension, got logits shape [65280,64] and labels shape [2720]
- loss -> SparseCategoricalCrossentropy(from_logits=True), data -> (16(batch). 170)형태. | 65280, 64, 2720의 출처가 어디인지 파악 필요.
- 2. tensorflow.python.framework.errors_impl.InvalidArgumentError:  Incompatible shapes: [2,16,12,170] vs. [16,170] 
- loss -> SparseCategoricalCrossentropy() | 양립할 수 없는 형태. 익숙한 숫자가 몇 보임. 정확히 어느 단계에서 나는 오류인지 파악되지 않음. 순전파 과정 아닐까 추측.
- 3. ValueError: Shapes (None, 170) and (None, 170, 51200) are incompatible
- loss -> CategoricalCrossentropy(from_logits=True) | (None, 170, 51200)이 어디에서 나오는 건지 파악 필요(loss함수에서 나는 오류). 아마 모델의 반환값?
- 4. ValueError: Shapes (None, 170) and (None, 170, 51200) are incompatible
- loss -> CategoricalCrossentropy() | from_logit의 여부와 관계없이 같은 오류 -> 1. logit이 모델 output때 쓰이지 않는다, 2. 둘다 괜찮다. 

- 해결시도 : 다른 챗봇모델의 구조를 참고함([참고](https://github.com/haven-jeon/KoGPT2-chatbot)) -> 다른 챗봇의 구조를 참고 + 내 방식으로 시도해봄. 
- 다른 챗봇의 파인튜닝(kogpt 기반) : <q>Q<sent>S + <a>A</s> 형태의 데이터. | max_len을 정한 뒤 q와 a를 토큰화, `q_len + a_len > max_len`인지 판단 후, 넘는다면 
  `max_len - q_len <= 0`인지 판단, 그렇다면 q를 [-(max_len/2) : \]로 나누며. 이 후 a를 [:max_len - q_len\]로 나눔. 
  이후 ( token_ids(<q>Q+<a>A</s>), mask(A 부분만 1, 나머진 0), labels(Q길이만큼 masking +<bos>A</s>)를 반환함.
- 의문점 : Loss func가 Crossentropy? | train_x에 A가 포함되는 이유 -> token_ids를 모델에 넣고, 반환값에 mask를 적용해 label과의 손실을 측정함 -> 나는 오류는 형태의 오류라 지금은 상관 없을 듯.
