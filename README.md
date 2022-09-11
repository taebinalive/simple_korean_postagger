# Simple Korean Part-Of-Speech Tagger


# Model Structure
* BiLSTM
* BiLSTM-LAN [[Hierarchically-Refined Label Attention Network for Sequence Labeling](https://arxiv.org/pdf/1908.08676.pdf) (EMNLP 2019)]
* CNN




* 위 3가지 모델로 음절 정렬 코퍼스를 이용하여 tag classification을 수행 및 성능 비교
* 규칙, 매핑 사전 기반으로 음절과 형태소 정렬


# Corpus
* 세종 형태소 말뭉치 + 뉴스데이터 정제 말뭉치
* 전체 문장 수: 826,573
* 전체 어절 수: 9,896,840

# Requirement
* Python >= 3.6
* PyTorch >= 1.10

# Preprocess
* Build syllable-level aligned corpus from Sejong corpus
* Running Script
```
python align.py --data_dir [SOURCE_CORPUS_DIR] --dic_path [RESTORE_DIC_PATH] --vocab_path [VOCAB_PATH] --unmapped [UNMAPPED_FILE_DIR] --aligned [ALIGNED_FILE_DIR] --output [OUTPUT_DIR]'
```

* split aligned corpus to train / valid / test (80%/10%/10%)
```
python split_data.py --data_dir [OUTPUT_DIR]
```

# Train models
* Run below:
```
python trainer.py --config [CONFIG_PATH]
```

# Result
* ### Environment
  * #### 1 RTX 3090
  * #### AMD Ryzen Threadripper 3960X 24-Core Processor

| Model | # Hidden Size | # Layers | # Parameters | Morph Acc | Inference time / sent (GPU) | Inference time / sent (CPU)
| --------- | --- | --- | --- | --- | --- | ---
| BiLSTM | 600 | 4 | 8.3M | 98.58 | 15.4 ms | 172 ms
| BiLSTM-LAN | 400 | 5 | 11M | **98.72** | 22.1 ms | 180.7 ms
| CNN | 200 | - | 2.1M | 98.03 | 1.3 ms | 4.2 ms

# Test

```
python test.py --config config.json --model CNN
```

```
model loaded
input >> 
우리 집에선 회사까지 1시간이 걸렸다.
우리	우리/NP
집에선	집/NNG + 에서/JKB + ㄴ/JX
회사까지	회사/NNG + 까지/JX
1시간이	1/SN + 시간/NNB + 이/JKS
걸렸다.	걸리/VV + 었/EP + 다/EF + ./SF
elapsed: 0.31 s
input >> 
```


## TODO
- [ ] 후처리용 사용자 사전 추가 (분해 또는 조합)
- [ ] 에러 케이스 로그 파일로 출력
- [ ] vocab 최소 빈도수 적용
- [ ] F1 score

# Reference
[1] [Hierarchically-Refined Label Attention Network for Sequence Labeling](https://arxiv.org/pdf/1908.08676.pdf) (EMNLP 2019)
<br>
[2] [khaiii (Kakao Hangul Analyzer III)](https://github.com/kakao/khaiii)
<br>
[3] [HMM 한국어 형태소 분석기](https://github.com/lovit/hmm_postagger)

