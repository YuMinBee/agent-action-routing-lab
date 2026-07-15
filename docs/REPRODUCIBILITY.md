# Reproducibility

## Scope

이 공개 저장소는 competition data와 trained weights를 재배포하지 않습니다. 대신 학습/증강/후처리/양자화/추론의 source와 최종 configuration을 제공합니다.

완전 재현에는 다음 외부 자산이 필요합니다.

1. 대회 train/test data
2. `intfloat/multilingual-e5-base`
3. KLUE RoBERTa checkpoint/tokenizer
4. AU paraphrase 또는 이를 생성할 API/model

## Reproduction Order

```text
1. data schema validation
2. root-session fold generation
3. safe history mining
4. AU paraphrase generation and QC
5. SIM-A / SIM-B 5-fold diagnostics
6. AU-A / AU-B 5-fold diagnostics
7. component-specific full training
8. OOF class-bias fitting
9. KLUE tie-break sweep
10. preserved-int8 quantization
11. Linux offline smoke test
```

## Required Records

각 run은 최소한 다음을 저장해야 합니다.

- git commit
- command line
- full config
- random seed
- data/input hashes
- fold assignment hash
- label distribution
- epoch metrics
- checkpoint hash
- inference builder version
- quantizer version

## Reproduction Boundary

최종 public score는 고정된 제출 artifact의 결과입니다. 같은 recipe를 재학습해도 GPU kernel, library version, data generation API 등의 차이로 작은 편차가 생길 수 있습니다. 따라서 score만이 아니라 artifact hash와 code/config pair를 함께 관리해야 합니다.
