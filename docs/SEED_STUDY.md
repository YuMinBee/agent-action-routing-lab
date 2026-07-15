# Seed Study

## Source Artifact

마지막에 추가된 로컬 파일:

```text
share_sim_seedtop3_fp32_0714.zip
```

archive 크기는 약 2.56GB이며 model weights를 포함하므로 GitHub에는 올리지 않습니다. 대신 검증 가능한 요약과 fp32 hash를 문서로 보존합니다.

## Setup

- 동일한 SIM champion recipe
- full-train 64,975 rows
- epoch 2
- max length 512
- batch size 16
- seed만 변경
- 평가값은 training rows 전체 재채점

## Results

| Rank | Seed | Macro-F1 | Accuracy | Rare-4 Macro-F1 |
|---:|---:|---:|---:|---:|
| 1 | 100 | 0.8047 | 0.7908 | 0.8373 |
| 2 | 43 | 0.8019 | 0.7902 | 0.8315 |
| 3 | 45 | 0.7999 | 0.7883 | 0.8261 |
| 4 | 102 | 0.7998 | 0.7902 | 0.8228 |
| 5 | 46 | 0.7987 | 0.7877 | 0.8229 |
| 6 | 44 | 0.7970 | 0.7855 | 0.8229 |
| 7 | 104 | 0.7968 | 0.7872 | 0.8181 |
| 8 | 42 | 0.7935 | 0.7855 | 0.8139 |
| 9 | 103 | 0.7921 | 0.7835 | 0.8097 |
| 10 | 101 | 0.7914 | 0.7841 | 0.8084 |
| 11 | 47 | 0.7913 | 0.7840 | 0.8062 |

```text
mean macro = 0.7970
std macro  = 0.0043
seed100 - seed42 = +0.0111
```

## Interpretation

이 실험은 training randomness가 작은 문제가 아니라는 사실을 보여 줍니다. 같은 recipe도 initialization, shuffle, dropout에 따라 prediction boundary가 달라졌습니다.

하지만 이 표의 metric은 OOF가 아니라 full-train in-sample score입니다. 따라서 다음을 의미하지 않습니다.

```text
seed100 has better train fit
!= seed100 generalizes better
!= seed100 should replace the public champion
```

실제로 final high-score SIM-A hash는 seed100 제출의 int8 hash와 다릅니다. seed sweep은 ensemble diversity 후보를 찾는 탐색 도구로 사용했고, 최종 채택은 public one-variable A/B와 bounded inference logic을 기준으로 했습니다.

## FP32 Hashes

| Seed | SHA256 |
|---:|---|
| 100 | `9d4a14472e3aa06ee9229dd61a41bcd8002c9463dc6384df183a13f639813a4f` |
| 43 | `6e59be109b0ee5a09f742afc578b918907216f826ca7ba3924c143c07c5d72e3` |
| 45 | `8f9c8032d2070861ca18663d4fc0de1e5110898314ddff472dd31a656925cdaf` |
