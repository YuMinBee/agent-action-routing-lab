# Experiments

## Public Score Landmarks

| Public Macro-F1 | Experiment |
|---:|---|
| ~0.7259 | 초기 E5 단일 모델 |
| 0.7561 | SIM/AU domain split |
| ~0.779 | structured E5 full-train |
| 0.7837498 | seeded reproduction stack |
| 0.7846018 | inspect specialist ensemble |
| 0.78642 | SIM A/B diversity blend |
| 0.7878589 | stronger AU replacement |
| 0.7882105 | stable two-SIM/two-AU stack |
| 0.7887945 | alternative SIM combination |
| 0.7891094 | SIM per-class bias |
| 0.7895945 | conservative KLUE tie-break |
| 0.7898346 | wider candidate margin |
| **0.7899327** | margin `0.08`, confidence `0.55` |

점수는 단일한 선형 실험 로그가 아니라 여러 branch의 public submission에서 모은 landmark입니다. 같은 점수대의 모델도 seed, augmentation, epoch, quantizer가 다를 수 있으므로 파일명만으로 같은 모델이라 가정하지 않았습니다.

## Strong Positive Findings

### 1. Domain-specific training

SIM과 AU는 같은 label space를 쓰지만 생성 과정과 유효 신호가 달랐습니다. source별 모델과 text builder를 나누자 초기 대비 가장 큰 점프가 나왔습니다.

### 2. Diversity over raw capacity

E5-large는 학습·추론 비용에 비해 안정적인 이득을 주지 못했습니다. 반면 같은 E5-base라도 데이터 구성과 seed가 다른 모델의 오류가 달라 앙상블 이득이 생겼습니다.

### 3. AU paraphrase

AU의 작은 표본과 자연어 변형에 paraphrase가 유효했습니다. 언어 보존, 파일/코드 토큰 보존, 지시어 보존, 길이비 검사로 품질을 관리했습니다.

### 4. Conservative postprocessing

모든 fold에서 양의 gain이 확인된 class bias와, 기존 후보 중 하나만 선택하는 KLUE tie-break가 작지만 실제 public score를 올렸습니다.

## Hardest Classes

inspect 네 클래스는 group routing보다 group 내부 판별이 훨씬 어려웠습니다.

- `list_directory`는 초기 step에서 강한 prior를 가집니다.
- `read_file`과 `grep_search`는 같은 경로와 비슷한 자연어에 자주 충돌합니다.
- `glob_pattern`은 filename/extension 탐색과 content search의 경계가 흔들립니다.
- step 1에는 flow 정보가 거의 없어 prompt와 weak state만 남습니다.

동일하거나 매우 유사한 prompt/history가 서로 다른 label을 가진 사례도 있어, 모델 capacity만 키우는 것으로는 천장이 잘 움직이지 않았습니다.

## Negative Or Inconclusive Findings

### Multi-head specialists

group/inspect/modify/validate/reason head를 공유 embedding에 붙였지만, 실질적으로 같은 표현을 다시 선형 분리하는 데 그쳤습니다. head 수가 늘어도 정보가 늘지는 않았고 쉬운 group task가 capacity를 가져가기도 했습니다.

### Multi-stream encoders

semantic/flow/tool을 각각 E5에 넣고 weighted concat 또는 FusionMLP를 시도했습니다. 비용은 크게 늘었지만 single-stream common text를 안정적으로 넘지 못했습니다. field-level interaction이 encoder 내부 token interaction보다 약해진 것이 한 원인으로 보입니다.

### Prototype and kNN reranking

inspect-only prototype, class mean, text prototype, kNN posterior를 만들었습니다. OOF의 일부 fold에서는 상승했지만 full-train/LB로 안정적으로 전이되지 않았습니다. 애매한 클래스의 embedding cluster 자체가 겹쳤습니다.

### Larger backbones

E5-large, KLUE-large 등은 첫 fold에서 희망적인 구간이 있었지만 학습 시간, submission 크기, 추론 제한까지 고려하면 효율이 낮았습니다. 큰 모델이 더 많은 noisy pattern까지 외우는 문제도 보였습니다.

### Aggressive rules

step-1 prior, keyword rescue, list-directory binary detector, fixed class overrides를 다양하게 실험했습니다. local subset F1이 올라가도 전체 macro-F1 또는 public score가 떨어지는 경우가 많았습니다. precision 손실이 다른 step의 gain을 상쇄했습니다.

### OOF-only model selection

OOF가 높아진 새 SIM/AU가 public score를 낮춘 사례가 반복됐습니다. 원인은 domain proportion 차이, selected-fold optimism, leakage-prone split, augmentation/test mismatch, seed variance가 복합적이었습니다.

## Lessons

1. 모델 하나의 평균 F1보다 **모델 간 disagreement에서 누가 맞는지**가 앙상블에 더 중요합니다.
2. macro-F1 후처리는 class recall을 올리는 동시에 다른 class precision을 깎습니다. subset F1만 보면 안 됩니다.
3. full-train loss는 validation metric이 아닙니다. epoch 선택은 GroupKFold에서 해야 합니다.
4. 새로운 feature는 test에도 동일한 의미와 분포로 존재하는지 먼저 확인해야 합니다.
5. 제출 환경 검증은 모델링과 별도의 필수 단계입니다. 경로 한 줄이 모든 실험을 무효화할 수 있습니다.
