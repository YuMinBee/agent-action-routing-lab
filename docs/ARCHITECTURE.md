# Architecture

## Problem Formulation

각 row는 현재 user prompt와 그 시점까지의 agent history, 세션/워크스페이스 상태를 담습니다. 목표는 다음 14개 action 중 다음 행동 하나를 고르는 것입니다.

```text
apply_patch        ask_user           edit_file
glob_pattern       grep_search        lint_or_typecheck
list_directory     plan_task          read_file
respond_only       run_bash           run_tests
web_search         write_file
```

가장 어려운 경계는 `read_file`, `grep_search`, `glob_pattern`, `list_directory`로 구성된 inspect 그룹이었습니다. 서로 다른 행동이 같은 자연어 요청에 모두 합리적일 수 있어 prompt semantics만으로는 분리가 어려웠습니다.

## Domain Router

데이터 생성 방식이 다른 두 source를 하나의 모델에 강제로 맞추지 않았습니다.

```python
domain = "au" if str(row["id"]).startswith("sess_au") else "sim"
```

- SIM은 action flow와 state가 강하고, 긴 context가 유효해 `max_length=512`를 사용했습니다.
- AU는 자연어 표현과 직접적인 action hint 비중이 높고 표본이 적어 `max_length=448`, paraphrase 증강을 사용했습니다.

분기는 inference 때도 항상 가능하고 안정적인 `id` prefix만 사용합니다.

## Text Input

원본 필드를 그대로 이어 붙이지 않고 다음 의미 블록으로 정규화합니다.

```text
[CURRENT_PROMPT]
...

[ACTION_TRACE]
last=... prev=... tail=...

[LAST_RESULT]
status=... type=... compact_hint=...

[ARGS]
paths=... patterns=... commands=... targets=...

[SESSION_STATE]
turn=... budget=... open_files=... git_dirty=... ci=...
```

학습에서는 같은 row를 다섯 개 view로 표현합니다. 일부 view는 prompt를 줄이고 flow를 강조하며, 일부는 compact state만 유지합니다. validation은 원본 view만 사용하고 증강 view는 train partition에만 들어갑니다.

## Structured Features

텍스트와 별개로 93차원 feature vector를 만듭니다.

| Block | Dimension |
|---|---:|
| Action counts | 14 |
| Last action one-hot | 14 |
| Previous action one-hot | 14 |
| User tier one-hot | 4 |
| Language preference one-hot | 4 |
| CI status one-hot | 5 |
| Primary language one-hot | 24 |
| Numeric/log features | 14 |
| **Total** | **93** |

수치 feature에는 action/user turn 수, remaining budget, turn index, elapsed time, LOC, git 상태, open file 수, prompt 길이, history 길이, 실패/patch/test count 등이 포함됩니다.

## Classifier

E5 encoder의 마지막 hidden state에서 첫 토큰과 attention-mask mean을 함께 사용합니다.

```python
cls = hidden[:, 0]
mean = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
features = feature_norm(structured_features)
fused = torch.cat([cls, mean, features], dim=-1)
logits = classifier(fused)
```

분류기는 `1629 -> 768 -> 256 -> 14` MLP입니다. inspect specialist, multi-head, prototype head도 실험했지만 최종 제출에서는 단순한 global head가 가장 안정적이었습니다.

## Ensemble

SIM과 AU 모두 probability space에서 결합합니다.

```text
SIM = 0.60 * P(sim_a) + 0.40 * P(sim_b)
AU  = 0.50 * P(au_a)  + 0.50 * P(au_b)
```

SIM에는 OOF로 fitting한 class log-probability bias를 더합니다.

```python
adjusted = log(blended_probability + eps) + class_bias
```

그 뒤에도 다음 조건을 모두 만족하는 SIM row만 KLUE prediction으로 교체합니다.

1. SIM-A와 SIM-B의 top-1이 서로 다릅니다.
2. raw blend의 top-2 margin이 `0.08`보다 작습니다.
3. KLUE top-1이 A 또는 B의 후보 중 하나입니다.
4. KLUE confidence가 `0.55` 이상입니다.

세 번째 모델을 독립적인 결정권자로 쓰지 않고, 두 주 모델이 이미 제시한 후보를 재판정하는 역할만 맡겼습니다.

## Deployment

1GB 제한을 맞추기 위해 큰 2D weight는 per-tensor symmetric int8로 저장하고, bias·LayerNorm 같은 작은/1D tensor는 fp16으로 보존합니다. 공통 텐서와의 차이만 저장하는 q-delta도 사용했습니다.

추론은 domain별 모델을 순차적으로 로드하고 메모리를 회수합니다. output은 row position이 아니라 `id`로 sample submission에 결합합니다.
