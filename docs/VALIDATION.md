# Validation And Leakage

## Split Unit

같은 session의 step은 강하게 연결돼 있습니다. row random split을 쓰면 validation의 미래 step이 training에 들어가 과거 action trace를 통해 정답 패턴이 새어 나갈 수 있습니다.

기본 split은 root session 기준 5-fold GroupKFold입니다.

```python
splitter = GroupKFold(n_splits=5)
for train_idx, valid_idx in splitter.split(rows, labels, groups=root_session_ids):
    ...
```

다음 조건을 assert합니다.

```text
train id ∩ valid id = empty
train root_session ∩ valid root_session = empty
valid source row used as augmentation base = 0
valid augmented rows = 0
```

## Augmentation Boundary

fold를 먼저 나눈 뒤 training partition만 증강합니다.

```text
WRONG: full data -> augment -> split
RIGHT: full data -> split -> augment(train only)
```

같은 원본의 variant가 train과 validation 양쪽에 들어가면 거의 같은 문장을 맞히는 문제가 됩니다.

## History Mining

history에 과거 user/action pair가 있지만 top-level training row로 존재하지 않는 경우 이를 복원할 수 있습니다. 다음 조건을 지킵니다.

1. target step보다 앞선 event만 사용합니다.
2. 같은 `(root_session, step)`에서 label conflict가 있으면 폐기합니다.
3. 미래 시점의 open files, CI status, budget, result를 과거 row에 복사하지 않습니다.
4. split 이후 train session에서만 mining합니다.
5. 기존 label과 겹치는 key는 100% 일치하는지 검사합니다.

[`scripts/mine_history_pairs.py`](../scripts/mine_history_pairs.py)는 안전한 기본값으로 dynamic metadata를 비웁니다.

## Seed

fold assignment seed와 training seed는 다른 개념입니다. fold가 고정돼도 다음은 매번 달라질 수 있습니다.

- classifier initialization
- dropout mask
- DataLoader shuffle order
- CUDA kernel behavior
- augmentation sampling

최소한 Python, NumPy, PyTorch CPU/CUDA와 DataLoader generator를 같은 seed로 고정합니다. 완전한 bitwise determinism은 성능과 속도를 바꿀 수 있으므로, deterministic flag 사용 여부도 config에 기록합니다.

## OOF Discipline

- base logits는 반드시 해당 row를 학습하지 않은 fold model에서 생성합니다.
- prototype, kNN index, calibrator도 train-fold에서만 fitting합니다.
- class bias/threshold sweep 결과는 fold별 gain과 전체 cross-fit gain을 함께 봅니다.
- best epoch를 고른 fold의 점수는 낙관적일 수 있습니다.
- 한 fold의 최고점보다 5-fold 평균, 표준편차, positive-fold 수를 우선합니다.

## OOF Versus Leaderboard

OOF는 모델 선택의 필수 조건이지만 충분 조건은 아니었습니다. 특히 source 비율과 생성 방식이 로컬 train과 test에서 달랐습니다.

새 실험 채택 기준은 다음처럼 운영했습니다.

1. 기존 stack과 동일 split에서 비교합니다.
2. 평균 Macro-F1뿐 아니라 class별 변화와 disagreement를 봅니다.
3. 최소 4/5 fold에서 유지 또는 개선되는지 확인합니다.
4. test-only feature나 불안정한 metadata에 의존하지 않습니다.
5. public submission은 가능한 한 한 변수만 바꿉니다.

## Common Failure Modes

| Failure | Symptom | Prevention |
|---|---|---|
| Session overlap | validation이 비정상적으로 높음 | root-session assert |
| Validation augmentation | train 0.99, valid도 과도하게 높음 | split before augmentation |
| Future metadata reuse | mined row 성능 급등 | dynamic state reset |
| Row-order output | 정상 실행인데 점수 붕괴 | `id` map join |
| Input-builder mismatch | 새 checkpoint만 LB 하락 | train/infer text byte comparison |
| Selected-fold optimism | 한 fold만 매우 높음 | full 5-fold report |
| Seed drift | 재학습 점수 불안정 | complete seed/config logging |
