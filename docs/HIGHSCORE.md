# High-Score Artifact

## Confirmed Result

```text
Public Macro-F1: 0.7899327253
Final rank:      46 / 269 teams
Submission:      submit_champion_789834_gamble1_klue_m008_c055_20260715
```

이 점수는 `0.7898346366` anchor에서 KLUE candidate margin을 `0.08`로 넓힌 제출에서 나왔습니다. 현재 [`reference/inference/script.py`](../reference/inference/script.py), [`reference/inference/build_manifest.json`](../reference/inference/build_manifest.json)이 이 구성을 가리킵니다.

## Exact Logic

### SIM

```text
P_sim = 0.60 * P_sim_a + 0.40 * P_sim_b
P_sim = softmax(log(P_sim) + class_bias_sim)
```

KLUE override는 다음 조건을 모두 만족할 때만 적용합니다.

```text
pred_sim_a != pred_sim_b
raw_blend_top2_margin < 0.08
pred_klue in {pred_sim_a, pred_sim_b}
confidence_klue >= 0.55
```

### AU

```text
P_au = 0.50 * P_paraphrase_plus_mined_e4
     + 0.50 * P_paraphrase_only_e6
```

AU class bias는 OOF에서 nonzero 설정이 모두 나빠 비활성화했습니다.

## Artifact Hashes

| Component | SHA256 |
|---|---|
| SIM-A int8 | `2df045c73c74bc5ae10db6189184b821c5764a2a9b6f627d71543b48b789ef23` |
| SIM-B reconstructed int8 | `6be3136f320bce14766e7e8e5cd3626164682f2c2d0ca15b1a0d27e288335f35` |
| AU-A int8 | `75c637ba49839de613d7d7bb762e3929ed634133799be0794507db1213952a9e` |
| AU-B reconstructed int8 | `fcca0d8ff71dae6c25d7e10d1dd4d4e3e600e449f0af67ce0a6c5767dadc6c44` |
| KLUE int8 | `96e49a7c4ba053f52d37faf40cf4f4f7ea816ed3d9a94890fe943e43563cb0e2` |

가중치는 공개 저장소에 포함하지 않지만 manifest hash를 남겨 component mix-up을 방지합니다.

## Why This Was The Winner

| Change | Public Macro-F1 | Interpretation |
|---|---:|---|
| Two-SIM/two-AU anchor | 0.7887945263 | 다양성 기반 안정 anchor |
| + SIM class bias | 0.7891094113 | 모든 SIM fold에서 작은 gain |
| + conservative KLUE gate | 0.7895944696 | ambiguous disagreement만 재판정 |
| + wider candidate margin | 0.7898346366 | intervention coverage 확대 |
| + margin `0.08`, confidence `0.55` | **0.7899327253** | 최종 최고점 |

중요한 점은 KLUE를 더 강하게 섞은 것이 아니라 **개입 가능한 row를 명시적으로 제한한 것**입니다.

## Non-Winner Follow-up

`submit_champion_789932_push_klue_m009_c055_20260715`는 최고점 이후 margin을 `0.09`로 넓힌 후속 후보입니다. local OOF에서는 189개 row를 바꾸고 두 확인 fold가 모두 positive였지만, 확인된 `0.7899327253` 결과의 artifact는 아닙니다. 공개 저장소에서는 검증된 m0.08 설정만 final로 표기합니다.
