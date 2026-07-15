# Decision Log

이 문서는 실험을 나열하기보다, 어떤 관찰이 어떤 결정으로 이어졌는지 기록합니다.

## D1. Hierarchy보다 group 내부가 병목이었다

**관찰**

- 4개 상위 group routing은 약 86%까지 맞았습니다.
- group별 최종 action 정확도는 inspect 44.63%, modify 81.53%, validate 59.54%, reason 68.94%였습니다.
- 상위 confusion pair 대부분이 `read_file`, `grep_search`, `glob_pattern`, `list_directory`에 몰렸습니다.

**가설**

group router를 더 복잡하게 만드는 것보다 inspect 내부의 모호성과 representation을 다루는 편이 전체 Macro-F1에 더 큰 영향을 줄 것입니다.

**실험**

- hierarchical model
- shared E5 multi-head
- inspect specialist
- prototype reranker
- inspect-only kNN

**결정**

specialist head를 final stack에 넣지 않았습니다. 같은 embedding에 head만 늘려서는 새로운 정보가 생기지 않았고, full-train/LB에서 안정적인 개선이 없었습니다. 최종에는 global 14-class head와 독립 모델 간 diversity를 사용했습니다.

## D2. Random split 점수보다 session-safe OOF를 신뢰했다

**관찰**

한 session의 여러 step이 서로의 history에 나타납니다. row random split에서는 validation step의 미래 window가 train에 들어갈 수 있습니다.

**조치**

- root session 기준 GroupKFold
- train/valid id와 session overlap assert
- fold split 후 train partition만 증강
- mined row의 source session이 validation에 없는지 검사

**결과**

일부 높은 local score가 leakage 또는 selected-fold optimism에 의존한다는 것을 확인했습니다. AU 외부 fold에서 session overlap 139~153개를 발견한 뒤 split을 다시 만들었습니다.

## D3. SIM과 AU를 다른 문제로 취급했다

**관찰**

같은 label space인데도 source별 class 분포, prompt 스타일, flow/state의 영향이 달랐습니다. 한 모델의 개선이 다른 source 성능을 떨어뜨리는 경우도 있었습니다.

**실험**

- source를 합친 global model
- source token만 추가한 model
- SIM/AU 독립 model과 inference router
- cross-domain evaluation

**결정**

`sess_au` prefix로 안정적으로 분기하고 source별 max length, augmentation, checkpoint를 유지했습니다. 초기 public score를 약 0.726에서 0.756으로 올린 가장 큰 구조적 변화였습니다.

## D4. 데이터 증강을 source별로 설계했다

**SIM**

flow/state가 중요해 같은 row를 여러 structured view로 표현했습니다. history에서 누락된 과거 user/action pair를 mining하되 미래 dynamic metadata는 복사하지 않았습니다.

**AU**

표본이 적고 semantics가 중요해 paraphrase를 사용했습니다. 언어, 파일명, 경로, 코드 토큰, 지시어를 보존하고 QC 통과 row만 사용했습니다.

**결정**

validation은 언제나 원본 row만 유지했습니다. aggressive oversampling은 minority recall을 올려도 precision 손실이 커 채택하지 않았습니다.

## D5. 큰 모델보다 disagreement를 선택 기준으로 삼았다

**관찰**

E5-large와 KLUE-large는 단독 fold에서 좋아 보여도 비용 대비 public gain이 없었습니다. 반면 성능이 비슷한 E5-base 두 개가 서로 다른 row를 맞혔습니다.

**결정**

단독 F1뿐 아니라 다음을 함께 측정했습니다.

```text
prediction agreement
disagreement row accuracy
class별 complementary gain
confidence sharpness
inference time and archive size
```

그 결과 SIM `0.60 : 0.40`, AU `0.50 : 0.50`의 두-model blend를 final anchor로 정했습니다.

## D6. Calibration은 모든 class를 뒤집지 않게 만들었다

**관찰**

class imbalance 때문에 argmax가 minority class recall을 억제했지만, hand-written rule은 다른 class precision을 쉽게 무너뜨렸습니다.

**조치**

OOF log-probability에 class별 additive bias를 fitting하고 fold별 shrink를 선택했습니다.

**채택 기준**

- SIM 5/5 fold에서 positive gain
- AU는 모든 nonzero setting이 악화되어 bias 비활성화
- label order와 bias vector 길이를 runtime에 검사

public Macro-F1은 `0.7887945 -> 0.7891094`로 올랐습니다.

## D7. KLUE를 third voter가 아닌 bounded arbiter로 사용했다

**관찰**

KLUE를 고정 비율로 섞으면 E5가 맞힌 row까지 바뀌었습니다. 하지만 E5-A/B가 갈리는 일부 row에서는 KLUE가 유용했습니다.

**설계**

```text
A != B
and blend margin < 0.08
and KLUE prediction in {A, B}
and KLUE confidence >= 0.55
```

KLUE는 후보를 새로 만들 수 없고 두 E5 후보 중 하나만 선택합니다. 개입 범위를 제한한 덕분에 public score가 `0.7891094 -> 0.7899327`까지 올랐습니다.

## D8. 높은 seed 재채점을 그대로 채택하지 않았다

마지막에 추가된 `share_sim_seedtop3_fp32_0714.zip`에는 동일 recipe의 11-seed full-train 재채점 결과가 있습니다. seed100은 macro 0.8047로 seed42보다 0.0111 높았습니다.

하지만 이는 holdout/OOF가 아닌 **학습 전량에 대한 in-sample score**였습니다. 따라서 이를 “더 좋은 generalization”의 증거로 간주하지 않았습니다. seed100 조합은 별도 제출에서 champion을 안정적으로 넘지 못했고, 최종 최고점 구성의 SIM hash도 seed100과 다릅니다.

이 판단은 좋은 숫자를 버린 것이 아니라, 측정 대상이 다른 숫자를 구분한 것입니다. 상세 표는 [`SEED_STUDY.md`](SEED_STUDY.md)에 있습니다.

## D9. 제출물을 하나의 운영 시스템으로 검증했다

**실패 유형**

- `/app/model/sim` 경로 해석 실패
- Windows ZIP entry의 역슬래시
- train/inference input builder 불일치
- sample submission row-order 가정
- 1GB archive 제한과 10분 timeout

**조치**

- POSIX archive builder
- relative local-model resolution
- id mapping output
- preserved-int8와 q-delta
- offline Linux Docker smoke test
- artifact SHA256 manifest

모델링 결과가 실제 평가 서버에서 재현되는 마지막 구간까지 프로젝트 범위로 다뤘습니다.
