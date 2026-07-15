# AgentFlowLens

[![repository-checks](https://github.com/YuMinBee/AgentFlowLens/actions/workflows/ci.yml/badge.svg)](https://github.com/YuMinBee/AgentFlowLens/actions/workflows/ci.yml)

**AI Agent Next-Action Decision Routing Case Study**

> **모델을 키우는 문제로 시작해, 데이터 생성 과정과 오류 구조를 찾아내는 문제로 다시 정의했습니다.**

AI agent의 현재 요청, 행동 이력, 도구 결과, 세션 상태를 바탕으로 다음 action을 예측한 프로젝트입니다. [2026 AI·SW중심대학 디지털 경진대회 AI부문: AI Agent 행동 의사결정 예측 챌린지](https://dacon.io/competitions/official/236694/overview/description)에 참가하며 만든 코드와 의사결정 과정을 공개용 case study로 정리했습니다.

## Outcome

| Metric | Result |
|---|---:|
| Initial public Macro-F1 | ~0.7259 |
| Best public Macro-F1 | **0.7899327253** |
| Absolute improvement | **+0.0641** |
| Final rank | **46 / 269 teams (top 17.1%)** |
| Inference time | 약 6분 30초 |
| Serving constraint | offline, T4 16GB, 3 vCPU, RAM 12GB, ZIP 1GB 이하 |

최고점은 하나의 큰 모델에서 나오지 않았습니다. **도메인별 모델링, 서로 다른 오류를 내는 모델의 앙상블, OOF 기반 class calibration, 제한적으로만 개입하는 tie-break**를 순서대로 결합한 결과입니다.

## Executive Summary

처음에는 backbone과 feature를 바꾸면 해결될 일반적인 14-class text classification으로 보였습니다. 하지만 오류를 group과 step 단위로 분해하자 다른 문제가 드러났습니다.

- 초기 local OOF에서 4개 상위 group routing은 약 86%까지 맞았지만, inspect 내부 최종 action 정확도는 약 44.6%였습니다.
- `read_file`, `grep_search`, `glob_pattern`, `list_directory`는 같은 prompt에서도 모두 합리적인 선택이 될 수 있었습니다.
- SIM과 AU는 label space는 같지만 유효한 신호와 데이터 생성 방식이 달랐습니다.
- row random split은 같은 session의 미래 step을 train에 넣어 validation을 부풀릴 수 있었습니다.
- 모델 파일 1GB, offline inference 10분이라는 배포 제약이 있어 정확도만 최적화할 수 없었습니다.

그래서 문제를 다음처럼 바꿨습니다.

```text
Before: 어떤 backbone이 14개 action을 가장 잘 분류하는가?

After:  source별로 무엇이 다음 행동을 결정하는가?
        어떤 오류는 학습으로 줄이고, 어떤 모호성은 앙상블로 다룰 것인가?
        OOF 개선 중 실제 test로 전이되는 것은 무엇인가?
        제한된 환경에서 이를 어떻게 재현 가능하게 배포할 것인가?
```

## End-to-End Scope

| Area | What was built |
|---|---|
| Data diagnosis | source/group/step/class별 confusion과 label conflict 분석 |
| Validation | root-session GroupKFold, leakage assert, OOF artifact alignment |
| Data pipeline | structured text builder, 93 features, safe history mining, AU paraphrase QC |
| Modeling | E5 text+tabular fusion, source-specific models, multi-view training |
| Decision system | diversity ensemble, OOF class calibration, confidence-gated arbitration |
| Experiment operations | seed/config/hash 기록, one-variable submission 비교, 실패 실험 보존 |
| Deployment | preserved-int8, q-delta, POSIX ZIP, id join, offline Docker smoke test |
| Repository quality | 공개 범위 audit, unit tests, GitHub Actions CI |

사용 기술: Python, PyTorch, Hugging Face Transformers, scikit-learn, pandas, Docker, PowerShell, GitHub Actions.

## Problem-Solving Journey

### 1. 평균 점수를 오류 구조로 분해했습니다

전체 Macro-F1만 보지 않고 다음 축으로 OOF를 다시 분석했습니다.

- source: SIM / AU
- semantic group: inspect / modify / validate / reason
- action class
- session step
- 모델 간 disagreement

이 분석으로 group router를 더 복잡하게 만드는 대신, **inspect 내부 경계와 source shift가 핵심 병목**이라는 결론을 내렸습니다. 이후 모든 실험은 “어느 class, 어느 source, 어느 step에서 무엇이 바뀌었는가”를 보고 채택했습니다.

### 2. 모델보다 먼저 validation을 고쳤습니다

같은 session의 여러 step은 독립 표본이 아닙니다. row random split에서는 validation row의 미래 step이 training에 들어갈 수 있어 root session 기준 `GroupKFold`로 전환했습니다.

```text
train id ∩ valid id = 0
train session ∩ valid session = 0
valid augmentation rows = 0
valid source row used as augmentation base = 0
```

history mining, prototype, kNN, calibrator까지 모두 fold 내부에서만 fitting했습니다. 이 과정에서 AU의 기존 fold에 session overlap이 있다는 것도 찾아냈고, leakage 검사를 로그가 아닌 assert로 승격했습니다.

### 3. source별로 다른 정보 구조를 모델링했습니다

SIM은 action flow와 workspace state가 강했고, AU는 prompt semantics와 표현 다양성이 더 중요했습니다. 하나의 모델과 동일 입력을 강제하지 않고 source router 뒤에 독립 모델을 배치했습니다.

- SIM: `max_length=512`, flow/state 중심 multi-view augmentation
- AU: `max_length=448`, meaning-preserving paraphrase와 safe history mining
- 공통: E5의 CLS + masked mean pooling, 93 structured features, global 14-class MLP

history 전체를 무작정 넣는 대신 prompt, action trace, result, args, state를 명시적 블록으로 정규화했습니다.

### 4. 성능보다 오류 다양성으로 앙상블을 선택했습니다

큰 backbone과 세 번째 모델의 단순 추가는 대부분 실패했습니다. 단독 F1이 높은 모델보다 **기존 모델과 다른 row를 맞히는 모델**이 더 유용했습니다.

최종적으로 SIM과 AU에 각각 두 개의 E5-base를 사용했습니다.

```text
SIM = 0.60 * E5-A + 0.40 * E5-B
AU  = 0.50 * E5-paraphrase+mined + 0.50 * E5-paraphrase
```

모델 수를 늘리는 대신 두 모델의 disagreement를 측정하고, marginal gain이 확인된 조합만 남겼습니다.

### 5. 세 번째 모델은 예측기가 아니라 판정자로 제한했습니다

KLUE를 고정 비율로 섞으면 전체 분포를 흔들어 점수가 내려갔습니다. 그래서 KLUE의 역할을 다음 조건을 모두 만족하는 애매한 SIM row의 tie-break로 축소했습니다.

1. E5-A와 E5-B의 top-1이 다릅니다.
2. blend top-2 margin이 `0.08` 미만입니다.
3. KLUE가 A 또는 B의 후보 중 하나에 동의합니다.
4. KLUE confidence가 `0.55` 이상입니다.

즉 KLUE는 새로운 class를 주장하지 못하고, 이미 나온 두 후보 중 하나만 선택합니다. 이 bounded intervention이 최종 public score `0.7899327253`을 만들었습니다.

### 6. 제출 실패도 시스템 문제로 다뤘습니다

모델 성능과 별개로 Windows path, Linux archive layout, Hugging Face local model resolution, output row order 때문에 제출이 실패할 수 있었습니다. 이를 다음 방식으로 제거했습니다.

- 모든 model path를 `Path(__file__)` 기준으로 계산
- ZIP entry를 POSIX `/`로 고정
- sample submission을 row order가 아닌 `id`로 join
- 큰 matrix는 int8, 작은/1D tensor는 fp16으로 보존
- q-delta로 중복 모델 크기 절감
- offline Linux Docker에서 end-to-end smoke test

## Decision Log

| Observation | Hypothesis | Experiment | Decision |
|---|---|---|---|
| Group routing은 높지만 inspect 내부가 낮음 | hierarchy가 아니라 내부 표현이 병목 | specialist, multi-head, prototype 비교 | global head 유지, 다양성 앙상블 채택 |
| SIM/AU의 오류 양상이 다름 | 하나의 분포로 학습하면 서로 방해 | source별 OOF와 cross-domain 평가 | domain router와 별도 recipe 채택 |
| row random split 점수는 높지만 불안정 | session future leakage 가능 | root-session overlap audit | GroupKFold와 leakage assert 적용 |
| AU 표본이 적고 자연어 변형에 민감 | 의미 보존 증강이 generalization에 도움 | paraphrase QC 5-fold 비교 | 언어·파일 토큰 보존 paraphrase 채택 |
| 큰 모델과 3-model 평균이 LB에서 하락 | capacity보다 calibration과 diversity가 중요 | disagreement와 confidence 분석 | E5 두 개 + 조건부 KLUE로 축소 |
| OOF 상승이 반복해서 LB로 전이되지 않음 | test source 비율과 생성 방식이 다름 | one-variable public A/B | 5-fold 일관성과 bounded change를 우선 |
| seed100 full-train 재채점이 매우 높음 | 더 좋은 seed일 가능성 | 11-seed sweep과 제출 조합 비교 | in-sample 수치만으로 champion 교체하지 않음 |

자세한 근거는 [`docs/DECISION_LOG.md`](docs/DECISION_LOG.md)에 있습니다.

## Final System

```text
                           +-------------------------+
input row ---------------->| source router: SIM / AU |
                           +------------+------------+
                                        |
                   +--------------------+--------------------+
                   |                                         |
                SIM rows                                  AU rows
                   |                                         |
          +--------+--------+                       +--------+--------+
          |                 |                       |                 |
       E5-base A         E5-base B               E5-base A         E5-base B
          | 0.60            | 0.40                  | 0.50            | 0.50
          +--------+--------+                       +--------+--------+
                   |                                         |
       OOF-fitted class log-probability bias             probability blend
                   |
       disagreement-only KLUE tie-break
                   |
             final 14-class action
```

정확한 최고점 구성은 [`docs/HIGHSCORE.md`](docs/HIGHSCORE.md), [`configs/final_stack.json`](configs/final_stack.json), [`reference/inference/build_manifest.json`](reference/inference/build_manifest.json)에 고정했습니다.

## Model Architecture

```text
multilingual-e5-base
  -> CLS pooling (768)
  -> masked mean pooling (768)

structured features (93)
  -> LayerNorm

concat: 768 + 768 + 93 = 1629
  -> LayerNorm
  -> Linear 1629 -> 768 -> GELU -> Dropout
  -> Linear 768 -> 256  -> GELU -> Dropout
  -> Linear 256 -> 14 actions
```

93개 feature에는 action counts, 직전/이전 action one-hot, user tier, language preference, CI 상태, 주 언어, turn, budget, open files 등의 수치형 상태가 포함됩니다. 상세 내용은 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)를 참고하세요.

## Score Progression

| Stage | Public Macro-F1 | Key change |
|---|---:|---|
| Early baseline | ~0.7259 | 단일 text classifier |
| Domain routing | 0.7561 | SIM/AU 분리 |
| Structured E5 | ~0.779 | text + 93 features + full-train |
| Reproducible stack | 0.7837 | seeded component recipes |
| Diverse SIM ensemble | 0.7882 | E5-base A/B blend + stronger AU |
| Class calibration | 0.7891 | 5/5 positive-fold SIM class bias |
| Conservative arbitration | 0.7896 | disagreement-only KLUE |
| Final margin tuning | **0.7899327253** | margin `0.08`, confidence `0.55` |

## What Did Not Work

실패 실험도 결과만큼 중요하게 남겼습니다.

- E5-large, KLUE-large 단순 교체
- shared embedding 위의 group/specialist multi-head
- semantic/flow/tool 2-stream 및 3-stream fusion
- inspect prototype reranker와 global kNN
- aggressive step-1 rule, list-directory binary override
- 세 번째 backbone의 고정 비율 평균
- 한 fold 또는 full-train 재채점만으로 한 model selection

이 실험들은 local metric을 올리기도 했지만 inference cost 또는 public generalization에서 탈락했습니다. 코드와 판단 근거는 [`experiments/archived`](experiments/archived), [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)에 남겼습니다.

## Repository Map

```text
augmentation/          AU paraphrase 생성 및 QC
configs/               최종 stack, training recipe, class bias
data/                   데이터 배치 안내 (실데이터 미포함)
docs/                   case study, 구조, 검증, 제출 문서
experiments/archived/   실패/보류한 연구 코드
models/                 체크포인트 배치 안내 (가중치 미포함)
reference/training/     OOF 및 full-train 레퍼런스
reference/inference/    최고점 추론 로직 레퍼런스
reference/postprocess/  class bias와 KLUE gate sweep
reference/packaging/    양자화 및 submit.zip 검증
scripts/                safe mining과 repository audit
tests/                  무결성 테스트
```

## Quick Start

Python 3.11 환경을 권장합니다. CUDA 버전에 맞는 PyTorch를 먼저 설치한 뒤 나머지 패키지를 설치하세요.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

대회 데이터를 [`data/README.md`](data/README.md)의 구조로 배치하고 backbone을 로컬에 준비합니다.

```bash
python reference/training/run_domain_input_global_oof.py --help
python reference/training/run_au_augmented_8_2.py --help
```

외부 데이터나 모델 없이 실행 가능한 repository 검증은 다음과 같습니다.

```bash
python scripts/audit_repository.py
python -m unittest discover -s tests -v
python -m compileall -q augmentation reference experiments scripts tests
```

## Public Scope

대회 원본 데이터, 생성된 row-level data, 학습된 model weights, 실제 submit ZIP은 라이선스와 용량 문제로 포함하지 않습니다. 이를 재생성하고 검증하기 위한 코드, schema, configuration, decision record를 제공합니다.

split은 root session 기준 GroupKFold를 사용하고 validation row는 증강하지 않습니다. 자세한 leakage와 재현성 기준은 [`docs/VALIDATION.md`](docs/VALIDATION.md), 배포 검증은 [`docs/SUBMISSION.md`](docs/SUBMISSION.md)에 있습니다.

## License

코드는 [MIT License](LICENSE)로 공개합니다. 데이터와 pretrained backbone에는 각 원저작자의 라이선스가 별도로 적용됩니다.
