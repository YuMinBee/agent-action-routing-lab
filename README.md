# Agent Action Routing Lab

AI agent의 현재 요청, 행동 이력, 도구 결과, 세션 상태를 바탕으로 **다음 action을 14개 클래스 중 하나로 예측**한 실험 기록입니다.

이 저장소는 [2026 AI·SW중심대학 디지털 경진대회 AI부문: AI Agent 행동 의사결정 예측 챌린지](https://dacon.io/competitions/official/236694/overview/description)에 참가하며 만든 코드와 시행착오를 공개용으로 다시 정리한 것입니다.

> **Public Macro-F1 0.7899327253 · 46 / 269 teams**

순위보다 더 오래 남길 만했던 것은, 불명확한 라벨 경계와 강한 도메인 차이, OOF와 리더보드의 괴리, 제한된 제출 환경을 함께 다뤄 본 과정이었습니다. 이 저장소에는 잘된 실험뿐 아니라 실패한 구조도 남겨 두었습니다.

## Final Stack

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
       SIM class log-probability bias                    probability blend
                   |
       conservative KLUE-base tie-break
                   |
             final 14-class action
```

최종 추론 스택의 핵심은 다음과 같습니다.

- **도메인 분리**: `sess_sim`과 `sess_au`를 서로 다른 모델과 입력 길이로 처리합니다.
- **SIM 앙상블**: 서로 다른 데이터 레시피로 학습한 E5-base 두 개를 `0.60 : 0.40`으로 결합합니다.
- **AU 앙상블**: paraphrase 기반 모델과 mined-history 기반 모델을 `0.50 : 0.50`으로 결합합니다.
- **구조화 입력**: 텍스트 encoder 출력의 CLS/mean pooling과 93개 structured feature를 결합합니다.
- **제한적 후처리**: OOF에서 검증된 SIM class bias와 매우 보수적인 KLUE tie-break만 사용합니다.
- **배포 최적화**: 큰 행렬은 int8, 작은 텐서와 1D 파라미터는 fp16으로 보존합니다.

정확한 조합은 [`configs/final_stack.json`](configs/final_stack.json), 클래스 보정치는 [`configs/class_bias_final.json`](configs/class_bias_final.json)에 있습니다.

## Model

기본 분류기는 다음 구조입니다.

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

93개 feature에는 action count, 직전/이전 action one-hot, user tier, language preference, CI 상태, 주 언어, turn·budget·open files 등의 수치형 상태가 포함됩니다. 상세 내용은 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)를 참고하세요.

## Score Journey

| Stage | Public Macro-F1 | What changed |
|---|---:|---|
| Early baseline | ~0.726 | 단일 text classifier |
| Domain routing | 0.756 | SIM/AU 분리 |
| Strong E5 stack | ~0.779 | structured input, augmentation, full-train |
| Reproducible stack | 0.7837 | seeded SIM/AU recipe |
| Diverse SIM ensemble | 0.7882 | E5-base A/B blend + stronger AU |
| Class calibration | 0.7891 | OOF-fitted SIM class bias |
| Conservative tie-break | **0.7899327253** | KLUE agreement gate, margin `0.08`, confidence `0.55` |

리더보드와 OOF는 완전히 같은 방향으로 움직이지 않았습니다. 이 저장소는 그 차이를 숨기지 않습니다. 자세한 실험 결과와 기각된 가설은 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)에 정리했습니다.

## What Worked

1. **SIM/AU 분리**가 가장 큰 구조적 개선이었습니다.
2. backbone 크기보다 **입력 빌더와 데이터 레시피의 다양성**이 더 중요했습니다.
3. 단일 specialist보다 **서로 다른 오류를 내는 두 모델의 앙상블**이 안정적이었습니다.
4. OOF에서 모든 fold가 동의한 **작은 class bias**는 리더보드에서도 살아남았습니다.
5. 강한 override보다 **두 주 모델 중 하나와 동의할 때만 개입하는 tie-break**가 유효했습니다.
6. 모델 크기 제한 아래에서는 **선택적 int8 + fp16 보존 양자화**가 실용적이었습니다.

## What Did Not Transfer

- 큰 backbone을 단순 교체하는 것
- 같은 embedding 위에 specialist head를 여러 개 얹는 multi-head 구조
- 2/3-stream encoder와 강한 field weighting
- prototype reranker, global kNN, 무조건적인 규칙 보정
- OOF만 보고 선택한 공격적인 step-1 보정
- 세 번째 모델을 고정 비율로 단순 추가하는 앙상블

이 실험들은 완전히 무의미했다기보다, **로컬 개선이 실제 테스트 분포로 전이되지 않는 문제**를 보여 줬습니다. 구현은 비교와 재사용을 위해 [`experiments/archived`](experiments/archived)에 보존했습니다.

## Repository Map

```text
augmentation/          AU paraphrase 생성 및 조립
configs/               최종 앙상블, 학습 레시피, class bias
data/                   데이터 배치 안내 (실데이터 미포함)
docs/                   구조, 실험, 검증, 제출 문서
experiments/archived/   실패/보류한 연구 코드
models/                 체크포인트 배치 안내 (가중치 미포함)
reference/training/     OOF 및 full-train 레퍼런스
reference/inference/    최종 추론 로직 레퍼런스
reference/postprocess/  class bias와 KLUE gate sweep
reference/packaging/    양자화 및 submit.zip 검증
scripts/                mining과 저장소 안전 감사 도구
tests/                  가벼운 무결성 테스트
```

## Quick Start

Python 3.11 환경을 권장합니다. CUDA 버전에 맞는 PyTorch를 먼저 설치한 뒤 나머지 패키지를 설치하세요.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

대회 데이터를 [`data/README.md`](data/README.md)의 구조로 배치하고 backbone을 로컬에 준비합니다. 이후 대표적인 OOF 실행은 다음 형태입니다.

```bash
python reference/training/run_domain_input_global_oof.py --help
python reference/training/run_au_augmented_8_2.py --help
```

공개 전 안전 감사와 테스트는 외부 데이터나 모델 없이 실행됩니다.

```bash
python scripts/audit_repository.py
python -m unittest discover -s tests -v
python -m compileall -q augmentation reference experiments scripts tests
```

제출 패키지 구성과 Linux/Docker 검증 순서는 [`docs/SUBMISSION.md`](docs/SUBMISSION.md)에 있습니다.

## Data And Weights

대회 원본 데이터, 생성된 행 단위 데이터, 학습된 모델 가중치, 실제 제출 ZIP은 라이선스와 용량 문제로 포함하지 않습니다. 저장소에는 이를 **재생성하고 검증하기 위한 코드, 형식, 설정**만 들어 있습니다.

## Reproducibility Notes

- split은 row random split이 아니라 **root session 기준 GroupKFold**를 사용합니다.
- validation row는 증강하지 않고, 증강은 해당 fold의 train partition에만 적용합니다.
- seed는 fold뿐 아니라 Python, NumPy, PyTorch, CUDA, DataLoader shuffle에 모두 고정합니다.
- history mining은 현재 시점 이전의 user/action pair만 복원하며, 미래 상태 메타데이터를 복사하지 않습니다.
- 제출 결과는 `sample_submission.csv` 행 순서를 가정하지 않고 `id`로 결합합니다.

재현성 함정과 leakage 점검표는 [`docs/VALIDATION.md`](docs/VALIDATION.md)에 있습니다.

## License

코드는 [MIT License](LICENSE)로 공개합니다. 데이터와 pretrained backbone에는 각 원저작자의 라이선스가 별도로 적용됩니다.
