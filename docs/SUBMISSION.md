# Submission Packaging

대회 평가 서버는 Linux, offline, NVIDIA T4 16GB, 3 vCPU, 12GB RAM 환경이며 제출 ZIP은 1GB 이하여야 했습니다. 실제 제한은 대회 페이지가 우선입니다.

## Required Layout

```text
submit.zip
|-- model/
|   |-- sim/
|   |-- sim_alt/
|   |-- au/
|   |-- au_alt/
|   `-- sim_klue/
|-- script.py
|-- klue_infer.py
`-- requirements.txt
```

평가 시 서버가 다음을 추가합니다.

```text
data/test.jsonl
data/sample_submission.csv
output/submission.csv
```

## Path Rules

- ZIP entry는 Windows 역슬래시가 아닌 POSIX `/`를 사용합니다.
- model path는 `Path(__file__).resolve().parent / "model" / ...` 기준으로 찾습니다.
- Hugging Face loader에는 `/app/model/sim` 디렉터리를 넘기기 전에 `config.json` 존재를 assert합니다.
- archive root 아래에 불필요한 상위 폴더가 한 겹 더 생기지 않도록 합니다.

## Output Join

test와 sample submission의 행 순서가 같다고 가정하지 않습니다.

```python
pred_by_id = dict(zip(test_ids, predictions))
sample["action"] = sample["id"].astype(str).map(pred_by_id)
assert sample["action"].notna().all()
```

## Quantization

[`reference/packaging/quantize_preserved_int8.py`](../reference/packaging/quantize_preserved_int8.py)는 큰 matrix만 int8로 바꾸고 작은/1D tensor는 fp16으로 보존합니다.

```bash
python reference/packaging/quantize_preserved_int8.py \
  --src models/sim_fp32 \
  --dst build/submit/model/sim
```

반드시 다음을 확인합니다.

- state key 집합 일치
- quantized/dequantized round-trip error
- model config/tokenizer 동반 여부
- 최종 archive 크기

## Static Verification

```bash
python reference/packaging/verify_submit_package.py build/submit.zip
```

최종 추론은 대회 이미지와 같은 컨테이너에서 offline으로 한 번 끝까지 실행해야 합니다. 예시는 다음과 같습니다.

```powershell
$zip = Resolve-Path build\submit.zip
$data = Resolve-Path data\raw

docker run --rm --network none --gpus all --cpus=3 --memory=12g `
  -v "${zip}:/tmp/submit.zip:ro" `
  -v "${data}:/tmp/data:ro" `
  <competition-image> `
  bash -lc "set -eux; rm -rf /app; mkdir -p /app; unzip -q /tmp/submit.zip -d /app; mkdir -p /app/data; cp /tmp/data/test.jsonl /app/data/test.jsonl; cp /tmp/data/sample_submission.csv /app/data/sample_submission.csv; cd /app; python script.py; test -f output/submission.csv"
```

## Final Checklist

- [ ] ZIP 1GB 이하
- [ ] root에 `script.py`, `requirements.txt`, `model/`
- [ ] 모든 ZIP entry가 `/` 사용
- [ ] 인터넷 없이 실행
- [ ] 10분 이내 실행
- [ ] 최대 12GB RAM
- [ ] model directory와 `config.json` 존재
- [ ] output row 수와 id 집합 일치
- [ ] label이 허용된 14개 클래스에만 속함
- [ ] Docker end-to-end smoke test 통과
