# Model Placement

Pretrained backbones and trained checkpoints are intentionally excluded.

The reference inference code expects a structure similar to:

```text
models/
|-- sim/
|-- sim_alt/
|-- au/
|-- au_alt/
`-- sim_klue/
```

Each Hugging Face-style directory needs its matching `config.json` and tokenizer files. Quantized submission directories additionally contain `model_int8.pt` or the manifest-specific q-delta files.

Do not mix a checkpoint with a text builder, maximum length, or classifier architecture from another run.
