# Inference Reference

`script.py` is a readable reference for the final domain router, E5 ensembles, class bias, and output id mapping. `klue_infer.py` implements the conditional KLUE tie-break path.

The accompanying model files are not included. To adapt this code:

1. Place models according to `models/README.md`.
2. Verify that train and inference input builders match.
3. Set model paths relative to `Path(__file__)`.
4. Confirm the final settings in `configs/final_stack.json`.
5. Run an offline Linux/Docker smoke test.

`build_manifest.json` records the exact high-score inference combination and model hashes, but the hashes cannot be verified without the private model artifacts.
