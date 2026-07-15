# AU Paraphrase Augmentation

AU rows benefited from meaning-preserving paraphrase, but uncontrolled translation or code-token changes created label noise.

The generator/QC pipeline enforces:

- preserve the original language; English input stays English
- preserve file names, paths, backticks, and quoted terms
- preserve deictic terms such as `아까`, `방금`, `그때`, `이거`, `저거`, `그거`
- reject empty or identical outputs
- use relaxed length-ratio limits only for very short prompts
- treat action-keyword mismatch as a review flag, not an automatic hard rejection
- keep original ids available for audit

`make_au_paraphrase.py` supports external model APIs through environment variables. Never commit API keys or generated private data. `assemble_au_para.py` validates and joins accepted rows.
