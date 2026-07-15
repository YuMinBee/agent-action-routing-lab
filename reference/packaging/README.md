# Packaging Reference

- `quantize_preserved_int8.py` quantizes large matrices while preserving small and 1D tensors in fp16.
- `verify_submit_package.py` checks archive structure, required files, size, and POSIX paths.

Static checks are necessary but not sufficient. Always run the final archive in the competition-like offline Linux container.
