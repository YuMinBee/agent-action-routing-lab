# Archived Experiments

This directory keeps research branches that were useful diagnostically but were not part of the final stack.

| File | Idea |
|---|---|
| `hierarchical_oof.py` | Group router followed by specialist classification |
| `multihead_e5_oof.py` | Shared E5 with multiple task heads |
| `twostream_fusion_e5_oof.py` | Semantic/context streams with learned fusion |
| `prototype_rerank.py` | Inspect prototype similarity reranking |
| `knn_oof.py` | OOF-safe nearest-neighbor probabilities |
| `selective_multiview_tta.py` | Extra views only on uncertain examples |
| `candidate_conditioned_inspect.py` | State/action pairwise scoring |

These scripts may require OOF artifacts and local models that are not included. They are preserved to document the search space, not advertised as production entry points.
