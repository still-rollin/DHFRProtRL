# Draft rule files (not used by the code)

These came from a separate folder of supporting files. None is read by the optimizer.

| File | Status |
|--|--|
| `active_site_rules.yaml` | Not referenced anywhere in the code. The active-site penalties it describes were never applied. |
| `conservation_rules.yaml` | Not referenced anywhere in the code. The code uses the raw entropy values directly. |
| `size_rules_draft_unparsed_tokens.yaml` | Differs from the root `size_rules.yaml` for all 20 residues and uses tokens such as `hydrophobes`/`aromatics` that the code cannot interpret. Treated as an earlier draft. |
| `blosum_rules_draft_invalid_yaml.yaml` | Not valid YAML (keys like `≥high:`), so it cannot be loaded. The root `blosum_rules.yaml` is the valid one. |

Which `size_rules.yaml` a given historical run loaded is not recorded. The root version is the one the code read when the later PPO/SAC runs were launched, which is an inference from file dates.
