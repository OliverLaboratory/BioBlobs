# M-CSA data

- `mcsa_single_chain.json` (892 entries, ~966 KiB): one chain per M-CSA entry, as used
  for the paper (`pdb_id`, `chain_used`, `ec_number`, `enzyme_name`, `catalytic_residues`).
  The chain-selection step is not part of this release; see `preprocessing/mcsa/README.md`.
  Consumed by `bioblobs.data.mcsa.load_mcsa`.
- `heldout_ids.txt` (79 ids): M-CSA chains with no SwissProt EC-L3 training protein in the
  same MMseqs2 cluster. See the file header for the exact clustering rule.
- `evaluation_ids.txt` (74 ids): the 74 held-out chains evaluated in Figure 3 and Appendix
  Table 9. See the file header for the exact selection rule.
