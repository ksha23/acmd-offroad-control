# Data and large artifacts

This repository tracks small files only: code, configuration, the training
corpora behind every published checkpoint, the neural checkpoints themselves,
and the canonical figures and CSVs. The **large regenerate-or-recover
artifacts** live off-machine as tar-split **GitHub Release assets** on a private
companion repository, so they are backed up and fetchable anywhere without
bloating git:

- **Data repository:** `ksha23/terrain-aware-offroad-control-data` (private)
- **Tool:** [`data_sync/data_sync.sh`](data_sync/data_sync.sh) (uses `gh`)

## Restore the backed-up data

```bash
# into the current repo tree (default), from a snapshot tag:
data_sync/data_sync.sh list                         # see snapshot tags
data_sync/data_sync.sh pull snapshot-2026-07-02     # download + reassemble + extract
```

`pull` recreates the raw `benchmarking/results/…` generations the paper
publisher selects, together with the additional rig collections.

## Take a new snapshot

```bash
# tar+split every path in data_sync/data_snapshot.list and upload as release assets
DATA_ROOT=/path/to/populated/tree data_sync/data_sync.sh push          # tag = snapshot-YYYYMMDD
DATA_ROOT=/path/to/populated/tree data_sync/data_sync.sh push my-tag
```

`DATA_ROOT` is the tree the listed paths are relative to, and defaults to this
repository root. Files over 2 GB are split into parts of at most 1.9 GB (the
Release asset size limit); `pull` reassembles them.

## What the snapshot contains

**In the snapshot** (`data_sync/data_snapshot.list`): the raw result
generations behind the published evidence — the source folders named by
`my_paper/paper_figures/publish_manifest.json`, plus the frozen estimator
replay traces, the probe sessions, and the human-in-the-loop demonstration
session cited by the provenance registry in the paper's evidence bundle. It
also carries `data/tire_rig/`, the wider set of rig collections that no
published checkpoint is trained on. These are multi-GB and cannot live in git;
the snapshot is their off-machine backup.

**Tracked in git directly**, so the training provenance of every checkpoint the
paper uses is self-contained in the repository:

- `data/tire_rig_commanded/` — the DERIVED corpus behind both deployed
  checkpoints, with its `MANIFEST.json`;
- `data/tire_rig_static/train.csv` — the PRIMARY corpus behind the Table 2
  scalar-parent checkpoint;
- `data/paths/*.csv` — PRIMARY reference-path geometry the runtime loads
  directly;
- `data/5g_generated/` and `data/latency_profiles/` — the traffic traces and
  the profile that replays them;
- the canonical figures, CSVs, manifests, and checkpoints.

`benchmarking/verify_provenance_chain.py` checks the tracked corpora against
the hashes each checkpoint records.

**Not in the snapshot** (local-only, restore on demand or regenerate):

- any local `archive/<YYYY-MM-DD_label>/` directory — snapshot it separately if
  needed:
  `printf 'archive\n' > /tmp/l; LIST=/tmp/l data_sync/data_sync.sh push archive-<date>`
- the full `benchmarking/results/` tree (about 25 GB); only the paper-cited
  subset is snapshotted, and the rest is regenerable through
  `benchmarking/run.py`.

The snapshot set is scoped to the off-git inputs a reproduction of this paper
needs, and excludes smoke runs and unrelated debug generations.

## Snapshotting, not differential sync

`push` re-tars and re-uploads the **whole** backup set under a new tag; `pull`
downloads a tag's tarballs and **extracts over** the destination tree. There is
no per-file delta and no merge:

- adding one file still re-uploads the full set on the next `push`;
- `pull` **overwrites** rather than merges, so it can clobber un-pushed local
  changes.

Treat snapshots as periodic backups rather than a live two-machine sync. A
differential, multi-machine, edit-and-sync workflow — only changed files
transferred, versioned alongside the code — is better served by git-LFS
(transparent, paid above the 1 GB free tier) or DVC (free, bring-your-own cloud
bucket). The Release-asset approach here is chosen for zero-setup, free,
infrequent full backups.
