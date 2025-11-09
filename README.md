# tesa_2025

This repository now tracks every long-lived branch directly inside `main` so
each pull request can focus on a single top-level folder:

- `hardware/` – full Raspberry Pi / hardware bring-up scripts, datasets, and
  the original `hardware` branch history.
- `matlab/` – MATLAB Engine tooling and documentation from the `matlab`
  branch.
- `server/` – streaming, cron, and remote-visualization notes coming from the
  `server` branch.

To update a branch snapshot, rerun:

```bash
git checkout main
rm -rf <branch>
mkdir -p <branch> && git archive <branch> | tar -x -C <branch>
git commit -am "sync <branch> snapshot"
```
