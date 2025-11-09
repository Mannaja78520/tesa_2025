
This branch isolates the MATLAB/Simulink tooling so it can be reviewed
independently from the rest of `main`. The centerpiece is the
`matlab/matlab_drone_sim.py` runner, which drives a Simulink model from Python
via the MATLAB Engine.

Quick start
-----------

```bash
python matlab/matlab_drone_sim.py \
  --model Quadrotor.slx \
  --stop 10 \
  --param mass=1.45 \
  --param inertia=[0.02,0.02,0.04] \
  --signal logsout \
  --out logs
```

What the script does
--------------------

- Boots a MATLAB Engine session.
- Loads a user-supplied `.slx` model.
- Applies parameter overrides passed via `--param`.
- Runs the simulation for `--stop` seconds.
- Extracts workspace variables or `logsout` entries requested through
  `--signal` and writes them to the output directory (`--out`).

Before running
--------------

1. Install MATLAB locally and enable the Engine for Python:
   ```bash
   cd $MATLABROOT/extern/engines/python
   python -m pip install .
   ```
2. Provide a Simulink model (`--model`) that exposes the signals you want via
   logging or To Workspace blocks.
3. (Optional) Install `numpy` so CSV exports include vector data.

Keeping the branch focused
--------------------------

All MATLAB-specific code lives under the `matlab/` directory, which keeps the
pull request diff scoped to this integration.
# tesa_2025
