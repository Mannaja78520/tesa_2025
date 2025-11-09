#!/usr/bin/env python3
"""
MATLAB/Simulink drone simulation runner from Python.

This script uses the MATLAB Engine for Python to:
 - Start MATLAB
 - Load a Simulink model (e.g., a drone/quadrotor model)
 - Set parameters in the base workspace
 - Run the simulation for a specified stop time
 - Fetch logged signals or workspace variables
 - Optionally export results to CSV

Requirements:
 - MATLAB installed on this machine
 - MATLAB Engine for Python installed
   See: matlabroot/extern/engines/python and run: `python -m pip install .`
 - A Simulink model (.slx) configured to log the signals you want, or
   To Workspace blocks writing variables in the base workspace.

Example usage:
  python matlab_drone_sim.py \
    --model path/to/Quadrotor.slx \
    --stop 10 \
    --param mass=1.45 \
    --param inertia=[0.02,0.02,0.04] \
    --signal logsout \
    --signal x \
    --out out_results

Notes:
 - If you pass `--signal logsout`, the script will attempt to extract
   timeseries from the Simulink `logsout` object into simple arrays.
 - If you pass names like `x` or `u` that exist in the base workspace
   after the simulation (e.g., via To Workspace blocks), they will be
   exported directly.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore


def _try_import_matlab():
    try:
        import matlab  # type: ignore
        import matlab.engine  # type: ignore
        return matlab, matlab.engine
    except Exception as e:  # pragma: no cover - runtime environment dependent
        raise RuntimeError(
            "MATLAB Engine for Python is not available. Install it via:\n"
            "  cd <matlabroot>/extern/engines/python && python -m pip install .\n"
            f"Original error: {e}"
        )


def _to_matlab_value(matlab_mod, value: Any):
    """Best-effort conversion of Python values to MATLAB engine values.

    - Scalars (int/float/bool/str) pass through.
    - 1D list/tuple -> column vector (n x 1) matlab.double
    - 2D list of lists -> (m x n) matlab.double
    - Dicts / others are passed as-is (may fail if MATLAB can't accept it).
    """
    # Simple scalars/strings are fine
    if isinstance(value, (int, float, bool, str)):
        return value

    # List/tuple handling
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return matlab_mod.double([])
        first = value[0]
        # 2D: list of lists
        if isinstance(first, (list, tuple)):
            mtx = [[float(x) for x in row] for row in value]
            return matlab_mod.double(mtx)
        # 1D: make column vector by default
        col = [[float(x)] for x in value]
        return matlab_mod.double(col)

    # Pass-through for dicts or unknown
    return value


def _mat_to_py(value: Any) -> Any:
    """Convert MATLAB engine types to Python/numpy when feasible.

    - matlab.double -> numpy array (if numpy) else nested lists
    - MATLAB struct -> Python dict (engine already maps to dict)
    - timeseries/timetable can't cross boundary; we expect pre-conversion in MATLAB
    - Others returned as-is
    """
    # matlab.double -> numpy/list
    try:
        import matlab  # type: ignore

        if isinstance(value, matlab.double):  # type: ignore[attr-defined]
            # value is sequence-like of rows
            rows = [[float(x) for x in row] for row in value]
            if np is not None:
                return np.array(rows)
            return rows
    except Exception:
        pass

    # dict (MATLAB struct maps to dict)
    if isinstance(value, dict):
        return {k: _mat_to_py(v) for k, v in value.items()}

    return value


def start_matlab_engine() -> Any:
    matlab_mod, matlab_engine = _try_import_matlab()
    eng = matlab_engine.start_matlab()
    # Keep both on engine for convenience where needed
    eng.__dict__["_matlab_mod"] = matlab_mod
    return eng


def add_model_path(eng: Any, model_path: Path) -> None:
    eng.addpath(str(model_path.parent), nargout=0)


def load_model(eng: Any, model_path: Path) -> str:
    """Load Simulink model and return the model name (without extension)."""
    eng.load_system(str(model_path), nargout=0)
    return model_path.stem


def set_base_params(eng: Any, params: Mapping[str, Any]) -> None:
    """Assign parameter variables into MATLAB base workspace."""
    matlab_mod = getattr(eng, "_matlab_mod")
    for name, val in params.items():
        eng.workspace[name] = _to_matlab_value(matlab_mod, val)


def run_sim(eng: Any, model_name: str, stop_time: Union[int, float, str]) -> Any:
    """Run the simulation and return Simulink.SimulationOutput."""
    return eng.sim(model_name, 'StopTime', str(stop_time), nargout=1)


def _extract_from_logsout(eng: Any, sim_out: Any) -> Dict[str, Any]:
    """Attempt to convert logsout to a Python dict {signal_name: {'time': ..., 'data': ...}}.

    Runs a small MATLAB snippet to transform timeseries entries to numeric arrays
    before fetching them across the engine boundary.
    """
    # Prepare MATLAB-side helper to transform logsout to a struct of arrays
    eng.eval(r"""
      function s = __py_extract_logsout(simOut)
        s = struct();
        if ~isfield(simOut, 'logsout') && ~exist('simOut','var')
          return;
        end
        try
          lo = simOut.logsout;
        catch
          try
            lo = simOut.get('logsout');
          catch
            s = struct();
            return;
          end
        end
        try
          n = lo.numElements;
        catch
          n = length(lo);
        end
        for i = 1:n
          try
            el = lo{i};
          catch
            el = lo.get(i);
          end
          try
            name = el.Name;
          catch
            try
              name = el.get('Name');
            catch
              name = sprintf('signal_%d', i);
            end
          end
          try
            ts = el.Values;
          catch
            ts = [];
          end
          if ~isempty(ts)
            if isa(ts, 'timeseries')
              s.(name) = struct('time', ts.Time, 'data', ts.Data);
            elseif isa(ts, 'timetable')
              s.(name) = struct('time', posixtime(ts.Time), 'data', ts.Variables);
            else
              s.(name) = ts; % unknown type, hope it's numeric
            end
          else
            s.(name) = [];
          end
        end
      end
    """, nargout=0)

    eng.workspace['__sim_out'] = sim_out
    eng.eval("__logsout_struct = __py_extract_logsout(__sim_out);", nargout=0)
    logsout_struct = eng.workspace['__logsout_struct']
    return _mat_to_py(logsout_struct)


def fetch_signals(
    eng: Any,
    sim_out: Any,
    signal_names: Iterable[str],
    try_logsout: bool = True,
) -> Dict[str, Any]:
    """Fetch requested signals into Python-friendly data structures.

    Strategy:
      1) If requested and available, pull from logsout (converted to arrays)
      2) Otherwise, try base workspace variables with the given names
    """
    collected: Dict[str, Any] = {}
    names = list(signal_names)

    if try_logsout and ('logsout' in names or not names):
        try:
            logsout_map = _extract_from_logsout(eng, sim_out)
            if isinstance(logsout_map, dict):
                collected.update(logsout_map)
        except Exception:
            # ignore logsout extraction errors, fall back to workspace
            pass

    for name in names:
        if name == 'logsout':
            continue
        try:
            val = eng.workspace[name]
            collected[name] = _mat_to_py(val)
        except Exception:
            # ignore if not present
            pass

    return collected


def _ensure_out_dir(out_dir: Optional[Path]) -> Optional[Path]:
    if out_dir is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_csv(path: Path, arr: Any) -> None:
    if np is not None:
        a = np.asarray(arr)
        # Flatten 1D
        if a.ndim == 1:
            a = a.reshape((-1, 1))
        np.savetxt(path, a, delimiter=",")
        return
    # Fallback: csv writer with lists
    if isinstance(arr, list) and (len(arr) == 0 or not isinstance(arr[0], list)):
        # 1D list -> single column
        rows = [[x] for x in arr]
    else:
        rows = arr
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow(row if isinstance(row, list) else [row])


def export_results(out_dir: Optional[Path], results: Mapping[str, Any]) -> None:
    if out_dir is None:
        return
    meta = {}
    for name, val in results.items():
        try:
            if isinstance(val, dict) and 'time' in val and 'data' in val:
                # Save two files: *_time.csv and *_data.csv
                _save_csv(out_dir / f"{name}_time.csv", val['time'])
                _save_csv(out_dir / f"{name}_data.csv", val['data'])
                meta[name] = {"type": "timeseries", "files": [f"{name}_time.csv", f"{name}_data.csv"]}
            else:
                _save_csv(out_dir / f"{name}.csv", val)
                meta[name] = {"type": "array", "files": [f"{name}.csv"]}
        except Exception as e:
            meta[name] = {"type": "unknown", "error": str(e)}

    with (out_dir / "manifest.json").open('w') as f:
        json.dump(meta, f, indent=2)


def parse_param(s: str) -> Tuple[str, Any]:
    """Parse --param key=value where value is Python literal or string.

    Examples:
      mass=1.45
      inertia=[0.02, 0.02, 0.04]
      name='QuadX'
    """
    if '=' not in s:
        raise argparse.ArgumentTypeError("--param requires key=value format")
    k, v = s.split('=', 1)
    k = k.strip()
    v_str = v.strip()
    try:
        val = ast.literal_eval(v_str)
    except Exception:
        # Treat as raw string
        val = v_str
    return k, val


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run a MATLAB/Simulink drone simulation from Python")
    p.add_argument('--model', required=True, help='Path to the Simulink model (.slx)')
    p.add_argument('--stop', type=float, default=10.0, help='Simulation stop time (seconds)')
    p.add_argument('--param', action='append', default=[], help='Parameter assignment key=value (repeatable)')
    p.add_argument('--signal', action='append', default=[], help='Signal or workspace variable to fetch (repeatable). Include "logsout" to auto-extract logs.')
    p.add_argument('--out', type=str, default=None, help='Output directory to save CSVs (optional)')
    args = p.parse_args(argv)

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")

    out_dir = Path(args.out).resolve() if args.out else None
    _ensure_out_dir(out_dir)

    # Parse params
    params: Dict[str, Any] = {}
    for kv in args.param:
        k, v = parse_param(kv)
        params[k] = v

    # Start MATLAB and run
    eng = start_matlab_engine()
    add_model_path(eng, model_path)
    model_name = load_model(eng, model_path)
    set_base_params(eng, params)

    sim_out = run_sim(eng, model_name, args.stop)
    results = fetch_signals(eng, sim_out, args.signal or ['logsout'])
    export_results(out_dir, results)

    # Print a brief summary
    print(f"Simulated model '{model_name}' for {args.stop} s")
    if results:
        print("Fetched signals:")
        for k in results.keys():
            print(f" - {k}")
    else:
        print("No signals fetched. Ensure you used To Workspace blocks or logsout.")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

