# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

All development uses the `airo-mono` conda environment located at `~/Documents/thesis/airo-mono`:

```bash
conda activate airo-mono
```

The repo itself is installed as a package inside that environment. Submodules (`airo-mono/`, `airo-ipc/`, `lerobot/`) are vendored locally.

To set up from scratch:
```bash
git submodule update --init
conda env create -f environment.yaml   # creates 'robilglue' env
```

## Running tests

```bash
conda run -n airo-mono pytest test/
```

There is currently only a dummy test (`test/test_dummy.py`).

## Linting

```bash
conda run -n airo-mono flake8 .   # max-line-length 120, ignores E203/E501/E266/E402
```

## Architecture

The repo has two layers:

### 1. `robot_imitation_glue/` — generic framework

Abstract base classes in `base.py` define the three contracts everything builds on:
- `BaseEnv` — hardware environment (`get_observations`, `act`, `reset`, `get_joint_configuration`, `get_robot_pose_se3`, `get_gripper_opening`, `move_robot_to_tcp_pose`, `move_gripper`)
- `BaseAgent` — any policy or teleop device (`get_action(observation) → np.ndarray`)
- `BaseDatasetRecorder` — dataset writer (`start_episode`, `record_step`, `save_episode`)

`collect_data.py` and `eval_agent.py` implement the generic control loops (with keyboard listener, Rerun visualisation, precise timing via `utils.precise_wait`). Hardware station code calls these loops by passing concrete implementations of the three base classes.

Action format is always **absolute EEF pose + absolute gripper width**. Converter callables translate between teleop/policy action formats and this env format (see `docs/action-flow.png`).

Agents live in `robot_imitation_glue/agents/` (LeRobot ACT/DP, OpenVLA, Pi0, Gello, Spacemouse). Dataset utilities live in `robot_imitation_glue/lerobot_dataset/`.

### 2. `raspberry_IL/` — tactile pick experiment (added for this thesis)

A self-contained imitation learning pipeline for a UR3 + Robotiq gripper + AnySkin tactile sensor + raspberry pressure sensors + load cell. Does **not** use the generic `collect_data.py`/`eval_agent.py` loops — it has its own simpler loops.

**Data collection → training → eval flow:**

```
collect_raspberry_data.py   →  LeRobot dataset (on disk)
prepare_raspberry_dataset.py →  training_data/*.npz  (state + action arrays)
train_raspberry_bc.py        →  outputs/*.pt  checkpoint
eval_raspberry_bc.py         →  runs policy on hardware
```

**Key design decisions in `raspberry_IL/`:**

- `RaspberryPickEnv` (`uR3station/raspberry_pick_env.py`): the arm motion is **scripted** (linear interpolation GRASP_Q → PULL_Q via `_advance_pull_step`); only the gripper width is policy-controlled. `act()` is called each step and is **blocking** (two sequential `.wait()` calls), so effective Hz will be lower than the 10 Hz target once pulling starts.

- Observations (`get_observations`) return a dict with keys: `state` (29-dim concatenation), `gripper_state`, `raspberry_state`, `raspberry_diff`, `anyskin_mag`, `anyskin_slip`, `loadcell_state`, `phase`. The BC policy only reads `state`.

- `OnlineFeatureProcessor` (`raspberry_trial_utils.py`) computes running baselines, moving averages, slip proxies, and detach detection online. It must be `.reset()` between episodes.

- Sensor threads (raspberry serial, loadcell serial, AnySkin) run as daemon threads and persist across episodes. A `threading.Lock` protects all shared sensor state.

- `LeRobotDatasetRecorder` (`dataset_recorder.py`) wraps the vendored LeRobot `add_frame`/`save_episode` API. Requires `weights_only=False` in `torch.load` when loading checkpoints (PyTorch ≥ 2.6 broke the default).

- The heuristic agent (`agents/heuristic_raspberry_agent.py`) closes by `close_delta` each step and applies additional closing proportional to slip (`slip_gain < 0`).
