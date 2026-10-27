# G1 handover: official GR00T 46-D vs ours, robot session plan (2026-10-21)

Goal: compare on the robot three GR00T N1.7 models trained on the same 105 handover episodes
(`handover_bottle_merged_261020`, episodes 27 and 35 held out), plus NVIDIA's own inference client as a cross-check.

| Model | Init | State | Recipe | Workstation path |
|---|---|---|---|---|
| s10 (ours, 8/10 on 10-20) | HE-pretrained | 28-D arms + hands | LeRobot, `new_embodiment`, 16k × 32 | `/mnt/data/jihun/g1_models/wbt_groot_handover261020_he_full/pretrained_model` |
| no-HE control | `nvidia/GR00T-N1.7-3B` | 28-D | same as s10, only the init differs | `/mnt/data/jihun/g1_models/wbt_groot_handover261020_base_full/pretrained_model` |
| official 46-D | `nvidia/GR00T-N1.7-3B` | 46-D: legs, waist, arms, hands, projected gravity | Isaac-GR00T 2a08eec, `UNITREE_G1_SONIC`, 20k × 32 | `/mnt/data/jihun/g1_models/wbt_groot_official46_handover261020_base_20k` |

The three robot comparisons:
- **s10 vs no-HE** measures what HE pretraining adds on this dataset.
- **no-HE vs official** measures what NVIDIA's setup adds with the init held constant: the 46-D state, NVIDIA's pretrained G1 SONIC embodiment head, and teleop hand targets instead of `action.wbc`.

## Before the session

1. **No-HE copy.** It is still running detached. Its log `/mnt/data/jihun/g1_models/wbt_groot_handover261020_base_full/copy_from_h100.log` must end in `SHA MATCH`. If it does not, rerun `copy_from_h100.sh` in the same directory; it streams through the h100 container `jihun-lerobot-wbt-gpu7-261021`.
2. **Code.** Everything is on branch `feat/g1-groot-official46`:
   - `gr00t_official_server.py`;
   - `sonic46_state` in `stream_backends.py`;
   - `state_dim` handling and the `state46` log key in the streamer;
   - the `server46` launcher mode;
   - `official46_offline_check.py`.

   Merge it, or run from it, then copy the streamer files to PC2 with `g1_groot_real_run.sh sync`. The streamer stays compatible with 28-D servers (`state_dim` defaults to 28).
3. **Workstation environments.**
   - Official model: `/mnt/data/jihun/venvs/isaac-gr00t-2a08eec` with source `/mnt/data/jihun/code/isaac-gr00t-2a08eec`. These are the `server46` defaults.
   - LeRobot models (`server`): `uv run --project ../..` from a checkout that has a synced env. Running `server` from a worktree without one makes uv build a partial `.venv` there (one was made and deleted on 2026-10-21).
4. **GPU, one model at a time.** GPU 1 (RTX 3060, 12 GB) holds one model:
   - LeRobot GR00T needs about 9.4 GB.
   - The official model needs about 7.4 GB at 0.13 s per chunk.
   - Another session's s10 server may still be running as `groot-server-5561`. Stop a server (`PORT=… g1_groot_real_run.sh stop`) before starting the next.
   - GPU 0 runs a sim deploy and cannot hold the LeRobot GR00T.

## Session protocol (same for every model)

- `POLICY_FPS=50 MAX_S=30 TRIALS=10`, one attempt per trial, task `pick drink bottle from the table and handover`.
- Defaults: noise 0, `--chunk-blend-s 0.3`, replan 0.4 s.
- Use the same table, bottle and placement for all models, and alternate the order if more than one day is used.
- Use the next free session numbers (s13–s15 have since been used for the chunk-anchoring sessions): official46, no-HE, then an s10 rerun the same day. Rerunning s10 controls for day effects; s11 and s12 showed run-to-run variance from the stall issue.

```bash
cd examples/g1_dex3_training
# official 46-D through our streamer
MODEL=/mnt/data/jihun/g1_models/wbt_groot_official46_handover261020_base_20k GPU=1 PORT=5562 NOISE_SCALE=0 \
  ./g1_groot_real_run.sh server46
PORT=5562 POLICY_FPS=50 MAX_S=30 TRIALS=10 ./g1_groot_real_run.sh session sNN_official46_handover \
  "pick drink bottle from the table and handover"
PORT=5562 ./g1_groot_real_run.sh stop
# no-HE control (then s10 the same way, with its MODEL)
MODEL=/mnt/data/jihun/g1_models/wbt_groot_handover261020_base_full GPU=1 PORT=5560 ./g1_groot_real_run.sh server
PORT=5560 POLICY_FPS=50 MAX_S=30 TRIALS=10 ./g1_groot_real_run.sh session sNN_nohe_handover \
  "pick drink bottle from the table and handover"
```

Run `deploy` and `camera` as usual, then `pull` each session and review it in the eval logger. The official run's trial
logs carry `state46`, the exact 46-D state the model saw, for checks against the training distribution.

## NVIDIA's own client (cross-check, about 5 trials, after the three sessions)

Purpose: if the official model scores much worse through our streamer, this run tells whether our adapter or
blending is at fault or the model itself.

- **Server, workstation.** Run the stock Isaac-GR00T server in the same venv:
  `PYTHONPATH=/mnt/data/jihun/code/isaac-gr00t-2a08eec HF_HUB_OFFLINE=1 python gr00t/eval/run_gr00t_server.py --model-path <official46 dir> --embodiment-tag UNITREE_G1_SONIC --port 5550`.
  Offline it hits the same Cosmos-processor Hub lookup that `gr00t_official_server.py` patches (`build_local_processor`). Either apply that patch or start it once online.
- **Client, PC2.** Run `gear_sonic/scripts/launch_inference.py` from GR00T-WholeBodyControl branch `work/g1-handover-loop-20261020`. It contains fix 5e35112, which preserves the measured VLA hand observations; the local `vr3pt-cleanup-minimal-official` branch still overwrites the left-hand state. Options: `--policy-host 192.168.0.62 --policy-port 5550 --prompt "pick drink bottle from the table and handover"`, `action_horizon` 40 and publish rate 50 (defaults). It starts the deploy itself and uses NVIDIA's camera server on port 5555.
- **Unverified on PC2:** that the branch and `.venv_inference` exist (`install_scripts/install_inference.sh`), the camera server and device, the `initial_pose` choice (`planner_standing` is closest to our `--start planner`), and the deploy options it passes.
- **Conflict.** Stop our `g1_head_camera_server` and our deploy first; both clients want the same ports.
- **Controls** (keyboard pane): `k` starts the control loop, `i` returns to standing, `p` starts/pauses the policy. No eval logger, so note verdicts by hand. Its data exporter can record episodes (`c`/`s`/`f`).
- **Differences from ours:** no chunk blending (it jumps to the latency-compensated index of each new chunk), unscaled sampling noise, no streamer watchdogs.

## Offline results so far (workstation GPU 0, 2026-10-21)

- **State.** `sonic46_state` rebuilds the recorded 46-D state of all 107 episodes exactly (max error 1e-16). The hand order follows NVIDIA's robot model: Dex3 `hand_q[[3,4,5,6,0,1,2]]` for both hands.
- **Open loop.** Token MAE of 40-step chunks every 0.4 s through the adapter. s10 was measured through its server. "Hold" repeats the current token.

| Episode | official 46-D | s10 | hold |
|---|---|---|---|
| 27 (held out) | 0.0240 | 0.0237 | 0.0220 |
| 35 (held out) | 0.0317 | 0.0290 | 0.0302 |
| 0 (train) | 0.0094 | 0.0083 | 0.0177 |
| 50 (train) | 0.0096 | 0.0094 | 0.0218 |

Both models are close to "hold" on held-out episodes, so this metric cannot rank them; the robot has to.

## Related work and open items

- HE pretraining of every VLA with the 46-D state runs on h100_174 GPUs 4 and 5, managed by another session
  (memory note `g1-he-state46-pretrain.md`). Its 46-D handover fine-tunes are the next models for this comparison.
- Hub upload of the official and no-HE models is blocked. The token on h100 (`~/g1_hf_upload/hf.env`) gets 403
  creating repos in `mncai`. Fixed waiter and scripts: `~/g1_hf_upload/upload_official46_nohe.sh`,
  `official46_upload.py`, `nohe_cleanup.py`. Restart the waiter once the repos exist or the token can create them.
  The host weights must stay until then.
- Possible finger-order mismatch in our 28-D models: the WBT 28-D state names the left index and middle pairs in
  the opposite order from NVIDIA's robot model, and the HE 28-D right hand has the mirror case. It matters little
  for this mostly right-handed task, but check it before training more 28-D models.
