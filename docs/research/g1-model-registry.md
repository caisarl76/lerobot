# G1 Dex3 model registry

Generated 2026-10-08 02:14 UTC by `examples/g1_dex3_training/model_registry.py` (regenerate it after training or deleting runs; the CSV next to this file has the same rows).
Paths are host paths under `/mnt/data01/jhkim/model_weight/g1_dex3_20260922` on the named host.

## g1_wbt_handover_20261002

| Run | Policy | Action | Samples (micro-batch × steps, accum) | Excl. eps | Final | Exit | Finished (UTC) | GB | Host | Workstation copy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `groot_wbt50_base_full` | groot | 78D | 160,000 (32 × 5000, accum 1) | 2 | 005000 | 0 | 2026-10-04 17:57 | 12.6 | h100_174 |  |
| `groot_wbt50_he_full` | groot | 78D | 160,000 (32 × 5000, accum 1) | 2 | 005000 | 0 | 2026-10-05 02:12 | 12.6 | h100_174 | yes |
| `pi05_wbt50_base_full` | pi05 | 78D | 240,000 (4 × 60000, accum 8) | 2 | 060000 | 0 | 2026-10-05 01:15 | 16.6 | h100_174 |  |
| `pi05_wbt50_he_full` | pi05 | 78D | 240,000 (4 × 60000, accum 8) | 2 | 060000 | 0 | 2026-10-05 09:18 | 16.6 | h100_174 | yes |
| `psi0_wbt50_base_full` | psi0 | 78D | 160,000 (16 × 10000, accum 8) | 2 | 010000 | 0 | 2026-10-02 11:54 | 11.2 | h100_174 |  |
| `psi0_wbt50_he_full` | psi0 | 78D | 160,000 (16 × 10000, accum 8) | 2 | 010000 | 0 | 2026-10-05 13:39 | 11.2 | h100_174 |  |
| `xr1_wbt50_base_full` | xiaomi_robotics | 78D | 120,000 (16 × 7500, accum 3) | 2 | 007500 | 0 | 2026-10-04 17:08 | 10.2 | h100_174 |  |
| `xr1_wbt50_he_full` | xiaomi_robotics | 78D | 120,000 (16 × 7500, accum 3) | 2 | 007500 | 0 | 2026-10-05 11:43 | 10.2 | h100_174 |  |

## humanoid_everyday_g1_20260923

| Run | Policy | Action | Samples (micro-batch × steps, accum) | Excl. eps | Final | Exit | Finished (UTC) | GB | Host | Workstation copy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `act_sonic78sonicstate_ho5_official_full` | act | 78D | 1,600,000 (8 × 200000, accum 1) | 226 | 200000 | 0 | 2026-09-29 12:50 | 0.2 | h100 | yes |
| `diffusion_sonic78sonicstate_ho5_official_full` | diffusion | 78D | 12,800,000 (64 × 200000, accum 1) | 226 | 200000 | 0 | 2026-09-30 15:12 | 1.1 | h100 | yes |
| `fastwam_sonic78sonicstate_ho5_official_full` | fastwam | 78D | 480,000 (8 × 60000, accum 2) | 226 | 060000 | 0 | 2026-10-03 07:14 | 12.0 | h100 |  |
| `groot_joint28_ho5_official_full` | groot | 28D | 640,000 (32 × 20000, accum 1) | 226 | 020000 | 0 | 2026-10-02 15:30 | 12.6 | h100_174 |  |
| `groot_joint28_ho5_official_smoke` | groot | 28D | 640 (32 × 20, accum 1) | 226 | 000020 | 0 | 2026-10-02 12:16 | 12.6 | h100_174 |  |
| `groot_sonic78sonicstate_ho5_official_full` | groot | 78D | 640,000 (32 × 20000, accum 1) | 226 | 020000 | 0 | 2026-09-29 10:13 | 12.6 | h100 | yes |
| `groot_sonic78sonicstate_ho5_official_full` | groot | 78D | 640,000 (32 × 20000, accum 1) | 226 | 020000 | – | 2026-09-29 10:13 | 12.6 | h100_174 | yes |
| `molmoact2_sonic78sonicstate_ho5_official_full` | molmoact2 | 78D | 800,000 (8 × 100000, accum 2) | 226 | 100000 | 0 | 2026-10-02 05:02 | 12.0 | h100 |  |
| `pi05_sonic78sonicstate_ho5_official_full` | pi05 | 78D | 960,000 (8 × 120000, accum 4) | 226 | 120000 | 0 | 2026-09-30 07:21 | 16.6 | h100 | yes |
| `psi0_joint28_ho5_matched640k` | psi0 | 28D | 640,000 (16 × 40000, accum 8) | 226 | 040000 | 0 | 2026-10-02 17:34 | 11.2 | h100 |  |
| `psi0_joint28_ho5_official_smoke` | psi0 | 28D | 3,200 (16 × 200, accum 8) | 226 | 000200 | 0 | 2026-09-30 09:53 | 11.2 | h100 |  |
| `psi0_joint28_ho5_official_smoke_amo36` | psi0 | 28D | 6,400 (32 × 200, accum 4) | 226 | 000200 | 0 | 2026-09-30 03:37 | 6.3 | h100 |  |
| `psi0_sonic78sonicstate_ho5_jitter640k_full` | psi0 | 78D | 640,000 (16 × 40000, accum 8) | 226 | 040000 | 0 | 2026-10-06 14:04 | 11.2 | h100 |  |
| `psi0_sonic78sonicstate_ho5_jitter640k_smoke` | psi0 | 78D | 3,200 (16 × 200, accum 8) | 226 | 000200 | 0 | 2026-10-06 08:08 | 11.2 | h100 |  |
| `psi0_sonic78sonicstate_ho5_matched640k` | psi0 | 78D | 640,000 (16 × 40000, accum 8) | 226 | 040000 | 0 | 2026-10-02 11:39 | 11.2 | h100 | yes |
| `psi0_sonic78sonicstate_ho5_official_smoke` | psi0 | 78D | 3,200 (16 × 200, accum 8) | 226 | 000200 | 0 | 2026-09-30 03:50 | 11.2 | h100 |  |
| `vla_jepa_sonic78sonicstate_ho5_official_full` | vla_jepa | 78D | 960,000 (8 × 120000, accum 4) | 226 | 120000 | 0 | 2026-10-01 03:13 | 11.1 | h100 | yes |
| `xr1_joint28_ho5_official_full` | xiaomi_robotics | 28D | 480,000 (16 × 30000, accum 3) | 226 | 030000 | 0 | 2026-09-30 17:56 | 10.2 | h100 |  |
| `xr1_joint28_ho5_official_smoke` | xiaomi_robotics | 28D | 1,440 (16 × 90, accum 3) | 226 | 000090 | 0 | 2026-09-30 03:56 | 10.2 | h100 |  |
| `xr1_sonic78sonicstate_ho5_nostate_full` | xiaomi_robotics | 78D | 480,000 (16 × 30000, accum 3) | 226 | 030000 | 0 | 2026-10-07 09:54 | 10.2 | h100 | yes |
| `xr1_sonic78sonicstate_ho5_nostate_smoke` | xiaomi_robotics | 78D | 1,440 (16 × 90, accum 3) | 226 | 000090 | 0 | 2026-10-07 01:52 | 10.2 | h100 |  |
| `xr1_sonic78sonicstate_ho5_official_full` | xiaomi_robotics | 78D | 480,000 (16 × 30000, accum 3) | 226 | 030000 | 0 | 2026-10-01 01:55 | 10.2 | h100 | yes |
| `xr1_sonic78sonicstate_ho5_official_smoke` | xiaomi_robotics | 78D | 1,440 (16 × 90, accum 3) | 226 | 000090 | 0 | 2026-09-30 04:01 | 10.2 | h100 |  |
| `xr1_sonic78sonicstate_ho5_statedrop_full` | xiaomi_robotics | 78D | 480,000 (16 × 30000, accum 3) | 226 | 030000 | – | 2026-10-07 11:07 | 10.2 | h100 |  |
| `xr1_sonic78sonicstate_ho5_statedrop_full` | xiaomi_robotics | 78D | 480,000 (16 × 30000, accum 3) | 226 | 030000 | 0 | 2026-10-07 11:07 | 10.2 | h100_174 |  |
| `xr1_sonic78sonicstate_ho5_statedrop_smoke` | xiaomi_robotics | 78D | 1,440 (16 × 90, accum 3) | 226 | 000090 | 0 | 2026-10-07 01:54 | 10.2 | h100_174 |  |

## unitree

| Run | Policy | Action | Samples (micro-batch × steps, accum) | Excl. eps | Final | Exit | Finished (UTC) | GB | Host | Workstation copy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `act_sonic78sonicstate_ho5_official_full` | act | 78D | 1,600,000 (8 × 200000, accum 1) | 158 | 200000 | 0 | 2026-10-01 12:10 | 0.2 | h100 |  |
| `diffusion_sonic78sonicstate_ho5_official_full` | diffusion | 78D | 12,800,000 (64 × 200000, accum 1) | 158 | 200000 | 0 | 2026-10-01 21:50 | 1.1 | h100 |  |
| `fastwam_sonic78sonicstate_ho5_official_full` | fastwam | 78D | 480,000 (8 × 60000, accum 2) | 158 | 060000 | 0 | 2026-10-04 11:01 | 12.0 | h100 |  |
| `groot_sonic78sonicstate_ho5_official_full` | groot | 78D | 640,000 (16 × 40000, accum 2) | 158 | 040000 | 0 | 2026-10-01 09:03 | 12.6 | h100 |  |
| `molmoact2_sonic78sonicstate_ho5_official_full` | molmoact2 | 78D | 800,000 (8 × 100000, accum 2) | 158 | 100000 | 0 | 2026-10-04 10:47 | 12.0 | h100 |  |
| `pi05_sonic78sonicstate_ho5_official_full` | pi05 | 78D | 960,000 (4 × 240000, accum 8) | 158 | 240000 | 0 | 2026-10-03 03:05 | 16.6 | h100 |  |
| `psi0_joint28_ho5_matched640k` | psi0 | 28D | 640,000 (16 × 40000, accum 8) | 158 | 040000 | 0 | 2026-10-02 14:11 | 11.2 | h100_174 |  |
| `psi0_joint28_ho5_official_smoke` | psi0 | 28D | 3,200 (16 × 200, accum 8) | 158 | 000200 | 0 | 2026-10-01 02:31 | 11.2 | h100_174 |  |
| `psi0_joint28_ho5_official_smoke_amo36` | psi0 | 28D | 6,400 (32 × 200, accum 4) | 158 | 000200 | 0 | 2026-09-30 04:04 | 6.3 | h100 |  |
| `psi0_sonic78sonicstate_ho5_jitter640k_full` | psi0 | 78D | 640,000 (16 × 40000, accum 8) | 158 | 040000 | 0 | 2026-10-06 18:34 | 11.2 | h100_174 |  |
| `psi0_sonic78sonicstate_ho5_jitter640k_smoke` | psi0 | 78D | 3,200 (16 × 200, accum 8) | 158 | 000200 | 0 | 2026-10-06 08:11 | 11.2 | h100_174 |  |
| `psi0_sonic78sonicstate_ho5_matched640k` | psi0 | 78D | 640,000 (16 × 40000, accum 8) | 158 | 040000 | 0 | 2026-10-02 14:21 | 11.2 | h100_174 |  |
| `psi0_sonic78sonicstate_ho5_official_smoke` | psi0 | 78D | 3,200 (16 × 200, accum 8) | 158 | 000200 | 0 | 2026-09-30 04:09 | 11.2 | h100 |  |
| `vla_jepa_sonic78sonicstate_ho5_official_full` | vla_jepa | 78D | 960,000 (8 × 120000, accum 4) | 158 | 120000 | 0 | 2026-10-03 21:51 | 11.1 | h100 |  |
| `xr1_joint28_ho5_official_full` | xiaomi_robotics | 28D | 480,000 (8 × 60000, accum 6) | 158 | 060000 | 0 | 2026-10-01 19:10 | 10.2 | h100_174 |  |
| `xr1_joint28_ho5_official_smoke` | xiaomi_robotics | 28D | 720 (8 × 90, accum 6) | 158 | 000090 | 0 | 2026-09-30 05:38 | 10.2 | h100 |  |
| `xr1_sonic78sonicstate_ho5_official_full` | xiaomi_robotics | 78D | 480,000 (8 × 60000, accum 6) | 158 | 060000 | 0 | 2026-10-02 11:23 | 10.2 | h100_174 |  |
| `xr1_sonic78sonicstate_ho5_official_smoke` | xiaomi_robotics | 78D | 720 (8 × 90, accum 6) | 158 | 000090 | 0 | 2026-09-30 05:43 | 10.2 | h100 |  |

## Workstation copies without a run on any host

Their host runs were deleted (e.g. old recipes or the uncorrected Unitree data).

- `/home/jihun/work/g1_models/he_act_sonic78sonicstate_ho5_full`
- `/mnt/data/jihun/g1_models/he_groot_sonic78sonicstate_ho5_full`

Dataset, weight and config paths per run are in the CSV (`dataset`, `weights`, `config`, `workstation_copy`).
