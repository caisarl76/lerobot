# G1 Dex3 model registry

Generated 2026-10-14 00:59 UTC by `examples/g1_dex3_training/model_registry.py` (regenerate it after training or deleting runs; the CSV next to this file has the same rows).
Paths are host paths under `/mnt/data01/jhkim/model_weight/g1_dex3_20260922` on the named host.

## combined_261009

| Run | Policy | Action | Samples (micro-batch × steps, accum) | Excl. eps | Final | Exit | Finished (UTC) | GB | Host | Workstation copy | Hub |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `groot_combined_sonic78_ho5_official_full` | groot | 78D | 640,000 (32 × 20000, accum 1) | 384 | 020000 | 0 | 2026-10-09 20:20 | 12.6 | h100_174 |  | mncai/g1-combined-groot-combined-sonic78-ho5-official-full |
| `pi05_combined_sonic78_ho5_official_full` | pi05 | 78D | 960,000 (8 × 120000, accum 4) | 384 | 120000 | 0 | 2026-10-10 00:23 | 16.6 | h100 |  | mncai/g1-combined-pi05-combined-sonic78-ho5-official-full |
| `xr1_combined_sonic78_ho5_official_full` | xiaomi_robotics | 78D | 480,000 (16 × 30000, accum 3) | 384 | 030000 | 0 | 2026-10-10 05:43 | 10.2 | h100_174 |  | mncai/g1-combined-xr1-combined-sonic78-ho5-official-full |

## g1_wbt_dex_261009

| Run | Policy | Action | Samples (micro-batch × steps, accum) | Excl. eps | Final | Exit | Finished (UTC) | GB | Host | Workstation copy | Hub |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `groot_dex100_base_full` | groot | 78D | 160,000 (32 × 5000, accum 1) | 0 | 005000 | 0 | 2026-10-09 12:43 | 12.6 | h100_174 | yes | mncai/g1-wbt-dex-groot-dex100-base-full |
| `groot_dex100_he_full` | groot | 78D | 160,000 (32 × 5000, accum 1) | 0 | 005000 | 0 | 2026-10-09 11:01 | 12.6 | h100_174 | yes | mncai/g1-wbt-dex-groot-dex100-he-full |
| `groot_dex100_heu_full` | groot | 78D | 160,000 (32 × 5000, accum 1) | 0 | 005000 | 0 | 2026-10-11 09:39 | 12.6 | h100_174 |  | mncai/g1-wbt-dex-groot-dex100-heu-full |
| `groot_dex25_base_full` | groot | 78D | 160,000 (32 × 5000, accum 1) | 0 | 005000 | 0 | 2026-10-09 16:32 | 12.6 | h100_174 | yes | mncai/g1-wbt-dex-groot-dex25-base-full |
| `groot_dex25_he_full` | groot | 78D | 160,000 (32 × 5000, accum 1) | 0 | 005000 | 0 | 2026-10-09 14:17 | 12.6 | h100_174 | yes | mncai/g1-wbt-dex-groot-dex25-he-full |
| `groot_dex25_heu_full` | groot | 78D | 160,000 (32 × 5000, accum 1) | 0 | 005000 | 0 | 2026-10-11 13:21 | 12.6 | h100_174 |  | mncai/g1-wbt-dex-groot-dex25-heu-full |
| `groot_dex50_base_full` | groot | 78D | 160,000 (32 × 5000, accum 1) | 0 | 005000 | 0 | 2026-10-09 18:23 | 12.6 | h100_174 |  | mncai/g1-wbt-dex-groot-dex50-base-full |
| `groot_dex50_he_full` | groot | 78D | 160,000 (32 × 5000, accum 1) | 0 | 005000 | 0 | 2026-10-09 18:08 | 12.6 | h100_174 |  | mncai/g1-wbt-dex-groot-dex50-he-full |
| `groot_dex50_heu_full` | groot | 78D | 160,000 (32 × 5000, accum 1) | 0 | 005000 | 0 | 2026-10-11 10:23 | 12.6 | h100_174 |  | mncai/g1-wbt-dex-groot-dex50-heu-full |
| `pi05_dex100_base_full` | pi05 | 78D | 240,000 (4 × 60000, accum 8) | 0 | 060000 | 0 | 2026-10-10 07:08 | 16.6 | h100_174 | yes | mncai/g1-wbt-dex-pi05-dex100-base-full |
| `pi05_dex100_he_full` | pi05 | 78D | 240,000 (4 × 60000, accum 8) | 0 | 060000 | 0 | 2026-10-10 07:05 | 16.6 | h100_174 | yes | mncai/g1-wbt-dex-pi05-dex100-he-full |
| `pi05_dex100_heu_full` | pi05 | 78D | 240,000 (4 × 60000, accum 8) | 0 | 060000 | 0 | 2026-10-11 14:08 | 16.6 | h100 |  | mncai/g1-wbt-dex-pi05-dex100-heu-full |
| `pi05_dex25_base_full` | pi05 | 78D | 240,000 (4 × 60000, accum 8) | 0 | 060000 | 0 | 2026-10-10 12:56 | 16.6 | h100_174 | yes | mncai/g1-wbt-dex-pi05-dex25-base-full |
| `pi05_dex25_he_full` | pi05 | 78D | 240,000 (4 × 60000, accum 8) | 0 | 060000 | 0 | 2026-10-10 12:51 | 16.6 | h100_174 | yes | mncai/g1-wbt-dex-pi05-dex25-he-full |
| `pi05_dex25_heu_full` | pi05 | 78D | 240,000 (4 × 60000, accum 8) | 0 | 060000 | 0 | 2026-10-12 00:31 | 16.6 | h100 |  | mncai/g1-wbt-dex-pi05-dex25-heu-full |
| `pi05_dex50_base_full` | pi05 | 78D | 240,000 (4 × 60000, accum 8) | 0 | 060000 | 0 | 2026-10-10 18:42 | 16.6 | h100_174 |  | mncai/g1-wbt-dex-pi05-dex50-base-full |
| `pi05_dex50_he_full` | pi05 | 78D | 240,000 (4 × 60000, accum 8) | 0 | 060000 | 0 | 2026-10-10 18:41 | 16.6 | h100_174 |  | mncai/g1-wbt-dex-pi05-dex50-he-full |
| `pi05_dex50_heu_full` | pi05 | 78D | 240,000 (4 × 60000, accum 8) | 0 | 060000 | 0 | 2026-10-11 19:21 | 16.6 | h100 |  | mncai/g1-wbt-dex-pi05-dex50-heu-full |
| `xr1_dex100_base_full` | xiaomi_robotics | 78D | 120,000 (16 × 7500, accum 3) | 0 | 007500 | 0 | 2026-10-09 20:20 | 10.2 | h100_174 |  | mncai/g1-wbt-dex-xr1-dex100-base-full |
| `xr1_dex100_he_full` | xiaomi_robotics | 78D | 120,000 (16 × 7500, accum 3) | 0 | 007500 | 0 | 2026-10-09 20:24 | 10.2 | h100_174 |  | mncai/g1-wbt-dex-xr1-dex100-he-full |
| `xr1_dex100_heu_full` | xiaomi_robotics | 78D | 120,000 (16 × 7500, accum 3) | 0 | 007500 | 0 | 2026-10-11 15:34 | 10.2 | h100_174 |  | mncai/g1-wbt-dex-xr1-dex100-heu-full |
| `xr1_dex25_base_full` | xiaomi_robotics | 78D | 120,000 (16 × 7500, accum 3) | 0 | 007500 | 0 | 2026-10-09 22:42 | 10.2 | h100_174 |  | mncai/g1-wbt-dex-xr1-dex25-base-full |
| `xr1_dex25_he_full` | xiaomi_robotics | 78D | 120,000 (16 × 7500, accum 3) | 0 | 007500 | 0 | 2026-10-09 22:38 | 10.2 | h100_174 |  | mncai/g1-wbt-dex-xr1-dex25-he-full |
| `xr1_dex25_heu_full` | xiaomi_robotics | 78D | 120,000 (16 × 7500, accum 3) | 0 | 007500 | 0 | 2026-10-11 12:38 | 10.2 | h100_174 |  | mncai/g1-wbt-dex-xr1-dex25-heu-full |
| `xr1_dex50_base_full` | xiaomi_robotics | 78D | 120,000 (16 × 7500, accum 3) | 0 | 007500 | 0 | 2026-10-10 01:01 | 10.2 | h100_174 |  | mncai/g1-wbt-dex-xr1-dex50-base-full |
| `xr1_dex50_he_full` | xiaomi_robotics | 78D | 120,000 (16 × 7500, accum 3) | 0 | 007500 | 0 | 2026-10-10 01:02 | 10.2 | h100_174 |  | mncai/g1-wbt-dex-xr1-dex50-he-full |
| `xr1_dex50_heu_full` | xiaomi_robotics | 78D | 120,000 (16 × 7500, accum 3) | 0 | 007500 | 0 | 2026-10-11 17:48 | 10.2 | h100_174 |  | mncai/g1-wbt-dex-xr1-dex50-heu-full |

## g1_wbt_handover_20261002

| Run | Policy | Action | Samples (micro-batch × steps, accum) | Excl. eps | Final | Exit | Finished (UTC) | GB | Host | Workstation copy | Hub |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `groot_wbt20_base_full` | groot | 78D | 160,000 (32 × 5000, accum 1) | 0 | 005000 | 0 | 2026-10-13 06:27 | 12.6 | h100 | yes | mncai/g1-wbt-handover-groot-wbt20-base-full |
| `groot_wbt20_he_full` | groot | 78D | 160,000 (32 × 5000, accum 1) | 0 | 005000 | 0 | 2026-10-13 06:18 | 12.6 | h100_174 | yes | mncai/g1-wbt-handover-groot-wbt20-he-full |
| `groot_wbt50_base_full` | groot | 78D | 160,000 (32 × 5000, accum 1) | 2 | 005000 | 0 | 2026-10-04 17:57 | 12.6 | h100_174 | yes | mncai/g1-wbt-handover-groot-wbt50-base-full |
| `groot_wbt50_he_full` | groot | 78D | 160,000 (32 × 5000, accum 1) | 2 | 005000 | 0 | 2026-10-05 02:12 | 12.6 | h100_174 | yes | mncai/g1-wbt-handover-groot-wbt50-he-full |
| `pi05_wbt20_base_full` | pi05 | 78D | 240,000 (4 × 60000, accum 8) | 0 | 060000 | 0 | 2026-10-13 12:23 | 16.6 | h100 | yes | mncai/g1-wbt-handover-pi05-wbt20-base-full |
| `pi05_wbt20_he_full` | pi05 | 78D | 240,000 (4 × 60000, accum 8) | 0 | 060000 | 0 | 2026-10-13 13:38 | 16.6 | h100_174 | yes | mncai/g1-wbt-handover-pi05-wbt20-he-full |
| `pi05_wbt50_base_full` | pi05 | 78D | 240,000 (4 × 60000, accum 8) | 2 | 060000 | 0 | 2026-10-05 01:15 | 16.6 | h100_174 | yes | mncai/g1-wbt-handover-pi05-wbt50-base-full |
| `pi05_wbt50_he_full` | pi05 | 78D | 240,000 (4 × 60000, accum 8) | 2 | 060000 | 0 | 2026-10-05 09:18 | 16.6 | h100_174 | yes | mncai/g1-wbt-handover-pi05-wbt50-he-full |
| `psi0_wbt50_base_full` | psi0 | 78D | 160,000 (16 × 10000, accum 8) | 2 | 010000 | 0 | 2026-10-02 11:54 | 11.2 | h100_174 |  | mncai/g1-wbt-handover-psi0-wbt50-base-full |
| `psi0_wbt50_he_full` | psi0 | 78D | 160,000 (16 × 10000, accum 8) | 2 | 010000 | 0 | 2026-10-05 13:39 | 11.2 | h100_174 |  | mncai/g1-wbt-handover-psi0-wbt50-he-full |
| `xr1_wbt50_base_full` | xiaomi_robotics | 78D | 120,000 (16 × 7500, accum 3) | 2 | 007500 | 0 | 2026-10-04 17:08 | 10.2 | h100_174 |  | mncai/g1-wbt-handover-xr1-wbt50-base-full |
| `xr1_wbt50_he_full` | xiaomi_robotics | 78D | 120,000 (16 × 7500, accum 3) | 2 | 007500 | 0 | 2026-10-05 11:43 | 10.2 | h100_174 |  | mncai/g1-wbt-handover-xr1-wbt50-he-full |

## humanoid_everyday_g1_20260923

| Run | Policy | Action | Samples (micro-batch × steps, accum) | Excl. eps | Final | Exit | Finished (UTC) | GB | Host | Workstation copy | Hub |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `act_sonic78sonicstate_ho5_official_full` | act | 78D | 1,600,000 (8 × 200000, accum 1) | 226 | 200000 | 0 | 2026-09-29 12:50 | 0.2 | h100 | yes | mncai/g1-he-act-sonic78sonicstate-ho5-official-full |
| `diffusion_sonic78sonicstate_ho5_official_full` | diffusion | 78D | 12,800,000 (64 × 200000, accum 1) | 226 | 200000 | 0 | 2026-09-30 15:12 | 1.1 | h100 | yes | mncai/g1-he-diffusion-sonic78sonicstate-ho5-official-full |
| `fastwam_sonic78sonicstate_ho5_official_full` | fastwam | 78D | 480,000 (8 × 60000, accum 2) | 226 | 060000 | 0 | 2026-10-03 07:14 | 12.0 | h100 |  | mncai/g1-he-fastwam-sonic78sonicstate-ho5-official-full |
| `groot_joint28_ho5_official_full` | groot | 28D | 640,000 (32 × 20000, accum 1) | 226 | 020000 | 0 | 2026-10-02 15:30 | 12.6 | h100_174 |  | mncai/g1-he-groot-joint28-ho5-official-full |
| `groot_sonic78sonicstate_ho5_official_full` | groot | 78D | 640,000 (32 × 20000, accum 1) | 226 | 020000 | 0 | 2026-09-29 10:13 | 12.6 | h100 | yes | mncai/g1-he-groot-sonic78sonicstate-ho5-official-full |
| `groot_sonic78sonicstate_ho5_official_full` | groot | 78D | 640,000 (32 × 20000, accum 1) | 226 | 020000 | – | 2026-09-29 10:13 | 12.6 | h100_174 | yes |  |
| `molmoact2_sonic78sonicstate_ho5_official_full` | molmoact2 | 78D | 800,000 (8 × 100000, accum 2) | 226 | 100000 | 0 | 2026-10-02 05:02 | 12.0 | h100 |  | mncai/g1-he-molmoact2-sonic78sonicstate-ho5-official-full |
| `pi05_sonic78sonicstate_ho5_official_full` | pi05 | 78D | 960,000 (8 × 120000, accum 4) | 226 | 120000 | 0 | 2026-09-30 07:21 | 16.6 | h100 | yes | mncai/g1-he-pi05-sonic78sonicstate-ho5-official-full |
| `psi0_joint28_ho5_matched640k` | psi0 | 28D | 640,000 (16 × 40000, accum 8) | 226 | 040000 | 0 | 2026-10-02 17:34 | 11.2 | h100 |  | mncai/g1-he-psi0-joint28-ho5-matched640k |
| `psi0_sonic78sonicstate_ho5_jitter640k_full` | psi0 | 78D | 640,000 (16 × 40000, accum 8) | 226 | 040000 | 0 | 2026-10-06 14:04 | 11.2 | h100 |  | mncai/g1-he-psi0-sonic78sonicstate-ho5-jitter640k-full |
| `psi0_sonic78sonicstate_ho5_matched640k` | psi0 | 78D | 640,000 (16 × 40000, accum 8) | 226 | 040000 | 0 | 2026-10-02 11:39 | 11.2 | h100 | yes | mncai/g1-he-psi0-sonic78sonicstate-ho5-matched640k |
| `vla_jepa_sonic78sonicstate_ho5_official_full` | vla_jepa | 78D | 960,000 (8 × 120000, accum 4) | 226 | 120000 | 0 | 2026-10-01 03:13 | 11.1 | h100 | yes | mncai/g1-he-vla-jepa-sonic78sonicstate-ho5-official-full |
| `xr1_joint28_ho5_official_full` | xiaomi_robotics | 28D | 480,000 (16 × 30000, accum 3) | 226 | 030000 | 0 | 2026-09-30 17:56 | 10.2 | h100 |  | mncai/g1-he-xr1-joint28-ho5-official-full |
| `xr1_sonic78sonicstate_ho5_nostate_full` | xiaomi_robotics | 78D | 480,000 (16 × 30000, accum 3) | 226 | 030000 | 0 | 2026-10-07 09:54 | 10.2 | h100 | yes | mncai/g1-he-xr1-sonic78sonicstate-ho5-nostate-full |
| `xr1_sonic78sonicstate_ho5_official_full` | xiaomi_robotics | 78D | 480,000 (16 × 30000, accum 3) | 226 | 030000 | 0 | 2026-10-01 01:55 | 10.2 | h100 | yes | mncai/g1-he-xr1-sonic78sonicstate-ho5-official-full |
| `xr1_sonic78sonicstate_ho5_statedrop_full` | xiaomi_robotics | 78D | 480,000 (16 × 30000, accum 3) | 226 | 030000 | – | 2026-10-07 11:07 | 10.2 | h100 |  |  |
| `xr1_sonic78sonicstate_ho5_statedrop_full` | xiaomi_robotics | 78D | 480,000 (16 × 30000, accum 3) | 226 | 030000 | 0 | 2026-10-07 11:07 | 10.2 | h100_174 |  | mncai/g1-he-xr1-sonic78sonicstate-ho5-statedrop-full |

## unitree

| Run | Policy | Action | Samples (micro-batch × steps, accum) | Excl. eps | Final | Exit | Finished (UTC) | GB | Host | Workstation copy | Hub |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `act_sonic78sonicstate_ho5_official_full` | act | 78D | 1,600,000 (8 × 200000, accum 1) | 158 | 200000 | 0 | 2026-10-01 12:10 | 0.2 | h100 |  | mncai/g1-unitree-act-sonic78sonicstate-ho5-official-full |
| `diffusion_sonic78sonicstate_ho5_official_full` | diffusion | 78D | 12,800,000 (64 × 200000, accum 1) | 158 | 200000 | 0 | 2026-10-01 21:50 | 1.1 | h100 |  | mncai/g1-unitree-diffusion-sonic78sonicstate-ho5-official-full |
| `fastwam_sonic78sonicstate_ho5_official_full` | fastwam | 78D | 480,000 (8 × 60000, accum 2) | 158 | 060000 | 0 | 2026-10-04 11:01 | 12.0 | h100 |  | mncai/g1-unitree-fastwam-sonic78sonicstate-ho5-official-full |
| `groot_sonic78sonicstate_ho5_official_full` | groot | 78D | 640,000 (16 × 40000, accum 2) | 158 | 040000 | 0 | 2026-10-01 09:03 | 12.6 | h100 |  | mncai/g1-unitree-groot-sonic78sonicstate-ho5-official-full |
| `molmoact2_sonic78sonicstate_ho5_official_full` | molmoact2 | 78D | 800,000 (8 × 100000, accum 2) | 158 | 100000 | 0 | 2026-10-04 10:47 | 12.0 | h100 |  | mncai/g1-unitree-molmoact2-sonic78sonicstate-ho5-official-full |
| `pi05_sonic78sonicstate_ho5_official_full` | pi05 | 78D | 960,000 (4 × 240000, accum 8) | 158 | 240000 | 0 | 2026-10-03 03:05 | 16.6 | h100 |  | mncai/g1-unitree-pi05-sonic78sonicstate-ho5-official-full |
| `psi0_joint28_ho5_matched640k` | psi0 | 28D | 640,000 (16 × 40000, accum 8) | 158 | 040000 | 0 | 2026-10-02 14:11 | 11.2 | h100_174 |  | mncai/g1-unitree-psi0-joint28-ho5-matched640k |
| `psi0_sonic78sonicstate_ho5_jitter640k_full` | psi0 | 78D | 640,000 (16 × 40000, accum 8) | 158 | 040000 | 0 | 2026-10-06 18:34 | 11.2 | h100_174 |  | mncai/g1-unitree-psi0-sonic78sonicstate-ho5-jitter640k-full |
| `psi0_sonic78sonicstate_ho5_matched640k` | psi0 | 78D | 640,000 (16 × 40000, accum 8) | 158 | 040000 | 0 | 2026-10-02 14:21 | 11.2 | h100_174 |  | mncai/g1-unitree-psi0-sonic78sonicstate-ho5-matched640k |
| `vla_jepa_sonic78sonicstate_ho5_official_full` | vla_jepa | 78D | 960,000 (8 × 120000, accum 4) | 158 | 120000 | 0 | 2026-10-03 21:51 | 11.1 | h100 |  | mncai/g1-unitree-vla-jepa-sonic78sonicstate-ho5-official-full |
| `xr1_joint28_ho5_official_full` | xiaomi_robotics | 28D | 480,000 (8 × 60000, accum 6) | 158 | 060000 | 0 | 2026-10-01 19:10 | 10.2 | h100_174 |  | mncai/g1-unitree-xr1-joint28-ho5-official-full |
| `xr1_sonic78sonicstate_ho5_official_full` | xiaomi_robotics | 78D | 480,000 (8 × 60000, accum 6) | 158 | 060000 | 0 | 2026-10-02 11:23 | 10.2 | h100_174 |  | mncai/g1-unitree-xr1-sonic78sonicstate-ho5-official-full |

## Workstation copies without a run on any host

Their host runs were deleted (e.g. old recipes or the uncorrected Unitree data).

- `/mnt/data/jihun/g1_models/he_groot_sonic78sonicstate_ho5_full`

Dataset, weight and config paths per run are in the CSV (`dataset`, `weights`, `config`, `workstation_copy`).
