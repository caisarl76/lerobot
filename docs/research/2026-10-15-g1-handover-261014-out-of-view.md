# G1 handover 261014: out-of-view exclusions, combined training, auto-detection (2026-10-15)

51 new handover-bottle teleop episodes (new bottle positions and lighting, 50 Hz, ego view 640×480), recorded
2026-10-14 with GR00T-WBC SONIC. While the robot reaches for the bottle or hands it over, the right hand or the
bottle often leaves the ego-view camera. The reviewer cut those frames out with the dataset visualizer's excluded
intervals. This note covers the review, the combined dataset, the GR00T he run and the out-of-view detection tests.

## 1. Review

- Raw recording: `/mnt/data/jihun/datasets/G1_WBT_GR00T/handover_261014` (copy of
  `GR00T-WholeBodyControl/outputs/handover_261014`).
- Visualizer run: `official_annotations/workspace/runs/36a1f30ab44f451cab67150e5319bba7/run.json`
  (revision 142).
- **10 Failed**: short 1, 2, 42; failed 25, 45; out of frame 0, 27, 39, 49, 50. Ep 0 was kept with an empty
  tail and set to Failed on 10-15, to match ep 9, whose identical tail was excluded.
- **41 kept**: 23 of them have excluded intervals, 18 have none (mostly eps 3–21, where the hand stays in view).
  There are two patterns:
  - **Reach** (about 25–53 % of the episode; eps 40, 43, 44 and 48 start at frame 0): the bottle is in view, but
    the right hand is outside the right edge or only partly in view.
  - **Handover tail** (about 80–100 %): the hand and the bottle leave the view.
- The recording's task text is wrong ("place OBJECT on the table"). It came from the recording command line
  (`--task-prompt`), not from a default; see section 4.

## 2. Combined datasets

`examples/g1_dex3_training/prepare_wbt_sonic78.py` reads the review with `--annotations`:

- Failed episodes are dropped.
- Every kept frame range between excluded intervals becomes its own episode, so no action chunk spans a cut.
- It writes `<out>.segments.json` (the source episode and frame range of each output episode) and
  `<out>.heldout.json`.

| Dataset (`/mnt/data/jihun/datasets/G1_WBT_GR00T/`) | Episodes | Frames | Notes                                                                                                                                     |
| -------------------------------------------------- | -------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `handover_combined_sonic78_50hz_261014`            | 97       | 58,962 | Eps 0–42 are the old 41-episode set (identical to `handover_bottle_sonic78_50hz_261002`); 43–96 are 54 new segments. Trained (section 3). |
| `handover_combined_sonic78_50hz_261015`            | 96       | 58,134 | Same build with ep 0 Failed. Not trained yet.                                                                                             |

Both hold out episodes 27 and 35, which are old episodes, so the held-out pair is the same as in the 41-episode
runs. Build:

```bash
D=/mnt/data/jihun/datasets/G1_WBT_GR00T
cd examples/g1_dex3_training && uv run python prepare_wbt_sonic78.py \
  $D/handover_bottle_260930_subset_261002 $D/handover_261014 $D/handover_combined_sonic78_50hz_261015 \
  --annotations $D/official_annotations/workspace/runs/36a1f30ab44f451cab67150e5319bba7/run.json \
  --task "pick drink bottle from the table and handover" --heldout 27 35
```

Risks:

- 4 new episodes lose the whole approach (their exclusion starts at frame 0).
- Several segments start mid-task, so the reach start still comes mostly from the old 41 episodes.

## 3. GR00T he run

- Run: `groot_handover261014_he_full` on h100_174 GPU 7.
- Job root: `/mnt/data01/jhkim/model_weight/g1_dex3_20260922/g1_wbt_handover_261014`.
- Init: `mncai/g1-he-groot-sonic78sonicstate-ho5-official-full`, with its stats re-fitted to the dataset by
  `retarget_init_checkpoint.py`.
- Training: batch 32, 16,000 steps. That is ~8.8 passes over the data, the same as the 41-episode `wbt50` run.
  Started 10-14 09:46 UTC, ended 12:07 UTC with exit 0.
- Loss: 0.141 at step 100, 0.053 at 3k, 0.033 at 9k, 0.025 at 15k.
- Checkpoints at 5k, 10k, 15k and 16k. At 5k it has seen 160K samples, the same as the 41-episode run, so 5k is
  the equal-budget comparison point.
- Final weights: on the Hub as private `mncai/g1-wbt-handover-groot-handover261014-he-full`, and on the
  workstation in `/mnt/data/jihun/g1_models/wbt_groot_handover261014_he_full/pretrained_model`. The sha256 is
  the same on the host, the Hub and the workstation. The intermediate checkpoints exist only on h100_174.
- **Rerun without ep 0, `groot_handover261015_he_full` (started 10-15 07:25 UTC on h100_174 GPU 7):**
  - Same config, init (`init/groot_he`, retargeted to the 261014 stats) and 16,000 steps.
  - Same dataset files on h100_174; ep 0's segment (index 43 of `_261014`) is excluded together with the
    held-out 27/35. That gives 94 training episodes and 57,284 frames, the same data as `_261015`.
  - The normalization stats still include ep 0's 828 frames (1.4 %).
- **Robot (10-16, session s08, 16k weights): 9/10**, against 6/10 for `wbt_groot_he_full` (41 episodes, s06 + s07)
  on the same setup. Details in the
  [handover note](2026-10-02-g1-wbt-handover-training.md#real-robot-94-episode-groot-he-2026-10-1516). The 5k
  checkpoint (equal sample budget) has not been tested.
- **Merged set, `groot_handover261019_he_full` (10-19, h100_174 GPU 7):** dataset
  `handover_bottle_merged_261019` (the user's merge of reviewed runs: old 43, 25 segments of the 10-14 recording,
  7 episodes of the 10-19 recording) converted with `prepare_wbt_sonic78.py ... --heldout 27 35` to
  `handover_bottle_merged_sonic78_50hz_261019` (75 episodes; 73 / 60,526 frames for training). Job root
  `g1_wbt_handover_261019`, init `init/groot_he` copied and retargeted to the new stats, same 16,000 steps
  (8.5 passes). Started 10-19 09:16 UTC, exit 0 at 11:41 UTC, loss 0.141 at step 100 and 0.024 at 16k. On the Hub
  as private `mncai/g1-wbt-handover-groot-handover261019-he-full` and on the workstation in
  `/mnt/data/jihun/g1_models/wbt_groot_handover261019_he_full/pretrained_model` (sha256 `bb15edc3…`, same on the
  host, the Hub and the workstation).
  - **Robot (10-20, s09): 4/10**, against 9/10 for the 94-episode model; the reach stalls near the bottle. The
    10-19 episodes each hold several handover cycles (median 6), so cut them per cycle before training on them
    again. Details in the
    [handover note](2026-10-02-g1-wbt-handover-training.md#real-robot-75-episode-merged-groot-he-2026-10-20).
- **10-19 attempts cut, `groot_handover261020_he_full` (10-20, h100_174 GPU 2; GPU 7 was busy with Long-WAM):**
  `handover_bottle_merged_261020` (107 episodes: the old 43, the same 25 10-14 segments, 39 single attempts of the
  10-19 recording cut at the user's hand-marked exclusions) → `handover_bottle_merged_sonic78_50hz_261020` (105 /
  58,054 frames for training). Job root `g1_wbt_handover_261020`, same init and recipe (16,000 steps, 8.8 passes),
  loss 0.143 at step 100 and 0.023 at 16k. On the Hub as private
  `mncai/g1-wbt-handover-groot-handover261020-he-full` and on the workstation in
  `/mnt/data/jihun/g1_models/wbt_groot_handover261020_he_full/pretrained_model` (sha256 `b7ee9c8c…`, same as the
  Hub). The 261019 workstation copy was deleted (Hub only).
  - **Robot (10-20, s10): 8/10**; the reach is fixed, but successes are slower than s08's and one trial swayed after
    the lift. Details in the
    [handover note](2026-10-02-g1-wbt-handover-training.md#real-robot-105-episode-groot-he-with-the-10-19-attempts-cut-2026-10-20).

## 4. Recording command

The wrong task text came from the exporter command in the operator's shell history, which was reused from an
earlier session:

```bash
python -m gear_sonic.scripts.run_data_exporter --task-prompt 'place OBJECT on the table' ... --dataset-name hyelin_261002
```

For handover recordings, pass the training task text:

```bash
python -m gear_sonic.scripts.run_data_exporter --task-prompt 'pick drink bottle from the table and handover' \
  --data-collection-frequency 50 --camera-host 192.168.0.222 --camera-port 5555 \
  --state-zmq-host 192.168.0.222 --sonic-zmq-host 192.168.0.62 --dataset-name handover_<yymmdd>
```

## 5. Out-of-view auto-detection: what was tested

Target (the user's definition): flag frames where the right hand **or** the bottle is not visible.

Bar for building a visualizer filter: recall ≥ 0.9 at precision ≥ 0.7 against the reviewer's exclusions.

Test set: the 42 episodes kept at test time (before ep 0 was set to Failed), sampled at 5 fps. That is 4,643
frames, of which 13.5 % are excluded. Fitted methods use 5-fold out-of-episode cross-validation. The scripts were
throwaway and are not in the repo.

| Method                                                                                    | AUC  | Recall at precision 0.7       | Precision at recall 0.9 |
| ----------------------------------------------------------------------------------------- | ---- | ----------------------------- | ----------------------- |
| OWLv2 zero-shot ("robot hand" / "bottle")                                                 | 0.54 | 0                             | 0.15                    |
| OWLv2 zero-shot ("white robot arm")                                                       | 0.61 | 0                             | 0.16                    |
| Grounding DINO zero-shot                                                                  | 0.63 | 0                             | 0.16                    |
| Proprioception + episode progress                                                         | 0.74 | 0.03                          | 0.14                    |
| CLIP-L image features + logistic probe                                                    | 0.77 | 0                             | 0.17                    |
| CLIP + OWLv2 + proprioception                                                             | 0.79 | 0.08                          | 0.15                    |
| Genon GPT-5.6-Sol, 3 frames 1 s apart, 300-frame balanced sample, best rule ("hand none") | –    | recall 0.61 at precision 0.26 | –                       |
| GPT-5.6-Sol answers + proprioception (logistic, out-of-episode)                           | –    | 0                             | 0.15                    |

Precision for GPT-5.6-Sol is re-weighted from the balanced sample to the 13.5 % base rate. Time per frame:
OWLv2 0.08 s, Grounding DINO 0.29 s and CLIP 0.013 s on one GPU; GPT-5.6-Sol about 4 s per request.

Findings:

- **Open-vocabulary detectors don't see the Dex3 hand.** OWLv2 scores "robot hand" 0.03–0.15 whether or not the
  hand is in the frame. Grounding DINO gives about 0.45 everywhere.
- **GPT-5.6-Sol sees the hand, but per-frame visibility doesn't match the labels.**
  - It calls the hand "partial" in 105 of 150 kept frames: while carrying, the hand is normally cut by the right
    image edge, and the reviewer kept those frames.
  - It calls the hand "none" in 40 kept frames: idle starts and ends, where no hand is expected.
  - It can't judge the phase when the hand is out of view and calls those frames "idle" or "done".
- **So the exclusions mean "the hand is absent while the arm is reaching or carrying", not "the hand is not
  fully visible".** A per-frame visibility judge needs the arm phase from proprioception, and a 300-frame VLM
  sample was too small to fit that combination.
- **Label noise:** eps 0 and 9 had the same empty tail with different labels; resolved on 10-15 by setting ep 0
  to Failed.

## 6. Plan: small Dex3 hand + bottle detector (paused 10-15)

**Paused by the user on 10-15.** The 261014 review already covers this data by hand, and the next recordings
will keep the hand and the bottle in view instead:

- keep bottle placements inside the area where the reach stays in view (as in eps 3–21);
- hand the bottle over in front of the camera and stop recording right after the release.

Resume the plan only if out-of-view frames show up again, or if a warning during recording is wanted. Everything
below stays in place: the 457 sampled frames, the Label Studio project (no labels yet; restart it with
`start_label_studio.sh`) and `hand_detector.py`.

Goal: a reliable per-frame hand/bottle signal on our ego view. Combined with the arm phase from proprioception,
it should flag the reach and handover-tail exclusions. If it does, it becomes the scorer of the visualizer
out-of-view filter (Phase 2, which needs its own spec and review).

1. **Frames (about 500).**
   - Sample at 1 fps from all 51 `handover_261014` episodes; the Failed ones help because they contain long
     out-of-view stretches.
   - Add about 100 frames from `handover_bottle_260930_subset_261002` for the older lighting.
   - Stratify by the GPT-5.6-Sol answers (hand none / partial / full), so the rare "full" and the edge cases are
     well covered.
   - Split by episode: about 10 test episodes, the rest train.
2. **Labels (operator, about 1 hour).**
   - Boxes for `right_hand` (Dex3 hand including the wrist) and `bottle`, with a `truncated` flag when the box
     touches the image edge.
   - Frames with nothing visible need no boxes.
   - Tool: Label Studio, run once via `uvx label-studio` on the workstation; it is not a project dependency.
     Export as COCO.
3. **Model.**
   - torchvision `fasterrcnn_mobilenet_v3_large_fpn` (COCO-pretrained), fine-tuned for 2 classes at 640×480 on
     workstation GPU 0. torchvision is already installed and BSD-licensed.
   - Expected cost: about 30 epochs in about 15 minutes, under 15 ms per frame.
   - Ultralytics YOLO is the fallback if small-box recall is poor; it adds a dependency and is AGPL-licensed.
4. **Evaluation.**
   - (a) Box AP50 and per-class presence accuracy on the test episodes; target ≥ 0.9 presence accuracy for the
     hand.
   - (b) The real target: frame score = (hand absent or truncated to under ~30 % of its usual area, or bottle
     absent) × arm active (wrist speed or distance from the home pose, from proprioception), smoothed (0.5 s
     minimum, gaps under 0.3 s merged). Score it against the reviewer's exclusions on the 41 kept episodes with
     the same bar as above, and also count how many excluded intervals are hit.
   - Expect (b) to be below (a): the labels are partly judgment calls. A miss in (b) with a pass in (a) means
     the rule needs work, not the detector.
5. **Decide.**
   - If (b) meets the bar: write the Phase 2 spec. The visualizer `out_of_view` import filter would sit next to
     `transition_pauses`, run in a separate GPU runner process (the backend venv has CPU-only torch), and
     produce suggestions for review only.
   - If only (a) passes: use the detector for a recording-time warning (hand out of view while the arm moves)
     instead of auto-annotation.

### Status and commands (10-15)

- Steps 1–2 are ready.
  - `hand_detector.py sample` picked 457 frames: 385 for training and 72 for testing, split by episode. 93 of
    them come from excluded intervals or out-of-frame Failed episodes.
  - Frame selection used the reviewer's labels instead of GPT-5.6-Sol answers, so no frames went to Genon.
  - Data: `/mnt/data/jihun/datasets/G1_WBT_GR00T/dex3_hand_detector` (`images/`, `frames.json`).
- Labeling: `start_label_studio.sh` in that folder starts Label Studio at <http://127.0.0.1:8090> (localhost only,
  tmux session `hand-labels`). Project 1 holds the 457 tasks. The login is in `label_studio/.login`, which only
  the owner can read.
- Steps 3–4: training and evaluation are implemented and pass a smoke test with fake boxes. Once the labels are
  in, run from `examples/g1_dex3_training/`:

```bash
uv run python hand_detector.py labels /mnt/data/jihun/datasets/G1_WBT_GR00T/dex3_hand_detector   # Label Studio -> labels.json
uv run python hand_detector.py train  /mnt/data/jihun/datasets/G1_WBT_GR00T/dex3_hand_detector 30
uv run python hand_detector.py eval   /mnt/data/jihun/datasets/G1_WBT_GR00T/dex3_hand_detector
```

Step 4(b) (detector × arm activity against the reviewer's exclusions) is added after 4(a) passes.
