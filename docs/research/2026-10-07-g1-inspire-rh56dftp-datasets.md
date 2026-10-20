# Unitree G1 + Inspire RH56DFTP datasets for VLA training

Checked **2026-10-07**. Target hardware: Unitree G1 with two Inspire **RH56DFTP** hands, six actuator commands per hand. This follows the broader [G1 dataset survey](2026-10-05-unitree-g1-manipulation-dataset-survey.md) and narrows hardware compatibility. Real data rank first; simulation, access restrictions, missing licenses and uncertain hand models remain visible.

**Recommendation:** use Unitree's WBT Inspire collection as the largest candidate for whole-body continued pretraining, after confirming its exact hand model and mappings. Add the explicitly RH56DFTP CloudWalk sneaker and precision-manipulation recordings for hardware-specific supervision. The exact-model community sources are small and have substantial finger-feedback or action-diversity limitations. This is a recommendation for adapting an existing VLA into a G1/Inspire base policy; the survey does not establish enough diverse data to train a general foundation model from scratch.

## Prioritized real-data shortlist

“Exact” means the publisher explicitly names RH56DFTP. “Inspire, SKU unspecified” is compatible-family evidence, not proof of the exact physical model. Counts refer to releases; the notes distinguish source recordings from split clips and derivatives.

| Priority / dataset | Hardware evidence | Released volume | Actions useful to this project | License and limitation |
| --- | --- | --- | --- | --- |
| 1. [Unitree WBT collection](https://huggingface.co/collections/unitreerobotics/unifolm-wbt-dataset), Inspire subset — 29 repositories listed below | Real G1 + Inspire; exact SKU unspecified in inspected cards | 7,375 episodes; 7,376,196 frames; 68.30 hours at 30 Hz, summed before deduplication | 25 repos: body29 + hand12 in named fields. Four legacy repos need a separate mapping | 10 cards Apache-2.0; 19 have no license declaration in Hub metadata. Largest breadth and strongest whole-body candidate |
| 2. [CloudWalk sneaker v4](https://huggingface.co/datasets/cloudwalk-research/psi0-g1-sneaker-385ep-v4) | **Exact RH56DFTP**, real G1 | 385 episodes; 1.10 hours | 36D export; extract arms14 + hands12, remove verified padding; torso/navigation are separate | Apache-2.0. 325 positive/recovery + 60 negative examples. Six fingers move together per hand; stationary |
| 3. [jren313 five-task pipette teleop](https://huggingface.co/datasets/jren313/g1-pipette-2view-teleop0925-5task-eerel) | **Exact RH56DFTP** in [recorder schema](https://huggingface.co/datasets/jren313/g1-pipette-2view-teleop0925-5task-eerel/raw/main/meta/source_schema.json), real G1 | 237 clips from 143 retained recordings; 162,623 frames; 45.2 minutes at 60 Hz | High-level body/root35 + hands12; separate `action.low_level`29 available | Apache-2.0. Two views and five workflow steps; hand feedback, tactile and timestamps need auditing |
| 4. [MLeggiero pipette-tip ejection](https://huggingface.co/datasets/MLeggiero/g1-inspire-pipette-tip-twist2) | **Exact RH56DFTP**, real G1 | 68 episodes; 61,609 frames; 17.1 minutes at 60 Hz | TWIST2 action49 = body29 + root6 + hands12 + neck2 | MIT. Frozen hand feedback; mainly right-thumb commands |
| 5. [MLeggiero nut into box using tweezers](https://huggingface.co/datasets/MLeggiero/g1-inspire-nut-in-box-twist2) | **Exact RH56DFTP**, real G1 | 41 episodes; 84,552 frames; 23.5 minutes at 60 Hz | Same 49D high-level contract; three camera views | MIT. Right hand feedback constant; limited finger diversity |
| 6. [MLeggiero page turning](https://huggingface.co/datasets/MLeggiero/g1-inspire-turn-page-twist2) | **Exact RH56DFTP**, real G1 | 38 episodes; 20,990 frames; 5.8 minutes at 60 Hz | Same 49D contract; head view | MIT. Unusable hand feedback; mostly fixed fingers |
| 7. CloudWalk bottle originals: [105 episodes](https://huggingface.co/datasets/cloudwalk-research/gr00t-g1-grab-bottle-right-hand-105ep-v1) + [115 episodes](https://huggingface.co/datasets/cloudwalk-research/gr00t-g1-grab-bottle-right-hand-worst-positions-empty-115ep-v3) | Real G1 + Inspire FTP; full RH56DFTP SKU unspecified | 220 acquired episodes; 149,301 frames; 49.8 minutes at 50 Hz | `action.wbc`43 with Inspire6 + seventh channel per hand; SONIC token64 also present | Apache-2.0. Misleading inherited Dex3 finger names; mostly stationary right-hand grasping |
| 8. [jren313 pipette HIL293](https://huggingface.co/datasets/jren313/g1-pipette-3view-hil293) | Real G1 + Inspire family; exact SKU not independently confirmed for this older release | 293 episodes; 363,006 frames at 60 Hz | High-level body/root and separate hands; inspect its own contract | Apache-2.0. First 46 episodes have reconstructed hand commands; frozen right-hand feedback |

The exact-model rows 2–6 comprise **769 published episodes/clips from 675 retained source episodes, about 2.63 hours**. This is a lower bound from these verified sources, not a total for all publicly available RH56DFTP data. No combined total is given across the whole table because official version overlap and community derivatives need source-level deduplication.

The latest useful additions in this shortlist include the September 24–25 five-step pipette captures, converted September 26, and official WBT metadata updated through late September. Publication or modification date does not imply a new independent recording. [Pipette collection provenance](https://huggingface.co/datasets/jren313/g1-pipette-2view-teleop0925-5task-eerel/raw/main/meta/source_provenance.jsonl), [official inventory with dates and revisions](2026-10-07-g1-inspire-official-datasets.csv).

## How to use the official collection

The live [official Inspire inventory](https://huggingface.co/api/datasets?author=unitreerobotics&search=Inspire&limit=1000&full=true) had 29 repositories on the check date. Their revisions were unchanged from the October 5 pinned metadata survey. The accompanying [CSV](2026-10-07-g1-inspire-official-datasets.csv) records each exact revision, metadata URL, action/state/image fields, license declaration and hardware-verification limit.

The 25 modern schemas total **5,673 episodes / 61.39 hours**. They provide lower body15, left arm7, right arm7 and hand commands12, plus base/EEF fields. Lower body15 already includes legs and waist: do not append the redundant leg/waist columns again. Camera coverage commonly includes head stereo and both wrists; some repos also include IR. [Modern publisher schema](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Put_Clothes_into_Washing_Machine).

The four legacy schemas total **1,702 episodes / 6.91 hours** and expose `robot_q_desired`36, hand commands12 and EEF action12. Do not label their 36D vector as the canonical 29 body joints without recovering its mapping. Their internal directories can say `Dex5` although the published repository is Inspire. [Legacy publisher source](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Collect_Clothes_MainCamOnly).

For an initial subset with explicit Apache-2.0 declarations and the modern split schema, use these **six datasets, 1,800 episodes / 16.50 hours**:

- `Clean_The_Living_Room`
- `Pick_Up_Drinks_v0`
- `Put_Clothes_Into_Basket`
- `Put_Clothes_into_Washing_Machine`
- `Put_Drinks_Into_Fridge_v0`
- `Put_Vegetables_Into_Basket_v0`

All ten Apache-tagged repositories total 3,502 episodes / 23.41 hours, including the four legacy schemas. The remaining 19 repositories total 3,873 episodes / 44.88 hours with no license declared in the inspected Hub metadata; public access alone does not establish reuse terms. Similar task names, `v0` variants and `MainCamOnly` versions may overlap. Counts below are repository sums, not unique demonstrations or unique tasks.

| Official dataset | Episodes | Hours | Schema | License declaration |
| --- | ---: | ---: | --- | --- |
| [Clean_Living_Room](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Clean_Living_Room) | 200 | 1.79 | Body29 + hands12 | Not declared |
| [Clean_The_Living_Room](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Clean_The_Living_Room) | 300 | 2.6 | Body29 + hands12 | Apache-2.0 |
| [Collect_Clothes_MainCamOnly](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Collect_Clothes_MainCamOnly) | 336 | 0.82 | Legacy q_desired36 + hands12 | Apache-2.0 |
| [Empty_Folder_In_Cabinet](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Empty_Folder_In_Cabinet) | 200 | 2.44 | Body29 + hands12 | Not declared |
| [Hang_Up_Clothes](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Hang_Up_Clothes) | 200 | 1.23 | Body29 + hands12 | Not declared |
| [Pick_Cloth_On_Bed_And_Put_In_Laundry](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Pick_Cloth_On_Bed_And_Put_In_Laundry) | 200 | 1.82 | Body29 + hands12 | Not declared |
| [Pick_Up_Basket_And_Put_Vegetables_Into_Basket](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Pick_Up_Basket_And_Put_Vegetables_Into_Basket) | 199 | 3.18 | Body29 + hands12 | Not declared |
| [Pick_Up_Clothes](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Pick_Up_Clothes) | 200 | 1.42 | Body29 + hands12 | Not declared |
| [Pick_Up_Drinks](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Pick_Up_Drinks) | 200 | 2.16 | Body29 + hands12 | Not declared |
| [Pick_Up_Drinks_v0](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Pick_Up_Drinks_v0) | 300 | 2.99 | Body29 + hands12 | Apache-2.0 |
| [Pick_Up_Fruit](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Pick_Up_Fruit) | 200 | 1.55 | Body29 + hands12 | Not declared |
| [Pick_Up_Multiple_Cushions](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Pick_Up_Multiple_Cushions) | 200 | 2.4 | Body29 + hands12 | Not declared |
| [Pickup_Pillow_MainCamOnly](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Pickup_Pillow_MainCamOnly) | 609 | 1.45 | Legacy q_desired36 + hands12 | Apache-2.0 |
| [Plastic_Box_Parts](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Plastic_Box_Parts) | 200 | 6.97 | Body29 + hands12 | Not declared |
| [Push_Wheelchair](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Push_Wheelchair) | 200 | 1.49 | Body29 + hands12 | Not declared |
| [Put_Clothes_Into_Basket](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Put_Clothes_Into_Basket) | 300 | 2.12 | Body29 + hands12 | Apache-2.0 |
| [Put_Clothes_into_Washing_Machine](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Put_Clothes_into_Washing_Machine) | 300 | 4.5 | Body29 + hands12 | Apache-2.0 |
| [Put_Clothes_into_Washing_Machine_MainCamOnly](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Put_Clothes_into_Washing_Machine_MainCamOnly) | 457 | 1.11 | Legacy q_desired36 + hands12 | Apache-2.0 |
| [Put_Drinks_Into_Fridge](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Put_Drinks_Into_Fridge) | 200 | 1.31 | Body29 + hands12 | Not declared |
| [Put_Drinks_Into_Fridge_v0](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Put_Drinks_Into_Fridge_v0) | 300 | 1.74 | Body29 + hands12 | Apache-2.0 |
| [Put_Plate_Into_Dishwasher](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Put_Plate_Into_Dishwasher) | 200 | 2.34 | Body29 + hands12 | Not declared |
| [Put_Sink_Towel](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Put_Sink_Towel) | 100 | 1.5 | Body29 + hands12 | Not declared |
| [Put_Vegetables_Into_Basket](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Put_Vegetables_Into_Basket) | 200 | 1.71 | Body29 + hands12 | Not declared |
| [Put_Vegetables_Into_Basket_v0](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Put_Vegetables_Into_Basket_v0) | 300 | 2.56 | Body29 + hands12 | Apache-2.0 |
| [Sort_Fruit](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Sort_Fruit) | 200 | 2.42 | Body29 + hands12 | Not declared |
| [Supermarket_Shelf_Organizing](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Supermarket_Shelf_Organizing) | 200 | 1.92 | Body29 + hands12 | Not declared |
| [Take_Plates_Into_Dishwasher](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Take_Plates_Into_Dishwasher) | 300 | 3.53 | Legacy q_desired36 + hands12 | Apache-2.0 |
| [Tidy_Living_Room_Slippers](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Tidy_Living_Room_Slippers) | 200 | 2.1 | Body29 + hands12 | Not declared |
| [Walk_To_Table_Put_Cups_And_Rack_Plates_In_Dishwasher](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Walk_To_Table_Put_Cups_And_Rack_Plates_In_Dishwasher) | 374 | 5.12 | Body29 + hands12 | Not declared |

## Other public candidates with incomplete hardware evidence

| Dataset | Volume | Available schema | What remains unresolved |
| --- | ---: | --- | --- |
| [eastflag/g1_can_paperbox](https://huggingface.co/datasets/eastflag/g1_can_paperbox) | 69 episodes | 26D arms/hands, four cameras, Apache-2.0 | Metadata says `Unitree_G1_Inspire`; exact hand model, units and real/sim provenance unstated |
| [eastflag/g1_can_twohand](https://huggingface.co/datasets/eastflag/g1_can_twohand) | 30 episodes | Same 26D family and Apache-2.0 | Same uncertainty; weak task text |
| [eunjuri/empty_can_img_depth](https://huggingface.co/datasets/eunjuri/empty_can_img_depth) | 82 episodes | 26D arms/hands, RGB and depth-labelled video, Apache-2.0 | Exact model and real capture unstated; depth video declares `is_depth_map=false` |
| [MLeggiero/g1-gr00t-inspire-red-ball](https://huggingface.co/datasets/MLeggiero/g1-gr00t-inspire-red-ball) | Not anonymously verified | Exact RH56DFTP named; GR00T-style export | Gated; license `other` |
| [MLeggiero/g1-pack-red-ball-high-level](https://huggingface.co/datasets/MLeggiero/g1-pack-red-ball-high-level) | Not anonymously verified | Related TWIST2 recording family | Gated; Apache-2.0 tag; do not add overlapping variants |
| [MLeggiero/g1-gr00t-inspire-pick_and_place](https://huggingface.co/datasets/MLeggiero/g1-gr00t-inspire-pick_and_place) | Not anonymously verified | Related Inspire family | Manually gated; license `other`; model/volume need accessible metadata |

The two eastflag exports order arms first, then hands; eunjuri interleaves each arm with its hand. Finger labels also differ. A matching 26D shape is insufficient. [eastflag metadata](https://huggingface.co/datasets/eastflag/g1_can_paperbox/raw/main/meta/info.json), [eunjuri metadata](https://huggingface.co/datasets/eunjuri/empty_can_img_depth/raw/main/meta/info.json), [MLeggiero access inventory](https://huggingface.co/api/datasets?author=MLeggiero&limit=100).

## Simulation supplements

These are useful augmentation candidates, with weaker hardware and license evidence than the explicitly matched real sources. Do not count simulated descendants as new real demonstrations.

| Dataset | Episodes / rate | Action contract | Limitations / license |
| --- | ---: | --- | --- |
| [birbirll/g1-inspire-pipette-sim-v1](https://huggingface.co/datasets/birbirll/g1-inspire-pipette-sim-v1) | 52 / 50 Hz | 27D = waist yaw1 + arms14 + hand12 joint targets, radians | Scripted Isaac Lab workflow, three views, varying fingers, fixed legs. Exact RH56DFTP suffix and license unspecified |
| [ytung/g1-pipette-augmented](https://huggingface.co/datasets/ytung/g1-pipette-augmented) | 544 / 30 Hz | Body29 motor targets + hand12 = 41D | Descendants of only 12 real parents; grip reconstruction/taped-finger provenance. No new license declaration |
| [ytung/g1-pipette-augmented-0923](https://huggingface.co/datasets/ytung/g1-pipette-augmented-0923) | 500 / 30 Hz | 41D motor targets + hands | HIL293-derived; no minimum-lift success requirement. No new license declaration |
| [ytung/g1-pipette-augmented0927](https://huggingface.co/datasets/ytung/g1-pipette-augmented0927) | 625 / 25 Hz | Body/root reference35 + hands12 = 47D | HIL293-derived; low-level29 retained separately in diagnostics. No new license declaration |
| [MichaelGan0625 piston bimanual](https://huggingface.co/datasets/MichaelGan0625/g1-inspire-piston-bi-manual-3cam-20260819) | 87 / 50 Hz | 30D = arms14 + hands12 + height1 + navigation3 | Isaac Lab replay; semantic gates documented; exact hand suffix unspecified; license `other` |
| [MichaelGan0625 piston drop success](https://huggingface.co/datasets/MichaelGan0625/g1-inspire-piston-drop-3cam-success) | 62 / 50 Hz | Same 30D family | 50 originals + 12 recovery; tactile placeholders; license `other` |
| [MichaelGan0625 piston hybrid](https://huggingface.co/datasets/MichaelGan0625/g1-inspire-piston-drop-3cam-peiyu-hybrid-20260820) | 54 / 50 Hz | Same 30D family | Replay-fidelity filtering does not establish every semantic task stage; license `other` |
| [birbirll piston pick/place](https://huggingface.co/datasets/birbirll/g1-inspire-piston-pick-place) | 102 / 50 Hz | Same 30D family | Weaker replay provenance; Apache-2.0 described as an ecosystem default, leaving upstream rights uncertain |

The ytung releases retain parent lineage and have incompatible action meanings across versions. Keep augmentations and their real parents in the same train/validation partition. Older 67-episode piston long-horizon variants reuse trajectories with different text/views. Isaac Lab hand radians also require a different mapping from physical RH56 raw registers. [ytung original contract](https://huggingface.co/datasets/ytung/g1-pipette-augmented/raw/main/README.md), [September 27 contract](https://huggingface.co/datasets/ytung/g1-pipette-augmented0927/raw/main/README.md).

The official [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab) supports G129-Inspire collection environments. No official downloadable Inspire simulation demonstration corpus was verified in the checked inventory: the named public simulation demonstrations were Dex1, while the USD repository contains assets. Environment support alone is not a demonstration dataset release.

## Conversion to this project's SONIC / VLA pipeline

Recommended target representations, assuming all 29 body joints are used:

| Policy output | Dimensions | Meaning |
| --- | ---: | --- |
| Arm manipulation | 26 | Arms14 + RH56DFTP actuator12 |
| Whole-body joints | 41 | Body29 + RH56DFTP actuator12 |
| SONIC plus hands | 76 | SONIC latent64 + RH56DFTP actuator12, with hands bypassing the body codec |

These are proposed contracts, **not implemented adapters**. The project's Dex3-oriented 28D/43D datasets include 14 hand coordinates. Replacing those with Inspire requires an embodiment-aware mapping, not truncation. A source43D vector can still contain Inspire6 + padding per side, as CloudWalk bottle illustrates. Its seventh channel's exact semantics require inspection before removal. A supplied SONIC64 token can be reused only if encoder/checkpoint, normalization and timing match this project's codec.

Three source conventions must remain distinct:

- Official WBT cards: per-hand order **index, middle, ring, little, thumb open/close, thumb lateral tilt**, normalized **0=open, 1=closed**. [Publisher hand convention](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Put_Clothes_into_Washing_Machine#hand-convention).
- TWIST2 RH56DFTP recordings: **pinky, ring, middle, index, thumb bend, thumb rotation**, raw **0–1000, 1000=open**. Body joints are radians; the hand values are not. [Recorded vector contract](https://huggingface.co/datasets/MLeggiero/g1-inspire-turn-page-twist2#units--conventions).
- CloudWalk sneaker: the latter finger order with normalized **[0,1]** and a seventh zero; all six actuators share a trigger scalar. Verify actuator endpoint polarity from its controller before applying the official WBT normalization. [Canonical hardware and mapping](https://huggingface.co/datasets/cloudwalk-research/psi0-g1-sneaker-205ep-v2-source/raw/main/README.md).

Map named actuators, endpoint polarity and thumb semantics before computing shared statistics. Do not assume a normalized hand scalar, simulated finger angle and physical register are interchangeable. Retain base pose/velocity and task timing where present: projecting everything to arm26 discards locomotion supervision. High-level TWIST2 targets and low-level motor/PD targets must not share a label contract solely because both have 29 body coordinates.

For the precision sets, mask unusable measured hand/neck channels instead of replacing them with commands and calling them feedback. Decode the documented tactile byte order if tactile is used. The newest jren card has a conflict between its frozen-feedback narrative and episode-count table; run per-episode finite-value, variation and timestamp checks. The survey inspected metadata and publisher quality notes, not all parquet values.

## Deduplication and inclusion decisions

CloudWalk sneaker v4 contains its 94/205/325-episode predecessors, including an earlier held-out split. Bottle v11 has 355 edited segments from the two original acquisition sets; it is not another 355 demonstrations. Use one selected representation per source episode. [Sneaker lineage](https://huggingface.co/datasets/cloudwalk-research/psi0-g1-sneaker-385ep-v4/raw/main/README.md), [bottle v11 lineage](https://huggingface.co/datasets/cloudwalk-research/gr00t-g1-grab-bottle-right-hand-v11/raw/main/README.md).

MLeggiero's GR00T pipette68 and TWIST2 pipette68 are the same capture. Page-turning v2.1 retains the same38 episodes; nut-success36 is a filtered child of nut41. HIL194 and HG-DAgger46 are subsets of HIL293; the first46 HIL hand labels were reconstructed. September24/25 five-task clips have explicit source IDs for keeping pick/return siblings together. [Pipette alternate export](https://huggingface.co/datasets/MLeggiero/g1-inspire-pipette-tip), [page conversion](https://huggingface.co/datasets/birbirll/g1-inspire-turn-page-v21), [nut filtering provenance](https://huggingface.co/datasets/birbirll/g1-inspire-nut-in-box-v21-success36/raw/main/meta/source_provenance.json), [HIL293 provenance](https://huggingface.co/datasets/jren313/g1-pipette-3view-hil293/raw/main/README.md).

Do not include these as native RH56DFTP supervision without additional evidence or retargeting:

- Humanoid Everyday's **G1 Dex3** recordings, or its **H1 Inspire** recordings: one matches the body and the other the hand family, but neither establishes G1 + RH56DFTP native commands. See the [earlier hardware survey](2026-10-05-unitree-g1-manipulation-dataset-survey.md).
- Dex1, Dex3, BrainCo and generic five-finger G1 releases. The checked [RoboCOIN five-finger card](https://huggingface.co/datasets/RoboCOIN/Unitree_G1edu_u3_pour_drink) does not name Inspire; five fingers alone are not sufficient evidence.
- The earlier `DaoyuanZhu/wb_add_knee_toy` / `real_inspair` lead: current Hub API and metadata returned 404. Its prior episode count and inferred fingers are not counted as currently verified data.
- Human motion retargeted to G1 without paired manipulation images/language and physical Inspire commands: useful auxiliary motion data, but insufficient by itself for this hardware-specific VLA objective.

A practical first mixture is the six modern Apache-tagged official datasets plus CloudWalk sneaker and the three MLeggiero captures, adding the September five-task set after resolving feedback/timing quality. Then expand into the rest of WBT as hand mappings, legacy schemas and license terms are established. Balance by task/source and retain failures as labeled examples; do not let many nearly identical converted or synthetic copies dominate sampling. Source cards support the inclusion facts above; this mixture is an engineering recommendation, not an experimentally validated recipe.

## Verification scope

Sources were publisher dataset cards, live Hugging Face API inventories, revision-pinned small metadata and official repositories. The official CSV has 29 unique repository IDs; totals use `frames / fps / 3600`. Publicly downloadable, declared open license, exact hardware match and suitable control semantics are separate criteria. No bulk dataset download, sample-by-sample numeric validation, adapter implementation or training was performed. Search coverage is broad but cannot prove that every community upload or private/gated release has been discovered.
