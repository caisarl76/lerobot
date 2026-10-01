# 06. 공식 레시피로 전 정책 재학습

- 기간: 2026-09-29 ~ 진행 중
- 한 줄 요약: 7개 정책을 HE와 Unitree 두 데이터셋에서 각 정책의 공식 배치 × 업데이트 수로 다시 학습하고 있습니다. H100 GPU 두 장에 큐를 하나씩 두고 돌립니다. 10-01 오전까지 HE GR00T, ACT, Pi0.5가 끝났습니다.

## 무엇을 다시 학습하나

[05](05-groot-jerk-cause.md)에서 모든 파운데이션 모델이 공식 레시피보다 4~190배 적은 샘플로 학습된 걸 확인했습니다. 7개 정책 × 2개 데이터셋, 모두 14개 학습을 다시 합니다. 데이터는 재표기 상태 데이터셋(`sonic78_nolimit_sonicstate`)이고, 미학습 분리와 시드는 이전과 같습니다.

| 데이터셋 | 카메라 | 프레임 |
| --- | --- | --- |
| HE | 1대 | 178만 |
| Unitree | 2대 | 259만 |

잘못된 레시피로 학습한 기존 모델은 09-29에 H100에서 모두 지웠습니다. 이전 결과는 문서에만 남아 있고, 일부 모델만 평가용 워크스테이션에 사본이 있습니다.

## 레시피

| 정책 | 업데이트당 배치 (마이크로 × 누적) | 업데이트 수 | 근거와 주요 설정 |
| --- | --- | --- | --- |
| GR00T | HE 32 × 1, Unitree 16 × 2 | 20K | NVIDIA SONIC 가이드. lr 1e-4 코사인, 워밍업 5%, 고친 전처리 |
| ACT | 8 × 1 | 200K | 원저자 코드의 배치 8. 나머지는 이전과 같음 |
| Diffusion | 64 × 1 | 200K | Chi et al. 이미지 레시피. AdamW, EMA, ResNet18 처음부터 + GroupNorm |
| Pi0.5 | HE 8 × 4, Unitree 4 × 8 | 30K | openpi. fp32 가중치 + bf16 autocast, EMA 0.99, openpi 증강, 코사인 2.5e-5 → 2.5e-6 |
| VLA-JEPA | 8 × 4 | 30K | 공식 모듈별 학습률 (기본 3e-5, Qwen 1e-5, 액션 헤드 1e-4, predictor 5e-4) |
| MolmoAct2 | 8 × 2 | 50K | 논문의 실기 파인튜닝 레시피, 전체 파인튜닝 (사용자 선택) |
| FastWAM | 8 × 2 | 30K | 논문의 실기 레시피, bf16, gradient checkpointing (사용자 선택) |

공식 레시피와 다른 점도 있습니다. 모두 GPU 한 장으로 돌려야 해서 생긴 차이입니다.

- Pi0.5와 VLA-JEPA는 fp32 배치 32가 80 GB에 안 들어가서 그래디언트 누적(8 × 4)을 씁니다.
- VLA-JEPA의 공식 배치는 256(GPU 8장 × 32)인데 우리는 32입니다.
- Diffusion은 스트리머에 맞춰 LeRobot의 horizon 64 / 액션 32를 유지했습니다.

## 코드 변경

| 커밋 | 내용 |
| --- | --- |
| `dfe36b45` | GR00T 프로세서: Hub ID에서도 NVIDIA 전처리 사용 ([05](05-groot-jerk-cause.md)) |
| `b01a01db` | VLA-JEPA 모듈별 학습률 `optimizer_module_lrs` |
| `ceb40b77` | `AdamWConfig.foreach` 옵션. 끄면 multi-tensor 임시 메모리가 없어져 fp32 Pi0.5 + EMA가 80 GB에 들어감 |
| `4b55559b` | GPU별 학습 큐 실행기 `official_queue.sh` |

## 큐 운영

- GPU 0과 GPU 6에 큐 파일이 하나씩 있습니다. 한 줄이 학습 하나입니다.
- 작업 하나가 끝날 때마다 큐 파일을 다시 읽으므로, 줄을 추가하면 이어서 돌아갑니다.
- 짧은 smoke 학습이 성공해야 본 학습이 돌고, 실패하면 본 학습은 `skip`으로 넘어갑니다.
- 학습이 끝나면 `.exit` 파일에 종료 코드를 씁니다. 다음 작업은 이 파일을 보고 시작합니다.
- MolmoAct2와 FastWAM은 활성값 전에도 48 GB 넘게 잡아서 GPU 0에서만 돌 수 있습니다. GPU 6은 다른 사용자가 28~33 GB를 쓰고 있습니다.

## 10-01 오전 상태

| 정책 | HE | Unitree |
| --- | --- | --- |
| GR00T | 완료 (20K). 실기·오픈루프 평가함 ([07](07-retrained-groot-and-rtc.md)) | 고친 데이터로 재학습 중 (10-01 10:30 KST 시작) |
| ACT | 완료 (200K). 평가 전 | GPU 6 대기 |
| Diffusion | GPU 6 대기 (메모리 누수로 처음부터 다시, [08](08-h100-freeze-torchcodec-leak.md)) | GPU 6 대기 |
| Pi0.5 | 완료 (09-30) | GPU 0 대기 |
| VLA-JEPA | GPU 0 학습 중 (10:30 KST에 120K 중 108K, 12시 전후 완료 예상) | GPU 0 대기 |
| MolmoAct2 | GPU 0 대기 | GPU 0 대기 |
| FastWAM | GPU 0 대기 | GPU 0 대기 |

Unitree 작업은 데이터 수정 때문에 한동안 멈췄다가 10-01에 다시 큐에 들어갔습니다([09](09-unitree-dataset-bug.md)). GPU 0 큐는 대략 8~9일이 걸리고, MolmoAct2와 FastWAM 속도는 아직 재지 않았습니다. 80 GB GPU를 한 장 더 쓸 수 있으면 Unitree MolmoAct2와 FastWAM을 옮겨 4일 정도 줄일 수 있습니다.

## 이번에 배운 함정

- **LeRobot 스케줄러는 마이크로 배치마다 한 스텝씩 갑니다.** `AcceleratorConfig`가 `step_scheduler_with_optimizer=False`로 `Accelerator`를 만들기 때문입니다. 누적이 K면 `steps`, 워밍업, 감쇠를 모두 K배로 잡아야 합니다. EMA는 옵티마이저 스텝마다 갱신됩니다. 첫 HE Pi0.5 학습은 이걸 놓쳐 스케줄이 4배 빨랐고, 약 1K 마이크로 스텝에서 멈추고 다시 시작했습니다. 로그에 찍히는 `lr`도 구간 평균입니다.
- **정책 프리셋은 `optimizer`/`scheduler` 설정을 덮어씁니다.** `foreach`나 다른 beta를 쓰려면 프리셋을 끄고 둘 다 직접 줘야 합니다. 그런데 프리셋을 끄면 모듈별 학습률 그룹이 사라집니다. 그래서 VLA-JEPA와 MolmoAct2는 프리셋을 켠 채로 둡니다.
- **GR00T 워밍업은 `--steps`가 아니라 `policy.max_steps`를 봅니다.** 둘을 같게 맞춰야 합니다. 이전 학습의 워밍업이 1.25%였던 이유가 이것입니다.
- **학습 컨테이너에는 `ps`, `pkill`, `pgrep`이 없습니다.** 작업을 멈추려면 `/proc/*/cmdline`으로 PID를 찾고, 큐 루프를 먼저 멈춘 다음 학습 프로세스와 그 자식 프로세스를 멈춥니다.
- **`.exit` 파일을 쓰는 감시 프로세스가 죽으면 큐가 멈춥니다.** 09-30에 HE Pi0.5는 16:22 KST에 끝났는데 `.exit`이 안 써져서 GPU 0이 18:45까지 놀았습니다. 손으로 써서 풀었습니다.

## 다음 확인할 것

1. 끝난 모델마다 오픈루프 부드러움(`openloop_smooth.py`)부터 봅니다. 그다음 시뮬레이션(플래너 시작, 테이블 25 cm), 마지막으로 실기입니다.
2. MolmoAct2 smoke 학습에서 전체 파인튜닝 8 × 2가 메모리에 안 들어가면 4 × 4로 바꾸고, 스텝·워밍업·감쇠를 새 K에 맞춰 늘립니다.
3. `skip`이 찍힌 작업은 smoke 로그를 읽고 설정을 고친 뒤 `.exit` 두 개를 지워 다시 돌립니다.

## 관련 코드

- `examples/g1_dex3_training/official_queue.sh`, `write_foundation_configs.py`
- `src/lerobot/policies/groot/processor_groot.py`, `src/lerobot/optim/optimizers.py`
- 상세 기록: `docs/research/2026-09-30-official-retraining-handover.md`, `2026-09-29-issue-groot-jerky-predictions.md`
