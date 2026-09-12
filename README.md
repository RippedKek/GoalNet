# Part 1 — Data pipeline (Stages 0, 5, 6)

Owner of everything between the raw Metrica files and the tensors the models
train on. Nothing here uses a neural network; it is parsing, measurement and
feature engineering.

## Install

To use this folder, copy its contents into the project root, merging with what
is already there:

```
pip install numpy pandas pyarrow scipy
```

## What is in here

| file | what it does |
|---|---|
| `src/stage0_gsr/load_metrica.py` | reads the raw Metrica files (CSV pairs for games 1-2, FIFA-EPTS text + XML for game 3) and writes one tidy `tracks.parquet` per match |
| `src/stage_style/descriptors.py` | measures the four style knobs (line height, press, width, tempo) from tracking, per 60-second chunk |
| `src/stage5_storage/common.py` | velocity smoothing, speed clamping, long-format rows to dense arrays |
| `src/stage5_storage/run_storage_metrica.py` | cuts the tracking into 6-second training windows, attaches each window's style, does the train/val split |
| `src/stage6_features/relational.py` | the 24-dimension per-agent relational feature vector |
| `src/stage6_features/run_features_gsr.py` | applies that encoder to every window, adds the ball row, writes the feature tensors |

## Run it

Run all three from the `src` directory.

```bash
python -m stage0_gsr.load_metrica --all
```

```bash
python -m stage5_storage.run_storage_metrica
```

```bash
python -m stage6_features.run_features_gsr --store ../output/storage/metrica --out ../output/features/metrica
```

## What it produces

```
output/metrica/game{1,2,3}/tracks.parquet      tidy tracking, one row per agent per frame
output/storage/metrica/windows_{train,val}.npz 13,725 train / 3,402 val windows
output/storage/metrica/style_chunks.json       the measured style knobs per chunk
output/features/metrica/features_{train,val}.npz  (N, 30, 23, 24) relational features
```

Part 2 trains on the last two.

## Things worth knowing (they cost debugging time)

- **Goalkeeper detection** uses the mean of the per-row `|x - 0.5|`, not
  `|mean(x) - 0.5|`. Teams swap ends at half time, so a keeper's mean x over a
  full match lands near the centre of the pitch and the naive test finds nobody.
- **Segments are halves.** No window may cross one, because the ends swap.
- **Target deltas are physics-capped** (`cap_deltas`). The ball is interpolated
  while out of play, which produces jumps of tens of metres in a single step;
  uncapped, those few frames dominate the squared loss.
- **The validation split is the last 20% of each segment**, and windows that
  straddle the boundary are dropped. Windows overlap, so without that drop the
  training frames leak into validation.



# Part 2 — Learned models (Stage 7)

Owner of both neural networks: the architectures, the training loops, the
losses. This is the pattern-recognition core of the project.

## Install

Copy the contents of this folder into the project root, merging with what is
already there.

```
pip install numpy pandas pyarrow torch
```

Needs Part 1's output (`output/storage/metrica`, `output/features/metrica`) and
the raw event CSVs under `data/metrica`.

## What is in here

### StyleNet — how the team moves

| file | what it does |
|---|---|
| `src/stage7_style/model_gsr.py` | the architecture: encoder, FiLM conditioning, three factored spatial/temporal transformer blocks, trajectory + style heads |
| `src/stage7_style/features_torch.py` | the Stage-6 relational encoder rewritten in torch so features can be rebuilt from the model's own predictions inside the training loop |
| `src/stage7_style/train_gsr.py` | training: trajectory MSE, style-consistency loss, scheduled-sampling rollout, physics priors |

FiLM is the part worth explaining: the four style knobs go through a small MLP
that outputs a scale and a shift for every channel of every agent embedding
(`h = h * (1 + gamma) + beta`). One 4-number vector therefore re-weights the
whole network instead of being four extra inputs it can ignore. The
style-consistency head, which has to recover the knobs from the generated
motion, is what forces it to actually listen.

### KickNet — what the player on the ball decides

| file | what it does |
|---|---|
| `src/stage7_style/build_kick_data.py` | turns the event CSVs into training samples: passes with their receiver, shots, and "hold" negatives from the frames just before each kick |
| `src/stage7_style/kick_model.py` | the pointer network: a shared MLP scores every teammate, a pooled head predicts hold / pass / shot |
| `src/stage7_style/train_kick.py` | training with class-weighted cross-entropy (shots are rare) plus cross-entropy over receivers |

The receiver is chosen by scoring the teammates who are actually on the pitch
rather than by picking from fixed classes, so substitutions and formation
changes do not break it. All geometry is expressed in the kicker's attacking
direction, which lets one model serve both teams and both halves.

## Run it

From the `src` directory.

```bash
python -m stage7_style.train_gsr --rollout --feat ../output/features/metrica --store ../output/storage/metrica --out ../output/training/metrica
```

```bash
python -m stage7_style.build_kick_data
```

```bash
python -m stage7_style.train_kick
```

## What it produces

```
output/training/metrica/style_model.pt   StyleNet checkpoint (config travels inside it)
output/training/metrica/train_log.json   per-epoch losses, for the training curve figure
output/kick/kick_data.npz                the KickNet dataset
output/kick/kick_model.pt                KickNet checkpoint
output/kick/kick_meta.json               includes the fitted pass-speed relation
```

Part 3 loads the two checkpoints.

## Numbers to quote

- StyleNet validation trajectory loss 1.91e-05 against 3.26e-05 on training, so
  it generalises rather than memorising.
- Free-running for 60 seconds with no reseeding, the players keep moving
  (median 0.63 m/s), the ball behaves (median 2.6 m/s), and the team spread
  grows from 17 m to 30 m. An earlier version collapsed everyone into a knot;
  that failure is gone.
- KickNet: receiver top-3 accuracy 82% against a 10-way choice, shot recall
  0.89. Hold against pass sits near chance, which is expected, because the two
  states look almost identical one frame apart. The simulator handles the
  timing with a release gate instead of asking the model to decide it.


# Part 3 — Simulator and evidence (Stage 8)

Owner of the thing people actually watch: the match, the ball physics, the live
visualisations, the statistics, and the ablation study that proves the learned
models are the ones playing.

**This part needs no data**, only the two trained checkpoints from Part 2:
`output/training/metrica/style_model.pt` and `output/kick/kick_model.pt` (plus
`kick_meta.json` beside it). Checkpoints live under `output/`, which is not
tracked by git, so copy them in or run Part 2 once. After that a match starts
immediately, with no training and no raw tracking data.

## Install

```
pip install numpy torch pygame matplotlib
```

## Run a match

From the `src` directory.

```bash
python -m stage8_sim.simulate_gsr
```

A menu appears: **L** watches a live match with the tactical sliders, **S**
simulates a full 90 minutes quickly and then replays the goals.

Keys during a live match: `1`-`4` select a knob, `UP`/`DOWN` adjust it (or drag
the sliders), `G` ghosts, `N` names, `C` carrier ring, `SPACE` pause, `ESC`
quit.

## Run the evidence suite

```bash
python -m stage8_sim.ablation
```

Six configurations, ten simulated minutes each, same seed. It writes the
comparison figures and a summary table to `output/telemetry/`.

| config | pass/min | shots/10min | goals | int/10min | player m/s | Argentina possession |
|---|---|---|---|---|---|---|
| full hybrid | 25.6 | 5.0 | 1 | 14.0 | 2.53 | 45% |
| random-kick (KickNet off) | 8.2 | 95.0 | 41 | 36.0 | 3.81 | 53% |
| frozen-players (StyleNet off) | 26.7 | 8.0 | 1 | 8.0 | 1.06 | 59% |
| no-structure | 0.5 | 1.0 | 1 | 0.0 | 0.13 | 2% |
| high-press knob | 26.4 | 3.0 | 2 | 14.0 | 2.66 | 75% |
| high-tempo knob | 25.9 | 1.0 | 1 | 9.0 | 2.30 | 50% |

How to read it. Replacing KickNet with random decisions gives 95 shots and 41
goals in ten minutes while passing collapses, so the sensible football is
coming from that model. Freezing StyleNet drops player speed from 2.53 to 1.06
m/s and shrinks the pitch, so the movement is coming from that one. Pushing the
press knob alone moves possession from 45% to 75%, which is the conditioning
working end to end. And removing the structure layer kills the game entirely,
which is the honest part: that layer is engineered, not learned, and the
figures say so.

## What is in here

| file | what it does |
|---|---|
| `src/stage8_sim/ball_engine.py` | the ball's state machine: carried, flight, loose, restart, goal. Interceptions, tackles, saves, throw-in style restarts |
| `src/stage8_sim/simulate_gsr.py` | the match itself: formations, kick-offs, halves, the structure layer, all the panels, the fast 90-minute mode and the goal replay viewer |
| `src/stage8_sim/telemetry.py` | the live box score and the per-tick record of what each model was doing, plus the figures |
| `src/stage8_sim/ablation.py` | the six-configuration runner above |
| `src/stage_style/knob_sweep.py` | drives one knob at a time and measures the physical response, to check the knobs are causal |
| `src/stage7_style/` | the three model files needed to run inference (architecture only, no training) |

## Live panels, and what each one shows

- **KickNet overlay** — the carrier's receiver probabilities as arrows, plus the
  live hold / pass / shot bars. The model is consulted about five times a
  second.
- **StyleNet ghosts** — the pure model free-rolling eight steps ahead with the
  ball engine and the structure layer switched off. That is the network's own
  motion intent, unaided.
- **FiLM authority meters** — rerun the model with each knob nudged up by 0.25
  and plot how much the motion changes. A live measurement of how much
  authority each dial has.
- **Attention panel** — the last transformer block's spatial attention, shown
  as deviation from uniform, with player names on both axes. Rows look at
  columns. Red means more attention than uniform, blue means less. Raw
  attention is close to uniform, so the deviation is the signal; the true
  magnitude is printed as `max |dev|`. Raise the tempo knob and the own-team
  block turns blue, because the attention mass moves to the opponents, which is
  what direct play looks like.

## Statistics

Possession, passes, completion percentage, shots, goals, interceptions,
tackles and saves for both teams, counted live and printed at full time.
Possession follows the carrier, and falls to the last team that touched the
ball while it is in flight or dead. Pass completion is worked out from what
happens to the flight, not from the event text.
