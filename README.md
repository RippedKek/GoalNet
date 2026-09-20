# GoalNet

A football match simulated by two neural networks trained on real player
tracking data, with four dials that change *how* a team plays.

Most football analytics predicts outcomes: who wins, whether a shot scores.
GoalNet models behaviour instead. Given the recent state of all 23 agents
(22 players and the ball), it generates how play continues, conditioned on a
playing-style vector. That makes counterfactuals possible: run the same team
with a higher pressing line and watch what changes.

Trained on the [Metrica Sports](https://github.com/metrica-sports/sample-data)
open tracking data — three professional matches at 25 fps, plus event files
listing every pass and shot.

---

## How it works

The problem splits into two halves with different mathematical character, so it
is solved by two models with losses matched to their output spaces, combined by
an explicit ball engine.

### StyleNet — where everyone moves

A factored spatio-temporal transformer. It consumes a 6-second window of all 23
agents and predicts each one's next displacement.

**Why a transformer and not a CNN.** A frame is a *set* of interacting agents,
not a grid. Self-attention is permutation-equivariant, which matches the fact
that slot indices are arbitrary labels, and its key-padding mask handles a
varying number of players directly — substituted or off-pitch players simply
drop out. A convolutional encoder would have to rasterise the pitch and learn
geometry we already hold exactly as coordinates.

**Why attention is factored across space and time.** Joint attention over all
`23 × 30 = 690` tokens costs `O(690²)`. Attending over agents and then over
time costs `O(23²·30 + 30²·23)`, roughly an order of magnitude less, while
stacked blocks still propagate information along both axes.

**Why FiLM conditioning.** Four style numbers concatenated to a 24-dimensional
feature vector are trivially ignorable by the optimiser. Feature-wise linear
modulation instead generates a per-channel scale and shift from the style
vector and applies `h ← h(1 + γ) + β` to *every* agent embedding, so the
conditioning cannot be dropped. A style-consistency head that must recover the
knobs from the generated motion enforces this further.

**The style dials are measured, not labelled.** Line height, press, width and
tempo are computed directly from the tracking in 60-second chunks, so the
conditioning signal is self-supervised and costs nothing to obtain.

### KickNet — what the player on the ball does

A pointer-style network that outputs an action (hold / pass / shoot) and a
distribution over receivers.

**Why a pointer network.** The receiver is one of a *varying set of teammates*,
not a fixed class index. A shared MLP scores each teammate, so the model is
permutation-equivariant and stays valid under substitutions and formation
changes. A fixed 11-way classifier would have to relearn "slot 4" in every
context.

**Why the geometry is canonicalised.** All features are expressed in the
kicker's attacking direction, so one model serves both teams and both halves.
This is invariance by construction rather than by augmentation.

**Why shots are weighted 25×.** Shots are 1.3 % of the training data. Under
plain cross-entropy the optimiser scores well by never predicting "shot", and
the resulting simulator would never score a goal. Inverse-frequency class
weights (`w = N / 3n_c`) make one missed shot cost as much as ~25 missed
passes.

### The ball engine — why the ball is scripted

The first version made the ball a 23rd learned agent trained on next-frame
regression. It failed structurally: a pass is a **multi-modal discrete
choice**, and squared error is minimised by predicting the conditional mean of
the available options, which is none of them. The ball drifted smoothly between
players instead of being kicked.

So the ball is a physical object with five modes — carried, flight, loose,
restart, goal — driven by KickNet's decisions. Passes have kickers and
receivers, flights are substepped so defenders on the path can intercept,
tackles and goalkeeper saves are contests, and possession changes hands
physically.

A structure layer supplies formation anchors and pressing, which the trajectory
model does not maintain over match-length horizons. **This layer is engineered,
not learned**, and the ablation below quantifies exactly how much it
contributes.

### How the pieces meet

Each 0.2 s tick: StyleNet moves the players, the structure layer corrects the
shape, KickNet decides for whoever has the ball, and the engine resolves the
consequences. The scripted ball position is then written back into the history
buffer, so the relational features the players observe next tick contain the
*physical* ball. The two models never communicate directly — only through the
shared world state.

---

## Does it actually work?

Removing one component at a time, with an identical seed, the same engine and
ten simulated minutes each:

| configuration | passes/min | shots/10min | goals | player m/s | possession T0 |
|---|---|---|---|---|---|
| full | 25.6 | 5.0 | 1 | 2.53 | 45 % |
| KickNet randomised | 8.2 | **95.0** | **41** | 3.81 | 53 % |
| StyleNet frozen | 26.7 | 8.0 | 1 | **1.06** | 59 % |
| structure layer off | **0.5** | 1.0 | 1 | 0.13 | 2 % |
| press dial at 0.95 | 26.4 | 3.0 | 2 | 2.66 | **75 %** |

Randomising the decision policy produces 95 shots and 41 goals per ten minutes
while passing collapses, so KickNet is what makes play sensible. Freezing the
trajectory model halves player speed, so StyleNet is what makes players move.
The press dial alone moves possession from 45 % to 75 %, so the conditioning
survives to the level of match outcome. Removing the structure layer kills the
game, which is the honest bound on the claim.

Model-level: StyleNet reaches a validation trajectory loss of `1.91e-05`
against `3.26e-05` on training. KickNet reaches 82 % receiver top-3 accuracy
against a 30 % chance baseline and a shot recall of 0.89.

### Known gaps

Measured against the source matches, simulated play is about 2.5× too fast and
too safe: 25.6 passes/min against 9.3, 93 % pass completion against 78 %, and
5.0 shots/10min against 2.6. Two causes: hold-versus-pass was not learnable
(recall 0.51, near chance), so the engine forces a release after ~2.5 s; and
the interception model ignores a defender's ability to move toward the ball
during flight.

---

## Running it

```bash
pip install -r requirements.txt
```

### Watch a match

Model checkpoints live under `output/`, which is not tracked by git. Either
train them (below) or copy `output/training/metrica/style_model.pt`,
`output/kick/kick_model.pt` and `output/kick/kick_meta.json` into place.

```bash
cd src
python -m stage8_sim.simulate_gsr
```

A menu offers **L** to watch live with interactive dials, or **S** to simulate
a full 90 minutes quickly and then replay the goals.

During a live match: `1`–`4` select a dial, `UP`/`DOWN` adjust it (or drag the
sliders), `G` toggles StyleNet ghosts, `N` names, `C` the carrier ring,
`SPACE` pauses, `ESC` quits.

### Reproduce the evidence

```bash
cd src
python -m stage8_sim.ablation
```

Runs the six configurations above and writes comparison figures, per-config
telemetry and a summary to `output/telemetry/`.

### Train from scratch

Needs the Metrica sample data under `data/metrica/`. Run from `src/`.

```bash
python -m stage0_gsr.load_metrica --all
```

```bash
python -m stage5_storage.run_storage_metrica
```

```bash
python -m stage6_features.run_features_gsr --store ../output/storage/metrica --out ../output/features/metrica
```

```bash
python -m stage7_style.train_gsr --rollout --feat ../output/features/metrica --store ../output/storage/metrica --out ../output/training/metrica
```

```bash
python -m stage7_style.build_kick_data
python -m stage7_style.train_kick
```

**Training note.** StyleNet is teacher-forced for the first ten epochs, then
switches to scheduled sampling: its own predictions are fed back as input with
a probability ramped from 0 to 0.5. Without this the model has an excellent
one-step error but drifts into a corner when run freely for thousands of steps,
which is the regime the simulator actually uses.

---

## Layout

```
src/stage0_gsr/        raw Metrica files -> tidy tracking parquet
src/stage_style/       style descriptors; knob-response validation
src/stage5_storage/    windowing, train/val split
src/stage6_features/   24-dim relational features per agent
src/stage7_style/      StyleNet and KickNet: architectures and training
src/stage8_sim/        ball engine, simulator, telemetry, ablation
config/                storage and training hyperparameters
```

The train/validation split is chronological — the last 20 % of each half — and
windows that straddle the boundary are dropped, since overlapping windows would
otherwise leak training frames into validation.
