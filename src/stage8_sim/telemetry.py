"""Match telemetry: evidence that the learned models drive the game.

MatchStats  -- live football box score (possession, passes, completion,
               shots, goals, interceptions, tackles, saves) counted from
               BallEngine events and mode transitions.
Telemetry   -- per-tick time series of model signals: KickNet action
               probabilities and receiver entropy, StyleNet motion vs the
               structure-layer correction, FiLM knob authority, attention
               deviation, possession, ball position, score.
make_match_figures -- render one match's time series into report figures.

Used by simulate_gsr (live HUD + post-match dump) and stage8_sim.ablation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MODE_CODE = {"carried": 0, "flight": 1, "loose": 2, "restart": 3, "goal": 4}
N_SLOTS = 11

COLS = ["tick", "mode", "poss", "ball_x", "ball_y",
        "p_hold", "p_pass", "p_shot", "recv_entropy",
        "stylenet_mps", "structure_mps", "player_mps", "x_spread",
        "sens_line", "sens_press", "sens_width", "sens_tempo",
        "attn_dev", "score0", "score1"]

# report palette (matches the docx figure scripts)
BLUE, AQUA, YELLOW, RED, GRAY = "#2a78d6", "#1baf7a", "#eda100", "#d3382c", "#8a8a8a"


class MatchStats:
    """Box score accumulated from engine events + mode transitions.

    Pass completion is inferred structurally: a PASS opens a pending flight;
    if the flight resolves into a carry by the same team it completed,
    anything else (interception, out, loose) is incomplete.
    """

    KEYS = ("poss", "pass_att", "pass_cmp", "shots", "goals",
            "intercepts", "tackles", "saves", "recoveries", "throw_ins")

    def __init__(self, names=("team0", "team1")):
        self.names = list(names)
        self.t = [dict.fromkeys(self.KEYS, 0) for _ in range(2)]
        self._pending = None          # team with a pass in the air
        self._prev_mode = None

    def _team_in(self, ev: str):
        for t, n in enumerate(self.names):
            if n in ev:
                return t
        return None

    def update(self, engine, events):
        team = engine.team_of(engine.carrier) if engine.carrier is not None \
            else engine.last_touch_team
        self.t[team]["poss"] += 1
        if self._prev_mode == "flight" and engine.mode != "flight" \
                and self._pending is not None:
            if engine.mode == "carried" \
                    and engine.team_of(engine.carrier) == self._pending:
                self.t[self._pending]["pass_cmp"] += 1
            self._pending = None
        for ev in events:
            t = self._team_in(ev)
            if t is None:
                continue
            if ev.startswith("PASS"):
                self.t[t]["pass_att"] += 1
                self._pending = t
            elif ev.startswith("SHOT"):
                self.t[t]["shots"] += 1
            elif ev.startswith("GOAL"):
                self.t[t]["goals"] += 1
            elif ev.startswith("INTERCEPTED"):
                self.t[t]["intercepts"] += 1
            elif ev.startswith("TACKLE"):
                self.t[t]["tackles"] += 1
            elif ev.startswith("SAVE"):
                self.t[t]["saves"] += 1
            elif ev.startswith("RECOVERY"):
                self.t[t]["recoveries"] += 1
            elif ev.startswith("OUT"):
                self.t[t]["throw_ins"] += 1
        self._prev_mode = engine.mode

    def possession(self):
        tot = max(self.t[0]["poss"] + self.t[1]["poss"], 1)
        return [self.t[0]["poss"] / tot, self.t[1]["poss"] / tot]

    def as_dict(self):
        return {"names": self.names, "possession": self.possession(),
                "teams": [dict(d) for d in self.t]}

    def table_str(self):
        ps = self.possession()
        hdr = (f"{'':14}{'poss':>6}{'pass':>7}{'cmp%':>7}{'shots':>7}"
               f"{'goals':>7}{'int':>6}{'tackle':>8}{'save':>6}{'recov':>7}")
        lines = [hdr]
        for t in (0, 1):
            d = self.t[t]
            cmp_pct = 100.0 * d["pass_cmp"] / max(d["pass_att"], 1)
            lines.append(
                f"{self.names[t][:13]:14}{ps[t] * 100:5.0f}%{d['pass_att']:7d}"
                f"{cmp_pct:6.0f}%{d['shots']:7d}{d['goals']:7d}{d['intercepts']:6d}"
                f"{d['tackles']:8d}{d['saves']:6d}{d['recoveries']:7d}")
        return "\n".join(lines)


class Telemetry:
    """Per-tick recorder. Cheap (one tuple append per step)."""

    def __init__(self):
        self.rows = []

    def record(self, sim):
        e = sim.engine
        if e is None:
            return
        q = e.last_query
        fresh = e.mode == "carried" and q is not None and q["carrier"] == e.carrier
        if fresh:
            pa = q["action"]
            pr = q["recv"][q["mask"] > 0]
            pr = pr[pr > 1e-9]
            ent = float(-(pr * np.log(pr)).sum() / max(np.log(len(pr)), 1e-9)) \
                if len(pr) > 1 else 0.0
        else:
            pa, ent = (np.nan, np.nan, np.nan), np.nan
        poss = e.team_of(e.carrier) if e.carrier is not None else -1
        attn_dev = float(np.abs(sim.attn[:N_SLOTS, :N_SLOTS] - 1.0 / 23.0).max())
        self.rows.append((
            sim._tick, MODE_CODE.get(e.mode, -1), poss,
            float(e.ball[0]), float(e.ball[1]),
            float(pa[0]), float(pa[1]), float(pa[2]), ent,
            float(sim.model_step_m.mean() * sim.fps),
            float(sim.struct_corr_m.mean() * sim.fps),
            float(sim.player_speed_m), float(np.std(sim.pos[:22, 0])),
            *[float(v) for v in sim.knob_sens],
            attn_dev, float(e.score[0]), float(e.score[1])))

    def array(self):
        return np.array(self.rows, np.float32) if self.rows \
            else np.zeros((0, len(COLS)), np.float32)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, data=self.array(), cols=np.array(COLS))
        return path


# --------------------------------------------------------------------- #
def _roll(v, w):
    if len(v) < max(w, 2):
        return v
    return np.convolve(v, np.ones(w) / w, mode="same")


def _load(npz_path):
    d = np.load(npz_path)
    cols = [str(c) for c in d["cols"]]
    return d["data"], {n: i for i, n in enumerate(cols)}


def make_match_figures(npz_path, outdir, fps=5, names=("team0", "team1"),
                       prefix="match"):
    """Render one recorded match into four evidence figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A, c = _load(npz_path)
    if not len(A):
        return []
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t = A[:, c["tick"]] / fps / 60.0          # match minutes
    w = int(30 * fps)                         # 30 s rolling window
    plt.rcParams.update({"figure.facecolor": "white", "font.size": 9})
    paths = []

    # 1 -- KickNet activity: action probs + receiver certainty per carried step
    m = ~np.isnan(A[:, c["p_pass"]])
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), dpi=140, sharex=True)
    for k, col, lab in (("p_hold", GRAY, "hold"), ("p_pass", BLUE, "pass"),
                        ("p_shot", RED, "shot")):
        axes[0].scatter(t[m], A[m, c[k]], s=3, color=col, label=lab, alpha=0.55)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("action probability")
    axes[0].legend(markerscale=4, loc="upper right")
    axes[0].set_title("KickNet output on every carried step (the model is asked "
                      "~5x/s whenever someone has the ball)")
    axes[1].scatter(t[m], A[m, c["recv_entropy"]], s=3, color=AQUA, alpha=0.55)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("receiver entropy\n(0 = locked-on, 1 = uniform)")
    axes[1].set_xlabel("match minutes")
    p = outdir / f"{prefix}_kicknet_activity.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # 2 -- who moves the players: StyleNet intent vs structure correction
    sty = _roll(A[:, c["stylenet_mps"]], w)
    stc = _roll(A[:, c["structure_mps"]], w)
    act = _roll(A[:, c["player_mps"]], w)
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=140)
    ax.stackplot(t, sty, stc, labels=["StyleNet motion (learned)",
                                      "structure correction (anchors/press)"],
                 colors=[BLUE, YELLOW], alpha=0.85)
    ax.plot(t, act, color="#222", lw=1.2, label="actual mean player speed")
    ax.set_xlabel("match minutes"); ax.set_ylabel("mean per-player m/s")
    ax.legend(loc="upper right")
    ax.set_title("Motion attribution (30 s rolling): learned model vs engineered layer")
    p = outdir / f"{prefix}_motion_share.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # 3 -- FiLM knob authority + attention deviation over the match
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=140)
    sens_cols = ["sens_line", "sens_press", "sens_width", "sens_tempo"]
    labs = ["line height", "press", "width", "tempo"]
    cols = [BLUE, RED, AQUA, YELLOW]
    S = A[:, [c[k] for k in sens_cols]]
    if np.any(S > 0):
        for k in range(4):
            ax.plot(t, _roll(S[:, k], w), color=cols[k], lw=1.4, label=labs[k])
        ax.set_ylabel("motion response (m/s per +0.25 knob)")
        ax.legend(loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(t, _roll(A[:, c["attn_dev"]], w), color=GRAY, lw=1.0, ls="--",
             label="max attention deviation")
    ax2.set_ylabel("attention |dev from uniform|", color=GRAY)
    ax.set_xlabel("match minutes")
    ax.set_title("FiLM style authority (live counterfactual probes) + spatial attention")
    p = outdir / f"{prefix}_film_attention.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # 4 -- possession flow + field tilt + goals
    poss = A[:, c["poss"]]
    p0 = (poss == 0).astype(np.float32)
    p1 = (poss == 1).astype(np.float32)
    share0 = _roll(p0, w) / np.maximum(_roll(p0 + p1, w), 1e-6)
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=140)
    ax.plot(t, share0, color=BLUE, lw=1.5, label=f"{names[0]} possession share")
    ax.axhline(0.5, color=GRAY, lw=0.8, ls=":")
    ax.fill_between(t, 0.5, share0, where=share0 >= 0.5, color=BLUE, alpha=0.18)
    ax.fill_between(t, share0, 0.5, where=share0 < 0.5, color=YELLOW, alpha=0.25)
    ax.set_ylim(0, 1); ax.set_ylabel("possession share (30 s rolling)")
    sc = A[:, c["score0"]] + A[:, c["score1"]]
    for i in np.where(np.diff(sc) > 0)[0]:
        team = 0 if A[i + 1, c["score0"]] > A[i, c["score0"]] else 1
        ax.axvline(t[i], color=BLUE if team == 0 else YELLOW, lw=1.4, alpha=0.9)
        ax.text(t[i], 1.01, "goal", fontsize=7, ha="center", color=GRAY)
    ax2 = ax.twinx()
    ax2.plot(t, _roll(A[:, c["ball_x"]], w), color=AQUA, lw=1.0, ls="--",
             label="ball x (field tilt)")
    ax2.set_ylabel("ball x (m)", color=AQUA); ax2.set_ylim(0, 105)
    ax.set_xlabel("match minutes")
    ax.set_title(f"Possession contest: {names[0]} vs {names[1]} (vertical lines = goals)")
    ax.legend(loc="upper left")
    p = outdir / f"{prefix}_possession.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); paths.append(p)
    return paths
