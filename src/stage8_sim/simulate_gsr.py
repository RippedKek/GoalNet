"""Stage 8/9 (GSR) -- style-controllable simulator + visualisation.

Drives 23 agents (22 players + a LEARNED ball, agent 22) with StyleNetGSR. Four
on-screen sliders set team0's style knobs live (line height, press, width, tempo);
the model restyles motion in real time. Features are built with the same
differentiable builder used in training (features_torch.build_frame), so sim and
training see identical inputs.

  python -m stage8_sim.simulate_gsr
  python -m stage8_sim.simulate_gsr --headless --steps 60   # screenshot self-test

Keys: 1-4 select knob, UP/DOWN adjust, SPACE pause, ESC quit. Mouse: drag sliders.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from stage7_style.model_gsr import StyleNetGSR
from stage7_style.features_torch import build_frame
from stage8_sim.telemetry import MatchStats, Telemetry

ROOT = Path(__file__).resolve().parents[2]
N_SLOTS, BALL = 11, 22
PITCH_L, PITCH_W = 105.0, 68.0
MARGIN = 40
PX_W, PX_H = 900, 600
PANEL_X = 980
WIN_W, WIN_H = 1460, 810

GRASS_A, GRASS_B = (34, 139, 34), (40, 150, 40)
LINE = (235, 235, 235)
T0, T1 = (108, 172, 228), (250, 220, 40)      # Argentina sky blue vs Brazil yellow
GK_C, GK_BORDER = (60, 200, 90), (255, 255, 255)
BALL_C = (250, 250, 250)
PANEL_BG = (18, 18, 24)
TEXT = (235, 235, 235)
KNOBS = ["line height", "press", "width", "tempo"]
# user-knob -> model descriptor-input polarity, calibrated by knob-sweep against
# the CURRENT model (stage_style.knob_sweep). Metrica-trained model: width came
# out perfectly monotonic but inverted (rank-corr -1.0), rest correct. The old
# GSR-broadcast model needed [-1,-1,-1,1].
KNOB_POLARITY = np.array([1.0, 1.0, -1.0, 1.0], np.float32)
ROLE_LABELS = ["GK", "RB", "CB", "CB", "LB", "RM", "CM", "CM", "LM", "ST", "ST"]

# slot 0 = GK; team-local frame: own goal at x=0, attacking +x, all inside own
# half (kick-off legal). y symmetric around 34.
FORMATIONS = {
    "4-4-2":   [(6, 34), (20, 10), (20, 25), (20, 43), (20, 58),
                (40, 10), (40, 26), (40, 42), (40, 58), (50, 26), (50, 42)],
    "4-3-3":   [(6, 34), (20, 10), (20, 25), (20, 43), (20, 58),
                (37, 20), (35, 34), (37, 48), (49, 12), (51, 34), (49, 56)],
    "4-2-3-1": [(6, 34), (20, 10), (20, 25), (20, 43), (20, 58),
                (33, 26), (33, 42), (45, 14), (46, 34), (45, 54), (51, 34)],
}
FORMATION_LABELS = {
    "4-4-2":   ["GK", "RB", "CB", "CB", "LB", "RM", "CM", "CM", "LM", "ST", "ST"],
    "4-3-3":   ["GK", "RB", "CB", "CB", "LB", "CM", "CM", "CM", "RW", "ST", "LW"],
    "4-2-3-1": ["GK", "RB", "CB", "CB", "LB", "DM", "DM", "RM", "AM", "LM", "ST"],
}

TEAM_NAMES = ("Argentina", "Brazil")
# squad names per formation, aligned slot-for-slot with FORMATION_LABELS
SQUADS = {
    "Argentina": {
        "4-4-2":   ["Dibu Martinez", "Molina", "Romero", "Otamendi", "Tagliafico",
                    "Messi", "De Paul", "Enzo", "Mac Allister", "J. Alvarez", "Lautaro"],
        "4-3-3":   ["Dibu Martinez", "Molina", "Romero", "Otamendi", "Tagliafico",
                    "De Paul", "Enzo", "Mac Allister", "Messi", "J. Alvarez", "Nico Gonzalez"],
        "4-2-3-1": ["Dibu Martinez", "Molina", "Romero", "Otamendi", "Tagliafico",
                    "Enzo", "De Paul", "Messi", "Mac Allister", "Nico Gonzalez", "J. Alvarez"],
    },
    "Brazil": {
        "4-4-2":   ["Alisson", "Danilo", "Marquinhos", "Militao", "Arana",
                    "Raphinha", "Bruno G.", "Paqueta", "Vinicius Jr", "Endrick", "Rodrygo"],
        "4-3-3":   ["Alisson", "Danilo", "Marquinhos", "Militao", "Arana",
                    "Bruno G.", "Paqueta", "Gerson", "Raphinha", "Endrick", "Vinicius Jr"],
        "4-2-3-1": ["Alisson", "Danilo", "Marquinhos", "Militao", "Arana",
                    "Bruno G.", "Andre", "Raphinha", "Paqueta", "Vinicius Jr", "Endrick"],
    },
}


def initial_positions():
    t0 = [(5, 34), (20, 14), (20, 27), (20, 41), (20, 54),
          (42, 14), (42, 27), (42, 41), (42, 54), (58, 27), (58, 41)]      # 4-4-2 defends x=0
    t1 = [(100, 34), (85, 14), (85, 27), (85, 41), (85, 54),
          (68, 27), (68, 41), (50, 14), (50, 34), (50, 54), (47, 34)]      # 4-2-3-1 defends x=105
    return np.array(t0 + t1 + [(52.5, 34.0)], np.float32)                  # (23,2)


class Sim:
    def __init__(self, model_path, seed=None, kick_path=None, pure=False):
        """kick_path + pure=False -> hybrid mode: scripted ball (BallEngine)
        with KickNet pass/shot decisions + formation anchors. pure=True -> the
        old fully-learned ball."""
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        ck = torch.load(model_path, map_location=self.dev, weights_only=False)
        self.cfg = ck["cfg"]
        self.norm = self.cfg["norm"]
        self.W = self.cfg["model"]["window"]
        self.model = StyleNetGSR(self.cfg).to(self.dev).eval()
        self.model.load_state_dict(ck["model"])
        self.fps = 5
        L, Wd = self.norm["pos_x_m"], self.norm["pos_y_m"]
        pm, bm = self.norm["player_max_speed_mps"], self.norm["ball_max_speed_mps"]
        fac = torch.zeros(23, 2, device=self.dev)
        fac[:BALL, 0] = L * self.fps / pm; fac[:BALL, 1] = Wd * self.fps / pm
        fac[BALL, 0] = L * self.fps / bm; fac[BALL, 1] = Wd * self.fps / bm
        self.fac = fac
        self.L, self.Wd = L, Wd
        self.attn = np.zeros((23, 23), np.float32)
        self.ref = torch.tensor([[0.32, 0.55, 0.34, 0.35]], device=self.dev)
        self.knobs = np.array([0.32, 0.58, 0.34, 0.35], np.float32)
        self.no_style = False
        self.engine = None
        self.anchors = None            # (22,2) metres; formation home positions
        self.anchor_k = 0.045          # per-step pull toward (dynamic) anchor when far from ball
        self.anchor_far = 12.0         # m from ball beyond which the pull applies
        self.follow_atk = 0.75         # attacking block follows the ball by this fraction (x)
        self.follow_def = 0.35         # defending block ditto (compresses deep / steps up)
        self.follow_y = 0.30           # lateral follow, both teams
        self.follow_gk = 0.08          # GK anchor barely moves
        # asymmetric caps on forward-axis block travel (m). In possession the
        # block advances freely but barely retreats (strikers stay high as
        # outlets); defending it drops freely but steps up conservatively.
        self.adv_cap_atk, self.ret_cap_atk = 35.0, 6.0
        self.adv_cap_def, self.ret_cap_def = 12.0, 35.0
        self.y_cap = 12.0
        self.n_pressers = 2            # defenders sent to press the carrier
        self.press_step = 0.35         # m/step base press speed bonus (scaled by press knob)
        self.press_standoff = 1.0      # m: inside TACKLE_R so pressers can actually tackle
        self.goal_pending = False
        self.ball_snap = False         # ball restarted: renderer must snap, not lerp
        self.events = []               # HUD tail
        self.event_log = []            # full log (headless stats)
        self.base_form = None          # (form0, form1) team-local formation coords
        self.labels = (ROLE_LABELS, ROLE_LABELS)
        self.hud_extra = ""            # match clock / half indicator
        self._tick = 0
        self.fast = False              # True = skip viz extras (fast simulation mode)
        self.ghost = None              # (n,23,2) StyleNet free-rollout trail (viz)
        self.knob_sens = np.zeros(4, np.float32)  # live knob authority (m/s per +0.25)
        self.stats = MatchStats()      # live box score (names set by set_formations)
        self.telemetry = Telemetry()   # per-tick model-signal recorder
        self.disable_stylenet = False  # ablation: ignore StyleNet player deltas
        self.disable_structure = False # ablation: skip anchors/press layer
        self.model_step_m = np.zeros(22, np.float32)   # |StyleNet delta| per player (m)
        self.struct_corr_m = np.zeros(22, np.float32)  # |structure correction| (m)
        self.player_speed_m = 0.0                      # realized mean speed (m/s)
        if not pure and kick_path is not None:
            from stage8_sim.ball_engine import BallEngine
            self.engine = BallEngine(kick_path, dt=1.0 / self.fps)
        if seed is not None:
            self.apply_seed(seed)                         # in-distribution real clip window
        else:
            self.apply_cold()                             # synthetic formation (degenerates)

    def apply_seed(self, seed, keep_knobs=False):
        agents = seed["agents"].astype(np.float32)        # (W,23,4) normalised
        self.hist = torch.from_numpy(agents[None]).to(self.dev)
        self.pres = torch.from_numpy(seed["presence"].astype(np.float32)[None]).to(self.dev)
        if not self.no_style:
            self.ref = torch.from_numpy(seed["style1"].astype(np.float32)[None]).to(self.dev)
            if not keep_knobs:
                self.knobs = seed["style0"].astype(np.float32).copy()
        self.pos = (agents[-1, :, :2] * [self.L, self.Wd]).astype(np.float32)
        self.cur_pres = self.pres[0, -1].cpu().numpy()
        # formation anchors = per-player mean position over the seed window
        pm = agents[:, :22, :2] * [self.L, self.Wd]           # (W,22,2)
        w = (seed["presence"][:, :22] > 0.2).astype(np.float32)[..., None]
        self.anchors = (pm * w).sum(0) / np.maximum(w.sum(0), 1e-6)
        if self.engine is not None:
            self.engine.reset(self.pos[:22], self.cur_pres[:22], self.pos[BALL])

    def set_formations(self, f0: str, f1: str):
        self.base_form = (np.array(FORMATIONS[f0], np.float32),
                          np.array(FORMATIONS[f1], np.float32))
        self.labels = (SQUADS[TEAM_NAMES[0]][f0], SQUADS[TEAM_NAMES[1]][f1])
        self.stats.names = list(TEAM_NAMES)
        if self.engine is not None:
            self.engine.names = TEAM_NAMES
            self.engine.slot_names = self.labels

    def apply_kickoff(self, kicking_team: int, second_half: bool = False):
        """Both teams to formation in their own half (ends swap in the second
        half), ball at centre carried by the kicking team's most central
        attacker. Used at match start, after every goal, and at half-time."""
        L, Wd = self.L, self.Wd
        pos = np.zeros((23, 2), np.float32)
        for team in (0, 1):
            world = self.base_form[team].copy()
            attacks_pos_x = (team == 0) != second_half     # team0 attacks +x in 1st half
            if not attacks_pos_x:
                world[:, 0] = L - world[:, 0]
                world[:, 1] = Wd - world[:, 1]
            pos[team * N_SLOTS:(team + 1) * N_SLOTS] = world
        centre = np.array([L / 2, Wd / 2], np.float32)
        sl = np.arange(kicking_team * N_SLOTS + 1, (kicking_team + 1) * N_SLOTS)
        kicker = int(sl[np.argmin(np.linalg.norm(pos[sl] - centre, axis=1))])
        pos[kicker] = centre
        pos[BALL] = centre
        self.anchors = pos[:22].copy()
        pos_n = pos / [L, Wd]
        row = np.concatenate([pos_n, np.zeros_like(pos_n)], -1).astype(np.float32)
        self.hist = torch.from_numpy(np.repeat(row[None], self.W, 0)[None]).to(self.dev)
        self.pres = torch.ones(1, self.W, 23, device=self.dev)
        self.pos = pos.copy()
        self.cur_pres = self.pres[0, -1].cpu().numpy()
        if self.engine is not None:
            e = self.engine
            e.ball = centre.copy()
            e.bvel[:] = 0
            e._capture(kicker)
            e.cooldown = 6                                 # ~1.2 s to organise
        self.ghost = None                                  # stale after reposition
        self.ball_snap = True
        ev = f"KICK-OFF {TEAM_NAMES[kicking_team]}"
        self.events = (self.events + [ev])[-6:]
        self.event_log.append(ev)

    def apply_cold(self):
        pos_m = initial_positions()
        pos_n = pos_m / [self.L, self.Wd]
        row = np.concatenate([pos_n, np.zeros_like(pos_n)], -1).astype(np.float32)
        self.hist = torch.from_numpy(np.repeat(row[None], self.W, 0)[None]).to(self.dev)
        self.pres = torch.ones(1, self.W, 23, device=self.dev)
        self.pos = pos_m.copy()
        self.cur_pres = self.pres[0, -1].cpu().numpy()
        self.anchors = pos_m[:22].copy()
        if self.engine is not None:
            self.engine.reset(self.pos[:22], self.cur_pres[:22], self.pos[BALL])

    def _adj_style(self):
        adj = np.where(KNOB_POLARITY < 0, 1.0 - self.knobs, self.knobs).astype(np.float32)
        return torch.tensor(adj[None], device=self.dev)

    def ghost_rollout(self, n: int = 8):
        """Free-roll the PURE StyleNet n steps ahead (no ball engine, no
        anchors) from the current state -- the network's own motion intent."""
        with torch.no_grad():
            hist = self.hist.clone()
            s0 = self._adj_style()
            out = []
            for _ in range(n):
                feat = build_frame(hist[0], self.norm).reshape(1, self.W, 23, 24)
                delta, _, _ = self.model(feat, self.pres, s0, self.ref)
                newpos = (hist[0, -1, :, :2] + delta[0, -1]).clamp(0, 1)
                newvel = (newpos - hist[0, -1, :, :2]) * self.fac
                hist = torch.cat([hist[:, 1:],
                                  torch.cat([newpos, newvel], -1)[None, None]], 1)
                out.append((newpos.cpu().numpy() * [self.L, self.Wd]).astype(np.float32))
        self.ghost = np.stack(out)

    def knob_sensitivity(self):
        """Mean instantaneous motion response of team0 (m/s) to +0.25 on each
        knob -- a live FiLM-authority meter."""
        with torch.no_grad():
            feat = build_frame(self.hist[0], self.norm).reshape(1, self.W, 23, 24)
            adj = np.where(KNOB_POLARITY < 0, 1.0 - self.knobs, self.knobs).astype(np.float32)
            base = self.model(feat, self.pres,
                              torch.tensor(adj[None], device=self.dev), self.ref)[0]
            sens = []
            for k in range(4):
                a2 = adj.copy()
                a2[k] = np.clip(a2[k] + 0.25, 0, 1)
                d2 = self.model(feat, self.pres,
                                torch.tensor(a2[None], device=self.dev), self.ref)[0]
                diff = (d2[0, -1, :N_SLOTS] - base[0, -1, :N_SLOTS]).cpu().numpy() \
                    * [self.L, self.Wd]
                sens.append(float(np.linalg.norm(diff, axis=1).mean() * self.fps))
        self.knob_sens = np.array(sens, np.float32)

    def _shape_and_press(self, pos_m, pres):
        """Possession-aware structure on top of StyleNet motion (in place).

        Team in possession (and no-possession states): hold formation — spring
        toward seed-window anchors when far from the ball. Defending team: its
        anchor block leans toward the ball (side-to-side shift), the nearest
        n_pressers outfielders close down the carrier at a speed scaled by the
        press knob, the rest hold the shifted shape."""
        e = self.engine
        ball = e.ball
        d_ball = np.linalg.norm(pos_m - ball, axis=1)
        present = pres[:22] > 0.2
        poss = e.team_of(e.carrier) if e.carrier is not None else None

        chasers = []
        if e.mode == "loose":          # 50/50 ball: nearest player of EACH team chases
            for sl in (slice(0, N_SLOTS), slice(N_SLOTS, 22)):
                idx = np.arange(22)[sl]
                cand = [i for i in idx if present[i]]
                if cand:
                    i = min(cand, key=lambda i: d_ball[i])
                    d = max(d_ball[i], 1e-6)
                    pos_m[i] += (ball - pos_m[i]) / d * min(self.press_step * 1.3, d)
                    chasers.append(i)
        elif e.mode == "restart":      # dead ball: only the awarded team collects
            sl = slice(0, N_SLOTS) if e.restart_team == 0 else slice(N_SLOTS, 22)
            cand = [i for i in np.arange(22)[sl] if present[i]]
            if cand:
                i = min(cand, key=lambda i: d_ball[i])
                d = max(d_ball[i], 1e-6)
                pos_m[i] += (ball - pos_m[i]) / d * min(self.press_step * 1.5, d)
                chasers.append(i)
        elif e.mode == "flight" and e.receiver is not None and present[e.receiver]:
            i = e.receiver             # receiver attacks the pass target
            to = e.target - pos_m[i]
            d = float(np.linalg.norm(to))
            if d > 0.3:
                pos_m[i] += to / d * min(self.press_step * 1.4, d)
            chasers.append(i)

        for team, sl in ((0, slice(0, N_SLOTS)), (1, slice(N_SLOTS, 22))):
            idx = np.arange(22)[sl]
            defending = poss is not None and team != poss
            # dynamic anchors: the whole block travels with the ball (attack
            # pushes up into the opposition half, defence compresses/steps up)
            base = self.anchors[sl]
            shift = ball - base.mean(0)
            d_fwd = 1.0 if pos_m[team * N_SLOTS, 0] < self.L / 2 else -1.0  # own GK side
            k = self.follow_def if defending else self.follow_atk
            adv, ret = ((self.adv_cap_def, self.ret_cap_def) if defending
                        else (self.adv_cap_atk, self.ret_cap_atk))
            dx = np.clip(k * shift[0] * d_fwd, -ret, adv) * d_fwd            # forward-axis cap
            dy = np.clip(self.follow_y * shift[1], -self.y_cap, self.y_cap)
            eff = base + [dx, dy]
            eff[0] = base[0] + shift * self.follow_gk
            np.clip(eff, [2.0, 2.0], [self.L - 2.0, self.Wd - 2.0], out=eff)
            pressers = list(chasers)
            if defending:
                knob = float(self.knobs[1]) if team == 0 else float(self.ref[0, 1])
                cand = sorted((i for i in idx if present[i] and i % N_SLOTS != 0),
                              key=lambda i: d_ball[i])
                pressers = cand[:self.n_pressers] + list(chasers)
                step_m = self.press_step * (0.4 + 1.2 * knob)
                for i in cand[:self.n_pressers]:
                    d = max(d_ball[i], 1e-6)
                    adv = min(step_m, max(d - self.press_standoff, 0.0))
                    pos_m[i] += (ball - pos_m[i]) / d * adv
            hold = present[sl] & (d_ball[sl] > self.anchor_far)
            for j, i in enumerate(idx):
                if hold[j] and i not in pressers and i != e.carrier:
                    pos_m[i] += self.anchor_k * (eff[j] - pos_m[i])

    def step(self):
        L, Wd = self.norm["pos_x_m"], self.norm["pos_y_m"]
        adj = np.where(KNOB_POLARITY < 0, 1.0 - self.knobs, self.knobs).astype(np.float32)
        s0 = torch.tensor(adj[None], device=self.dev)
        with torch.no_grad():
            feat = build_frame(self.hist[0], self.norm).reshape(1, self.W, 23, 24)
            delta, _, _, attn = self.model(feat, self.pres, s0, self.ref, return_attn=True)
        d = delta[0, -1]
        if self.disable_stylenet:              # ablation: model output ignored
            d = torch.zeros_like(d)
        self.model_step_m = np.linalg.norm(
            d[:22].cpu().numpy() * [L, Wd], axis=1).astype(np.float32)
        prev = self.hist[0, -1, :, :2]
        newpos = (prev + d).clamp(0, 1)

        if self.engine is not None:
            np_pos = newpos.cpu().numpy()
            pres = self.cur_pres
            pos_m = (np_pos[:22] * [L, Wd]).astype(np.float32)
            self.engine.tempo_knobs = (float(self.knobs[3]), float(self.ref[0, 3]))
            model_pos = pos_m.copy()
            if not self.disable_structure:
                self._shape_and_press(pos_m, pres)
            self.struct_corr_m = np.linalg.norm(pos_m - model_pos, axis=1)
            vel_m = ((pos_m / [L, Wd] - prev[:22].cpu().numpy()) * [L, Wd] * self.fps)
            new_ev = self.engine.step(pos_m, vel_m.astype(np.float32), pres[:22])
            if self.engine.jumped:
                self.ball_snap = True
                self.engine.jumped = False
            self.event_log.extend(new_ev)
            self.events = (self.events + new_ev)[-6:]
            self.stats.update(self.engine, new_ev)
            if self.engine.mode == "goal":
                self.goal_pending = True
            np_pos[:22] = pos_m / [L, Wd]
            np_pos[BALL] = self.engine.ball / [L, Wd]
            newpos = torch.from_numpy(np_pos.astype(np.float32)).to(self.dev).clamp(0, 1)

        newvel = (newpos - prev) * self.fac
        newrow = torch.cat([newpos, newvel], -1)
        self.hist = torch.cat([self.hist[:, 1:], newrow[None, None]], 1)
        self.pres = torch.cat([self.pres[:, 1:], self.pres[:, -1:]], 1)   # carry presence
        self.cur_pres = self.pres[0, -1].cpu().numpy()
        self.pos = (newpos.cpu().numpy() * [L, Wd]).astype(np.float32)
        self.player_speed_m = float(np.linalg.norm(
            self.pos[:22] - prev[:22].cpu().numpy() * [L, Wd], axis=1).mean() * self.fps)
        if attn is not None:
            self.attn = attn[0, -1].float().cpu().numpy()
        self._tick += 1
        self.telemetry.record(self)
        if not self.fast:                  # viz extras cost 8-9 forwards/second
            if self._tick % 5 == 0:
                self.ghost_rollout()
            if self._tick % 25 == 1:
                self.knob_sensitivity()


# --------------------------------------------------------------------------- #
def w2s(x, y):
    return int(MARGIN + x / PITCH_L * PX_W), int(MARGIN + y / PITCH_W * PX_H)


def heat(v):
    v = float(np.clip(v, 0, 1))
    return (int(255 * v), int(80 * (1 - abs(v - 0.5) * 2)), int(255 * (1 - v)))


def draw_pitch(screen, pg):
    for i in range(7):
        c = GRASS_A if i % 2 == 0 else GRASS_B
        pg.draw.rect(screen, c, (MARGIN + i * PX_W // 7, MARGIN, PX_W // 7 + 1, PX_H))
    pg.draw.rect(screen, LINE, (MARGIN, MARGIN, PX_W, PX_H), 2)
    pg.draw.line(screen, LINE, w2s(52.5, 0), w2s(52.5, 68), 2)
    pg.draw.circle(screen, LINE, w2s(52.5, 34), int(9.15 / PITCH_L * PX_W), 2)
    for gx in (0, 105):
        bx = 16.5 if gx == 0 else -16.5
        x1, y1 = w2s(gx, 13.84); x2, y2 = w2s(gx + bx, 54.16)
        pg.draw.rect(screen, LINE, (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)), 2)


SHOW_DEFAULT = {"ghost": True, "labels": True, "ring": True}


def render(screen, sim, pg, font, big, sel, t_ms, paused, show=None):
    show = show or SHOW_DEFAULT
    screen.fill(PANEL_BG)
    draw_pitch(screen, pg)
    thr = 0.2
    # StyleNet intent trail (pure model rollout). Players only: the model's own
    # ball prediction is vestigial under the scripted ball and free-drifts.
    if show["ghost"] and sim.ghost is not None:
        n = len(sim.ghost)
        for t_i in range(n):
            f = (t_i + 1) / n
            for i in range(22):
                if sim.cur_pres[i] < thr:
                    continue
                base = T0 if i < N_SLOTS else T1
                col = tuple(int(c * (0.30 + 0.55 * f)) for c in base)
                pg.draw.circle(screen, col, w2s(*sim.ghost[t_i, i]), 2)
    for i in range(22):
        if sim.cur_pres[i] < thr:          # hide players absent in the seed clip
            continue
        x, y = w2s(*sim.pos[i])
        is_gk = i in (0, N_SLOTS)
        col = GK_C if is_gk else (T0 if i < N_SLOTS else T1)
        pg.draw.circle(screen, col, (x, y), 9)
        if is_gk:
            pg.draw.circle(screen, GK_BORDER, (x, y), 9, 2)
        if show["labels"]:
            lbl = sim.labels[0 if i < N_SLOTS else 1][i % N_SLOTS]
            tw = font.size(lbl)[0]
            screen.blit(font.render(lbl, True, TEXT), (x - tw // 2, y - 22))
    bx, by = w2s(*sim.pos[BALL])
    pg.draw.circle(screen, BALL_C, (bx, by), 6)
    pg.draw.circle(screen, (0, 0, 0), (bx, by), 6, 1)
    if sim.engine is not None:
        e = sim.engine
        if show["ring"] and e.carrier is not None and sim.cur_pres[e.carrier] > 0.2:
            cx, cy = w2s(*sim.pos[e.carrier])
            pg.draw.circle(screen, (255, 220, 60), (cx, cy), 13, 2)
        clock = sim.hud_extra if sim.hud_extra else f"{t_ms//1000:02d}s"
        hud = (f"{TEAM_NAMES[0]} {e.score[0]} - {e.score[1]} {TEAM_NAMES[1]}   {clock}   "
               f"ball: {e.mode}" + (f"  |  {e.last_event}" if e.last_event else ""))
        for i, txt in enumerate(sim.events[-5:]):
            screen.blit(font.render(txt, True, (200, 200, 160)),
                        (MARGIN, 700 + i * 15))
        _stats_strip(screen, sim, pg, font, MARGIN, PX_H + MARGIN + 6)
    else:
        hud = (f"{TEAM_NAMES[0]} (blue) vs {TEAM_NAMES[1]} (yellow)   "
               f"{t_ms//1000:02d}s   LEARNED ball")
    screen.blit(big.render(hud, True, TEXT), (MARGIN, 8))
    if paused:
        screen.blit(big.render("PAUSED", True, (255, 210, 90)), (PX_W - 60, 8))

    # ---- panel: sliders + attention ----
    px = PANEL_X + 20
    screen.blit(big.render(f"style knobs ({TEAM_NAMES[0]})", True, TEXT), (px, 30))
    smax = max(float(sim.knob_sens.max()), 1e-6)
    for k in range(4):
        y = 70 + k * 46
        sim_slider(screen, pg, font, px, y, KNOBS[k], sim.knobs[k], k == sel)
        # FiLM authority meter: motion response (m/s) to +0.25 on this knob
        pg.draw.rect(screen, (55, 55, 65), (px, y + 12, SLIDER_W, 4))
        pg.draw.rect(screen, (235, 170, 70),
                     (px, y + 12, int(SLIDER_W * sim.knob_sens[k] / smax), 4))
        screen.blit(font.render(f"{sim.knob_sens[k]:.2f}", True, (235, 170, 70)),
                    (px + SLIDER_W + 8, y + 6))
    screen.blit(font.render("orange bars: FiLM authority (m/s per +0.25 knob)",
                            True, (160, 160, 170)), (px, 258))
    _panel_attn(screen, sim, pg, font, px, 284)
    _panel_kick(screen, sim, pg, font, px, 616)
    screen.blit(font.render("1-4 select  UP/DOWN adjust  G ghosts  N names  C ring  "
                            "SPACE pause  ESC quit", True, (255, 210, 90)),
                (MARGIN, WIN_H - 22))


SLIDER_W = 380


def _stats_strip(screen, sim, pg, font, x, y):
    """Live box score under the pitch, one row per team."""
    s = sim.stats
    ps = s.possession()
    hdr = (f"{'':13}{'poss':>5}{'pass':>7}{'cmp%':>7}{'shots':>7}"
           f"{'goals':>7}{'intercept':>11}{'tackles':>9}{'saves':>7}")
    screen.blit(font.render(hdr, True, (160, 160, 170)), (x, y))
    for t in (0, 1):
        d = s.t[t]
        cmp_pct = 100.0 * d["pass_cmp"] / max(d["pass_att"], 1)
        row = (f"{s.names[t][:12]:13}{ps[t] * 100:4.0f}%{d['pass_att']:7d}"
               f"{cmp_pct:6.0f}%{d['shots']:7d}{d['goals']:7d}"
               f"{d['intercepts']:11d}{d['tackles']:9d}{d['saves']:7d}")
        screen.blit(font.render(row, True, T0 if t == 0 else T1), (x, y + 15 + t * 14))


def sim_slider(screen, pg, font, x, y, label, val, selected):
    col = (255, 210, 90) if selected else TEXT
    screen.blit(font.render(f"{label}: {val:.2f}", True, col), (x, y - 16))
    pg.draw.rect(screen, (60, 60, 70), (x, y, SLIDER_W, 8))
    fillc = (120, 200, 120) if selected else (90, 140, 200)
    pg.draw.rect(screen, fillc, (x, y, int(SLIDER_W * val), 8))
    pg.draw.circle(screen, col, (x + int(SLIDER_W * val), y + 4), 7)


def slider_hit(mx, my, px):
    for k in range(4):
        y = 70 + k * 46
        if px <= mx <= px + SLIDER_W and y - 8 <= my <= y + 16:
            return k, float(np.clip((mx - px) / SLIDER_W, 0, 1))
    return None, None


def _short_name(name: str, n: int = 9) -> str:
    """Compact a squad name for the attention axes ('Dibu Martinez' -> 'Martinez')."""
    if len(name) <= n:
        return name
    parts = name.split()
    if len(parts) > 1:
        cand = parts[0] if (parts[-1].endswith(".") or len(parts[-1]) <= 2) else parts[-1]
        return cand[:n]
    return name[:n]


def _panel_attn(screen, sim, pg, font, x, y):
    """Attention DEVIATION from uniform (1/23): red = row player attends the
    column player more than uniform, blue = less. Raw attention is
    near-uniform, so normalising by the max painted everything red — the
    deviation is the signal. Axes carry the team0 player names; rows = who is
    looking, columns = who is looked at."""
    screen.blit(font.render(f"spatial attn vs uniform ({TEAM_NAMES[0]}, red=+ blue=-)",
                            True, TEXT), (x, y))
    A = sim.attn[:N_SLOTS, :N_SLOTS] - 1.0 / 23.0
    m = max(float(np.abs(A).max()), 1e-6)
    V = 0.5 + 0.5 * A / m
    cell = 20
    names = [_short_name(sim.labels[0][s]) for s in range(N_SLOTS)]
    gx, gy = x + 72, y + 84                    # room for row + rotated col labels
    for c in range(N_SLOTS):
        surf = pg.transform.rotate(font.render(names[c], True, (200, 200, 210)), 90)
        screen.blit(surf, (gx + c * cell + (cell - surf.get_width()) // 2,
                           gy - surf.get_height() - 4))
    for r in range(N_SLOTS):
        surf = font.render(names[r], True, (200, 200, 210))
        screen.blit(surf, (gx - surf.get_width() - 5,
                           gy + r * cell + (cell - surf.get_height()) // 2))
        for c in range(N_SLOTS):
            pg.draw.rect(screen, heat(V[r, c]), (gx + c * cell, gy + r * cell, cell, cell))
    pg.draw.rect(screen, LINE, (gx, gy, N_SLOTS * cell, N_SLOTS * cell), 1)
    screen.blit(font.render(f"max |dev| {m:.3f}", True, (160, 160, 170)),
                (gx + N_SLOTS * cell + 8, gy))
    screen.blit(font.render("rows look at columns", True, (160, 160, 170)),
                (gx + N_SLOTS * cell + 8, gy + 16))


def _panel_kick(screen, sim, pg, font, x, y):
    """KickNet live: the carrier's pass options (arrow brightness/width =
    receiver probability) and the hold/pass/shot action distribution."""
    screen.blit(font.render("KickNet: receiver probs + action", True, TEXT), (x, y - 16))
    w, h = 200, 110
    pg.draw.rect(screen, (24, 60, 24), (x, y, w, h))
    pg.draw.rect(screen, LINE, (x, y, w, h), 1)

    def mp(p):
        return int(x + p[0] / PITCH_L * w), int(y + p[1] / PITCH_W * h)

    q = sim.engine.last_query if sim.engine is not None else None
    if q is None:
        screen.blit(font.render("no carrier yet", True, TEXT), (x + 8, y + 8))
        return
    base = 0 if q["team"] == 0 else N_SLOTS
    tc = T0 if q["team"] == 0 else T1
    cxy = mp(sim.pos[q["carrier"]])
    pmax = max(float(q["recv"].max()), 1e-6)
    for s in range(N_SLOTS):
        i = base + s
        if q["mask"][s] == 0 or sim.cur_pres[i] < 0.2:
            continue
        p = q["recv"][s] / pmax
        v = int(90 + 165 * p)
        pg.draw.line(screen, (v, v, 60), cxy, mp(sim.pos[i]), 1 + int(2 * p))
        pg.draw.circle(screen, tc, mp(sim.pos[i]), 3)
    pg.draw.circle(screen, (255, 220, 60), cxy, 4)
    labels = ("hold", "pass", "shot")
    for k in range(3):
        bx = x + k * 68
        by = y + h + 8
        pg.draw.rect(screen, (55, 55, 65), (bx, by, 60, 8))
        pg.draw.rect(screen, (120, 200, 120), (bx, by, int(60 * q["action"][k]), 8))
        screen.blit(font.render(f"{labels[k]} {q['action'][k]:.2f}", True, TEXT),
                    (bx, by + 10))


def choose_mode(screen, pg, font, big, fpsclock, half_secs):
    """Pre-match menu. Returns 'live', 'sim', or None (quit)."""
    mm, ss = divmod(int(half_secs), 60)
    lines = [
        ("FOOTBALL SIM — choose match mode", big, TEXT, 120),
        (f"L   watch LIVE        (2 x {mm}:{ss:02d} halves, interactive knobs + viz)", big, (120, 200, 120), 200),
        ("S   SIMULATE full 90' (fast, 45+45; then watch the goals)", big, (90, 160, 240), 240),
        ("ESC quit", font, (255, 210, 90), 300),
    ]
    while True:
        screen.fill(PANEL_BG)
        for txt, fnt, col, y in lines:
            screen.blit(fnt.render(txt, True, col), (MARGIN + 40, y))
        pg.display.flip()
        for e in pg.event.get():
            if e.type == pg.QUIT:
                return None
            if e.type == pg.KEYDOWN:
                if e.key == pg.K_ESCAPE:
                    return None
                if e.key == pg.K_l:
                    return "live"
                if e.key == pg.K_s:
                    return "sim"
        fpsclock.tick(30)


def run_fast_match(sim, pg, screen, font, big, half_secs):
    """Simulate a whole match as fast as the model runs (no pacing, viz extras
    off), recording a clip around every goal. Returns the clip list."""
    from collections import deque
    sim.fast = True
    half, clock2, first_ko = 1, 0.0, 0
    buf = deque(maxlen=90)                       # last 18 s of frames
    clips, pending, meta = [], None, None
    step = 0
    while True:
        sim.step()
        clock2 += 1.0 / sim.fps
        buf.append((sim.pos.copy(), sim.cur_pres.copy()))
        if pending is not None:
            pending -= 1
            if pending <= 0:                     # aftermath captured -> cut clip
                clips.append({"frames": np.stack([f for f, _ in buf]),
                              "pres": np.stack([p for _, p in buf]), **meta})
                sim.goal_pending = False
                conceded = 1 - (sim.engine.last_goal_team or 0)
                sim.apply_kickoff(conceded, second_half=(half == 2))
                buf.clear()
                pending = None
        elif sim.goal_pending:
            mm, ss = divmod(int(clock2), 60)
            meta = {"team": sim.engine.last_goal_team,
                    "score": tuple(sim.engine.score),
                    "label": f"H{half} {mm:02d}:{ss:02d}"}
            pending = 12                         # keep ~2.4 s of celebration room
        if clock2 >= half_secs and pending is None:
            if half == 1:
                half, clock2 = 2, 0.0
                sim.apply_kickoff(1 - first_ko, second_half=True)
                buf.clear()
            else:
                break
        step += 1
        if step % 150 == 0:                      # progress screen + stay responsive
            abort = False
            for e in pg.event.get():
                if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                    abort = True
            if abort:
                break
            frac = (0.0 if half == 1 else 0.5) + 0.5 * min(clock2 / half_secs, 1.0)
            mm, ss = divmod(int(clock2), 60)
            screen.fill(PANEL_BG)
            screen.blit(big.render(
                f"SIMULATING...  H{half} {mm:02d}:{ss:02d} / 45:00     "
                f"{TEAM_NAMES[0]} {sim.engine.score[0]} - "
                f"{sim.engine.score[1]} {TEAM_NAMES[1]}", True, TEXT), (MARGIN, 60))
            pg.draw.rect(screen, (60, 60, 70), (MARGIN, 110, 900, 14))
            pg.draw.rect(screen, (120, 200, 120), (MARGIN, 110, int(900 * frac), 14))
            ps = sim.stats.possession()
            st = sim.stats.t
            screen.blit(font.render(
                f"possession {ps[0] * 100:.0f}%-{ps[1] * 100:.0f}%   "
                f"passes {st[0]['pass_att']}-{st[1]['pass_att']}   "
                f"shots {st[0]['shots']}-{st[1]['shots']}   "
                f"interceptions {st[0]['intercepts']}-{st[1]['intercepts']}",
                True, (160, 160, 170)), (MARGIN, 132))
            for i, txt in enumerate(sim.events[-8:]):
                screen.blit(font.render(txt, True, (200, 200, 160)), (MARGIN, 152 + 16 * i))
            screen.blit(font.render("ESC abort", True, (255, 210, 90)), (MARGIN, 300))
            pg.display.flip()
    sim.fast = False
    return clips


def replay_goals(screen, sim, pg, font, big, clips, score, fpsclock):
    """Post-match viewer: loop each goal clip; LEFT/RIGHT switch, SPACE restart."""
    interp = 6                                   # 5 Hz frames -> 30 fps display
    idx, running = 0, True
    while running:
        if not clips:
            screen.fill(PANEL_BG)
            screen.blit(big.render(f"FULL TIME  {TEAM_NAMES[0]} {score[0]} - "
                                   f"{score[1]} {TEAM_NAMES[1]}    "
                                   "no goals to replay", True, TEXT), (MARGIN, 80))
            screen.blit(font.render("ESC quit", True, (255, 210, 90)), (MARGIN, 120))
            pg.display.flip()
            for e in pg.event.get():
                if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                    running = False
            fpsclock.tick(30)
            continue
        clip = clips[idx]
        frames, pres = clip["frames"], clip["pres"]
        S, f, playing = len(frames), 0, True
        while playing and running:
            for e in pg.event.get():
                if e.type == pg.QUIT:
                    running = False
                elif e.type == pg.KEYDOWN:
                    if e.key == pg.K_ESCAPE:
                        running = False
                    elif e.key == pg.K_RIGHT:
                        idx = (idx + 1) % len(clips); playing = False
                    elif e.key == pg.K_LEFT:
                        idx = (idx - 1) % len(clips); playing = False
                    elif e.key == pg.K_SPACE:
                        f = 0
            s_i = min(f // interp, S - 2)
            frac = (f % interp) / interp
            pos = (1 - frac) * frames[s_i] + frac * frames[s_i + 1]
            screen.fill(PANEL_BG)
            draw_pitch(screen, pg)
            for i in range(22):
                if pres[s_i][i] < 0.2:
                    continue
                x, y = w2s(*pos[i])
                is_gk = i in (0, N_SLOTS)
                col = GK_C if is_gk else (T0 if i < N_SLOTS else T1)
                pg.draw.circle(screen, col, (x, y), 9)
                if is_gk:
                    pg.draw.circle(screen, GK_BORDER, (x, y), 9, 2)
            bx, by = w2s(*pos[BALL])
            pg.draw.circle(screen, BALL_C, (bx, by), 6)
            pg.draw.circle(screen, (0, 0, 0), (bx, by), 6, 1)
            hud = (f"FULL TIME {TEAM_NAMES[0]} {score[0]} - {score[1]} {TEAM_NAMES[1]}    "
                   f"GOAL {idx + 1}/{len(clips)}  {TEAM_NAMES[clip['team']]}  "
                   f"{clip['label']}  ->  {clip['score'][0]}-{clip['score'][1]}")
            screen.blit(big.render(hud, True, TEXT), (MARGIN, 8))
            screen.blit(font.render("LEFT/RIGHT switch goal   SPACE replay   ESC quit",
                                    True, (255, 210, 90)), (MARGIN, WIN_H - 22))
            pg.display.flip()
            f += 1
            if f >= (S - 1) * interp:
                f = 0                            # loop the clip
            fpsclock.tick(30)


def save_match_outputs(sim):
    """Post-match dump: box score to console, telemetry npz + stats json +
    evidence figures to output/telemetry/."""
    if sim.engine is None or not sim.telemetry.rows:
        return
    out = ROOT / "output" / "telemetry"
    out.mkdir(parents=True, exist_ok=True)
    npz = sim.telemetry.save(out / "last_match.npz")
    (out / "last_match_stats.json").write_text(
        json.dumps(sim.stats.as_dict(), indent=2))
    print(sim.stats.table_str())
    print(f"telemetry: {npz}")
    try:
        from stage8_sim.telemetry import make_match_figures
        for f in make_match_figures(npz, out, fps=sim.fps, names=sim.stats.names):
            print(f"fig: {f}")
    except Exception as ex:                    # matplotlib missing/broken: not fatal
        print(f"figures skipped: {ex}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default=str(ROOT / "output" / "training" / "metrica" / "style_model.pt"))
    ap.add_argument("--store", default=str(ROOT / "output" / "storage" / "metrica"),
                    help="windows dir for seeding (use .../storage/gsr for the old data)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--cold", action="store_true",
                    help="init from a synthetic formation (out-of-distribution, degenerates); "
                         "default seeds from a real GSR window")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--window", type=int, default=-1,
                    help="seed window index; -1 = highest-visibility window")
    ap.add_argument("--reseed-secs", type=float, default=15.0,
                    help="re-seed from a fresh real window every N seconds (0 = never)")
    ap.add_argument("--no-style", action="store_true",
                    help="unconditioned: hold style neutral, ignore sliders")
    ap.add_argument("--kick", default=str(ROOT / "output" / "kick" / "kick_model.pt"))
    ap.add_argument("--pure", action="store_true",
                    help="disable the hybrid ball engine (old fully-learned ball)")
    ap.add_argument("--seeded", action="store_true",
                    help="legacy demo mode: seed from real windows + auto-reseed "
                         "(default is a structured match from kick-off)")
    ap.add_argument("--half-secs", type=float, default=120.0,
                    help="length of each LIVE half in sim seconds")
    ap.add_argument("--formation0", default="4-4-2", choices=sorted(FORMATIONS))
    ap.add_argument("--formation1", default="4-3-3", choices=sorted(FORMATIONS))
    ap.add_argument("--mode", default="ask", choices=["ask", "live", "sim"],
                    help="pre-match choice: watch live or fast-simulate 90' + goal replays")
    args = ap.parse_args()
    match_mode = not args.seeded and not args.cold and not args.pure
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame as pg
    pg.init()
    screen = pg.display.set_mode((WIN_W, WIN_H))
    pg.display.set_caption("Style-controllable football sim")
    font = pg.font.SysFont("consolas", 12)
    big = pg.font.SysFont("consolas", 18)
    clock = pg.time.Clock()

    pool, rng = [], np.random.default_rng(0)
    if not args.cold and not match_mode:
        d = np.load(Path(args.store) / f"windows_{args.split}.npz")
        vis = d["presence"][:, -1, :22].sum(-1)
        idx = (np.argsort(vis)[-25:] if args.window < 0 else [args.window])  # top-vis windows
        pool = [{k: d[k][i] for k in ("agents", "presence", "style0", "style1")} for i in idx]
        print(f"seed pool: {len(pool)} windows, reseed every {args.reseed_secs}s")
    sim = Sim(args.model, seed=(pool[-1] if pool else None),
              kick_path=(None if args.pure else args.kick), pure=args.pure)
    match = {"half": 1, "clock": 0.0, "first_ko": 0, "full": False}
    if match_mode:
        sim.set_formations(args.formation0, args.formation1)
        sim.apply_kickoff(match["first_ko"])
        print(f"match: {args.formation0} vs {args.formation1}, "
              f"halves of {args.half_secs:.0f}s")

    def match_tick():
        """Advance match clock; kick-offs after goals, half-time end swap,
        full time. Returns True when sim state was re-positioned."""
        if match["full"]:
            return False
        match["clock"] += 1.0 / sim.fps
        mm, ss = divmod(int(match["clock"]), 60)
        sim.hud_extra = f"H{match['half']} {mm:02d}:{ss:02d}"
        if sim.goal_pending:
            sim.goal_pending = False
            conceded = 1 - (sim.engine.last_goal_team or 0)
            sim.apply_kickoff(conceded, second_half=(match["half"] == 2))
            return True
        if match["clock"] >= args.half_secs:
            if match["half"] == 1:
                match["half"], match["clock"] = 2, 0.0
                ht = f"HALF-TIME  {sim.engine.score[0]}-{sim.engine.score[1]}"
                sim.events = (sim.events + [ht])[-6:]
                sim.event_log.append(ht)
                sim.apply_kickoff(1 - match["first_ko"], second_half=True)
                return True
            match["full"] = True
            ft = f"FULL TIME  {sim.engine.score[0]}-{sim.engine.score[1]}"
            sim.events = (sim.events + [ft])[-6:]
            sim.event_log.append(ft)
            sim.hud_extra = ft
        return False
    if args.no_style:
        sim.knobs[:] = 0.5
        sim.ref = torch.full_like(sim.ref, 0.5)
        sim.no_style = True
    reseed_steps = int(args.reseed_secs * sim.fps) if (pool and args.reseed_secs > 0) else 0

    def maybe_reseed(n):
        if sim.goal_pending:                              # goal -> restart from a real moment
            sim.goal_pending = False
            if pool:
                sim.apply_seed(pool[rng.integers(len(pool))], keep_knobs=True)
            elif sim.engine is not None:
                sim.engine.reset(sim.pos[:22], sim.cur_pres[:22], sim.pos[BALL])
            return True
        if reseed_steps and n > 0 and n % reseed_steps == 0:
            sim.apply_seed(pool[rng.integers(len(pool))], keep_knobs=True)
            return True
        return False

    if args.headless:
        t = 0
        steps = int(2 * args.half_secs * sim.fps) + 25 if match_mode else args.steps
        done = 0
        for n in range(steps):
            if match_mode and match["full"]:
                break
            sim.step()
            done += 1
            match_tick() if match_mode else maybe_reseed(n + 1)
            render(screen, sim, pg, font, big, 0, t, False)
            t += 200
        out = ROOT / "output" / "sim" / "gsr"
        out.mkdir(parents=True, exist_ok=True)
        pg.image.save(screen, str(out / "screenshot.png"))
        xspread = float(np.std(sim.pos[:22, 0]))
        yspread = float(np.std(sim.pos[:22, 1]))
        print(f"headless {done} steps: attn sum {sim.attn.sum():.2f} "
              f"x-spread {xspread:.1f}m y-spread {yspread:.1f}m "
              f"ball@({sim.pos[BALL][0]:.0f},{sim.pos[BALL][1]:.0f}) "
              f"-> {out/'screenshot.png'}")
        if sim.engine is not None:
            from collections import Counter
            c = Counter(e.split()[0] for e in sim.event_log)
            mins = done / sim.fps / 60
            print(f"events over {mins:.1f}min: {dict(c)} "
                  f"score {sim.engine.score[0]}-{sim.engine.score[1]} "
                  f"pass rate {c.get('PASS', 0)/max(mins,1e-6):.1f}/min")
            save_match_outputs(sim)
        pg.quit()
        return

    if match_mode:
        mode = args.mode
        if mode == "ask":
            mode = choose_mode(screen, pg, font, big, clock, args.half_secs)
            if mode is None:
                pg.quit()
                return
        if mode == "sim":
            clips = run_fast_match(sim, pg, screen, font, big, half_secs=45 * 60)
            print(f"simulated 90': {sim.engine.score[0]}-{sim.engine.score[1]}, "
                  f"{len(clips)} goal clip(s)")
            save_match_outputs(sim)
            replay_goals(screen, sim, pg, font, big, clips,
                         tuple(sim.engine.score), clock)
            pg.quit()
            return

    show = dict(SHOW_DEFAULT)
    base_interp = max(1, 30 // sim.fps)
    running, paused, sel, t, frame, nstep = True, False, 0, 0, 0, 0
    prev = sim.pos.copy(); sim.step(); cur = sim.pos.copy()
    while running:
        for e in pg.event.get():
            if e.type == pg.QUIT:
                running = False
            elif e.type == pg.KEYDOWN:
                if e.key == pg.K_ESCAPE:
                    running = False
                elif e.key == pg.K_SPACE:
                    paused = not paused
                elif e.key == pg.K_g:
                    show["ghost"] = not show["ghost"]
                elif e.key == pg.K_n:
                    show["labels"] = not show["labels"]
                elif e.key == pg.K_c:
                    show["ring"] = not show["ring"]
                elif e.key in (pg.K_1, pg.K_2, pg.K_3, pg.K_4):
                    sel = e.key - pg.K_1
                elif e.key == pg.K_UP:
                    sim.knobs[sel] = float(np.clip(sim.knobs[sel] + 0.05, 0, 1))
                elif e.key == pg.K_DOWN:
                    sim.knobs[sel] = float(np.clip(sim.knobs[sel] - 0.05, 0, 1))
            elif e.type == pg.MOUSEBUTTONDOWN:
                k, v = slider_hit(*e.pos, PANEL_X + 20)
                if k is not None:
                    sel = k; sim.knobs[k] = v
            elif e.type == pg.MOUSEMOTION and e.buttons[0]:
                k, v = slider_hit(*e.pos, PANEL_X + 20)
                if k is not None:
                    sim.knobs[k] = v
        if not paused and not (match_mode and match["full"]):
            frac = (frame % base_interp) / base_interp
            sim.pos = (1 - frac) * prev + frac * cur
            if frame % base_interp == base_interp - 1:
                prev = cur.copy(); sim.step(); nstep += 1
                moved = match_tick() if match_mode else maybe_reseed(nstep)
                if moved:
                    prev = sim.pos.copy()          # snap (no interp) on kick-off/reseed
                cur = sim.pos.copy()
                if sim.ball_snap:                  # restart: don't lerp the ball across
                    prev[BALL] = cur[BALL]
                    sim.ball_snap = False
            frame += 1; t += int(1000 / args.fps)
        render(screen, sim, pg, font, big, sel, t, paused, show)
        pg.display.flip()
        clock.tick(args.fps)
    if match_mode:
        save_match_outputs(sim)
    pg.quit()


if __name__ == "__main__":
    main()
