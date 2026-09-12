"""Scripted ball engine for the hybrid simulator.

The ball is no longer a free learned agent (that gave "the ball passes
itself"). It has three physical modes:

  CARRIED  stuck to a carrier; each step KickNet decides hold / pass / shot.
  FLIGHT   ballistic run toward a target at a data-fitted speed; players on
           the path can intercept -> contests and turnovers are physical.
  LOOSE    decelerating roll; nearest player captures it.

Players stay StyleNet-driven; they see the scripted ball through their
features, so ball-chasing behaviour is preserved while the ball itself obeys
football semantics (passes have kickers and receivers, shots can score,
possession changes hands by tackle/interception).

Feature math mirrors stage7_style.build_kick_data.build_sample EXACTLY --
keep the two in sync.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from stage7_style.kick_model import KickNet

PITCH_L, PITCH_W = 105.0, 68.0
N_SLOTS = 11
GOAL_HALF_W = 3.66

CAPTURE_R = 1.2          # m: loose-ball / arrival capture radius
INTERCEPT_R = 1.2        # m: distance to flight path that allows interception
P_INTERCEPT_NEAR = 0.55  # squarely in the path (< 0.6 m)
P_INTERCEPT_FAR = 0.30   # grazing contact
TACKLE_R = 1.5           # m: opponent this close to carrier may tackle
P_TACKLE = 0.08          # per step (0.2 s)
COOLDOWN_STEPS = 3       # no kick decision / re-capture right after a kick
CATCH_SPEED = 8.0        # m/s: captured ball closes on the carrier's feet (no snap)
CONTROL_R = 0.9          # m: ball must be this close before the carrier may kick
KICK_GATE = 0.38         # base kick gate on p(pass)+p(shot)...
GATE_DECAY = 0.07        # ...decaying per carried step: KickNet decides when/to whom
GATE_FLOOR = 0.15        # within a window, but a real carry never lasts 3+ s
SHOT_SPEED = 24.0
DIRECT_GAIN = 1.6        # forward-bias on receiver sampling, scaled by the tempo knob
SHOT_NOISE = 3.2         # m std of shot aim around goal centre (half-width 3.66)
GK_SAVE_R = 2.5          # m: defending GK reach against a shot
P_SAVE = 0.7             # save chance per contact
RECV_TEMP = 0.7          # receiver sampling temperature


class BallEngine:
    def __init__(self, kick_model_path: str, dt: float, rng=None):
        ck = torch.load(kick_model_path, map_location="cpu", weights_only=False)
        self.net = KickNet()
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        meta = json.loads((Path(kick_model_path).parent / "kick_meta.json").read_text())
        f = meta["pass_speed_fit"]
        self.spd_a, self.spd_b, self.spd_clip = f["intercept"], f["slope"], f["clip"]
        self.dt = dt
        self.rng = rng or np.random.default_rng(0)
        self.mode = "loose"
        self.carrier = None          # agent idx 0-21
        self.ball = np.array([PITCH_L / 2, PITCH_W / 2], np.float32)
        self.bvel = np.zeros(2, np.float32)
        self.target = None
        self.receiver = None         # intended receiver agent idx
        self.speed = 0.0
        self.cooldown = 0
        self.kicker = None
        self.tried = set()           # players who already got an intercept roll this flight
        self.last_event = ""
        self.score = [0, 0]
        self.last_goal_team = None
        self.carry_steps = 0
        self.last_query = None       # latest KickNet output (for visualization)
        self.tempo_knobs = (0.5, 0.5)  # per-team tempo/directness (sim updates live)
        self.names = ("team0", "team1")
        self.slot_names = None       # optional [team][slot] player names for events
        self.jumped = False          # ball moved discontinuously this step (kick-off only)
        self.last_touch_team = 0
        self.restart_team = 0        # who picks up an out-of-bounds ball
        self.random_kick = False     # ablation: replace KickNet with random decisions

    # ------------------------------------------------------------------ #
    def reset(self, pos: np.ndarray, pres: np.ndarray, ball: np.ndarray):
        """Seed from a real window state: nearest present player captures."""
        self.ball = ball[:2].astype(np.float32).copy()
        self.bvel[:] = 0
        d = np.linalg.norm(pos - self.ball, axis=1)
        d[pres <= 0.2] = 1e9
        j = int(d.argmin())
        if d[j] < 3.0:
            self._capture(j)
        else:
            self.mode, self.carrier = "loose", None

    def _capture(self, j: int):
        self.mode, self.carrier = "carried", int(j)
        self.cooldown = COOLDOWN_STEPS
        self.carry_steps = 0
        self.last_touch_team = self.team_of(int(j))
        self.tried.clear()

    @staticmethod
    def team_of(idx: int) -> int:
        return 0 if idx < N_SLOTS else 1

    def _pname(self, team: int, slot: int) -> str:
        if self.slot_names is not None:
            return self.slot_names[team][slot]
        return str(slot)

    def _attack_dir(self, pos, pres, team) -> float:
        gk = pos[0] if team == 0 else pos[N_SLOTS]
        return 1.0 if gk[0] < PITCH_L / 2 else -1.0

    # ------------------------------------------------------------------ #
    def _fwd(self, xy, d):
        xy = np.asarray(xy, np.float32).copy()
        if d < 0:
            xy[..., 0] = PITCH_L - xy[..., 0]
            xy[..., 1] = PITCH_W - xy[..., 1]
        return xy

    def _kick_query(self, pos, pres, team, kslot, d):
        """Mirror of build_kick_data.build_sample on live arrays."""
        own = pos[:N_SLOTS] if team == 0 else pos[N_SLOTS:2 * N_SLOTS]
        own_p = pres[:N_SLOTS] if team == 0 else pres[N_SLOTS:2 * N_SLOTS]
        opp = pos[N_SLOTS:2 * N_SLOTS] if team == 0 else pos[:N_SLOTS]
        opp_p = pres[N_SLOTS:2 * N_SLOTS] if team == 0 else pres[:N_SLOTS]
        opp_xy = self._fwd(opp[opp_p > 0.2], d)
        if not len(opp_xy):
            return None
        k = self._fwd(own[kslot], d)
        goal = np.array([PITCH_L, PITCH_W / 2], np.float32)
        vg = goal - k
        dist_goal = float(np.linalg.norm(vg))
        d_opp = np.linalg.norm(opp_xy - k, axis=1)
        ctx = np.array([
            k[0] / PITCH_L, k[1] / PITCH_W, dist_goal / PITCH_L,
            vg[0] / max(dist_goal, 1e-6), vg[1] / max(dist_goal, 1e-6),
            min(d_opp.min(), 20) / 20, 1.0 if kslot == 0 else 0.0,
            (d_opp <= 10).sum() / 11.0,
        ], np.float32)
        mates = np.zeros((N_SLOTS, 6), np.float32)
        mask = np.zeros(N_SLOTS, np.float32)
        for s in range(N_SLOTS):
            if s == kslot or own_p[s] <= 0.2:
                continue
            m = self._fwd(own[s], d)
            rel = m - k
            od = float(np.linalg.norm(opp_xy - m, axis=1).min())
            mates[s] = [rel[0] / 50, rel[1] / 34, np.linalg.norm(rel) / 50,
                        min(od, 20) / 20, np.linalg.norm(goal - m) / PITCH_L,
                        1.0 if s == 0 else 0.0]
            mask[s] = 1.0
        with torch.no_grad():
            a, r = self.net(torch.from_numpy(ctx[None]),
                            torch.from_numpy(mates[None]),
                            torch.from_numpy(mask[None]))
        return (torch.softmax(a[0], -1).numpy(),
                torch.softmax(r[0] / RECV_TEMP, -1).numpy(), mask, mates)

    # ------------------------------------------------------------------ #
    def step(self, pos: np.ndarray, vel: np.ndarray, pres: np.ndarray) -> list[str]:
        """pos/vel (22,2) metres & m/s, pres (22,). Returns event strings."""
        ev = []
        if self.mode == "goal":
            return ev                      # frozen until the kick-off resets us
        if self.mode == "carried":
            ev += self._step_carried(pos, vel, pres)
        elif self.mode == "flight":
            ev += self._step_flight(pos, pres)
        elif self.mode == "restart":
            ev += self._step_restart(pos, pres)
        else:
            ev += self._step_loose(pos, pres)
        self.ball = np.clip(self.ball, 0, [PITCH_L, PITCH_W])
        if ev:
            self.last_event = ev[-1]
        return ev

    def _step_carried(self, pos, vel, pres):
        c = self.carrier
        v = vel[c]
        sp = np.linalg.norm(v)
        off = (v / sp * 0.5) if sp > 0.3 else 0.0
        # ball closes on the carrier's feet at a capped speed -- never snaps
        tgt = pos[c] + off
        gapv = tgt - self.ball
        gap = float(np.linalg.norm(gapv))
        max_step = CATCH_SPEED * self.dt
        self.ball = tgt if gap <= max_step else self.ball + gapv / gap * max_step
        self.bvel = v.copy()
        if gap > CONTROL_R:
            return []                      # still bringing it under control
        if self.cooldown > 0:
            self.cooldown -= 1
            return []

        team = self.team_of(c)
        # tackle contest: nearby opponent may poke it loose
        oppsl = slice(N_SLOTS, 2 * N_SLOTS) if team == 0 else slice(0, N_SLOTS)
        od = np.linalg.norm(pos[oppsl] - pos[c], axis=1)
        od[pres[oppsl] <= 0.2] = 1e9
        if od.min() < TACKLE_R and self.rng.random() < P_TACKLE:
            self.mode, self.carrier = "loose", None
            ang = self.rng.uniform(0, 2 * np.pi)
            self.bvel = np.array([np.cos(ang), np.sin(ang)], np.float32) * 3.0
            return [f"TACKLE by {self.names[1 - team]}"]

        kslot = c % N_SLOTS
        d = self._attack_dir(pos, pres, team)
        if self.random_kick:           # ablation baseline: no learned decisions
            own_p = pres[:N_SLOTS] if team == 0 else pres[N_SLOTS:2 * N_SLOTS]
            mask = (own_p > 0.2).astype(np.float32)
            mask[kslot] = 0.0
            if mask.sum() == 0:
                return []
            pa = self.rng.dirichlet(np.ones(3)).astype(np.float32)
            pr = (mask / mask.sum()).astype(np.float32)
        else:
            q = self._kick_query(pos, pres, team, kslot, d)
            if q is None:
                return []
            pa, pr, mask, mates = q
            # directness: bias receiver sampling toward forward options (attack-
            # frame dx is mates[:,0], 50 m units), strength = team's tempo knob
            tempo = self.tempo_knobs[team]
            pr = pr * np.exp(DIRECT_GAIN * (0.3 + 1.4 * tempo) * mates[:, 0]) * (mask > 0)
            pr = pr / max(pr.sum(), 1e-9)
        self.last_query = {"action": pa.copy(), "recv": pr.copy(), "mask": mask.copy(),
                           "carrier": c, "team": team}
        self.carry_steps += 1
        gate = KICK_GATE * max(GATE_FLOOR / KICK_GATE, 1.0 - GATE_DECAY * self.carry_steps)
        if pa[1] + pa[2] < gate:
            return []
        if pa[2] > pa[1] or pa[2] > 0.35:                   # ---- shot ----
            gx = PITCH_L if d > 0 else 0.0
            gy = PITCH_W / 2 + self.rng.normal(0, SHOT_NOISE)
            self.target = np.array([gx, gy], np.float32)
            self.speed = SHOT_SPEED + self.rng.uniform(-2, 4)
            self.receiver = None
            ev = [f"SHOT {self.names[team]} ({self._pname(team, kslot)})!"]
        else:                                               # ---- pass ----
            rslot = int(self.rng.choice(N_SLOTS, p=pr / pr.sum()))
            recv = rslot if team == 0 else N_SLOTS + rslot
            dist = float(np.linalg.norm(pos[recv] - pos[c]))
            self.speed = float(np.clip(self.spd_a + self.spd_b * dist, *self.spd_clip))
            t_fly = dist / self.speed
            lead = pos[recv] + vel[recv] * min(t_fly, 1.5) * 0.7
            self.target = np.clip(lead, 0, [PITCH_L, PITCH_W]).astype(np.float32)
            self.receiver = recv
            ev = [f"PASS {self.names[team]} {self._pname(team, kslot)} -> "
                  f"{self._pname(team, rslot)}"]
        self.mode = "flight"
        self.kicker = c
        self.carrier = None
        self.cooldown = COOLDOWN_STEPS
        self.tried.clear()
        return ev

    def _step_flight(self, pos, pres):
        to = self.target - self.ball
        dist = float(np.linalg.norm(to))
        step = self.speed * self.dt
        u = to / max(dist, 1e-6)
        new = self.ball + u * min(step, dist)
        shot = self.receiver is None

        # goal check (shots only): does the segment cross the goal line?
        if shot:
            gx = self.target[0]
            crossed = (self.ball[0] <= gx <= new[0]) if gx >= PITCH_L else \
                      (new[0] <= gx <= self.ball[0])
            if crossed and abs(self.target[1] - PITCH_W / 2) <= GOAL_HALF_W and dist <= step:
                team = self.team_of(self.kicker)
                self.score[team] += 1
                self.mode, self.carrier = "goal", None
                self.last_goal_team = team
                self.ball = new
                return [f"GOAL {self.names[team]}!  {self.score[0]}-{self.score[1]}"]

        # interception along the path
        n_sub = max(1, int(step / 0.8))
        for i in range(1, n_sub + 1):
            p = self.ball + u * (min(step, dist) * i / n_sub)
            if not (0 < p[0] < PITCH_L and 0 < p[1] < PITCH_W):
                break                          # rest of the path is out of bounds
            dd = np.linalg.norm(pos - p, axis=1)
            dd[pres <= 0.2] = 1e9
            if self.kicker is not None and self.cooldown > 0:
                dd[self.kicker] = 1e9
            cand = set(np.where(dd < INTERCEPT_R)[0].tolist())
            if shot:                                       # defending GK has real reach
                for g in (0, N_SLOTS):
                    if self.team_of(g) != self.team_of(self.kicker) and dd[g] < GK_SAVE_R:
                        cand.add(g)
            for j in sorted(cand):
                if j in self.tried:
                    continue
                self.tried.add(int(j))
                is_recv = self.receiver is not None and j == self.receiver
                is_save = shot and j in (0, N_SLOTS)
                prob = P_SAVE if is_save else \
                    (P_INTERCEPT_NEAR if dd[j] < 0.6 else P_INTERCEPT_FAR)
                if is_recv or self.rng.random() < prob:
                    self.ball = p.astype(np.float32)
                    self._capture(j)
                    if is_recv:
                        return []
                    side = self.names[self.team_of(int(j))]
                    return [f"SAVE by {side} GK"] if is_save else [f"INTERCEPTED by {side}"]
        self.cooldown = max(0, self.cooldown - 1)

        # out of bounds -> restart BEFORE the ball is ever placed outside
        # (writing the clipped position first is what flashed it into a corner)
        if (new[0] <= 0 or new[0] >= PITCH_L or new[1] <= 0 or new[1] >= PITCH_W):
            return self._out_restart(new)
        self.ball = new
        self.bvel = u * self.speed

        if dist <= step:                                    # arrived
            dd = np.linalg.norm(pos - self.ball, axis=1)
            dd[pres <= 0.2] = 1e9
            j = int(dd.argmin())
            if dd[j] < CAPTURE_R + 1.0:
                self._capture(j)
                return []
            self.mode = "loose"
            self.bvel = u * min(self.speed * 0.4, 6.0)
            return []
        return []

    def _step_loose(self, pos, pres):
        nb = self.ball + self.bvel * self.dt
        if (nb[0] <= 0 or nb[0] >= PITCH_L or nb[1] <= 0 or nb[1] >= PITCH_W):
            return self._out_restart(nb)           # never place the ball outside
        self.ball = nb
        self.bvel *= 0.88
        dd = np.linalg.norm(pos - self.ball, axis=1)
        dd[pres <= 0.2] = 1e9
        j = int(dd.argmin())
        if dd[j] < CAPTURE_R:
            self._capture(j)
            return [f"RECOVERY {self.names[self.team_of(j)]}"]
        return []

    def _out_restart(self, exit_point):
        """Ball went out: it STAYS at the exit point (clamped just inside) and
        the non-touching team's nearest player walks over to collect it -- a
        throw-in-style restart with zero teleporting."""
        self.mode, self.carrier = "restart", None
        self.restart_team = 1 - self.last_touch_team
        self.ball = np.clip(exit_point, 0.5, [PITCH_L - 0.5, PITCH_W - 0.5]).astype(np.float32)
        self.bvel[:] = 0
        return [f"OUT -- {self.names[self.restart_team]} ball"]

    def _step_restart(self, pos, pres):
        """Ball is dead at the touchline; only the restart team may collect."""
        sl = slice(0, N_SLOTS) if self.restart_team == 0 else slice(N_SLOTS, 2 * N_SLOTS)
        dd = np.linalg.norm(pos[sl] - self.ball, axis=1)
        dd[pres[sl] <= 0.2] = 1e9
        j = int(dd.argmin()) + (0 if self.restart_team == 0 else N_SLOTS)
        if dd.min() < CAPTURE_R:
            self._capture(j)
            self.cooldown = COOLDOWN_STEPS * 2
            return [f"RESTART {self.names[self.restart_team]}"]
        return []
