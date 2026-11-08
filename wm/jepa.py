"""Entity-token JEPA world model for WxMARL, trained under CTDE, and its planner interface.

Tokens. A frame is a set of entity tokens: one per aircraft slot and one per weather-cell slot.
Aircraft tokens carry the aircraft's own state (position, heading, speed, exit, alive). Cell tokens
carry what the radar showed at the last update (centre, radius, intensity, alive, radar age): the
*observable* weather. Nothing identifies an aircraft but its state and the action attached to it,
so the model is permutation-equivariant over aircraft and one set of weights serves any team size
(the design of the VMAS entity-token model).

CTDE. Training uses privileged *targets*: a state head decodes, from every encoded and predicted
latent, the true current aircraft features and the true current cells including their hidden
velocity and growth. The head is the planner's cost readout (``wm.cost.decoded_step_cost``): the
cost is the simulator's reward formula on decoded entities, never a latent distance (the Push-T
lesson). The predictor is conditioned on actions through an action embedding on each aircraft token.

Every aircraft's action is given to the predictor. At execution (``WorldModelDynamics``) each
decentralized planner gives its own action and its assumption about the others.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from config import EnvCfg
from core.geometry import bearing, wrap
from wm.config import DiagCfg
from wm.cost import decoded_step_cost, nearest_action
from wm.module import SIGReg, causal_frame_mask, mlp, transformer
from wm.snapshot import Snapshot

AIR_IN, CELL_IN, AIR_TGT, CELL_TGT = 8, 6, 8, 8


@dataclass
class Frame:
    """One decision step of ``B`` scenarios as token features."""

    air_in: Tensor  # [B, N, AIR_IN]  p/L, cos psi, sin psi, v/v_nom, e/L, alive
    cell_in: Tensor  # [B, K, CELL_IN] radar: q/L, rho/rho_max, I, alive, radar age (steps / radar_steps), 0
    air_tgt: Tensor  # [B, N, AIR_TGT] true, same layout as air_in
    cell_tgt: Tensor  # [B, K, CELL_TGT] true: q/L (2), rho/rho_max, I, u/v_nom (2), g/g0_hi, alive
    action_prev: Tensor  # [B, N] the action executed between the previous frame and this one

    def to(self, device) -> Frame:
        return Frame(**{k: v.to(device) for k, v in vars(self).items()})


def frame_from_snapshot(snap: Snapshot, cfg: EnvCfg) -> Frame:
    ac, cells, radar, w = snap.ac, snap.cells, snap.radar, cfg.weather
    L, v_nom = cfg.domain.L, cfg.aircraft.v_nom
    alive = ac.alive.unsqueeze(-1).float()
    air = torch.cat(
        (
            ac.p / L,
            torch.cos(ac.psi)[..., None],
            torch.sin(ac.psi)[..., None],
            (ac.v / v_nom)[..., None],
            ac.e / L,
            alive,
        ),
        -1,
    )
    age = torch.full_like(radar.rho, (snap.t % cfg.radar_steps) / cfg.radar_steps)
    r_alive = radar.alive.float()
    cell_in = (
        torch.stack((radar.q[..., 0] / L, radar.q[..., 1] / L, radar.rho / w.rho_max, radar.I, r_alive, age), -1)
        * r_alive[..., None]
    )
    c_alive = cells.alive.float()
    cell_tgt = (
        torch.stack(
            (
                cells.q[..., 0] / L,
                cells.q[..., 1] / L,
                cells.rho / w.rho_max,
                cells.I,
                cells.u[..., 0] / v_nom,
                cells.u[..., 1] / v_nom,
                cells.g / w.g0_hi,
            ),
            -1,
        )
        * c_alive[..., None]
    )
    cell_tgt = torch.cat((cell_tgt, c_alive[..., None]), -1)
    return Frame(
        air * alive, cell_in, air * alive, cell_tgt, nearest_action(ac.last_dpsi, cfg) * ac.alive + cfg.nominal_action * ~ac.alive
    )


class EntityJEPA(nn.Module):
    def __init__(self, m, n_actions: int) -> None:
        super().__init__()
        self.m, self.A = m, n_actions
        self.embed_air, self.embed_cell = mlp(AIR_IN, m.dim), mlp(CELL_IN, m.dim)
        self.type_emb = nn.Parameter(torch.randn(2, m.dim) * 0.02)
        self.encoder = transformer(m.dim, m.enc_layers, m.heads)
        self.proj = nn.Sequential(nn.LayerNorm(m.dim), nn.Linear(m.dim, m.latent))
        self.unproj = nn.Linear(m.latent, m.dim)
        self.act = nn.Embedding(n_actions + 1, m.dim)  # index n_actions: action unknown
        self.time = nn.Parameter(torch.randn(m.history, m.dim) * 0.02)
        self.predictor = transformer(m.dim, m.pred_layers, m.heads)
        self.head = nn.Sequential(nn.LayerNorm(m.dim), nn.Linear(m.dim, m.latent))
        self.state_head = nn.ModuleList([mlp(m.latent, AIR_TGT), mlp(m.latent, CELL_TGT)])
        self.sigreg = SIGReg()
        self.register_buffer("tgt_mean", torch.zeros(AIR_TGT + CELL_TGT), persistent=True)
        self.register_buffer("tgt_std", torch.ones(AIR_TGT + CELL_TGT), persistent=True)

    # ---------- encoder ----------
    def encode(self, air_in: Tensor, cell_in: Tensor) -> Tensor:
        """[..., N, AIR_IN], [..., K, CELL_IN] -> token latents [..., N + K, latent]."""
        lead, N = air_in.shape[:-2], air_in.shape[-2]
        x = torch.cat((self.embed_air(air_in.reshape(-1, N, AIR_IN)) + self.type_emb[0],
                       self.embed_cell(cell_in.reshape(-1, cell_in.shape[-2], CELL_IN)) + self.type_emb[1]), 1)  # fmt: skip
        return self.proj(self.encoder(x)).reshape(*lead, x.shape[1], -1)

    def decode(self, z: Tensor, N: int) -> tuple[Tensor, Tensor]:
        """Token latents [..., N + K, latent] -> decoded air [..., N, AIR_TGT], cells [..., K, CELL_TGT] (feature units)."""
        air = self.state_head[0](z[..., :N, :]) * self.tgt_std[:AIR_TGT] + self.tgt_mean[:AIR_TGT]
        cell = self.state_head[1](z[..., N:, :]) * self.tgt_std[AIR_TGT:] + self.tgt_mean[AIR_TGT:]
        return air, cell

    # ---------- predictor ----------
    def predict(self, z: Tensor, actions: Tensor, N: int) -> Tensor:
        """z [B, h, T, latent] latent history, actions [B, h, N] (index A = unknown) applied after each frame
        -> [B, T, latent] the next frame's token latents."""
        B, h, T, _ = z.shape
        x = self.unproj(z) + self.time[-h:, None]
        a = self.act(actions)  # [B, h, N, dim]
        x = torch.cat((x[:, :, :N] + a, x[:, :, N:]), 2)
        mask = causal_frame_mask(h, T, z.device)
        out = self.predictor(x.reshape(B, h * T, -1), mask=mask)
        return self.head(out.reshape(B, h, T, -1)[:, -1])

    def rollout(self, z_hist: Tensor, a_hist: Tensor, a_future: Tensor, N: int) -> Tensor:
        """z_hist [B, h, T, latent], a_hist [B, h - 1, N] actions between the history frames,
        a_future [B, H, N] -> predicted latents [B, H, T, latent]."""
        h = self.m.history
        z, a, out = z_hist, a_hist, []
        for k in range(a_future.shape[1]):
            a = torch.cat((a, a_future[:, k : k + 1]), 1)
            nz = self.predict(z[:, -h:], a[:, -h:], N)
            out.append(nz)
            z = torch.cat((z, nz[:, None]), 1)
        return torch.stack(out, 1)

    # ---------- training ----------
    def loss(self, f: dict[str, Tensor], gen: torch.Generator | None = None) -> tuple[Tensor, dict[str, float]]:
        """``f``: air_in [B, h+k, N, ·], cell_in [B, h+k, K, ·], air_tgt, cell_tgt, actions [B, h+k-1, N] (action
        after each frame), alive [B, N]. Prediction MSE in latent space + SIGReg + the state head on encoded and
        predicted latents against the privileged targets (z-scored per feature)."""
        h, k, N = self.m.history, self.m.k, f["air_in"].shape[2]
        z = self.encode(f["air_in"], f["cell_in"])  # [B, h+k, T, latent]
        acts = f["actions"]
        pred = (
            self.rollout(z[:, :h], acts[:, : h - 1], acts[:, h - 1 : h - 1 + k], N)
            if k > 1
            else self.predict(z[:, :h], acts[:, :h], N)[:, None]
        )
        tgt = z[:, h : h + k]
        l_pred = (pred - tgt).pow(2).mean()
        l_sig = self.sigreg(z.reshape(1, -1, z.shape[-1]))
        air_t = (f["air_tgt"] - self.tgt_mean[:AIR_TGT]) / self.tgt_std[:AIR_TGT]
        cell_t = (f["cell_tgt"] - self.tgt_mean[AIR_TGT:]) / self.tgt_std[AIR_TGT:]
        head_err = lambda zz, sl: ((self.state_head[0](zz[..., :N, :]) - air_t[:, sl]).pow(2).mean()
                                   + (self.state_head[1](zz[..., N:, :]) - cell_t[:, sl]).pow(2).mean())  # fmt: skip
        l_head = 0.5 * (head_err(z, slice(None)) + head_err(pred, slice(h, h + k)))
        loss = l_pred + self.m.sigreg_weight * l_sig + self.m.head_weight * l_head
        return loss, {"loss": loss.item(), "pred": l_pred.item(), "sigreg": l_sig.item(), "head": l_head.item()}


class WorldModelDynamics:
    """The learned model behind the same planner: candidate plans are scored on decoded entities."""

    futures = 1

    def __init__(self, model: EntityJEPA, diag: DiagCfg, env_cfg: EnvCfg, device) -> None:
        self.model, self.diag, self.cfg, self.device = model.to(device).eval(), diag, env_cfg, torch.device(device)
        self.h = model.m.history
        self.amp = bool(model.m.rollout_fp16) and self.device.type == "cuda"
        self.z_hist: Tensor | None = None  # [B, h, T, latent]
        self.a_hist: Tensor | None = None  # [B, h - 1, N]

    @classmethod
    def load(cls, diag: DiagCfg, env_cfg: EnvCfg, device) -> WorldModelDynamics:
        ckpt = torch.load(diag.checkpoint, map_location="cpu", weights_only=False)
        model = EntityJEPA(ckpt["model_cfg"], env_cfg.n_actions)
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, diag, env_cfg, device)

    @torch.no_grad()
    def observe(self, snap: Snapshot) -> None:
        """Push the current frame into the history (call once per decision step, before planning)."""
        f = frame_from_snapshot(snap, self.cfg).to(self.device)
        z = self.model.encode(f.air_in, f.cell_in)[:, None]  # [B, 1, T, latent]
        if self.z_hist is None or self.z_hist.shape[0] != z.shape[0] or snap.t == 0:
            self.z_hist = z.expand(-1, self.h, -1, -1).clone()
            self.a_hist = f.action_prev[:, None].expand(-1, self.h - 1, -1).clone()
        else:
            self.z_hist = torch.cat((self.z_hist[:, 1:], z), 1)
            self.a_hist = torch.cat((self.a_hist[:, 1:], f.action_prev[:, None]), 1)

    def _known(self, actions: Tensor, P: int, n: int, slots: Tensor | None) -> Tensor:
        """Mask the actions the model may see: the own model sees only the planning aircraft's."""
        if self.model.m.cond == "joint":
            return actions
        Bx, _, N = actions.shape
        if slots is None:  # joint planning with the own model: nothing is a single ego; show aircraft 0
            slots = torch.zeros(P, dtype=torch.long, device=actions.device)
        ego = slots.repeat_interleave(n).repeat(Bx // (P * n))  # [Bx]
        keep = torch.nn.functional.one_hot(ego, N).bool()[:, None, :]
        return torch.where(keep, actions, torch.full_like(actions, self.model.A))

    @torch.no_grad()
    def _rollout_cost(self, snap: Snapshot, actions: Tensor, direct_mask: Tensor | None) -> Tensor:
        """Per-step, per-aircraft decoded cost [Bx, H, N] of joint sequences ``actions`` (already masked).

        Rows are processed ``model.rollout_chunk`` at a time: the transformer's attention over
        ``history x tokens`` is the memory peak of a planning step.
        """
        Bx = actions.shape[0]
        reps = Bx // snap.B
        rep = lambda x: x.repeat_interleave(reps, 0)
        ac = snap.ac
        z, a = rep(self.z_hist), rep(self.a_hist)
        alive, e, last_dpsi = rep(ac.alive), rep(ac.e), rep(ac.last_dpsi)
        out = []
        for s in range(0, Bx, self.model.m.rollout_chunk):
            sl = slice(s, s + self.model.m.rollout_chunk)
            mask = None if direct_mask is None else direct_mask[sl]
            out.append(self._rollout_rows(actions[sl], mask, z[sl], a[sl], alive[sl], e[sl], last_dpsi[sl]))
        return torch.cat(out, 0)

    def _rollout_rows(
        self,
        actions: Tensor,
        direct_mask: Tensor | None,
        z: Tensor,
        a: Tensor,
        alive: Tensor,
        e: Tensor,
        last_dpsi: Tensor,
    ) -> Tensor:
        Bx, H, N = actions.shape
        L, w = self.cfg.domain.L, self.cfg.weather
        offs = torch.deg2rad(torch.tensor(self.cfg.action.heading_offsets_deg, device=self.device))
        n_speed = len(self.cfg.action.speed_deltas_kt)
        air, cell = self.model.decode(z[:, -1], N)
        p, psi = air[..., :2] * L, torch.atan2(air[..., 3], air[..., 2])
        cost = torch.zeros(Bx, H, N, device=self.device)
        for h in range(H):
            a_h = actions[:, h]
            if direct_mask is not None:  # the direct policy on the decoded state
                a_h = torch.where(direct_mask, nearest_action(wrap(bearing(p, e) - psi), self.cfg), a_h)
            a = torch.cat((a, a_h[:, None]), 1)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
                nz = self.model.predict(z[:, -self.h :], a[:, -self.h :], N).float()
            z = torch.cat((z, nz[:, None]), 1)
            air, cell = self.model.decode(nz, N)
            p_next, psi = air[..., :2] * L, torch.atan2(air[..., 3], air[..., 2])
            dpsi = offs[torch.where(a_h >= self.model.A, 0, a_h) // n_speed] * (a_h < self.model.A) + last_dpsi * (
                a_h >= self.model.A
            )
            q, rho, I, c_alive = (
                cell[..., :2] * L,
                cell[..., 2] * w.rho_max,
                cell[..., 3].clamp(0, 1),
                cell[..., 7] > 0.5,
            )
            cost[:, h] = decoded_step_cost(self.cfg, p, p_next, alive, e, q, rho, I, c_alive, dpsi)
            p = p_next
        return cost

    def evaluate(self, snap: Snapshot, actions: Tensor, P: int, n: int, direct_mask: Tensor | None = None,
                 slots: Tensor | None = None, seed: int | None = None) -> Tensor:  # fmt: skip
        """Same contract as ``Oracle.evaluate`` with F = 1: cost [B, 1, P, n, H, N]."""
        B, H, N = snap.B, actions.shape[1], actions.shape[2]
        return self._rollout_cost(snap, actions, direct_mask).view(B, 1, P, n, H, N)
