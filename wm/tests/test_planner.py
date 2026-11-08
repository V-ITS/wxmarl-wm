"""The planner on tiny sizes: shapes, legality and that planning beats flying blind on a forced conflict."""

import torch

from env import WxEnv
from wm.config import MOCK, load
from wm.oracle import Oracle
from wm.planner import Planner
from wm.snapshot import take
from wm.weather import WeatherForecaster, _radar_target, local_view, simulate


def _setup(mode, assume="shared", info="clairvoyant", N=2, B=2):
    diag = load(MOCK, {"n_agents": N, "planner": {"mode": mode, "assume": assume, "candidates": 16, "iters": 2, "horizon": 4},
                       "oracle": {"info": info, "futures": 2}})  # fmt: skip
    cfg = diag.env_cfg()
    env = WxEnv(cfg, batch_size=B)
    env.reset(seed=3)
    planner = Planner(diag, cfg, Oracle(diag, cfg), "cpu")
    planner.reset(B, 0)
    return cfg, env, planner


def test_joint_and_dec_return_legal_actions():
    for mode, assume in (("joint", "shared"), ("dec", "shared"), ("dec", "direct"), ("dec", "priority")):
        cfg, env, planner = _setup(mode, assume)
        a = planner.act(take(env))
        assert a.shape == (env.B, env.N) and a.dtype == torch.long
        assert int(a.min()) >= 0 and int(a.max()) < cfg.n_actions
        assert planner.plan.shape == (env.B, planner.H, env.N)


def test_belief_levels_run():
    for info in ("perfect", "gstale", "gpool"):
        _, env, planner = _setup("dec", "shared", info)
        a = planner.act(take(env))
        assert a.shape == (env.B, env.N)


def _llearned(tmp_path, assume):
    diag = load(MOCK, {"n_agents": 3, "planner": {"mode": "dec", "assume": assume, "candidates": 8, "iters": 2, "horizon": 4},
                       "oracle": {"info": "llearned", "weather_ckpt": str(tmp_path / "wx.pt")}})  # fmt: skip
    cfg = diag.env_cfg()
    model = WeatherForecaster(diag.forecast, cfg)
    torch.save({"cfg": diag.forecast, "state_dict": model.state_dict(), "epoch": 0}, tmp_path / "wx.pt")
    env = WxEnv(cfg, batch_size=2)
    env.reset(seed=3)
    planner = Planner(diag, cfg, Oracle(diag, cfg), "cpu")
    planner.reset(2, 0)
    return cfg, env, planner


def test_llearned_plans_decentralized(tmp_path):
    for assume in ("direct", "priority"):
        _, env, planner = _llearned(tmp_path, assume)
        assert planner.act(take(env)).shape == (env.B, env.N)


def test_llearned_feeds_each_population_its_own_view(tmp_path):
    cfg, env, planner = _llearned(tmp_path, "direct")
    snap, seen = take(env), []
    forecast = planner.dyn.weather.forecast

    def spy(radar, prev, age, c):
        seen.append(radar.alive.clone())
        return forecast(radar, prev, age, c)

    planner.dyn.weather.forecast = spy
    a = torch.full((env.B * env.N * 2, planner.H, env.N), cfg.nominal_action)
    cost = planner.dyn.evaluate(snap, a, env.N, 2, slots=torch.arange(env.N))
    assert cost.shape == (env.B, 1, env.N, 2, planner.H, env.N)
    for b in range(env.B):
        for i in range(env.N):
            p = snap.ac.p[b, i].unsqueeze(0)
            one = lambda c: type(c)(**{k: v[b : b + 1] for k, v in vars(c).items()})
            view, _ = local_view(one(snap.radar), one(snap.radar_prev), p, cfg.obs.sense_range_nm)
            assert bool((seen[0][b * env.N + i] == view.alive[0]).all())


def test_radar_targets_use_later_pictures_not_the_truth():
    diag = load(MOCK, {})
    cfg = diag.env_cfg()
    R = cfg.radar_steps
    hist = simulate(cfg, 4, 3 * R + 2, 7)
    radar = hist[R + 2][1]  # the picture of step R, 2 steps old
    q, rho, _, same = _radar_target(hist, radar, R, 2 * R, cfg)
    pic = hist[2 * R][1]
    assert bool((q == pic.q).all()) and bool((rho == pic.rho).all())  # at a refresh the target is that picture
    assert not bool((hist[2 * R - 1][0].q == pic.q).all())  # and the truth has moved on either side of it
    assert same.shape == radar.alive.shape


def test_local_view_drops_far_cells_on_each_picture():
    cfg, env, _ = _setup("dec", "direct")
    snap = take(env)
    far = torch.full((env.B, 2), 1e6)
    radar, prev = local_view(snap.radar, snap.radar_prev, far, cfg.obs.sense_range_nm)
    assert not radar.alive.any() and not prev.alive.any()
    assert snap.radar.alive.any()  # the snapshot itself is untouched


def test_best_responses_to_a_joint_plan_have_the_joint_plans_shape():
    _, env, planner = _setup("joint")
    snap = take(env)
    a_star, c_star = planner.joint_plan(snap)
    a_br, c_br = planner.best_responses(snap, a_star, None, a_star)
    assert a_star.shape == a_br.shape == (env.B, planner.H, env.N)
    assert c_star.shape == (env.B,) and c_br.shape == (env.B, env.N)


def test_joint_plan_costs_no_more_than_flying_direct_on_the_same_stream():
    cfg, env, planner = _setup("joint", N=3)
    snap = take(env)
    a_star, _ = planner.joint_plan(snap)
    direct = torch.full_like(a_star, cfg.nominal_action)
    both = torch.stack((a_star, direct), 1).reshape(env.B * 2, planner.H, env.N)
    c = planner.dyn.evaluate(snap, both, 1, 2).sum((-1, -2)).view(env.B, 2)
    assert bool((c[:, 0] <= c[:, 1] + 1e-4).all()), c
