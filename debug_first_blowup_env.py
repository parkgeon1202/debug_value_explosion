"""
debug_first_blowup_env.py  (v3)
========================================================================
4096 env 중 '처음' action 이 임계를 넘는 env 한 마리를 잠그고, 그 env 의
폭발 직전(ring)·직후 궤적을 추적한다.

v3 변경점:
  1) FB_WARN 기본값 상향 (정상 보행 action 출렁임을 건너뛰고 '진짜 폭발'만). 기본 100.
  2) 물리값을 못 읽으면 nan 대신 'none' 으로 명시 출력 (계산실패 nan 과 구분).
  3) 방식 B 추가: 자동미분으로 '순수' d(action)/d(last_action) (= 진짜 루프 게인)
     을 측정. 방식 A(차분 dA/dL)는 다른 obs 변화가 섞여 부정확하므로 B 로 확정.

측정값:
  GANG  : (방식B) d(action)/d(last_action) 자코비안 최대 절대성분. 순수 게인. 신뢰지표.
  dA/dL : (방식A) |Δaction|/|Δlast_action|. 근사(다른 obs 섞임).
  dA/dO : (방식A) |Δaction|/|Δobs전체|.

사용법:  train.py 에:  import debug_first_blowup_env   # noqa
환경변수:
  FB_WARN=100  FB_RING=60  FB_AFTER=60  FB_DIM=0(자동)  FB_JAC=1(방식B on)
"""

import os
import collections
import torch

FB_WARN  = float(os.environ.get("FB_WARN", "100"))
FB_RING  = int(float(os.environ.get("FB_RING", "60")))
FB_AFTER = int(float(os.environ.get("FB_AFTER", "60")))
FB_DIM   = int(float(os.environ.get("FB_DIM", "0")))
FB_JAC   = int(float(os.environ.get("FB_JAC", "1")))
FB_SKIP    = int(float(os.environ.get("FB_SKIP", "0")))      # 이 step 이전엔 감지 안 함(초기 스파이크 무시)
FB_REPEAT  = int(float(os.environ.get("FB_REPEAT", "0")))    # 1이면 한 번 추적 후 다시 감시(반복). 0이면 1회.
FB_PUSHWIN = int(float(os.environ.get("FB_PUSHWIN", "0")))   # >0이면 'push 발생 후 이 step 이내'에 터진 것만 감지
FB_COOL    = int(float(os.environ.get("FB_COOL", "200")))    # 반복 모드에서 추적 종료 후 재감시까지 쿨다운

_TERMS_45 = [
    ("proj_grav",  0,  3), ("vel_cmd",    3,  6), ("joint_pos",  6, 18),
    ("joint_vel", 18, 30), ("imu_ang_v", 30, 33), ("last_act",  33, 45),
]
_TERMS_42 = [
    ("vel_cmd",    0,  3), ("joint_pos",  3, 15), ("joint_vel", 15, 27),
    ("imu_ang_v", 27, 30), ("last_act",  30, 42),
]
_TERMS = {"cur": _TERMS_45, "dim": 45, "lact": (33, 45)}

_STEP = {"n": 0}
_RING = collections.deque(maxlen=FB_RING)
_LOCK = {"env": None, "left": 0, "done": False, "cool": 0}
_PUSH = {"step": -10, "ids": None, "mag": float("nan")}
_PREV = {"pobs": None, "act": None}
_GANG_DBG = {"printed": False}


def _resolve_dim(d):
    if FB_DIM in (42, 45):
        d = FB_DIM
    if d == 42:
        _TERMS["cur"], _TERMS["dim"], _TERMS["lact"] = _TERMS_42, 42, (30, 42)
    else:
        _TERMS["cur"], _TERMS["dim"], _TERMS["lact"] = _TERMS_45, 45, (33, 45)


def _fmtp(x):
    if x is None:
        return "none"
    try:
        if x != x:
            return "nan"
        return f"{x:.2e}"
    except Exception:
        return "none"


def _sens_A(snap, prev, e):
    if prev is None or prev.get("pobs") is None:
        return float("nan"), float("nan")
    try:
        lo, hi = _TERMS["lact"]
        d_act = (snap["act"][e] - prev["act"][e]).abs().max().item()
        d_lact = (snap["pobs"][e, lo:hi] - prev["pobs"][e, lo:hi]).abs().max().item()
        d_obs = (snap["pobs"][e] - prev["pobs"][e]).abs().max().item()
        eps = 1e-6
        return d_act / (d_lact + eps), d_act / (d_obs + eps)
    except Exception:
        return float("nan"), float("nan")


def _gain_jacobian(policy, full_obs, e):
    """d(actor mean)/d(last_action) 의 최대 절대성분 + frob norm.
    여러 호출 방식을 시도하고, 첫 실패 시 원인을 1회 출력한다(_GANG_DBG).
    """
    if not FB_JAC:
        return float("nan"), float("nan")
    lo, hi = _TERMS["lact"]
    try:
        with torch.enable_grad():
            # 1) actor 입력 obs 구성: get_actor_obs 가 있으면 쓰고, 없으면 full_obs 가
            #    이미 actor obs(tensor)라고 가정.
            if hasattr(policy, "get_actor_obs"):
                aobs = policy.get_actor_obs(full_obs)
            else:
                aobs = full_obs
            aobs = aobs.detach()
            x = aobs[e:e + 1].clone().requires_grad_(True)

            # 2) 정규화 통과(있으면)
            if hasattr(policy, "actor_obs_normalizer"):
                xn = policy.actor_obs_normalizer(x)
            else:
                xn = x

            # 3) actor mean 계산: 여러 경로 시도
            mean = None
            if hasattr(policy, "actor") and isinstance(policy.actor, torch.nn.Module):
                mean = policy.actor(xn)
            elif hasattr(policy, "evaluate"):
                mean = policy.evaluate(xn)
            else:
                # 최후: act_inference 류 (단 grad 유지 필요)
                mean = policy.act_inference(xn) if hasattr(policy, "act_inference") else None
            if mean is None:
                raise RuntimeError("no actor forward path found")
            if not mean.requires_grad:
                raise RuntimeError("mean has no grad (actor path detached)")

            A = mean.shape[-1]
            gmax = 0.0
            gsq = 0.0
            for a in range(A):
                g = torch.autograd.grad(mean[0, a], x, retain_graph=(a < A - 1),
                                        create_graph=False, allow_unused=True)[0]
                if g is None:
                    continue
                gla = g[0, lo:hi]
                gmax = max(gmax, gla.abs().max().item())
                gsq += float((gla * gla).sum().item())
            return gmax, gsq ** 0.5
    except Exception as ex:
        if not _GANG_DBG["printed"]:
            import traceback
            print(f"[fb][GANG 진단] 자동미분 실패 원인: {type(ex).__name__}: {ex}")
            print(f"[fb][GANG 진단] policy 타입: {type(policy).__name__}, "
                  f"get_actor_obs={hasattr(policy,'get_actor_obs')}, "
                  f"actor={hasattr(policy,'actor')}, "
                  f"normalizer={hasattr(policy,'actor_obs_normalizer')}")
            _GANG_DBG["printed"] = True
        return float("nan"), float("nan")



def _row_for_env(snap, e, prev=None, policy=None, full_obs=None):
    pobs = snap["pobs"]; act = snap["act"]
    parts = []
    for name, lo, hi in _TERMS["cur"]:
        try:
            v = pobs[e, lo:hi].abs().max().item()
        except Exception:
            v = float("nan")
        parts.append(f"{name}={v:.2e}")
    try:
        a = act[e].abs().max().item()
    except Exception:
        a = float("nan")
    dA_dL, dA_dO = _sens_A(snap, prev, e)
    if policy is not None and full_obs is not None:
        gang, gnorm = _gain_jacobian(policy, full_obs, e)
    else:
        gang, gnorm = float("nan"), float("nan")
    rv = snap["root_v"][e].item() if snap["root_v"] is not None else None
    av = snap["ang_v"][e].item() if snap["ang_v"] is not None else None
    ia = snap["imu_a"][e].item() if snap["imu_a"] is not None else None
    push = "*" if (snap["push_ids"] is not None and e in snap["push_ids"]) else " "
    done = "X" if (snap["dones"] is not None and bool(snap["dones"][e])) else " "
    gflag = "  <<<GANG>1" if (gang == gang and gang > 1.0) else \
            ("  <<<dA/dL>1" if (gang != gang and dA_dL == dA_dL and dA_dL > 1.0) else "")
    return (f"  s{snap['step']:>8d} p{push} d{done} | " + " ".join(parts) +
            f" -> ACT={a:.3e} | GANG={_fmtp(gang)} dA/dL={dA_dL:.2f} dA/dO={dA_dO:.2f}"
            f" | root_v={_fmtp(rv)} ang_v={_fmtp(av)} imu_a={_fmtp(ia)}{gflag}")


def _dump_locked(e, trigger_step):
    print(f"\n[fb] ===== env {e} first blowup (step={trigger_step}, thr={FB_WARN}) =====")
    print(f"[fb] past {len(_RING)} steps (GANG n/a) + next {FB_AFTER} steps (GANG on).")
    print(f"[fb] GANG=d(action)/d(last_action) pure gain(B,trust) | dA/dL=diff approx(A) | phys none=read fail")
    print(f"[fb] note: GANG > 1 = true divergence region. watch if GANG exceeds 1 after push.")
    print("  " + "-" * 138)
    prev = None
    for snap in _RING:
        print(_row_for_env(snap, e, prev))
        prev = snap
    _PREV["pobs"] = _RING[-1]["pobs"] if len(_RING) else None
    _PREV["act"] = _RING[-1]["act"] if len(_RING) else None
    print(f"  {'-'*48} > past / future {'-'*48}")


def _grab_phys(env):
    out = {"root_v": None, "ang_v": None, "imu_a": None}
    try:
        robot = env.scene["robot"]
        try:
            out["root_v"] = robot.data.root_lin_vel_w.detach().norm(dim=-1)
        except Exception:
            pass
        try:
            out["ang_v"] = robot.data.root_ang_vel_w.detach().norm(dim=-1)
        except Exception:
            pass
    except Exception:
        pass
    try:
        imu = env.scene["imu"]
        for attr in ("lin_acc_b", "lin_acc_w", "lin_acc"):
            if hasattr(imu.data, attr):
                out["imu_a"] = getattr(imu.data, attr).detach().norm(dim=-1)
                break
    except Exception:
        pass
    return out


def _install_push_hook():
    try:
        import functools
        import isaaclab.envs.mdp.events as ev
        if not hasattr(ev, "push_by_setting_velocity"):
            print("[fb] x push func not found")
            return False
        _orig = ev.push_by_setting_velocity

        @functools.wraps(_orig)
        def _patched(*args, **kwargs):
            try:
                _PUSH["step"] = _STEP["n"]
                env_ids = args[1] if len(args) > 1 else kwargs.get("env_ids")
                _PUSH["ids"] = set(env_ids.detach().tolist()) if env_ids is not None else None
            except Exception:
                _PUSH["ids"] = None
            try:
                vr = args[2] if len(args) > 2 else kwargs.get("velocity_range", {})
                xr = vr.get("x", (0, 0)); yr = vr.get("y", (0, 0))
                _PUSH["mag"] = max(abs(xr[0]), abs(xr[1]), abs(yr[0]), abs(yr[1]))
            except Exception:
                _PUSH["mag"] = float("nan")
            return _orig(*args, **kwargs)

        ev.push_by_setting_velocity = _patched
        print("[fb] v push hook ok")
        return True
    except Exception as e:
        print(f"[fb] x push hook fail: {e}")
        return False


def _install_act_hook():
    try:
        from rsl_rl.algorithms.ppo import PPO
        _orig_act = PPO.act

        def _patched_act(self, obs, *a, **kw):
            actions = _orig_act(self, obs, *a, **kw)
            try:
                _STEP["n"] += 1
                step = _STEP["n"]
                if _LOCK["done"]:
                    return actions

                pobs = self.policy.get_actor_obs(obs).detach()
                if _TERMS["dim"] != pobs.shape[-1]:
                    _resolve_dim(pobs.shape[-1])

                env = getattr(self, "env", None) or _ENV_REF.get("env")
                phys = _grab_phys(env) if env is not None else \
                    {"root_v": None, "ang_v": None, "imu_a": None}

                dones = None
                try:
                    dones = env.termination_manager.dones.detach()
                except Exception:
                    pass

                push_ids = _PUSH["ids"] if _PUSH["step"] >= step - 1 else None
                snap = {"step": step, "pobs": pobs, "act": actions.detach(),
                        "push_ids": push_ids, "dones": dones, **phys}

                if _LOCK["env"] is not None:
                    e = _LOCK["env"]
                    prev = {"pobs": _PREV["pobs"], "act": _PREV["act"]}
                    print(_row_for_env(snap, e, prev, policy=self.policy, full_obs=obs))
                    _PREV["pobs"] = snap["pobs"]; _PREV["act"] = snap["act"]
                    _LOCK["left"] -= 1
                    if _LOCK["left"] <= 0:
                        print("  " + "-" * 138)
                        print(f"[fb] ===== env {e} trace end (step={step}) =====\n")
                        if FB_REPEAT:
                            # 반복 모드: 잠금 해제하고 쿨다운 후 재감시
                            _LOCK["env"] = None
                            _LOCK["cool"] = FB_COOL
                            _PREV["pobs"] = None; _PREV["act"] = None
                        else:
                            _LOCK["done"] = True
                    return actions

                _RING.append(snap)
                # 쿨다운 중이면 감지 보류(반복 모드)
                if _LOCK.get("cool", 0) > 0:
                    _LOCK["cool"] -= 1
                    return actions
                # 초기 구간 무시
                if step < FB_SKIP:
                    return actions
                # push 창 모드: 최근 push 후 FB_PUSHWIN step 이내만 감지
                if FB_PUSHWIN > 0 and not (_PUSH["step"] >= step - FB_PUSHWIN):
                    return actions
                act_abs = actions.detach().abs().amax(dim=-1)
                over = (act_abs > FB_WARN).nonzero(as_tuple=True)[0]
                if over.numel() > 0:
                    e = int(over[act_abs[over].argmax()].item())
                    _LOCK["env"] = e
                    _LOCK["left"] = FB_AFTER
                    _dump_locked(e, step)
            except Exception as ex:
                print(f"[fb] err(ignored): {ex}")
            return actions

        PPO.act = _patched_act
        print(f"[fb] v act hook ok (WARN={FB_WARN}, RING={FB_RING}, AFTER={FB_AFTER}, JAC={FB_JAC})")
        return True
    except Exception as e:
        print(f"[fb] x act hook fail: {e}")
        return False


_ENV_REF = {"env": None}
def _install_env_ref():
    try:
        from rsl_rl.runners.on_policy_runner import OnPolicyRunner
        _orig_init = OnPolicyRunner.__init__
        def _patched_init(self, env, *a, **kw):
            _ENV_REF["env"] = env
            return _orig_init(self, env, *a, **kw)
        OnPolicyRunner.__init__ = _patched_init
    except Exception:
        pass


_install_env_ref()
_install_push_hook()
_ok = _install_act_hook()
if _ok:
    print(f"[fb] ready. FB_WARN={FB_WARN} FB_SKIP={FB_SKIP} FB_REPEAT={FB_REPEAT} "
          f"FB_PUSHWIN={FB_PUSHWIN} FB_AFTER={FB_AFTER} FB_JAC={FB_JAC}")
    print(f"[fb] >>> 위 값이 네가 준 환경변수와 같은지 확인! 다르면 새 파일이 안 옮겨졌거나 env var 미적용.")
