"""
debug_first_blowup_env.py  (v2: 민감도 ① 측정 추가)
========================================================================
"4096개 env 중 '처음으로' action 이 임계를 넘는 env 하나를 감지 → 그 env id 를
잠그고 → 그 env 의 폭발 '직전' 궤적과 '직후' 궤적을 따라간다."
+ 추가: 그 env 의 매 스텝 '정책 민감도 ①' 를 같이 찍는다.

민감도 ① (방식 A — 스텝 간 비율):
    d_act_d_lact = |action(t) - action(t-1)|_max / |last_act(t) - last_act(t-1)|_max
        → 이게 곧 ∂action/∂last_action 의 근사 = "루프 게인".
          1 을 넘으면 'last_action 이 조금 변할 때 action 이 더 크게 변한다'
          = 자기증폭(발산) 영역. push 직후 이 값이 1 을 넘어 커지는지 본다.

    d_act_d_obs = |Δaction|_max / |Δ(policy obs 전체)|_max
        → 전체 입력 변화 대비 출력 변화. 작은 입력 변화에 출력이 튀면 크다.

    ※ 방식 A 는 '근사'다(두 스텝의 차분). 정확한 야코비안(자동미분)은 방식 B 로
      별도. 먼저 A 로 1 넘는 게 보이면 B 로 확정하는 순서를 권장.

읽는 법:
    잠긴 env 한 마리의 시간축 표. 폭발(▶) 기준 위(과거)/아래(이후).
      - dA/dL (=d_act_d_lact) 이 push 직후 1 을 넘어 커지나? → 민감영역 진입 = 가설 확정
      - dA/dL 이 매 스텝 일정? → 일정 게인의 기하급수 발산
      - 입력(joint_*/imu_*)은 작게 변하는데 ACT 만 크게? → 출력 증폭(민감)
      - push=★ 가 dA/dL 상승 몇 스텝 전인가? → push→민감화 지연

────────────────────────────────────────────────────────────────────────
사용법:
    train.py 에:  import debug_first_blowup_env   # noqa

환경변수:
    FB_WARN  = 5.0   action |max| 가 이 값 넘으면 "첫 폭발"로 감지·잠금
    FB_RING  = 60    폭발 감지 시 되짚을 직전 스텝 수
    FB_AFTER = 60    폭발 감지 후 따라갈 스텝 수
    FB_DIM   = 0     policy obs 차원 강제(0=자동). 45(proj_grav 활성) 또는 42.
"""

import os
import collections
import torch

FB_WARN  = float(os.environ.get("FB_WARN", "5.0"))
FB_RING  = int(float(os.environ.get("FB_RING", "60")))
FB_AFTER = int(float(os.environ.get("FB_AFTER", "60")))
FB_DIM   = int(float(os.environ.get("FB_DIM", "0")))

# policy obs 항목 경계 — 45차원(projected_gravity 활성) 기준.
_TERMS_45 = [
    ("proj_grav",  0,  3),
    ("vel_cmd",    3,  6),
    ("joint_pos",  6, 18),
    ("joint_vel", 18, 30),
    ("imu_ang_v", 30, 33),
    ("last_act",  33, 45),
]
_TERMS_42 = [
    ("vel_cmd",    0,  3),
    ("joint_pos",  3, 15),
    ("joint_vel", 15, 27),
    ("imu_ang_v", 27, 30),
    ("last_act",  30, 42),
]
# lact_lo/hi: last_action 의 obs 내 인덱스 (민감도 분모용)
_TERMS = {"cur": _TERMS_45, "dim": 45, "lact": (33, 45)}

_STEP = {"n": 0}
_RING = collections.deque(maxlen=FB_RING)
_LOCK = {"env": None, "left": 0, "done": False}
_PUSH = {"step": -10, "ids": None, "mag": float("nan")}
# 잠긴 env 의 직전 스냅샷(민감도 차분용)
_PREV = {"pobs": None, "act": None}


def _resolve_dim(d):
    if FB_DIM in (42, 45):
        d = FB_DIM
    if d == 42:
        _TERMS["cur"], _TERMS["dim"], _TERMS["lact"] = _TERMS_42, 42, (30, 42)
    else:
        _TERMS["cur"], _TERMS["dim"], _TERMS["lact"] = _TERMS_45, 45, (33, 45)


def _sens(snap, prev, e):
    """env e 의 스텝 간 민감도 (dA/dL, dA/dObs). prev 없으면 nan."""
    if prev is None or prev.get("pobs") is None:
        return float("nan"), float("nan")
    try:
        lo, hi = _TERMS["lact"]
        d_act = (snap["act"][e] - prev["act"][e]).abs().max().item()
        d_lact = (snap["pobs"][e, lo:hi] - prev["pobs"][e, lo:hi]).abs().max().item()
        d_obs = (snap["pobs"][e] - prev["pobs"][e]).abs().max().item()
        eps = 1e-6
        dA_dL = d_act / (d_lact + eps)
        dA_dO = d_act / (d_obs + eps)
        return dA_dL, dA_dO
    except Exception:
        return float("nan"), float("nan")


def _row_for_env(snap, e, prev=None):
    pobs = snap["pobs"]
    act = snap["act"]
    terms = _TERMS["cur"]
    parts = []
    for name, lo, hi in terms:
        try:
            v = pobs[e, lo:hi].abs().max().item()
        except Exception:
            v = float("nan")
        parts.append(f"{name}={v:.2e}")
    try:
        a = act[e].abs().max().item()
    except Exception:
        a = float("nan")
    dA_dL, dA_dO = _sens(snap, prev, e)
    rv = snap["root_v"][e].item() if snap["root_v"] is not None else float("nan")
    av = snap["ang_v"][e].item() if snap["ang_v"] is not None else float("nan")
    ia = snap["imu_a"][e].item() if snap["imu_a"] is not None else float("nan")
    push = "★" if (snap["push_ids"] is not None and e in snap["push_ids"]) else " "
    done = "✗" if (snap["dones"] is not None and bool(snap["dones"][e])) else " "
    flag = "  <<<게인>1" if (dA_dL == dA_dL and dA_dL > 1.0) else ""  # nan-safe
    return (f"  s{snap['step']:>8d} p{push} d{done} | " + " ".join(parts) +
            f" → ACT={a:.3e} | dA/dL={dA_dL:.2f} dA/dO={dA_dO:.2f}"
            f" | root_v={rv:.2e} ang_v={av:.2e} imu_a={ia:.2e}{flag}")


def _dump_locked(e, trigger_step):
    print(f"\n[fb] ===== env {e} 첫 폭발 감지 (step={trigger_step}, 임계 {FB_WARN}) =====")
    print(f"[fb] 이 env 한 마리의 과거 {len(_RING)}스텝 + 이후 {FB_AFTER}스텝.")
    print(f"[fb] p★=push  d✗=done | 각 obs단계 |max| → ACT | "
          f"dA/dL=Δaction/Δlast_action(민감도①,루프게인) dA/dO=Δaction/Δobs | 물리")
    print(f"[fb] ※ dA/dL > 1 이면 '민감영역(발산)'. push 직후 이게 1 넘어 커지는지 보세요.")
    print("  " + "-" * 128)
    prev = None
    for snap in _RING:
        print(_row_for_env(snap, e, prev))
        prev = snap
    # 이후 추적용 prev 시드
    _PREV["pobs"] = _RING[-1]["pobs"] if len(_RING) else None
    _PREV["act"] = _RING[-1]["act"] if len(_RING) else None
    print(f"  {'─'*45} ▶ 위=과거 / 아래=이후 {'─'*45}")


def _grab_phys(env):
    out = {"root_v": None, "ang_v": None, "imu_a": None}
    try:
        robot = env.scene["robot"]
        out["root_v"] = robot.data.root_lin_vel_w.detach().norm(dim=-1)
        out["ang_v"] = robot.data.root_ang_vel_w.detach().norm(dim=-1)
    except Exception:
        pass
    try:
        imu = env.scene["imu"]
        out["imu_a"] = imu.data.lin_acc_b.detach().norm(dim=-1)
    except Exception:
        pass
    return out


def _install_push_hook():
    try:
        import isaaclab.envs.mdp.events as ev
        if not hasattr(ev, "push_by_setting_velocity"):
            print("[fb] ✗ push 함수 못 찾음")
            return False
        _orig = ev.push_by_setting_velocity

        def _patched(env, env_ids, velocity_range, asset_cfg=None, **kw):
            _PUSH["step"] = _STEP["n"]
            try:
                _PUSH["ids"] = set(env_ids.detach().tolist()) if env_ids is not None else None
            except Exception:
                _PUSH["ids"] = None
            try:
                xr = velocity_range.get("x", (0, 0)); yr = velocity_range.get("y", (0, 0))
                _PUSH["mag"] = max(abs(xr[0]), abs(xr[1]), abs(yr[0]), abs(yr[1]))
            except Exception:
                _PUSH["mag"] = float("nan")
            if asset_cfg is not None:
                return _orig(env, env_ids, velocity_range, asset_cfg, **kw)
            return _orig(env, env_ids, velocity_range, **kw)

        ev.push_by_setting_velocity = _patched
        print("[fb] ✓ push 후킹 완료")
        return True
    except Exception as e:
        print(f"[fb] ✗ push 후킹 실패: {e}")
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
                    print(_row_for_env(snap, e, prev))
                    _PREV["pobs"] = snap["pobs"]
                    _PREV["act"] = snap["act"]
                    _LOCK["left"] -= 1
                    if _LOCK["left"] <= 0:
                        print("  " + "-" * 128)
                        print(f"[fb] ===== env {e} 추적 종료 =====\n")
                        _LOCK["done"] = True
                    return actions

                _RING.append(snap)
                act_abs = actions.detach().abs().amax(dim=-1)
                over = (act_abs > FB_WARN).nonzero(as_tuple=True)[0]
                if over.numel() > 0:
                    e = int(over[act_abs[over].argmax()].item())
                    _LOCK["env"] = e
                    _LOCK["left"] = FB_AFTER
                    _dump_locked(e, step)
            except Exception as ex:
                print(f"[fb] 처리 오류(무시): {ex}")
            return actions

        PPO.act = _patched_act
        print(f"[fb] ✓ act 후킹 완료 (WARN={FB_WARN}, RING={FB_RING}, AFTER={FB_AFTER})")
        return True
    except Exception as e:
        print(f"[fb] ✗ act 후킹 실패: {e}")
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
    print("[fb] 준비 완료. 처음 터지는 env 한 마리를 잠가 과거·이후 + 민감도(dA/dL)를 추적합니다.")
