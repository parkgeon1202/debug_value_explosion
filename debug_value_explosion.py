"""
debug_value_explosion.py
========================================================================
Isaac Lab 2.3.0 / Isaac Sim 5.1.0 + rsl-rl-lib(>=3.0.1) 학습에서
    RuntimeError: normal expects all elements of std >= 0.0
가 "가치함수(critic) 폭발" 때문에 발생하는 경로를 코드 베이스 단위로 추적한다.

이 에러는 두 가지 경우에 모두 뜬다.
  (A) std 파라미터가 음수로 밀려남  (noise_std_type="scalar" 일 때 self.std 는 raw 파라미터)
  (B) std 가 NaN/Inf 가 됨          (NaN >= 0 == False 이므로 동일한 에러)
critic 폭발은 보통 (B) → returns/advantage NaN → loss NaN → std NaN 경로로 이어진다.

추적 단계 (rsl_rl 실제 코드 흐름 그대로):
  STAGE-1 REWARD : PPO.process_env_step   reward + time_out value-bootstrap
  STAGE-2 RETURN : PPO.compute_returns    storage.returns / advantages / values
  STAGE-3 FWD    : ActorCritic.update_distribution  actor mean(mu), sigma
  STAGE-4 GRAD   : optimizer step-post-hook  pre-clip grad norm, per-group grad norm
  STAGE-5 PARAM  : optimizer step-post-hook  self.std / self.log_std 값 (← 폭발 직격탄)
  STAGE-7 DIST   : update_distribution      std 음수/NaN 사전 감지 → 명확한 에러로 abort
  CRASH          : ActorCritic.act          torch.normal 크래시 캡처 + 직전 이력 덤프

사용법:
    train.py 맨 위 (AppLauncher import 부근, rsl_rl import 이후면 어디든) 에 한 줄:

        import debug_value_explosion   # noqa

    또는 실행 시:
        PYTHONPATH=. python train.py ...  (파일을 train.py 와 같은 폴더 또는 PYTHONPATH 경로에 둠)

환경변수로 동작 조절:
    DBG_ABORT=1      std 이상 감지 즉시 abort (기본 1). 0이면 발산 궤적만 로깅하고 계속.
    DBG_EVERY=25     STAGE-1~5 주기 로깅 간격 (기본 25 update)
    DBG_WARN=1e3     이 값 이상이면 "폭발 예고" 경고
    DBG_HIST=12      크래시 시 되돌아볼 직전 optimizer step 개수
"""

import os
import sys
import math
import traceback
from collections import deque

import torch
import torch.nn as nn

# ────────────────────────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────────────────────────
ABORT   = os.environ.get("DBG_ABORT", "1") == "1"
EVERY   = int(os.environ.get("DBG_EVERY", "25"))
WARN    = float(os.environ.get("DBG_WARN", "1e3"))
HIST_N  = int(os.environ.get("DBG_HIST", "12"))

_update_idx = 0          # PPO.update 호출 횟수 (≈ learning iteration)
_step_idx   = 0          # optimizer.step 호출 횟수 (minibatch 단위)
_last_preclip_norm = float("nan")
_history = deque(maxlen=HIST_N)   # 직전 step 진단 스냅샷
_hook_registered = False


# ────────────────────────────────────────────────────────────────────
# 텐서 진단 유틸
# ────────────────────────────────────────────────────────────────────
def _stat(t, name, warn=WARN):
    if t is None:
        return f"  {name:26s}: None"
    t = t.detach().float()
    n_nan = int(torch.isnan(t).sum())
    n_inf = int(torch.isinf(t).sum())
    finite = t[torch.isfinite(t)]
    if finite.numel() == 0:
        return f"  {name:26s}: ALL non-finite!  NaN={n_nan} Inf={n_inf}"
    mx = finite.abs().max().item()
    flag = ""
    if n_nan or n_inf:
        flag = "  <<< NaN/Inf"
    elif mx > warn:
        flag = f"  <<< |max|>{warn:.0e}"
    return (f"  {name:26s}: min={finite.min().item():+.3e} "
            f"max={finite.max().item():+.3e} "
            f"mean={finite.mean().item():+.3e} "
            f"|max|={mx:.3e} NaN={n_nan} Inf={n_inf}{flag}")


def _is_bad(t, allow_neg=True):
    """NaN/Inf 또는 (allow_neg=False 일 때) 음수 포함 여부."""
    if t is None:
        return False
    t = t.detach()
    bad = bool(torch.isnan(t).any() or torch.isinf(t).any())
    if not allow_neg:
        bad = bad or bool((t < 0).any())
    return bad


def _grad_group_norms(policy):
    """actor / critic / noise(std) 그룹별 grad norm (현재 .grad 기준)."""
    groups = {"actor": 0.0, "critic": 0.0, "noise(std)": 0.0, "other": 0.0}
    for name, p in policy.named_parameters():
        if p.grad is None:
            continue
        g2 = float(p.grad.detach().float().pow(2).sum())
        if name.startswith("actor"):
            groups["actor"] += g2
        elif name.startswith("critic"):
            groups["critic"] += g2
        elif name in ("std", "log_std") or name.endswith(".std") or name.endswith(".log_std"):
            groups["noise(std)"] += g2
        else:
            groups["other"] += g2
    return {k: math.sqrt(v) for k, v in groups.items()}


def _noise_param(policy):
    """noise_std_type 에 따라 실제 std 또는 log_std 파라미터와 유효 std 반환."""
    nst = getattr(policy, "noise_std_type", "scalar")
    if nst == "scalar" and hasattr(policy, "std"):
        raw = policy.std
        eff = raw                      # scalar 모드: std = self.std (그대로, 음수 가능!)
        return nst, raw, eff
    if nst == "log" and hasattr(policy, "log_std"):
        raw = policy.log_std
        eff = torch.exp(raw)           # log 모드: std = exp(log_std) (항상 양수)
        return nst, raw, eff
    return nst, None, None


def _dump_history(tag):
    print(f"\n──── 직전 {len(_history)} optimizer step 발산 궤적 ({tag}) ────")
    print("  step    | preclip_grad | g_actor   g_critic  g_noise | std|min  std|max  | lr")
    for h in _history:
        print("  {step:7d} | {pre:11.3e} | {ga:8.2e} {gc:8.2e} {gn:8.2e} | "
              "{smin:+.3e} {smax:+.3e} | {lr:.2e}".format(**h))
    print("─" * 78)


# ────────────────────────────────────────────────────────────────────
# Patch 0: clip_grad_norm_ 래핑 → pre-clip total norm 캡처
#   (rsl_rl PPO.update 가 매 minibatch 마다 호출. 반환값 = 클리핑 전 전체 norm)
# ────────────────────────────────────────────────────────────────────
_orig_clip = nn.utils.clip_grad_norm_

def _clip_wrap(parameters, max_norm, *a, **kw):
    global _last_preclip_norm
    total = _orig_clip(parameters, max_norm, *a, **kw)
    try:
        _last_preclip_norm = float(total)
    except Exception:
        _last_preclip_norm = float("nan")
    return total

nn.utils.clip_grad_norm_ = _clip_wrap
print("[dbg] ✓ clip_grad_norm_ 래핑 (pre-clip grad norm 캡처)")


# ────────────────────────────────────────────────────────────────────
# Patch 1~5: rsl_rl 클래스 패치
# ────────────────────────────────────────────────────────────────────
try:
    from rsl_rl.modules.actor_critic import ActorCritic
    from rsl_rl.algorithms.ppo import PPO

    # ---- optimizer step-post-hook (STAGE-4 GRAD / STAGE-5 PARAM) ----
    def _make_step_hook(ppo):
        def _hook(optimizer, *args, **kwargs):
            global _step_idx
            _step_idx += 1
            nst, raw, eff = _noise_param(ppo.policy)
            gn = _grad_group_norms(ppo.policy)
            smin = float(eff.detach().min()) if eff is not None else float("nan")
            smax = float(eff.detach().max()) if eff is not None else float("nan")
            lr = optimizer.param_groups[0]["lr"]
            snap = dict(step=_step_idx, pre=_last_preclip_norm,
                        ga=gn["actor"], gc=gn["critic"], gn=gn["noise(std)"],
                        smin=smin, smax=smax, lr=lr)
            _history.append(snap)

            # 즉시 위험 감지: 유효 std 가 음수이거나 NaN/Inf
            bad_neg = (eff is not None) and bool((eff.detach() < 0).any())
            bad_nan = (eff is not None) and _is_bad(eff)
            if bad_neg or bad_nan:
                print("\n" + "!" * 78)
                print(f"[dbg][STAGE-5 PARAM] optimizer.step 직후 std 손상 감지 "
                      f"(update={_update_idx}, step={_step_idx}, mode={nst})")
                print(f"  bad_negative={bad_neg}  bad_nan/inf={bad_nan}")
                print(_stat(raw, f"raw param ({'std' if nst=='scalar' else 'log_std'})"))
                print(_stat(eff, "effective std"))
                print(f"  pre-clip grad norm = {_last_preclip_norm:.4e}  "
                      f"(max_grad_norm={ppo.max_grad_norm})")
                print(f"  grad norms  actor={gn['actor']:.3e}  critic={gn['critic']:.3e}  "
                      f"noise(std)={gn['noise(std)']:.3e}")
                _dump_history("STAGE-5")
                if ABORT:
                    print("[dbg] DBG_ABORT=1 → 종료")
                    os._exit(1)
        return _hook

    # ---- ActorCritic.update_distribution (STAGE-3 FWD / STAGE-7 DIST) ----
    _orig_update_dist = ActorCritic.update_distribution

    def _patched_update_dist(self, obs):
        nst, raw, eff = _noise_param(self)
        # std 사전 검사 — Normal() 생성 전에 명확한 에러를 던진다
        if eff is not None and (_is_bad(eff) or bool((eff.detach() < 0).any())):
            print("\n" + "=" * 78)
            print(f"[dbg][STAGE-7 DIST] Normal() 생성 직전 std 이상 "
                  f"(update={_update_idx}, step={_step_idx}, mode={nst})")
            mean_preview = None
            try:
                mean_preview = self.actor(obs)
            except Exception as e:
                print(f"  (actor forward 실패: {e})")
            print(_stat(obs, "actor input obs"))
            print(_stat(mean_preview, "actor output mean(mu)"))
            print(_stat(raw, "raw noise param"))
            print(_stat(eff, "effective std"))
            _dump_history("STAGE-7")
            print("=" * 78)
            if ABORT:
                os._exit(1)
        # 주기적 정상 로깅
        if _update_idx % EVERY == 0 and _step_idx % 50 == 0:
            mean = self.actor(obs)
            print(f"[dbg][STAGE-3 FWD] update={_update_idx} "
                  f"mu|max={mean.detach().abs().max():.3e} "
                  f"obs|max={obs.detach().abs().max():.3e} "
                  f"std|max={(eff.detach().max() if eff is not None else float('nan')):.3e}")
        return _orig_update_dist(self, obs)

    ActorCritic.update_distribution = _patched_update_dist

    # ---- ActorCritic.act (CRASH 캡처) ----
    _orig_act = ActorCritic.act

    def _patched_act(self, obs, **kwargs):
        try:
            return _orig_act(self, obs, **kwargs)
        except RuntimeError as e:
            if "std" not in str(e):
                raise
            nst, raw, eff = _noise_param(self)
            print("\n" + "#" * 78)
            print(f"[dbg][CRASH] torch.normal std>=0 위반 "
                  f"(update={_update_idx}, step={_step_idx}, mode={nst})")
            try:
                a_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
                print(_stat(a_obs, "actor input obs"))
                print(_stat(self.actor(a_obs), "actor output mean(mu)"))
            except Exception as inner:
                print(f"  (obs/actor 재계산 실패: {inner})")
            print(_stat(raw, "raw noise param"))
            print(_stat(eff, "effective std"))
            _dump_history("CRASH")
            print("#" * 78)
            traceback.print_exc()
            os._exit(1)

    ActorCritic.act = _patched_act

    # ---- PPO.process_env_step (STAGE-1 REWARD) ----
    _orig_proc = PPO.process_env_step

    def _patched_proc(self, obs, rewards, dones, extras):
        if _is_bad(rewards) or (rewards.detach().abs().max() > WARN):
            print(f"[dbg][STAGE-1 REWARD] update={_update_idx} reward 이상")
            print(_stat(rewards, "rewards"))
            if "time_outs" in extras:
                # time_out bootstrap 은 value 를 사용 → critic 폭발 시 reward 로 전염
                print(_stat(self.transition.values, "transition.values (bootstrap 원천)"))
            _dump_history("STAGE-1")
            if ABORT and _is_bad(rewards):
                os._exit(1)
        return _orig_proc(self, obs, rewards, dones, extras)

    PPO.process_env_step = _patched_proc

    # ---- PPO.compute_returns (STAGE-2 RETURN) ----
    _orig_compret = PPO.compute_returns

    def _patched_compret(self, obs):
        last_values = self.policy.evaluate(obs).detach()
        if _is_bad(last_values) or (last_values.abs().max() > WARN):
            print(f"[dbg][STAGE-2 RETURN] update={_update_idx} last_values(critic) 폭발")
            print(_stat(last_values, "last_values (critic out)"))
            print(_stat(self.policy.get_critic_obs(obs), "critic input obs"))
            _dump_history("STAGE-2(pre)")
            if ABORT and _is_bad(last_values):
                os._exit(1)
        out = _orig_compret(self, obs)
        # storage 의 returns/advantages 검사 (normalize 이후)
        st = self.storage
        for attr in ("values", "returns", "advantages"):
            t = getattr(st, attr, None)
            if t is not None and (_is_bad(t) or t.detach().abs().max() > WARN):
                print(f"[dbg][STAGE-2 RETURN] update={_update_idx} storage.{attr} 이상")
                print(_stat(t, f"storage.{attr}"))
                _dump_history("STAGE-2")
                if ABORT and _is_bad(t):
                    os._exit(1)
        return out

    PPO.compute_returns = _patched_compret

    # ---- PPO.update (counter + hook 등록 + loss 검사) ----
    _orig_update = PPO.update

    def _patched_update(self):
        global _update_idx, _hook_registered
        _update_idx += 1
        if not _hook_registered:
            try:
                self.optimizer.register_step_post_hook(_make_step_hook(self))
                _hook_registered = True
                print("[dbg] ✓ optimizer step-post-hook 등록 (STAGE-4/5)")
            except Exception as e:
                print(f"[dbg] step-post-hook 등록 실패(무시): {e}")

        loss_dict = _orig_update(self)

        # update 결과 loss 검사 (key: value_function / surrogate / entropy)
        vf  = loss_dict.get("value_function", float("nan"))
        sur = loss_dict.get("surrogate", float("nan"))
        ent = loss_dict.get("entropy", float("nan"))
        nst, raw, eff = _noise_param(self.policy)
        std_max = float(eff.detach().max()) if eff is not None else float("nan")
        std_min = float(eff.detach().min()) if eff is not None else float("nan")

        warn = ""
        if not math.isfinite(vf) or abs(vf) > WARN:
            warn += "  <<< value_function 폭발"
        if _update_idx % EVERY == 0 or warn:
            print(f"[dbg][SUMMARY] update={_update_idx:6d} "
                  f"value_fn={vf:.3e} surrogate={sur:.3e} entropy={ent:.3e} "
                  f"| std[{std_min:+.2e},{std_max:+.2e}] lr={self.learning_rate:.2e}{warn}")
        if (not math.isfinite(vf)) and ABORT:
            print("[dbg] value_function NaN/Inf → 종료")
            _dump_history("SUMMARY")
            os._exit(1)
        return loss_dict

    PPO.update = _patched_update

    print("[dbg] ✓ ActorCritic / PPO 패치 완료\n"
          f"[dbg]   ABORT={ABORT} EVERY={EVERY} WARN={WARN:.0e} HIST={HIST_N}\n")

except ImportError as e:
    print(f"[dbg] ✗ rsl_rl import 실패 — train.py 의 rsl_rl import 이후에 두세요: {e}")
