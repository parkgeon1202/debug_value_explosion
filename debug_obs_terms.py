"""
debug_obs_terms.py
========================================================================
critic obs(284차원)를 항목별로 쪼개서, 폭발 순간 "어느 ObsTerm이 큰지"를
100% 확정한다. debug_value_explosion.py 와 함께(또는 단독으로) 사용.

사용법:
    train.py 에서 debug_value_explosion import 바로 "아래" 줄에:
        import debug_obs_terms   # noqa

    (debug_value_explosion 없이 단독으로도 동작함)

환경변수:
    OBS_WARN=1e3   critic obs |max| 가 이 값 넘으면 항목별 분해 출력 (기본 1e3)

원리:
    이 패치는 PPO.compute_returns 를 감싸, critic obs 를 한 번 읽어
    rough_env_cfg.py 의 CriticCfg 항목 순서대로 인덱스를 잘라
    각 항목의 |max| 를 찍는다. 읽기 전용(detach), 학습에 영향 없음.
"""

import os
import torch

OBS_WARN = float(os.environ.get("OBS_WARN", "1e3"))

# ── critic obs 항목 순서·차원 (rough_env_cfg.py CriticCfg 의 concat 순서) ──
#   합계 284. 순서가 바뀌면 이 표도 바꿔야 함.
_CRITIC_TERMS = [
    ("base_lin_vel",           3),
    ("base_ang_vel",           3),
    ("projected_gravity",      3),
    ("velocity_commands",      3),
    ("actions(last_action)",  12),
    ("height_scan",          187),
    ("imu_projected_gravity",  3),
    ("imu_ang_vel",            3),
    ("imu_lin_acc",            3),   # ← 1순위 용의자
    ("joint_torques",         12),
    ("body_poses",            21),
    ("joint_pos_accurate",    12),
    ("joint_vel_accurate",    12),
    ("base_pos",               1),
    ("root_lin_vel_w",         3),   # ← 속도 캡(1000) 도달 여부 확인용
    ("root_ang_vel_w",         3),
]
_TOTAL = sum(d for _, d in _CRITIC_TERMS)  # 284


def _term_breakdown(critic_obs: torch.Tensor) -> str:
    """284차원 critic obs 를 항목별 |max|/min/max 로 분해."""
    t = critic_obs.detach().float()
    dim = t.shape[-1]
    if dim != _TOTAL:
        return (f"  [경고] critic obs dim={dim} != 표 합계 {_TOTAL}. "
                f"CriticCfg 가 바뀌었으니 _CRITIC_TERMS 표를 갱신하세요.")
    lines = ["  ── critic obs 항목별 분해 (|max| 큰 순) ──"]
    rows = []
    i = 0
    for name, d in _CRITIC_TERMS:
        seg = t[..., i:i + d]
        finite = seg[torch.isfinite(seg)]
        if finite.numel() == 0:
            amax, smin, smax = float("inf"), float("nan"), float("nan")
        else:
            amax = finite.abs().max().item()
            smin = finite.min().item()
            smax = finite.max().item()
        n_bad = int((~torch.isfinite(seg)).sum())
        rows.append((amax, name, i, i + d, smin, smax, n_bad))
        i += d
    # |max| 큰 순 정렬
    for amax, name, lo, hi, smin, smax, n_bad in sorted(rows, key=lambda r: -r[0]):
        flag = "  <<<" if amax > OBS_WARN else ""
        bad = f" non-finite={n_bad}" if n_bad else ""
        lines.append(f"    {name:24s} [{lo:3d}:{hi:3d}]  "
                     f"|max|={amax:.3e}  (min={smin:+.2e} max={smax:+.2e}){bad}{flag}")
    return "\n".join(lines)


try:
    from rsl_rl.algorithms.ppo import PPO
    _orig_compret = PPO.compute_returns

    def _patched_compret_terms(self, obs):
        try:
            critic_obs = self.policy.get_critic_obs(obs)
            amax = critic_obs.detach().abs().max().item()
            if amax > OBS_WARN or not torch.isfinite(critic_obs).all():
                print(f"\n[dbg-obs] critic obs |max|={amax:.3e} → 항목별 분해:")
                print(_term_breakdown(critic_obs))
        except Exception as e:
            print(f"[dbg-obs] 분해 실패(무시): {e}")
        return _orig_compret(self, obs)

    PPO.compute_returns = _patched_compret_terms
    print(f"[dbg-obs] ✓ critic obs 항목별 분해 패치 완료 (OBS_WARN={OBS_WARN:.0e}, dim={_TOTAL})")

except ImportError as e:
    print(f"[dbg-obs] ✗ rsl_rl import 실패: {e}")
