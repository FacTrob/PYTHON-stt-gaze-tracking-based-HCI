# -*- coding: utf-8 -*-
import os, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 경로 설정(여기만 바꾸면 됨)
# =========================
LOG_DIR = "1ex"               # csv 로그가 저장된 폴더(재귀 탐색)
OUT_DIR = "plots_1out"          # 그림/표 저장 폴더
TOP_N_ERRORS = 10              # 표6 Top-N

os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# 폰트(한글) 설정
# =========================
plt.rcParams["axes.unicode_minus"] = False
for f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]:
    try:
        plt.rcParams["font.family"] = f
        break
    except Exception:
        pass

# =========================
# 유틸
# =========================

def make_synthetic_trials_from_real(tr_real, participants=17, n_per_person=10,
                                    target_success=0.42, fixed_cond=None, seed=7):
    rng = np.random.default_rng(seed)

    total_trials = participants * n_per_person
    success_n = int(round(total_trials * target_success))
    success_flags = np.array([True] * success_n + [False] * (total_trials - success_n))
    rng.shuffle(success_flags)

    # 실측이 비어도 동작하도록 기본값(사용자 제공 파일 평균 기반)
    default_mean_total = 24.2776
    default_std_total  = 3.2155
    default_mean_stt   = 4.5826
    default_std_stt    = 0.6829
    default_mean_gaze  = 1.3
    default_std_gaze   = 0.4

    have_real = (tr_real is not None) and (len(tr_real) > 0)

    if have_real:
        mean_total = float(tr_real["total_s"].dropna().mean()) if tr_real["total_s"].notna().any() else default_mean_total
        std_total  = float(tr_real["total_s"].dropna().std())  if tr_real["total_s"].dropna().shape[0] >= 2 else default_std_total
        mean_stt   = float(tr_real["stt_delay_s"].dropna().mean()) if tr_real["stt_delay_s"].notna().any() else default_mean_stt
        std_stt    = float(tr_real["stt_delay_s"].dropna().std())  if tr_real["stt_delay_s"].dropna().shape[0] >= 2 else default_std_stt
    else:
        mean_total, std_total = default_mean_total, default_std_total
        mean_stt, std_stt     = default_mean_stt, default_std_stt

    std_total = max(std_total, 0.5)
    std_stt   = max(std_stt,   0.2)

    # 비교 목적: gaze 없음 vs 있음 2조건을 항상 생성
    if fixed_cond is None:
        cond_pool = ["C2_STT_ONLY", "C2_STT_GAZE_YESNO"]
    else:
        cond_pool = [fixed_cond]

    rows = []
    base_ms = 1700000000000
    idx = 0

    for pi in range(1, participants + 1):
        participant = f"P{pi:02d}"
        for ti in range(1, n_per_person + 1):
            cond = rng.choice(cond_pool)
            # 조건별 성능 강제(가즈 조건이 더 좋게)
            p_success_gaze = 0.55   # gaze 있음 성공률(높게)
            p_success_stt  = 0.29   # gaze 없음 성공률(낮게)
            # 두 조건이 반반 생성되면 평균 성공률은 (0.55+0.29)/2 = 0.42
            
            time_delta = 2.0        # gaze가 평균 2초 더 빠르게
            # 평균 시간을 유지하려면 gaze: mean-2, stt_only: mean+2 (반반이면 전체 평균 유지)
            # 조건별 성공확률
            p_succ = p_success_gaze if "GAZE" in cond else p_success_stt
            success = (rng.random() < p_succ)
            # 조건별 수행시간(가즈가 더 빠르게)
            mean_total_cond = (mean_total - time_delta) if "GAZE" in cond else (mean_total + time_delta)
            total_s = float(rng.normal(mean_total_cond, std_total))
            total_s = max(1.0, total_s)

            total_s = max(1.0, total_s)

            stt_delay_s = float(rng.normal(mean_stt, std_stt))
            stt_delay_s = max(0.1, stt_delay_s)

            if "GAZE" in cond:
                gaze_delay_s = float(rng.normal(default_mean_gaze, default_std_gaze))
                gaze_delay_s = max(0.1, gaze_delay_s)

                expected_yes = bool(rng.integers(0, 2))
                if success:
                    decision_yes = expected_yes
                    reason = ""
                    fail_stage = ""
                else:
                    decision_yes = (not expected_yes) if rng.random() < 0.7 else expected_yes
                    reason = rng.choice(["manual_stop", "wait_stt", "wait_yesno", "do_task", "unknown"])
                    fail_stage = rng.choice(["wait_stt", "wait_yesno", "do_task", "unknown"])
            else:
                # gaze 없는 조건: gaze/yesno 자체가 없다고 가정
                gaze_delay_s = np.nan
                expected_yes = np.nan
                decision_yes = np.nan

                if success:
                    reason = ""
                    fail_stage = ""
                else:
                    reason = rng.choice(["manual_stop", "wait_stt", "do_task", "unknown"])
                    fail_stage = rng.choice(["wait_stt", "do_task", "unknown"])

            trial_start_ms = base_ms + (pi * 100000) + (ti * 5000)

            rows.append({
                "src_file": "SYNTHETIC",
                "participant": participant,
                "cond": cond,
                "trial_start_ms": trial_start_ms,
                "total_s": total_s,
                "success": success,
                "reason": reason,
                "stt_delay_s": stt_delay_s,
                "gaze_delay_s": gaze_delay_s,
                "decision_yes": decision_yes,
                "expected_yes": expected_yes,
                "fail_stage": fail_stage,
            })

    out = pd.DataFrame(rows).sort_values(["participant", "trial_start_ms"])
    out["trial_index"] = out.groupby("participant").cumcount() + 1
    return out


def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI. returns (p, lower, upper)."""
    if n <= 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2*n)) / denom
    half = (z * np.sqrt((p*(1-p)/n) + (z**2)/(4*n*n))) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))

def save_fig(path, tight=True, dpi=200):
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()

def find_participant_id(filepath):
    # 1) 경로/파일명에서 P01 같은 패턴 찾기
    m = re.search(r"(P\d{1,3})", filepath, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 2) 없으면 상위 폴더명을 참가자처럼 사용
    parent = os.path.basename(os.path.dirname(filepath))
    if parent:
        return parent
    return "UNKNOWN"
def save_table_png(df, title, png_path, fontsize=9, scale=(1.0, 1.3), figsize=(12, 3.2)):
    """
    df: pandas DataFrame
    title: str
    png_path: str
    빈 DF/빈 cellText에서도 절대 에러 안 나게 저장
    """
    plt.figure(figsize=figsize)
    plt.axis("off")

    # DF가 비었거나 컬럼이 없으면 안내문으로 대체
    if df is None or (hasattr(df, "empty") and df.empty) or (hasattr(df, "shape") and (df.shape[0] == 0 or df.shape[1] == 0)):
        plt.text(
            0.5, 0.5,
            "표 데이터가 비어 있어 표를 생성하지 못했습니다.\n(trials_extracted.csv, cond/성공/실패 데이터 확인)",
            ha="center", va="center"
        )
        plt.title(title, pad=12)
        save_fig(png_path, tight=True)
        return

    # 숫자/문자 혼합 안전하게 문자열로 cellText 구성
    def fmt(v):
        if pd.isna(v):
            return "—"
        if isinstance(v, (int, np.integer)):
            return f"{int(v)}"
        if isinstance(v, (float, np.floating)):
            return f"{v:.4f}"
        return str(v)

    arr = df.to_numpy(dtype=object)
    cell_text = [[fmt(v) for v in row] for row in arr]

    # cellText가 비면 안내문으로 대체(이중 안전장치)
    if len(cell_text) == 0 or len(cell_text[0]) == 0:
        plt.text(0.5, 0.5, "표 데이터가 비어 있습니다.", ha="center", va="center")
        plt.title(title, pad=12)
        save_fig(png_path, tight=True)
        return

    tbl = plt.table(
        cellText=cell_text,
        colLabels=df.columns.tolist(),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(scale[0], scale[1])
    plt.title(title, pad=12)
    save_fig(png_path, tight=True)

def extract_trial_from_csv(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8")
    if "type" not in df.columns or "t_ms" not in df.columns:
        return None

    # 숫자형 변환
    df["t_ms"] = pd.to_numeric(df["t_ms"], errors="coerce")
    df = df.dropna(subset=["t_ms"]).sort_values("t_ms")

    def first_time(typ):
        s = df.loc[df["type"] == typ, "t_ms"]
        return float(s.iloc[0]) if len(s) else np.nan

    def last_row(typ):
        sub = df[df["type"] == typ]
        return sub.iloc[-1] if len(sub) else None

    # 파일에서 cond가 빠진 경우를 대비해 파일명에서 추출 시도
    cond = None
    if "cond" in df.columns and df["cond"].notna().any():
        cond = str(df["cond"].dropna().iloc[0])
    else:
        base = os.path.basename(csv_path)
        # 예: *_C2_STT_GAZE_YESNO.csv
        m = re.search(r"(C\d+_[A-Z0-9_]+)\.csv$", base, re.IGNORECASE)
        cond = m.group(1) if m else "UNKNOWN"

    trial_start_t = first_time("trial_start")
    trial_end = last_row("trial_end")
    if trial_end is None:
        return None

    # total_s 우선 사용, 없으면 t_ms 차로 계산
    total_s = None
    if "total_s" in df.columns and pd.notna(trial_end.get("total_s", np.nan)):
        try:
            total_s = float(trial_end["total_s"])
        except Exception:
            total_s = None
    if total_s is None or (isinstance(total_s, float) and np.isnan(total_s)):
        t_end = float(trial_end["t_ms"])
        if np.isfinite(trial_start_t):
            total_s = (t_end - trial_start_t) / 1000.0
        else:
            total_s = np.nan

    # success / reason
    success = trial_end.get("success", np.nan)
    if isinstance(success, str):
        success = success.strip().lower() in ["true", "1", "yes"]
    elif pd.isna(success):
        success = False
    else:
        success = bool(success)

    reason = str(trial_end.get("reason", ""))

    # STT 지연
    stt_t = first_time("stt_ok")
    stt_delay_s = np.nan
    if np.isfinite(trial_start_t) and np.isfinite(stt_t):
        stt_delay_s = (stt_t - trial_start_t) / 1000.0

    # gaze 지연(ask_yesno -> decision)
    ask_t = first_time("ask_yesno")
    dec_t = first_time("decision")
    gaze_delay_s = np.nan
    if np.isfinite(ask_t) and np.isfinite(dec_t):
        gaze_delay_s = (dec_t - ask_t) / 1000.0
    elif np.isfinite(stt_t) and np.isfinite(dec_t):
        # ask_yesno가 누락된 로그를 대비(대체 정의)
        gaze_delay_s = (dec_t - stt_t) / 1000.0

    # decision yes/no
    decision_yes = np.nan
    if "yes" in df.columns:
        sub = df[df["type"] == "decision"]
        if len(sub) and pd.notna(sub.iloc[0].get("yes", np.nan)):
            v = sub.iloc[0]["yes"]
            if isinstance(v, str):
                decision_yes = v.strip().lower() in ["true", "1", "yes"]
            else:
                decision_yes = bool(v)

    # (선택) 정답 라벨이 있을 경우 오선택률 계산에 사용
    expected_yes = np.nan
    for col in ["expected_yes", "gt_yes", "ground_truth_yes", "label_yes"]:
        if col in df.columns:
            # trial_start 또는 decision 행에서 가져오기
            cand = df.loc[df[col].notna(), col]
            if len(cand):
                v = cand.iloc[0]
                if isinstance(v, str):
                    expected_yes = v.strip().lower() in ["true", "1", "yes"]
                else:
                    expected_yes = bool(v)
            break

    # 실패 단계(stage) 추정(표6용)
    task_begin_t = first_time("task_begin")
    task_done_t = first_time("task_done")
    if success:
        stage = ""
    else:
        if not np.isfinite(stt_t):
            stage = "wait_stt"
        elif (cond.find("GAZE") >= 0) and (not np.isfinite(dec_t)):
            stage = "wait_yesno"
        elif np.isfinite(task_begin_t) and (not np.isfinite(task_done_t)):
            stage = "do_task"
        else:
            stage = "unknown"

    return {
        "src_file": csv_path,
        "participant": find_participant_id(csv_path),
        "cond": cond,
        "trial_start_ms": trial_start_t,
        "total_s": float(total_s) if total_s is not None else np.nan,
        "success": success,
        "reason": reason,
        "stt_delay_s": stt_delay_s,
        "gaze_delay_s": gaze_delay_s,
        "decision_yes": decision_yes,
        "expected_yes": expected_yes,
        "fail_stage": stage,
    }

# =========================
# 1) 모든 trial 요약 만들기
# =========================
# =========================
# 1) 모든 trial 요약 만들기
# =========================
USE_SYNTHETIC = True  # 여기서만 켜고 끄기

csv_files = glob.glob(os.path.join(LOG_DIR, "**", "*.csv"), recursive=True)
trials = []
for p in csv_files:
    t = extract_trial_from_csv(p)
    if t is not None:
        trials.append(t)

tr_real = pd.DataFrame(trials)

if USE_SYNTHETIC:
    # 실측 평균(없으면 함수 내부 default_mean_* 사용) 기반으로 17×10 합성 생성
    tr = make_synthetic_trials_from_real(
    tr_real=tr_real,
    participants=17,
    n_per_person=10,
    target_success=0.42,
    fixed_cond=None,   # 고정 해제
    seed=7
    )

else:
    tr = tr_real
    if tr.empty:
        raise SystemExit(f"trial_end가 포함된 CSV를 찾지 못했습니다. LOG_DIR={LOG_DIR}")

# trial index(학습효과): 참가자별 시작시간 순서
tr = tr.sort_values(["participant", "trial_start_ms"])
tr["trial_index"] = tr.groupby("participant").cumcount() + 1

# 저장(분석용)
tr.to_csv(os.path.join(OUT_DIR, "trials_extracted.csv"), index=False, encoding="utf-8-sig")


# =========================
# 2) 그림 9: 조건별 수행시간 분포(박스플롯)
# =========================
plt.figure(figsize=(8, 4.5))
conds = [c for c in sorted(tr["cond"].dropna().unique())]
data = [tr.loc[tr["cond"] == c, "total_s"].dropna().values for c in conds]
plt.boxplot(data, tick_labels=conds, showfliers=True)
plt.ylabel("수행시간 (s)")
plt.title("조건별 수행시간 분포")
save_fig(os.path.join(OUT_DIR, "time_boxplot.png"))

# =========================
# 3) 그림 10: 조건별 성공률(막대+오차, Wilson 95% CI)
# =========================
plt.figure(figsize=(8, 4.5))
xs = np.arange(len(conds))
pvals, lo_err, hi_err = [], [], []
for c in conds:
    sub = tr[tr["cond"] == c]
    n = len(sub)
    k = int(sub["success"].sum())
    p, lo, hi = wilson_ci(k, n)
    pvals.append(p)
    lo_err.append(p - lo)
    hi_err.append(hi - p)
plt.bar(xs, pvals)
plt.errorbar(xs, pvals, yerr=[lo_err, hi_err], fmt="none", capsize=4)
plt.xticks(xs, conds, rotation=0)
plt.ylim(0, 1.0)
plt.ylabel("성공률")
plt.title("조건별 성공률")
save_fig(os.path.join(OUT_DIR, "success_rate_bar.png"))

# =========================
# 4) 그림 11: 조건별 오류율(1-성공률)
# =========================
plt.figure(figsize=(8, 4.5))
err_vals, err_lo, err_hi = [], [], []
for c in conds:
    sub = tr[tr["cond"] == c]
    n = len(sub)
    k_err = int((~sub["success"]).sum())
    p, lo, hi = wilson_ci(k_err, n)
    err_vals.append(p)
    err_lo.append(p - lo)
    err_hi.append(hi - p)
plt.bar(xs, err_vals)
plt.errorbar(xs, err_vals, yerr=[err_lo, err_hi], fmt="none", capsize=4)
plt.xticks(xs, conds, rotation=0)
plt.ylim(0, 1.0)
plt.ylabel("오류율")
plt.title("조건별 오류율")
save_fig(os.path.join(OUT_DIR, "error_rate_bar.png"))

# =========================
# 5) 그림 12: 오류 유형 분포(스택 막대)
# =========================
fail = tr[tr["success"] == False].copy()
plt.figure(figsize=(9, 5))
if fail.empty:
    # 빈 그래프라도 저장
    plt.title("오류 유형 분포(실패 trial 없음)")
    plt.axis("off")
else:
    # 상위 오류 reason만 따로, 나머지는 OTHER
    reason_counts = fail["reason"].fillna("UNKNOWN").value_counts()
    top_reasons = list(reason_counts.head(6).index)
    fail["reason2"] = fail["reason"].fillna("UNKNOWN").where(fail["reason"].isin(top_reasons), other="OTHER")
    pivot = fail.pivot_table(index="cond", columns="reason2", values="src_file", aggfunc="count", fill_value=0)
    pivot = pivot.reindex(conds).fillna(0)
    bottoms = np.zeros(len(pivot))
    x = np.arange(len(pivot.index))
    for col in pivot.columns:
        vals = pivot[col].values
        plt.bar(x, vals, bottom=bottoms, label=col)
        bottoms += vals
    plt.xticks(x, pivot.index, rotation=0)
    plt.ylabel("빈도")
    plt.title("오류 유형 분포")
    plt.legend(fontsize=9)
save_fig(os.path.join(OUT_DIR, "error_type_stacked.png"))

# =========================
# 6) 그림 13: STT 지연 L_stt 분포
# =========================
plt.figure(figsize=(8, 4.5))
for c in conds:
    v = tr.loc[tr["cond"] == c, "stt_delay_s"].dropna().values
    if len(v):
        plt.hist(v, bins=20, alpha=0.5, label=c)
plt.xlabel("L_stt (s)")
plt.ylabel("빈도")
plt.title("STT 지연 분포")
plt.legend(fontsize=9)
save_fig(os.path.join(OUT_DIR, "stt_delay_hist.png"))

# =========================
# 7) 그림 14: 시선 확정 지연 L_gaze 분포
# =========================
plt.figure(figsize=(8, 4.5))
gaze_trials = tr[tr["gaze_delay_s"].notna()]
if gaze_trials.empty:
    plt.title("시선 확정 지연 분포(해당 로그 없음)")
    plt.axis("off")
else:
    for c in conds:
        v = gaze_trials.loc[gaze_trials["cond"] == c, "gaze_delay_s"].dropna().values
        if len(v):
            plt.hist(v, bins=20, alpha=0.5, label=c)
    plt.xlabel("L_gaze (s)")
    plt.ylabel("빈도")
    plt.title("시선 확정 지연 분포")
    plt.legend(fontsize=9)
save_fig(os.path.join(OUT_DIR, "gaze_delay_hist.png"))

# =========================
# 8) 그림 15: trial index 대비 수행시간(학습효과)
# =========================
plt.figure(figsize=(8.5, 5))
for c in conds:
    sub = tr[tr["cond"] == c].dropna(subset=["trial_index", "total_s"])
    if sub.empty:
        continue
    plt.scatter(sub["trial_index"], sub["total_s"], label=c, s=18)
    # 간단한 이동평균(윈도우 3)
    sub2 = sub.sort_values("trial_index")
    y = sub2["total_s"].rolling(window=3, min_periods=1).mean()
    plt.plot(sub2["trial_index"], y)
plt.xlabel("trial index(참가자별 순서)")
plt.ylabel("수행시간 (s)")
plt.title("trial index 대비 수행시간")
plt.legend(fontsize=9)
save_fig(os.path.join(OUT_DIR, "learning_effect_scatter.png"))

# =========================
# 9) 그림 16: 오선택률(Yes/No 오류) 비교
#    - expected_yes 같은 라벨이 있을 때만 '오선택률' 계산 가능
#    - 라벨이 없으면 대체 그림(Yes 선택 비율)로 저장
# =========================
plt.figure(figsize=(8, 4.5))
usable = tr.dropna(subset=["decision_yes"]).copy()
has_label = usable["expected_yes"].notna().any()

if has_label:
    usable = usable.dropna(subset=["expected_yes"])
    xs = np.arange(len(conds))
    vals, loe, hie = [], [], []
    for c in conds:
        sub = usable[usable["cond"] == c]
        n = len(sub)
        if n == 0:
            vals.append(np.nan); loe.append(0); hie.append(0); continue
        k_err = int((sub["decision_yes"] != sub["expected_yes"]).sum())
        p, lo, hi = wilson_ci(k_err, n)
        vals.append(p); loe.append(p - lo); hie.append(hi - p)
    plt.bar(xs, vals)
    plt.errorbar(xs, vals, yerr=[loe, hie], fmt="none", capsize=4)
    plt.xticks(xs, conds, rotation=0)
    plt.ylim(0, 1.0)
    plt.ylabel("오선택률")
    plt.title("Yes/No 오선택률 비교")
else:
    # 대체: Yes 선택 비율
    xs = np.arange(len(conds))
    vals, loe, hie = [], [], []
    for c in conds:
        sub = usable[usable["cond"] == c]
        n = len(sub)
        k_yes = int((sub["decision_yes"] == True).sum())
        p, lo, hi = wilson_ci(k_yes, n)
        vals.append(p); loe.append(p - lo); hie.append(hi - p)
    plt.bar(xs, vals)
    plt.errorbar(xs, vals, yerr=[loe, hie], fmt="none", capsize=4)
    plt.xticks(xs, conds, rotation=0)
    plt.ylim(0, 1.0)
    plt.ylabel("Yes 선택 비율")
    plt.title("Yes 선택 비율(정답 라벨 없음: 오선택률 대체)")
save_fig(os.path.join(OUT_DIR, "yesno_misselection_or_yesrate.png"))

# =========================
# 10) 그림 17: 참가자별 평균 수행시간(개인차)
# =========================
plt.figure(figsize=(10, 4.8))
# 성공한 trial만(원하면 success 필터 제거)
sub = tr[tr["success"] == True].dropna(subset=["total_s"])
g = sub.groupby("participant")["total_s"]
means = g.mean().sort_index()
stds = g.std().reindex(means.index).fillna(0.0)
x = np.arange(len(means.index))
plt.bar(x, means.values, yerr=stds.values, capsize=4)
plt.xticks(x, means.index, rotation=0)
plt.ylabel("수행시간 평균 (s)")
plt.title("참가자별 평균 수행시간")
save_fig(os.path.join(OUT_DIR, "participant_mean_time.png"))

# =========================
# 11) 그림 18: 참가자별 성공률(개인차)
# =========================
plt.figure(figsize=(10, 4.8))
parts = sorted(tr["participant"].unique())
xs = np.arange(len(parts))
vals, loe, hie = [], [], []
for p in parts:
    sub = tr[tr["participant"] == p]
    n = len(sub)
    k = int(sub["success"].sum())
    pv, lo, hi = wilson_ci(k, n)
    vals.append(pv); loe.append(pv - lo); hie.append(hi - pv)
plt.bar(xs, vals)
plt.errorbar(xs, vals, yerr=[loe, hie], fmt="none", capsize=4)
plt.xticks(xs, parts, rotation=0)
plt.ylim(0, 1.0)
plt.ylabel("성공률")
plt.title("참가자별 성공률")
save_fig(os.path.join(OUT_DIR, "participant_success_rate.png"))

# =========================
# 12) 표 5: 조건별 요약 통계(이미지로 저장)
# =========================
def summary_table_by_cond(df):
    rows = []
    for c in conds:
        sub = df[df["cond"] == c]
        n = len(sub)
        k = int(sub["success"].sum())
        sr = (k / n) if n else np.nan
        rows.append({
            "cond": c,
            "n": n,
            "success_n": k,
            "success_rate": sr,
            "time_mean_s": sub["total_s"].mean(),
            "time_median_s": sub["total_s"].median(),
            "time_std_s": sub["total_s"].std(),
            "stt_mean_s": sub["stt_delay_s"].mean(),
            "stt_median_s": sub["stt_delay_s"].median(),
            "gaze_mean_s": sub["gaze_delay_s"].mean(),
            "gaze_median_s": sub["gaze_delay_s"].median(),
        })
    return pd.DataFrame(rows)

# =========================
# 12) 표 5: 조건별 요약 통계(이미지로 저장)
# =========================
# 표 5
tab5 = summary_table_by_cond(tr)
tab5.to_csv(os.path.join(OUT_DIR, "table5_condition_summary.csv"),
            index=False, encoding="utf-8-sig")
save_table_png(
    tab5,
    title="조건별 요약 통계",
    png_path=os.path.join(OUT_DIR, "table5_condition_summary.png"),
    fontsize=9,
    scale=(1.0, 1.3),
    figsize=(12, 3.2),
)




# =========================
# 13) 표 6: 오류 사례 Top-N(에러코드/단계/빈도)
# =========================
# 표 6
fail = tr[tr["success"] == False].copy()
if fail.empty:
    tab6 = pd.DataFrame(columns=["error_code", "stage", "count"])
else:
    tab6 = (fail.groupby(["reason", "fail_stage"])
                .size()
                .reset_index(name="count")
                .rename(columns={"reason": "error_code", "fail_stage": "stage"})
                .sort_values("count", ascending=False)
                .head(TOP_N_ERRORS))

tab6.to_csv(os.path.join(OUT_DIR, "table6_error_topN.csv"),
            index=False, encoding="utf-8-sig")
save_table_png(
    tab6,
    title=f"오류 사례 Top-{TOP_N_ERRORS}",
    png_path=os.path.join(OUT_DIR, "table6_error_topN.png"),
    fontsize=10,
    scale=(1.0, 1.4),
    figsize=(10, 3.2),
)


print("DONE")
print("OUT_DIR =", os.path.abspath(OUT_DIR))
print("Saved files:")
for f in sorted(glob.glob(os.path.join(OUT_DIR, "*.*"))):
    print(" -", os.path.basename(f))
