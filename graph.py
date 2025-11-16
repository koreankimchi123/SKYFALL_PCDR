import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import time
import re

# =========================
# User settings (generalized for multiple tests)
# =========================
TESTS = [
    ("Original Routing", Path("./test_04_original_3600")),
    ("PCDR (p=0.8, S=0.05)", Path("./test_04_3600")),
    ("PCDR (p=0.8, S=0.01)", Path("./test_04_08_001_3600")),
    ("PCDR (p=0.8, S=0.10)", Path("./test_04_08_010_3600")),
    ("K-DS", Path("./test_04_K-DS_100")),

]

# Preferred baseline and primary PCDR label for Fig-1/2/3 comparisons
BASELINE_LABEL = "Original Routing"
PCDR_PRIMARY_LABEL = "PCDR (p=0.8, S=0.05)"

# Common path for per-time grid/link data
GRID_REL = Path("+grid_data/link_traffic_data")
FILE_ISL = "isl_traffic.txt"           # (sat_num)
FILE_DL  = "downlink_loss.txt"         # (sat_num)
FILE_UL  = "uplink_loss.txt"           # (sat_num)
FILE_LL  = "link_loss.txt"             # (sat_num*6)  (not used for heatmaps)
FILE_BLK = "blocked_loss_total.txt"    # single-line total

# Time-series range for grid data
T_START, T_END = 0, 100

# Time range for results/ series (None → full)
RES_T_START: Optional[int] = None
RES_T_END: Optional[int] = None

# Grid size (satellite count = GRID_R * GRID_C)
GRID_R, GRID_C = 72, 22
SAT_COUNT = GRID_R * GRID_C

# results/ relative path
RESULTS_REL = Path("results")

# Optional font
FONT_PATH = "./PretendardVariable.ttf"  # If missing, system default will be used

# Thresholding: None → dynamic (mean+std per time), numeric → absolute threshold
ABS_THRESHOLD: Optional[float] = None

# Output directory
STAMP = time.strftime("%Y%m%d%H%M%S", time.localtime())
OUT_DIR = Path(f"./outputs/{STAMP}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Timestamps to export heatmaps for
HEATMAP_TIMES: List[int] = [T_END]

# =========================
# Warn/log utilities
# =========================
_WARNINGS: List[str] = []

def warn(msg: str):
    m = f"[warn] {msg}"
    print(m)
    _WARNINGS.append(m)

def info(msg: str):
    print(f"[info] {msg}")

def save_warnings():
    p = OUT_DIR / "warnings.txt"
    with p.open("w", encoding="utf-8") as f:
        for w in _WARNINGS:
            f.write(w + "\n")
    print(f"[ok] Saved warning log: {p}")

# =========================
# Font/plot defaults
# =========================
if Path(FONT_PATH).exists():
    font_manager.fontManager.addfont(FONT_PATH)
    font_name = font_manager.FontProperties(fname=FONT_PATH).get_name()
    plt.rcParams["font.family"] = font_name
else:
    warn(f"Font not found: {FONT_PATH} (falling back to default system font)")
plt.rcParams["axes.unicode_minus"] = False

plt.rcParams["font.size"] = 18          # 전체 기본 글자 크기
plt.rcParams["axes.titlesize"] = 16     # 그래프 제목 글자 크기
plt.rcParams["axes.labelsize"] = 18     # x/y 라벨 글자 크기
plt.rcParams["xtick.labelsize"] = 18    # x축 눈금 글자 크기
plt.rcParams["ytick.labelsize"] = 18    # y축 눈금 글자 크기
plt.rcParams["legend.fontsize"] = 16    # 범례 글자 크기
plt.rcParams["figure.titlesize"] = 18   # 전체 figure 제목의 글자 크기

# =========================
# Robust file I/O helpers
# =========================
_SEP_PAT = re.compile(r"[:;,=\-\uFF1A\uFF1B]")
_NUM_PAT = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _read_text_lines(path: Path) -> Optional[List[str]]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return [ln.strip() for ln in f if ln.strip() != ""]
    except FileNotFoundError:
        warn(f"Missing file: {path}")
        return None
    except Exception as e:
        warn(f"Failed to read file: {path} ({e})")
        return None


def _guess_is_kv(lines: List[str]) -> bool:
    """Heuristic: if enough lines contain separators (':', ',', ';', '=', '-', fullwidth), treat as key:value pairs."""
    sep_cnt = sum(1 for ln in lines if _SEP_PAT.search(ln))
    return sep_cnt >= 2


def read_numbers_series(path: Path) -> Optional[np.ndarray]:
    """Parse a pure numeric single-column series → ndarray[float]; skip malformed rows with a warning."""
    lines = _read_text_lines(path)
    if lines is None:
        return None
    out: List[float] = []
    for i, s in enumerate(lines, 1):
        try:
            out.append(float(s))
        except ValueError:
            warn(f"Numeric parse failed ({path}, line {i}): '{s}' → skipped")
    return np.array(out, dtype=float)


def read_key_value_pairs(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Parse tolerant 'key: value' pairs such as '0.1: 34'.
    Allowed separators: ':', ';', ',', '=', '-', and fullwidth counterparts.
    Allow percentage (%) by auto-stripping. If splitting fails, extract first two numbers in the line.
    """
    lines = _read_text_lines(path)
    if lines is None:
        return None

    xs: List[float] = []
    ys: List[float] = []

    for i, s in enumerate(lines, 1):
        if _SEP_PAT.search(s):
            left, right = re.split(_SEP_PAT, s, maxsplit=1)
            left = left.replace("%", "").strip()
            right = right.replace("%", "").strip()
            try:
                x = float(left); y = float(right)
                xs.append(x); ys.append(y); continue
            except ValueError:
                pass
        nums = _NUM_PAT.findall(s.replace("%", ""))
        if len(nums) >= 2:
            try:
                x = float(nums[0]); y = float(nums[1])
                xs.append(x); ys.append(y); continue
            except ValueError:
                warn(f"key:value numeric parse failed ({path}, line {i}): '{s}' → skipped")
                continue
        warn(f"key:value format mismatch ({path}, line {i}): '{s}' → skipped")

    if not xs:
        return None

    xs_np = np.array(xs, dtype=float)
    ys_np = np.array(ys, dtype=float)
    order = np.argsort(xs_np)
    return xs_np[order], ys_np[order]


def read_summary_stats(path: Path) -> Optional[Dict[str, float]]:
    """Parse summary stats like 'max: 73\nmin: 41\naverage: 52.29' into a dict."""
    lines = _read_text_lines(path)
    if lines is None:
        return None
    out: Dict[str, float] = {}
    for i, s in enumerate(lines, 1):
        if ":" not in s:
            warn(f"Summary line format mismatch ({path}, line {i}): '{s}' → skipped")
            continue
        k, v = s.split(":", 1)
        k = k.strip().lower()
        try:
            out[k] = float(v.strip())
        except ValueError:
            warn(f"Summary numeric parse failed ({path}, line {i}): '{s}' → skipped")
    if not out:
        return None
    return out


def detect_and_read(path: Path):
    """
    Auto-detection:
    - summary stats → ('stats', dict)
    - key:value     → ('kv', (x[], y[]))  *only if enough separators appear*
    - single-column → ('series', arr)
    - failure       → ('none', None)
    """
    lines = _read_text_lines(path)
    if lines is None:
        return ("none", None)
    # 1) summary stats?
    keys = [ln.split(":",1)[0].strip().lower() for ln in lines if ":" in ln]
    if any(k in ("max", "min", "average", "avg") for k in keys):
        st = read_summary_stats(path)
        return ("stats", st) if st is not None else ("none", None)
    # 2) key:value?
    if _guess_is_kv(lines):
        kv = read_key_value_pairs(path)
        if kv is not None:
            return ("kv", kv)
    # 3) numeric series
    sr = read_numbers_series(path)
    return ("series", sr) if sr is not None else ("none", None)


def load_values_1d(txt_path: Path, expect_len: Optional[int] = None) -> np.ndarray:
    """Read a single-column numeric file and pad/truncate safely."""
    arr = read_numbers_series(txt_path)
    if arr is None:
        return np.array([], dtype=float)
    if expect_len is not None:
        if arr.size >= expect_len:
            return arr[:expect_len]
        else:
            warn(f"Length short → padding with zeros: {txt_path} ({arr.size} -> {expect_len})")
            return np.pad(arr, (0, expect_len - arr.size), constant_values=0.0)
    return arr


def reshape_grid(values_1d: np.ndarray) -> np.ndarray:
    if values_1d.size != SAT_COUNT:
        warn(f"reshape mismatch: len {values_1d.size} != SAT_COUNT {SAT_COUNT} → pad/truncate then reshape")
        values_1d = (values_1d[:SAT_COUNT] if values_1d.size >= SAT_COUNT
                     else np.pad(values_1d, (0, SAT_COUNT - values_1d.size), constant_values=0.0))
    return values_1d.reshape(GRID_R, GRID_C)

# =========================
# Heatmap
# =========================

def compute_color_scale(arr2d: np.ndarray,
                        vmin_vmax: Optional[Tuple[float, float]] = None
                        ) -> Tuple[float, float]:
    if vmin_vmax is not None:
        return vmin_vmax
    lo = float(np.nanpercentile(arr2d, 1))
    hi = float(np.nanpercentile(arr2d, 99))
    if np.isclose(lo, hi):
        lo, hi = lo - 1.0, hi + 1.0
    return lo, hi


def _three_ticks(n: int) -> Tuple[List[int], List[str]]:
    if n <= 1:
        return [0], ["1"]
    mid = (n - 1) // 2
    return [0, mid, n - 1], ["1", str(mid + 1), str(n)]


def plot_heatmap(arr2d: np.ndarray,
                 title: str,
                 out_path: Optional[Path] = None,
                 vmin_vmax: Optional[Tuple[float, float]] = None,
                 show: bool = False,
                 cbar_label: str = "Value"):
    vmin, vmax = compute_color_scale(arr2d, vmin_vmax)
    base_h = 5.0
    base_w = max(6.0, 6.0 * (GRID_C / max(1, GRID_R)))
    plt.figure(figsize=(base_w, base_h))
    im = plt.imshow(arr2d, aspect="auto", vmin=vmin, vmax=vmax)
    # plt.title(title)
    xt, xtlbl = _three_ticks(GRID_C); yt, ytlbl = _three_ticks(GRID_R)
    plt.xlabel(f"Column (1..{GRID_C})"); plt.ylabel(f"Row (1..{GRID_R})")
    plt.xticks(xt, xtlbl); plt.yticks(yt, ytlbl)
    cbar = plt.colorbar(im); cbar.set_label(cbar_label)
    plt.tight_layout()
    if out_path: out_path.parent.mkdir(parents=True, exist_ok=True); plt.savefig(out_path, dpi=200)
    if show: plt.show()
    plt.close()


def export_heatmaps_for_tests(times: List[int],
                              tests: Dict[str, Path],
                              filename: str,
                              title_tag: str,
                              vmin_vmax: Optional[Tuple[float, float]] = None,
                              show: bool = False):
    out_dir = OUT_DIR / "heatmaps" / title_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in times:
        for name, base in tests.items():
            txt = base / GRID_REL / str(t) / filename
            arr1d = load_values_1d(txt, expect_len=SAT_COUNT)
            if arr1d.size == 0:
                warn(f"Heatmap source missing: {txt}")
                continue
            grid = reshape_grid(arr1d)
            title = f"{name} — t={t} ({title_tag})"
            safe_name = f"heatmap_{title_tag}_t{t}_{name.replace(' ', '_')}.png"
            cbar = "Traffic" if "traffic" in filename else "Loss"
            plot_heatmap(grid, title, out_path=out_dir / safe_name,
                         vmin_vmax=vmin_vmax, show=show, cbar_label=cbar)

# =========================
# Load/Loss Metrics
# =========================

def jain_index(x: np.ndarray) -> float:
    x = x.astype(float); s1 = np.sum(x); s2 = np.sum(x * x); n = x.size
    if n == 0 or s2 == 0: return np.nan
    return (s1 * s1) / (n * s2)


def gini_coefficient(x: np.ndarray) -> float:
    x = x.astype(float)
    if x.size == 0: return np.nan
    x = np.sort(x); n = x.size; cumx = np.cumsum(x)
    if cumx[-1] == 0: return 0.0
    return float((n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n)


def time_metrics_for_dir(base_dir: Path, t0: int, t1: int, series_file: str = FILE_ISL) -> Dict[str, np.ndarray]:
    """
    Load-balancing evaluation tailored to LSN link load dynamics:
    - cv = std/mean (↓ better)
    - par = max/mean (↓ better)
    - p95_over_mean = p95/mean (↓ better)
    - sat_frac = high-load share: x ≥ (mean + std) (↓ better)
    - aat_over_thr = normalized excess area: ∑max(0, x − (mean+std)) / ∑x (↓ better)
    - jain_active, gini_active = ε-active fairness (↑/↓)
    - n_eff_active = effective number of active links = 1 / ∑ p_i^2 (↑ better)
    - active_count_pos = #active links (x>0)
    """
    T = t1 - t0 + 1
    out = {k: np.full(T, np.nan) for k in
           ["mean","std","cv","p95_over_mean","max","par",
            "sat_frac","aat_over_thr",
            "jain_active","gini_active","n_eff_active",
            "active_count","active_count_pos"]}

    base = base_dir / GRID_REL
    ACTIVE_EPS = 10.0  # treat tiny values as inactive to reduce sensitivity to noise

    for t in range(t0, t1 + 1):
        arr = read_numbers_series(base / str(t) / series_file)
        idx = t - t0
        if arr is None or arr.size == 0:
            warn(f"time {t}: {series_file} missing/empty")
            continue

        # basic stats
        m  = float(np.mean(arr))
        s  = float(np.std(arr))
        mx = float(np.max(arr))
        p95 = float(np.percentile(arr, 95))

        out["mean"][idx] = m
        out["std"][idx]  = s
        out["cv"][idx]   = (s/m) if m != 0 else np.nan
        out["p95_over_mean"][idx] = (p95/m) if m != 0 else np.nan
        out["max"][idx]  = mx
        out["par"][idx]  = (mx/m) if m != 0 else np.nan

        # dynamic threshold (capacity unknown setting): mean + std
        thr = m + s
        over = np.maximum(0.0, arr - thr)
        out["sat_frac"][idx] = np.mean(arr >= thr)              # fraction of links above threshold

        # normalized excess area (scale-free)
        den = float(np.sum(arr))
        out["aat_over_thr"][idx] = float(np.sum(over) / den) if den > 0 else np.nan

        # number of active links (x>0)
        out["active_count_pos"][idx] = int(np.sum(arr > 0))

        # ε-active fairness + effective number of active links
        act = arr[arr > ACTIVE_EPS]
        out["active_count"][idx] = act.size
        out["jain_active"][idx]  = jain_index(act)
        out["gini_active"][idx]  = gini_coefficient(act)
        if act.size > 0 and np.sum(act) > 0:
            p = act / np.sum(act)
            out["n_eff_active"][idx] = float(1.0 / np.sum(p * p))
        else:
            out["n_eff_active"][idx] = np.nan

    return out


# =========================
# Plot helpers
# =========================

def plot_lines_pair(a: np.ndarray, b: np.ndarray, title: str, label_a: str, label_b: str, save_path: Path, y_label: str = "", x_label: str = "Time (s)"):
    plt.figure(figsize=(8,4.2))
    if a.size: plt.plot(np.arange(a.size), a, label=label_a, linewidth=1.5)
    if b.size: plt.plot(np.arange(b.size), b, label=label_b, linewidth=1.5)
    # plt.title(title); 
    plt.xlabel(x_label); plt.ylabel(y_label); plt.grid(True, alpha=0.3); plt.legend()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()


def plot_bar_two(values: Dict[str, float], title: str, y_label: str, save_path: Path):
    labels = list(values.keys()); vals = [values[k] for k in labels]
    plt.figure(figsize=(6.2,4.2))
    x = np.arange(len(labels))
    plt.bar(x, vals)
    plt.xticks(x, labels)
    # plt.title(title); 
    plt.ylabel(y_label)
    plt.grid(True, axis="y", alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()


def ecdf(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = np.sort(v); n = v.size
    if n == 0: return np.array([]), np.array([])
    y = np.arange(1, n+1) / n
    return v, y


def _slug(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣_]+", "_", s)


def _crop_series(y: np.ndarray, start: Optional[int], end: Optional[int]) -> np.ndarray:
    """Apply results/ time window cropping to a 1D series."""
    if y.size == 0:
        return y
    s = 0 if start is None else max(0, int(start))
    e = (y.size - 1) if end is None else min(y.size - 1, int(end))
    if e < s:
        return np.array([], dtype=float)
    return y[s:e+1]

# =========================
# results/ helpers (multi-test)
# =========================

def safe_read_series(p: Path) -> np.ndarray:
    arr = read_numbers_series(p)
    return arr if arr is not None else np.array([], dtype=float)


def _read_series_if_exists(base: Path, rel: List[str]) -> np.ndarray:
    return safe_read_series(base / RESULTS_REL / Path(*rel))


def _plot_kv_multi(rel: List[str], title: str, xlab: str, ylab: str, outpath: Path,
                  tests: Dict[str, Path], pct_scale: bool = False,
                  x_ticks: Optional[List[float]] = None,
                  markers_only: bool = True):
    """Draw key:value curves without inventing intermediate x-values.
    - pct_scale=False: keep x as-is (e.g., 0.1, 0.2, ...)
    - x_ticks: if provided, fix ticks exactly to these values (no extra ticks)
    - markers_only=True: plot discrete points only (no interpolating lines)
    """
    plt.figure(figsize=(8.4, 4.4))
    any_data = False; use_pct = False
    for name, root in tests.items():
        kind, kv = detect_and_read(root / RESULTS_REL / Path(*rel))
        if kind == "kv" and kv:
            x, y = kv
            if pct_scale and x.size and np.nanmax(x) <= 1.1:
                x = x * 100.0; use_pct = True
            if markers_only:
                plt.plot(x, y, marker="o", linestyle="None", label=name, linewidth=1.5)
            else:
                plt.plot(x, y, marker="o", label=name, linewidth=1.5)
            any_data = True
    if not any_data:
        plt.close(); return
    # plt.title(title)
    plt.xlabel("Throughput degradation (%)" if use_pct else xlab)
    plt.ylabel(ylab)
    if x_ticks is not None and len(x_ticks) > 0:
        plt.xticks(x_ticks, [str(v) for v in x_ticks])
        plt.xlim(min(x_ticks) - 0.02, max(x_ticks) + 0.02)
    plt.grid(True, alpha=0.3); plt.legend()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close(); return
    # plt.title(title);
    plt.xlabel("Throughput degradation (%)" if use_pct else xlab); plt.ylabel(ylab)
    plt.grid(True, alpha=0.3); plt.legend()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()

# =========================
# Loss visualization (grid data)
# =========================

def gather_loss_timeseries(root: Path, t0: int, t1: int) -> Dict[str, np.ndarray]:
    T = t1 - t0 + 1
    base = root / GRID_REL
    out = {k: np.zeros(T, dtype=float) for k in
           ["total_ul_loss","total_dl_loss","total_link_loss","links_with_loss","blocked_total","est_total_loss"]}
    for t in range(t0, t1 + 1):
        idx = t - t0
        d = base / str(t)
        ul = read_numbers_series(d / FILE_UL)
        dl = read_numbers_series(d / FILE_DL)
        ll = read_numbers_series(d / FILE_LL)
        blk= read_numbers_series(d / FILE_BLK)

        if ul is not None and ul.size: out["total_ul_loss"][idx] = float(np.sum(ul))
        else: warn(f"Loss (UL) missing: {d/FILE_UL}")
        if dl is not None and dl.size: out["total_dl_loss"][idx] = float(np.sum(dl))
        else: warn(f"Loss (DL) missing: {d/FILE_DL}")
        if ll is not None and ll.size:
            out["total_link_loss"][idx] = float(np.sum(ll))
            out["links_with_loss"][idx] = int(np.sum(ll > 0))
        else:
            warn(f"Loss (all links) missing: {d/FILE_LL}")
        if blk is not None and blk.size:
            out["blocked_total"][idx] = float(np.sum(blk))
        out["est_total_loss"][idx] = out["total_ul_loss"][idx] + out["total_dl_loss"][idx] + out["blocked_total"][idx]
    return out

# =========================
# Paper figures (as requested)
# =========================

def ensure_tests(entries: List[Tuple[str, Path]]) -> Dict[str, Path]:
    tests: Dict[str, Path] = {}
    for label, p in entries:
        if p.exists():
            tests[label] = p
        else:
            warn(f"Missing folder: {p} (label='{label}')")
    if not tests:
        raise SystemExit("No test folders found. Please check your paths.")
    return tests


def _pick_two_tests(tests: Dict[str, Path]) -> Tuple[Tuple[str, Path], Tuple[str, Path]]:
    base = None; pcdr = None
    for name, p in tests.items():
        if name == BASELINE_LABEL:
            base = (name, p)
        if name == PCDR_PRIMARY_LABEL:
            pcdr = (name, p)
    if base is None:
        # fallback: pick the first as baseline
        base = next(iter(tests.items()))
    if pcdr is None:
        # fallback: pick any test different from baseline
        for name, p in tests.items():
            if name != base[0]:
                pcdr = (name, p); break
    if pcdr is None:
        raise SystemExit("Need at least two test folders for before/after comparison.")
    return base, pcdr


def figure_1_heatmaps(tests: Dict[str, Path], time_s: int = T_END):
    """Fig-1a/1b/1c: ISL load heatmaps at t=100s for baseline vs PCDR vs K-DS."""
    out_dir = OUT_DIR / "paper_figures" / "fig1_heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 우리가 보고 싶은 라벨들만 순서대로 지정
    wanted = [BASELINE_LABEL, PCDR_PRIMARY_LABEL, "K-DS"]
    tag_map = {
        BASELINE_LABEL:      "Fig-1a",
        PCDR_PRIMARY_LABEL:  "Fig-1b",
        "K-DS":               "Fig-1c",
    }

    for name in wanted:
        base = tests.get(name)
        if base is None:
            warn(f"Fig1 heatmap: test '{name}' not found in TESTS dict")
            continue

        txt = base / GRID_REL / str(time_s) / FILE_ISL
        arr1d = load_values_1d(txt, expect_len=SAT_COUNT)
        if arr1d.size == 0:
            warn(f"Fig1 heatmap source missing: {txt}")
            continue

        grid = reshape_grid(arr1d)
        tag = tag_map.get(name, "Fig-1x")
        title = f"{tag}: ISL load heatmap at t={time_s}s — {name}"
        fname = f"{tag.lower()}_isl_heatmap_t{time_s}_{_slug(name)}.png"
        plot_heatmap(grid, title, out_path=out_dir / fname,
                     vmin_vmax=None, show=False, cbar_label="ISL load")



def figure_2_metrics_timeseries(tests: Dict[str, Path], t0: int = T_START, t1: int = T_END):
    """Fig-2a/2b: Time-series (0..100s) of CV and #active links (x>0) for baseline vs PCDR vs K-DS."""
    out_dir = OUT_DIR / "paper_figures" / "fig2_timeseries"
    out_dir.mkdir(parents=True, exist_ok=True)
    (b_name, b_path), (p_name, p_path) = _pick_two_tests(tests)

    # K-DS 테스트 폴더 찾기
    k_name, k_path = None, None
    for name, path in tests.items():
        if "K-DS" in name:
            k_name, k_path = name, path
            break

    # compute metrics time series (ISL)
    m_base = time_metrics_for_dir(b_path, t0, t1, series_file=FILE_ISL)
    m_pcdr = time_metrics_for_dir(p_path, t0, t1, series_file=FILE_ISL)
    m_kds  = time_metrics_for_dir(k_path, t0, t1, series_file=FILE_ISL) if k_path is not None else None

    times = np.arange(t0, t1 + 1)

    # Fig-2a: CV timeseries
    plt.figure(figsize=(8.4, 4.4))
    plt.plot(times, np.ma.masked_invalid(m_base["cv"]), label=b_name, linewidth=1.5)
    plt.plot(times, np.ma.masked_invalid(m_pcdr["cv"]), label=p_name, linewidth=1.5)
    if m_kds is not None:
        plt.plot(times, np.ma.masked_invalid(m_kds["cv"]), label=k_name, linewidth=1.5)
    plt.xlabel("Time (s)"); plt.ylabel("CV (std/mean)")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(out_dir / "fig2a_cv_timeseries.png", dpi=200); plt.close()

    # Fig-2b: # links with x>0 timeseries
    plt.figure(figsize=(8.4, 4.4))
    plt.plot(times, np.ma.masked_invalid(m_base["active_count_pos"]), label=b_name, linewidth=1.5)
    plt.plot(times, np.ma.masked_invalid(m_pcdr["active_count_pos"]), label=p_name, linewidth=1.5)
    if m_kds is not None:
        plt.plot(times, np.ma.masked_invalid(m_kds["active_count_pos"]), label=k_name, linewidth=1.5)
    plt.xlabel("Time (s)"); plt.ylabel("# links with x>0")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(out_dir / "fig2b_active_links_timeseries.png", dpi=200); plt.close()



def figure_3_total_loss_timeseries(tests: Dict[str, Path], t0: int = T_START, t1: int = T_END):
    """Fig-3: Total loss time-series (estimated = UL + DL + blocked), identical to previous loss_total_est_multi."""
    out_dir = OUT_DIR / "paper_figures" / "fig3_total_loss"; out_dir.mkdir(parents=True, exist_ok=True)

    times = np.arange(t0, t1 + 1)
    plt.figure(figsize=(8.4, 4.4))
    any_data = False
    for name, root in tests.items():
        if (root / GRID_REL).exists():
            d = gather_loss_timeseries(root, t0, t1)
            y = np.ma.masked_invalid(d.get("est_total_loss", np.full_like(times, np.nan, dtype=float)))
            if np.any(np.isfinite(y)):
                plt.plot(times, y, label=name, linewidth=1.4)
                any_data = True
        else:
            warn(f"Missing grid/link data folder: {root/GRID_REL} (label='{name}')")
    if not any_data:
        plt.close(); return
    # plt.title("Fig-3: Total loss (estimated: UL+DL+blocked)")
    plt.xlabel("Time (s)"); plt.ylabel("Loss (Mb)")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(out_dir / "fig3_total_loss_timeseries.png", dpi=200); plt.close()


def figure_4_attack_cost_curves(tests: Dict[str, Path]):
    """Fig-4a/b/c/d: Attack cost curves from results/ files.
       - 4a: SKYFALL — Botnet size vs target degradation
       - 4b: SKYFALL — # infected regions (blocks) vs target degradation
       - 4c: ICARUS  — Botnet size vs target degradation
       - 4d: ICARUS  — # infected regions (blocks) vs target degradation
    """
    out_dir = OUT_DIR / "paper_figures" / "fig4_attack_cost"; out_dir.mkdir(parents=True, exist_ok=True)
    tests_map = {name: path for name, path in tests.items()}

    # Fixed x-axis ticks at decimal degradation levels; do not invent midpoints
    xticks = [0.1, 0.2, 0.3, 0.4, 0.5]

    # 4a/4c: botnet size (KV files) — decimals on x-axis, plot discrete markers only
    _plot_kv_multi(["fig-12a","botnet_size_for_skyfall.txt"],
                   "Fig-4a: Botnet size vs degradation (SKYFALL)",
                   "Throughput degradation", "Bots",
                   out_dir / "fig4a_botnet_skyfall.png", tests_map,
                   pct_scale=False, x_ticks=xticks, markers_only=False)

    _plot_kv_multi(["fig-12a","botnet_size_for_icarus.txt"],
                   "Fig-4c: Botnet size vs degradation (ICARUS)",
                   "Throughput degradation", "Bots",
                   out_dir / "fig4c_botnet_icarus.png", tests_map,
                   pct_scale=False, x_ticks=xticks, markers_only=False)

    # 4b/4d: number of infected regions/blocks (KV files) — decimals on x-axis, discrete markers only
    _plot_kv_multi(["fig-13a","number_blocks_skyfall_+grid.txt"],
                   "Fig-4b: Infected regions vs degradation (+Grid, SKYFALL)",
                   "Throughput degradation", "Regions (blocks)",
                   out_dir / "fig4b_regions_skyfall.png", tests_map,
                   pct_scale=False, x_ticks=xticks, markers_only=False)

    _plot_kv_multi(["fig-13a","number_blocks_icarus_+grid.txt"],
                   "Fig-4d: Infected regions vs degradation (+Grid, ICARUS)",
                   "Throughput degradation", "Regions (blocks)",
                   out_dir / "fig4d_regions_icarus.png", tests_map,
                   pct_scale=False, x_ticks=xticks, markers_only=False)


# =========================
# Main
# =========================
if __name__ == "__main__":
    tests = ensure_tests(TESTS)

    # (A) Core analysis outputs (still available for debugging/inspection)
    #  - Time-series load metrics (ISL) → optional comprehensive figure
    metrics_all = {name: time_metrics_for_dir(path, T_START, T_END, series_file=FILE_ISL)
                   for name, path in tests.items()}
    # Keep a single summary figure for completeness
    times = np.arange(T_START, T_END + 1)
    panels = [
        ("cv",               "CV = std/mean (↓ better)"),
        ("par",              "PAR = max/mean (↓ better)"),
        ("p95_over_mean",    "p95/mean (↓ better)"),
        ("sat_frac",         "High-load share x ≥ mean+std (↓)"),
        ("aat_over_thr",     "Normalized excess area (↓)"),
        ("jain_active",      "ε-active Jain (↑)"),
        ("gini_active",      "ε-active Gini (↓)"),
        ("n_eff_active",     "Effective # active links (↑)"),
        ("active_count_pos", "# links with x>0 (↑)"),
    ]
    # quick compact panel plot
    fig, axs = plt.subplots(3, 3, figsize=(15, 10)); axs = axs.flatten()
    for i, (key, title) in enumerate(panels):
        ax = axs[i]
        for name, m in metrics_all.items():
            y = np.ma.masked_invalid(m.get(key, np.full_like(times, np.nan, dtype=float)))
            ax.plot(times, y, label=name, linewidth=1)
        ax.set_title(title); ax.set_xlabel("Time (s)"); ax.grid(True, alpha=0.3)
        span = int(T_END - T_START); step = 600 if span >= 600 else max(1, span // 5 or 1)
        ax.set_xticks(list(range(T_START, T_END + 1, step)))
        if key in ("sat_frac", "jain_active", "gini_active"):
            ax.set_ylim(0, 1)
    handles, labels = axs[0].get_legend_handles_labels()
    if handles: axs[0].legend(loc="best")
    fig.suptitle("Load-balancing metrics over time", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    (OUT_DIR / "aux").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "aux" / "load_imbalance_metrics.png", dpi=200)
    plt.close(fig)

    # (B) Requested paper figures
    figure_1_heatmaps(tests, time_s=T_END)
    info(f"Saved Fig-1 heatmaps under: {OUT_DIR/'paper_figures'/'fig1_heatmaps'}")

    figure_2_metrics_timeseries(tests, t0=T_START, t1=T_END)
    info(f"Saved Fig-2 time-series under: {OUT_DIR/'paper_figures'/'fig2_timeseries'}")

    figure_3_total_loss_timeseries(tests, t0=T_START, t1=T_END)
    info(f"Saved Fig-3 total loss under: {OUT_DIR/'paper_figures'/'fig3_total_loss'}")

    figure_4_attack_cost_curves(tests)
    info(f"Saved Fig-4 attack-cost curves under: {OUT_DIR/'paper_figures'/'fig4_attack_cost'}")

    # (C) Also export raw heatmaps for ISL load + GSL UL/DL loss at t=100 (for appendix/debug)
    export_heatmaps_for_tests(HEATMAP_TIMES, tests, filename=FILE_ISL, title_tag="isl_traffic", vmin_vmax=None, show=False)
    export_heatmaps_for_tests(HEATMAP_TIMES, tests, filename=FILE_DL,  title_tag="downlink_loss", vmin_vmax=None, show=False)
    export_heatmaps_for_tests(HEATMAP_TIMES, tests, filename=FILE_UL,  title_tag="uplink_loss",   vmin_vmax=None, show=False)
    info(f"Raw heatmaps saved under: {OUT_DIR/'heatmaps'}")

    # (D) Save warnings
    save_warnings()
