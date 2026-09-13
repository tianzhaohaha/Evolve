"""
EvoStream 三面板环形图：Isolated / Sequential / Interleave 三种 stream setting 横排。
数据用“方法 × benchmark”的表格形式填写，Avg 自动求均值。
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch, PathPatch
from matplotlib.transforms import Affine2D
from matplotlib.textpath import TextPath, text_to_path
from matplotlib.font_manager import FontProperties

# ============================================================
# 0. 字体：只用本机存在的，避免 findfont 警告
# ============================================================
def pick_font(candidates=("Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "FreeSans", "DejaVu Sans")):
    installed = {f.name for f in fm.fontManager.ttflist}
    return next((c for c in candidates if c in installed), "DejaVu Sans")

FONT = pick_font()
plt.rcParams["font.family"] = FONT
FP      = FontProperties(family=FONT)
FP_BOLD = FontProperties(family=FONT, weight="bold")

# ============================================================
# 1. 方法 / 颜色 / 数据（表格形式：每行一个方法，每列一个 benchmark）
# ============================================================
legend_order = ["EvoStream (Ours)", "SDAR", "RLSD", "SEED", "OPSD", "GRPO", "Skill-Prompt", "Vanilla"]
bar_order    = legend_order[::-1]        # 组内顺时针顺序，主角在最外侧
hero         = "EvoStream (Ours)"

colors = {
    "EvoStream (Ours)": "#5C54C7",
    "SDAR":             "#BBDDBD",
    "RLSD":             "#F4C5AB",
    "SEED":             "#9DB7D8",
    "OPSD":             "#DFA57E",
    "GRPO":             "#D3BEE3",
    "Skill-Prompt":     "#EFE0A3",
    "Vanilla":          "#CFCFCF",
}

benchmarks = ["AppWorld", "BFCL", "BrowseComp+", "HLE", "SWE", "Tau2"]

# ---- 占位数据：全是随手填的，请替换 ------------------------------------
#      列顺序 = benchmarks；Avg 会自动算，不用填
settings = {
    "Isolated": {
        "EvoStream (Ours)": [62.4, 78.6, 41.3, 24.8, 57.9, 69.5],
        "SDAR":             [55.1, 74.2, 35.7, 20.3, 51.2, 63.8],
        "RLSD":             [52.8, 73.0, 33.9, 19.5, 49.6, 62.1],
        "SEED":             [50.3, 71.5, 32.4, 18.7, 47.8, 60.4],
        "OPSD":             [48.9, 70.8, 30.6, 17.2, 46.1, 58.7],
        "GRPO":             [46.2, 69.1, 29.8, 16.9, 44.3, 57.3],
        "Skill-Prompt":     [41.7, 66.4, 25.1, 14.6, 39.5, 52.6],
        "Vanilla":          [36.5, 63.2, 21.4, 12.1, 34.8, 48.9],
    },
    "Sequential": {
        "EvoStream (Ours)": [60.1, 77.4, 39.8, 23.6, 56.2, 68.0],
        "SDAR":             [51.9, 72.0, 33.2, 18.9, 48.7, 61.5],
        "RLSD":             [49.4, 70.6, 31.5, 18.0, 46.9, 59.7],
        "SEED":             [47.2, 69.3, 29.9, 17.1, 45.0, 58.1],
        "OPSD":             [45.5, 68.4, 28.3, 15.8, 43.2, 56.3],
        "GRPO":             [43.0, 66.9, 27.4, 15.3, 41.6, 54.8],
        "Skill-Prompt":     [38.6, 64.1, 23.0, 13.2, 36.9, 50.2],
        "Vanilla":          [33.8, 61.0, 19.6, 11.0, 32.4, 46.5],
    },
    "Interleave": {
        "EvoStream (Ours)": [58.7, 76.5, 38.2, 22.9, 54.8, 66.9],
        "SDAR":             [49.3, 70.4, 31.0, 17.6, 46.1, 59.2],
        "RLSD":             [47.0, 69.1, 29.4, 16.8, 44.5, 57.6],
        "SEED":             [44.9, 67.8, 28.0, 15.9, 42.7, 55.9],
        "OPSD":             [43.1, 66.7, 26.6, 14.7, 40.9, 54.2],
        "GRPO":             [40.8, 65.2, 25.7, 14.1, 39.3, 52.7],
        "Skill-Prompt":     [36.2, 62.5, 21.5, 12.0, 34.6, 48.1],
        "Vanilla":          [31.4, 59.3, 18.2, 10.1, 30.2, 44.3],
    },
}

# 每个面板中心的标题 / 副标题（副标题占位，请替换）
panel_titles = {
    "Isolated":   ("Isolated",   "Stream setting (a)\n(replace me)"),
    "Sequential": ("Sequential", "Stream setting (b)\n(replace me)"),
    "Interleave": ("Interleave", "Stream setting (c)\n(replace me)"),
}

def table_to_data(table, benchmarks, methods):
    """方法×benchmark 表 → {metric: {method: value}}，并追加 Avg。"""
    data = {b: {m: float(table[m][i]) for m in methods} for i, b in enumerate(benchmarks)}
    data["Avg"] = {m: float(np.mean(table[m])) for m in methods}
    return data

metric_order = benchmarks + ["Avg"]      # 顺时针顺序；Avg 会被放到 12 点方向

def value_string(metric, value):
    return "0" if value == 0 else f"{value:.1f}"

# ============================================================
# 2. 文本工具
# ============================================================
def normalize_rotation(a):
    a = (a + 180) % 360 - 180
    if a > 90:  a -= 180
    if a < -90: a += 180
    return a

def ink_extents(s, fs, fp=FP):
    return TextPath((0, 0), s, size=fs, prop=fp).get_extents().extents
def text_width_pts(s, fs, fp=FP):  x0, _, x1, _ = ink_extents(s, fs, fp); return x1 - x0
def text_height_pts(s, fs, fp=FP): _, y0, _, y1 = ink_extents(s, fs, fp); return y1 - y0
def advance_pts(s, fs, fp=FP):
    if not s: return 0.0
    w, _, _ = text_to_path.get_text_width_height_descent(s, fp, ismath=False)
    return w * fs / (fp.get_size_in_points() or 12.0)

class GlyphPatch(PathPatch):
    """把 TextPath 轮廓贴到极坐标点 (theta, r)，anchor 是轮廓坐标系里对到该点的位置（pt）。"""
    def __init__(self, ax, theta, r, tp, anchor, rot_deg, **kw):
        self._base = Affine2D().translate(-anchor[0], -anchor[1]).rotate_deg(rot_deg).scale(1 / 72)
        self._ax, self._xy = ax, (theta, r)
        kw.setdefault("edgecolor", "none"); kw.setdefault("clip_on", False)
        super().__init__(tp, **kw)
    def get_transform(self):
        px, py = self._ax.transData.transform(self._xy)
        return self._base + self._ax.figure.dpi_scale_trans + Affine2D().translate(px, py)

def put_label_centered(ax, theta, r, s, fs, rot_deg, fp=FP, color="white", zorder=5):
    tp = TextPath((0, 0), s, size=fs, prop=fp)
    x0, y0, x1, y1 = tp.get_extents().extents
    ax.add_patch(GlyphPatch(ax, theta, r, tp, ((x0 + x1) / 2, (y0 + y1) / 2), rot_deg,
                            facecolor=color, zorder=zorder))

def put_arc_text(ax, center_deg, r_mid, s, fs, ppu, fp=FP_BOLD, color="black", zorder=6,
                 sup_scale=0.72, sup_raise=0.38):
    """沿弧逐字排版；整条标签按中心角统一决定是否翻转（下半圈翻转）。'*' 画成上标。"""
    flip = 90 < (center_deg % 360) < 270
    cap = text_height_pts("H", fs, fp)
    chars, adv = [], []
    for ch in s:
        if ch == "*":
            chars.append(("*", fs * sup_scale, sup_raise * fs)); adv.append(advance_pts("*", fs * sup_scale, fp))
        else:
            chars.append((ch, fs, 0.0)); adv.append(advance_pts(ch, fs, fp))
    total = sum(adv)
    r_base = r_mid - (cap / 2) / ppu if not flip else r_mid + (cap / 2) / ppu
    ang_per_pt = 1.0 / (r_base * ppu)
    d = -1 if flip else 1
    theta0 = np.deg2rad(center_deg) - d * total / 2 * ang_per_pt
    x = 0.0
    for (ch, size, raise_pt), a in zip(chars, adv):
        th = theta0 + d * (x + a / 2) * ang_per_pt
        rot = -np.rad2deg(th) + (180 if flip else 0)
        ax.add_patch(GlyphPatch(ax, th, r_base, TextPath((0, 0), ch, size=size, prop=fp),
                                (a / 2, -raise_pt), rot, facecolor=color, zorder=zorder))
        x += a

# ============================================================
# 3. 单个环形图
# ============================================================
def draw_radial(ax, data, metric_order, bar_order, colors, hero, fmt, font_scale=1.0,
                inner_radius=2.30, max_bar_height=2.60, zero_height=0.26,
                group_gap_deg=4.5, bar_gap_deg=0.45,
                fs_min=5.0, fs_max=10.5, label_pad_pts=2.2, zero_fill=0.62,
                metric_fs=6.4, arc_lw=1.3, avg_on_top=True):
    n_g, n_b = len(metric_order), len(bar_order)
    span = 360.0 / n_g
    bar_step = (span - group_gap_deg) / n_b
    bar_w = np.deg2rad(bar_step - bar_gap_deg)
    start = -(n_g - 1) * span if avg_on_top else -span / 2     # 最后一项落在 0°
    centers = {m: start + i * span for i, m in enumerate(metric_order)}

    fs_min, fs_max, metric_fs, label_pad_pts = (v * font_scale for v in (fs_min, fs_max, metric_fs, label_pad_pts))
    arc_lw *= font_scale

    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1); ax.set_axis_off()
    ax.set_ylim(0, inner_radius + max_bar_height + 0.35)
    fig = ax.figure
    p0, p1 = ax.transData.transform((0, 0)), ax.transData.transform((0, 1))
    ppu = np.hypot(*(p1 - p0)) * 72 / fig.dpi
    pad = label_pad_pts / ppu

    def font_for(h):
        t = np.clip((h - zero_height) / (max_bar_height - zero_height), 0, 1)
        return fs_min + (fs_max - fs_min) * t ** 0.6

    for metric in metric_order:
        c_deg = centers[metric]
        raw = np.array([data[metric][m] for m in bar_order], float)
        labels = [fmt(metric, v) for v in raw]
        fps = [FP_BOLD if m == hero else FP for m in bar_order]
        heights = np.where(raw > 0, zero_height + raw / raw.max() * (max_bar_height - zero_height), zero_height)
        zero_fs = {}
        for k, (v, s, fp) in enumerate(zip(raw, labels, fps)):
            heights[k] = max(heights[k], text_width_pts(s, fs_min, fp) / ppu + 2 * pad)
            if v == 0:
                zero_fs[k] = zero_fill * inner_radius * bar_w * ppu / (text_height_pts(s, 10, fp) / 10)
                heights[k] = text_width_pts(s, zero_fs[k], fp) / ppu + 2 * pad
        offs = (np.arange(n_b) - (n_b - 1) / 2) * bar_step
        for k, (m, s, h, off, fp) in enumerate(zip(bar_order, labels, heights, offs, fps)):
            td = c_deg + off; th = np.deg2rad(td)
            ax.bar(th, h, width=bar_w, bottom=inner_radius, color=colors[m],
                   edgecolor="white", linewidth=0.6 * font_scale, zorder=3)
            if k in zero_fs:
                fs, r_t = zero_fs[k], inner_radius + h / 2
            else:
                fs = font_for(h)
                avail = h * ppu - 2 * label_pad_pts
                if text_width_pts(s, fs, fp) > avail:
                    fs = max(fs_min, fs * avail / text_width_pts(s, fs, fp))
                L = text_width_pts(s, fs, fp) / ppu
                r_t = max(inner_radius + h - pad - L / 2, inner_radius + pad + L / 2)
                fs = min(fs, 0.9 * r_t * bar_w * ppu / (text_height_pts(s, fs, fp) / fs))
            put_label_centered(ax, th, r_t, s, fs, normalize_rotation(90 - td), fp=fp)
        half = (span - group_gap_deg) / 2
        at = np.deg2rad(np.linspace(c_deg - half, c_deg + half, 80))
        ax.plot(at, np.full_like(at, inner_radius - 0.09), color="black", lw=arc_lw, solid_capstyle="butt", zorder=4)
        put_arc_text(ax, c_deg, inner_radius - 0.29, metric, metric_fs, ppu)

def add_center_title(ax, title, subtitle, font_scale=1.0, color="#30259B", title_fs=22, sub_fs=7.4):
    common = dict(xy=(0, 0), xycoords="data", textcoords="offset points", ha="center", va="center",
                  fontweight="bold", fontstyle="italic", color=color, annotation_clip=False)
    ax.annotate(title, xytext=(0, 20 * font_scale), fontsize=title_fs * font_scale, **common)
    ax.annotate(subtitle, xytext=(0, -15 * font_scale), fontsize=sub_fs * font_scale, linespacing=1.08, **common)

# ============================================================
# 4. 三面板大图
# ============================================================
if __name__ == "__main__":
    panel_in = 6.0                      # 每个面板的边长（英寸）；整图缩到论文里时字号按比例缩小
    n = len(settings)
    gap_in, legend_in = 0.25, 1.0
    legend_fs = 14                      # 图例字号；色块尺寸随字号等比放大
    fig_w = n * panel_in + (n - 1) * gap_in + 0.3
    fig_h = panel_in + legend_in + 0.2
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    for i, (name, table) in enumerate(settings.items()):
        x0 = (0.15 + i * (panel_in + gap_in)) / fig_w
        ax = fig.add_axes([x0, (legend_in + 0.1) / fig_h, panel_in / fig_w, panel_in / fig_h], projection="polar")
        data = table_to_data(table, benchmarks, legend_order)
        draw_radial(ax, data, metric_order, bar_order, colors, hero, value_string)
        title, sub = panel_titles[name]
        add_center_title(ax, title, sub)

    handles = [Patch(facecolor=colors[m], edgecolor="none", label=m) for m in legend_order]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=len(legend_order),
               frameon=False, fontsize=legend_fs,
               handlelength=2.4, handleheight=1.0,      # 色块长/高，单位为字号倍数
               columnspacing=2.0, handletextpad=0.6)

    plt.savefig("evostream_radial_3panel.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.savefig("evostream_radial_3panel.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.show()