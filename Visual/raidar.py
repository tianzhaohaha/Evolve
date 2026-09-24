import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, PathPatch
from matplotlib.transforms import Affine2D
from matplotlib.textpath import TextPath, text_to_path
from matplotlib.font_manager import FontProperties

# ============================================================
# 0. 字体：优先 Arial/Helvetica，没有就退到 Liberation Sans（Arial 同度量）
# ============================================================
from matplotlib import font_manager as fm
def pick_font(candidates=("Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "FreeSans", "DejaVu Sans")):
    installed = {f.name for f in fm.fontManager.ttflist}
    return next((c for c in candidates if c in installed), "DejaVu Sans")

FONT = pick_font()
plt.rcParams["font.family"] = FONT
FP      = FontProperties(family=FONT)
FP_BOLD = FontProperties(family=FONT, weight="bold")

# ============================================================
# 1. Data
# ============================================================
# legend_order = ["EvoStream (Ours)", "SDAR", "RLSD", "SEED", "OPSD", "GRPO", "Skill-Prompt", "Vanilla"]
legend_order = ["BORDER", "SEED", "GRPO", "OPSD", "Vanilla"]

bar_order    = legend_order[::-1]          # 组内顺时针顺序：Vanilla ... EvoStream，主角在最外侧
hero         = "BORDER"          # 主角方法：数值标签加粗

# colors = {
#     "BORDER (Ours)": "#5C54C7",   # 主角深紫
#     # "SDAR":             "#BBDDBD",
#     # "RLSD":             "#F4C5AB",
#     "SEED":             "#9DB7D8",
#     "OPSD":             "#DFA57E",
#     "GRPO":             "#D3BEE3",
#     # "Skill-Prompt":     "#EFE0A3",
#     "Vanilla":          "#CFCFCF",   # 基线用灰色
# }

colors = {
    "BORDER": "#5C54C7",   # 主角深紫
    # "SDAR":             "#BBDDBD",
    # "RLSD":             "#F4C5AB",
    "SEED":             "#BBDDBD",
    "OPSD":             "#F4C5AB",
    "GRPO":             "#9DB7D8",
    # "Skill-Prompt":     "#EFE0A3",
    "Vanilla":          "#CFCFCF",   # 基线用灰色
}



# ---- 占位数据：两种模式暂时沿用原有分数，请分别替换成真实结果（0–100）----
benchmark_order = ["BFCL", "BrowseComp+", "Tau2"]
mode_order = ["Isolated", "Interleaved"]
data = {
    "BFCL": {
        "Isolated":    {"BORDER": 78.6, "SEED": 71.5, "OPSD": 70.8, "GRPO": 69.1, "Vanilla": 63.2},
        "Interleaved": {"BORDER": 78.6, "SEED": 71.5, "OPSD": 70.8, "GRPO": 69.1, "Vanilla": 63.2},
    },
    "BrowseComp+": {
        "Isolated":    {"BORDER": 41.3, "SEED": 32.4, "OPSD": 30.6, "GRPO": 29.8, "Vanilla": 21.4},
        "Interleaved": {"BORDER": 41.3, "SEED": 32.4, "OPSD": 30.6, "GRPO": 29.8, "Vanilla": 21.4},
    },
    "Tau2": {
        "Isolated":    {"BORDER": 69.5, "SEED": 60.4, "OPSD": 58.7, "GRPO": 57.3, "Vanilla": 48.9},
        "Interleaved": {"BORDER": 69.5, "SEED": 60.4, "OPSD": 58.7, "GRPO": 57.3, "Vanilla": 48.9},
    },
}
# Avg 只有一组：对三个 benchmark × 两种模式等权求均值。
# 如果有自己的 Avg 数值，直接用 data["Avg"] = {方法名: 分数, ...} 覆盖即可。
data["Avg"] = {
    m: float(np.mean([data[b][mode][m] for b in benchmark_order for mode in mode_order]))
    for m in legend_order
}

# 按模式分区，每个模式依次排列三个 benchmark；None 表示 Avg 不区分模式。
group_order = [(b, mode) for mode in mode_order for b in benchmark_order] + [("Avg", None)]

# ============================================================
# 2. 几何参数（六个模式分组 + 一个 Avg，360° 均分）
# ============================================================
n_groups, n_bars = len(group_order), len(bar_order)
group_span_deg = 360.0 / n_groups
group_gap_deg  = 4.5
bar_gap_deg    = 0.45
bar_step_deg   = (group_span_deg - group_gap_deg) / n_bars
bar_width_deg  = bar_step_deg - bar_gap_deg
bar_width      = np.deg2rad(bar_width_deg)
start_deg      = -(n_groups - 1) * group_span_deg   # 最后一组 Avg 居于 0°（正上方）
group_center_deg = {group: start_deg + i * group_span_deg for i, group in enumerate(group_order)}

inner_radius   = 3.00      # 为双层标注留空间，同时保留中心标题的空白区
max_bar_height = 2.60
zero_height    = 0.26

fs_min, fs_max = 5.0, 10.5
label_pad_pts  = 2.2
zero_fill      = 0.62

mode_fs        = 8
mode_text_color = "#666666"      # 深灰模式名，与黑色 benchmark 区分且保持可读性
benchmark_fs   = 8
arc_lw         = 0.6
benchmark_arc_color = "#D0D0D5"   # 外层细浅灰线，弱化装饰的视觉权重
benchmark_arc_r = inner_radius - 0.09
benchmark_label_r = inner_radius - 0.29   # 外层：紧邻柱子的 benchmark 名（包括 Avg）
mode_label_r   = inner_radius - 0.68   # 模式文字位置保持不变
mode_band_color = "#F2F2F5"        # 两种模式共用浅灰色，避免与方法配色混淆
mode_band_inner_r = mode_label_r - 0.16
mode_band_outer_r = mode_label_r + 0.16

# ============================================================
# 3. 文本工具
# ============================================================
def normalize_rotation(a):
    a = (a + 180) % 360 - 180
    if a > 90:  a -= 180
    if a < -90: a += 180
    return a

def value_string(value):
    return "0" if value == 0 else f"{value:.1f}"

def ink_extents(s, fs, fp=FP):
    return TextPath((0, 0), s, size=fs, prop=fp).get_extents().extents

def text_width_pts(s, fs, fp=FP):  x0, _, x1, _ = ink_extents(s, fs, fp); return x1 - x0
def text_height_pts(s, fs, fp=FP): _, y0, _, y1 = ink_extents(s, fs, fp); return y1 - y0

def advance_pts(s, fs, fp=FP):
    """字符串的排版前进宽度（pt），用于沿弧排字。"""
    if not s: return 0.0
    w, _, _ = text_to_path.get_text_width_height_descent(s, fp, ismath=False)
    return w * fs / fp.get_size_in_points() if fp.get_size_in_points() else w * fs / 12.0

class GlyphPatch(PathPatch):
    """
    把一段 TextPath 轮廓贴到极坐标点 (theta, r) 上：
    anchor 是轮廓自身坐标系里（pt）要对到该点的位置，rot_deg 是屏幕旋转角。
    直接画轮廓避免了 ax.text 旋转时的取整/hinting 偏移。
    """
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

def put_arc_text(ax, center_deg, r_mid, s, fs, pts_per_unit, fp=FP_BOLD, color="black", zorder=6,
                 sup_scale=0.72, sup_raise=0.38):
    """
    沿半径 r_mid 的圆弧逐字排版，文字中线落在 r_mid。
    上半圈顺时针读、字头朝外；下半圈逆时针读、字头朝内（都保证不倒）。
    末尾的 '*' 以上标形式绘制。
    """
    flip = normalize_rotation(-center_deg) != (-center_deg + 180) % 360 - 180  # 是否翻转（下半圈）
    cap  = text_height_pts("H", fs, fp)                                        # 大写字高
    # 逐字符的前进量（'*' 上标缩小）
    chars, adv = [], []
    for ch in s:
        if ch == "*":
            chars.append(("*", fs * sup_scale, sup_raise * fs)); adv.append(advance_pts("*", fs * sup_scale, fp))
        else:
            chars.append((ch, fs, 0.0)); adv.append(advance_pts(ch, fs, fp))
    total = sum(adv)
    # 字基线所在半径：文字中线 r_mid 上下各半个字高
    r_base = r_mid - (cap / 2) / pts_per_unit if not flip else r_mid + (cap / 2) / pts_per_unit
    ang_per_pt = 1.0 / (r_base * pts_per_unit)               # 弧长 1pt 对应的弧度
    direction = 1 if not flip else -1                        # 顺时针 / 逆时针
    theta0 = np.deg2rad(center_deg) - direction * total / 2 * ang_per_pt
    x = 0.0
    for (ch, size, raise_pt), a in zip(chars, adv):
        th = theta0 + direction * (x + a / 2) * ang_per_pt
        # 整个词采用一致朝向，避免跨过侧边时个别字符单独翻转。
        rot = -np.rad2deg(th) + (180 if flip else 0)
        tp = TextPath((0, 0), ch, size=size, prop=fp)
        # 锚点：本字前进宽度中点、基线（上标则再抬高）
        ax.add_patch(GlyphPatch(ax, th, r_base, tp, (a / 2, -raise_pt), rot,
                                facecolor=color, zorder=zorder))
        x += a

# ============================================================
# 4. 画布
# ============================================================
# 压缩圆图与单行图例之间的留白，不改变内部半径和分组间距。
fig = plt.figure(figsize=(6.2, 6.3), facecolor="white")
ax = fig.add_axes([0.02, 0.1, 0.96, 0.89], projection="polar")
ax.set_theta_zero_location("N"); ax.set_theta_direction(-1); ax.set_axis_off()
ax.set_ylim(0, inner_radius + max_bar_height + 0.25)

fig.canvas.draw()   # 先确定极坐标轴的实际尺寸，再将文字 pt 换算为半径单位
_p0, _p1 = ax.transData.transform((0, 0)), ax.transData.transform((0, 1))
pts_per_unit = np.hypot(*(_p1 - _p0)) * 72 / fig.dpi
pad_units = label_pad_pts / pts_per_unit

def font_for_height(h):
    t = np.clip((h - zero_height) / (max_bar_height - zero_height), 0, 1)
    return fs_min + (fs_max - fs_min) * t ** 0.6

# ============================================================
# 5. 柱子 + 数值 + 外层 benchmark 标注（包括 Avg）
# ============================================================
for benchmark, mode in group_order:
    center_deg = group_center_deg[(benchmark, mode)]
    scores = data[benchmark] if mode is None else data[benchmark][mode]
    raw = np.array([scores[m] for m in bar_order], float)
    labels = [value_string(v) for v in raw]
    fps = [FP_BOLD if m == hero else FP for m in bar_order]

    scale = raw.max() if raw.max() > 0 else 1.0
    heights = np.where(raw > 0, zero_height + raw / scale * (max_bar_height - zero_height), zero_height)
    zero_fs = {}
    for k, (v, s, fp) in enumerate(zip(raw, labels, fps)):
        heights[k] = max(heights[k], text_width_pts(s, fs_min, fp) / pts_per_unit + 2 * pad_units)
        if v == 0:
            tangential = inner_radius * bar_width * pts_per_unit
            zero_fs[k] = zero_fill * tangential / (text_height_pts(s, 10, fp) / 10)
            heights[k] = text_width_pts(s, zero_fs[k], fp) / pts_per_unit + 2 * pad_units

    offsets_deg = (np.arange(n_bars) - (n_bars - 1) / 2) * bar_step_deg
    for k, (method, s, h, off, fp) in enumerate(zip(bar_order, labels, heights, offsets_deg, fps)):
        theta_deg = center_deg + off; theta = np.deg2rad(theta_deg)
        ax.bar(theta, h, width=bar_width, bottom=inner_radius, color=colors[method],
               edgecolor="white", linewidth=0.6, zorder=3)
        if k in zero_fs:
            fs, r_text = zero_fs[k], inner_radius + h / 2
        else:
            fs = font_for_height(h)
            avail = h * pts_per_unit - 2 * label_pad_pts
            if text_width_pts(s, fs, fp) > avail:
                fs = max(fs_min, fs * avail / text_width_pts(s, fs, fp))
            L = text_width_pts(s, fs, fp) / pts_per_unit
            r_text = max(inner_radius + h - pad_units - L / 2, inner_radius + pad_units + L / 2)
            fs = min(fs, 0.9 * r_text * bar_width * pts_per_unit / (text_height_pts(s, fs, fp) / fs))
        put_label_centered(ax, theta, r_text, s, fs, normalize_rotation(90 - theta_deg), fp=fp)

    half = (group_span_deg - group_gap_deg) / 2
    at = np.deg2rad(np.linspace(center_deg - half, center_deg + half, 80))
    ax.plot(at, np.full_like(at, benchmark_arc_r), color=benchmark_arc_color, lw=arc_lw,
            solid_capstyle="butt", zorder=4)
    put_arc_text(ax, center_deg, benchmark_label_r, benchmark, benchmark_fs, pts_per_unit)

# 内层：用无描边的浅灰环带替代粗黑弧线，跨度和文字位置保持不变。
# Avg 不属于任何模式，其下方不绘制环带，只保留外层名称和细线。
for mode in mode_order:
    centers = [group_center_deg[group] for group in group_order if group[1] == mode]
    center_deg = (centers[0] + centers[-1]) / 2
    half = (centers[-1] - centers[0] + group_span_deg - group_gap_deg) / 2
    at = np.deg2rad(np.linspace(center_deg - half, center_deg + half, 160))
    ax.fill_between(at, mode_band_inner_r, mode_band_outer_r,
                    facecolor=mode_band_color, edgecolor="none", linewidth=0, zorder=1)
    put_arc_text(ax, center_deg, mode_label_r, mode, mode_fs, pts_per_unit, color=mode_text_color)

# ============================================================
# 6. 中心标题 + 图例
# ============================================================
title_color = "#30259B"
cx, cy = fig.transFigure.inverted().transform(ax.transData.transform((0, 0)))
fig.text(cx, cy, "BORDER", ha="center", va="center", fontsize=22,
         fontweight="bold", fontstyle="italic", color=title_color)
# fig.text(cx, cy - 0.030, "Subtitle line 1,\nSubtitle line 2\n(replace me)",
#          ha="center", va="center", fontsize=7.4, fontweight="bold", fontstyle="italic",
#          linespacing=1.08, color=title_color)

handles = [Patch(facecolor=colors[m], edgecolor="none", label=m) for m in legend_order]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.006), ncol=5,
           borderaxespad=0,
           frameon=False, fontsize=12, handlelength=2.1, handleheight=0.9,
           columnspacing=1.8, handletextpad=0.55, labelspacing=0.25)

plt.savefig("evostream_radial.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.savefig("evostream_radial.pdf", bbox_inches="tight", pad_inches=0.1)
plt.show()