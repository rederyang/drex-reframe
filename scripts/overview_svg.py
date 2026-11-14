"""Generate the overview diagram (inline SVG) for the results page.

Colors come from page classes (.o-*), so the SVG follows the page theme.
"""
from __future__ import annotations

import math

VW, VH = 1120, 468


def node(x, y, w, h, title, sub, glyph=None):
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" class="o-node"/>'
    tx = x + w / 2
    if glyph == "land":
        s += f'<rect x="{tx - 20}" y="{y + 12}" width="40" height="22" rx="2" class="o-glyph"/>'
    elif glyph == "port":
        s += f'<rect x="{tx - 10}" y="{y + 8}" width="18" height="30" rx="2" class="o-glyph o-crop-s"/>'
    ty = y + (52 if glyph else 36)
    s += f'<text x="{tx}" y="{ty}" text-anchor="middle" class="o-t1">{title}</text>'
    s += f'<text x="{tx}" y="{ty + 20}" text-anchor="middle" class="o-t2">{sub}</text>'
    return s


def arrow(x1, x2, y, label, sub, drex=False):
    cls = "o-arrow o-drex" if drex else "o-arrow"
    s = f'<line x1="{x1 + 6}" y1="{y}" x2="{x2 - 8}" y2="{y}" class="{cls}" marker-end="url(#oa{"d" if drex else ""})"/>'
    mx = (x1 + x2) / 2
    s += f'<text x="{mx}" y="{y - 12}" text-anchor="middle" class="o-lab{" o-drex-t" if drex else ""}">{label}</text>'
    s += f'<text x="{mx}" y="{y + 22}" text-anchor="middle" class="o-t2">{sub}</text>'
    return s


def build() -> str:
    s = [f'<svg viewBox="0 0 {VW} {VH}" role="img" aria-label="流程：整场比赛先用解说和欢呼声筛出候选时刻，Drex 判断哪些是高光，再逐帧在栅格上选出要保留的区域和球的位置，最后在每个镜头内平滑成不丢球的 9:16 运镜。">']
    s.append('<defs>'
             '<marker id="oa" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="o-head"/></marker>'
             '<marker id="oad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="o-head-d"/></marker>'
             '</defs>')
    # ---- row 1: the funnel
    nw, nh, y0 = 132, 96, 30
    gap = (VW - 40 - 5 * nw) / 4
    xs = [20 + i * (nw + gap) for i in range(5)]
    nodes = [("整场比赛", "30 分钟横屏", "land"), ("候选时刻", "20 个", None), ("高光片段", "8 段 × 22 秒", None),
             ("逐帧决策", "344 个关键帧", None), ("竖屏高光", "8 条 9:16 视频", "port")]
    for x, (t, sub, g) in zip(xs, nodes):
        s.append(node(x, y0, nw, nh, t, sub, g))
    ay = y0 + nh / 2
    arrows = [("解说 + 欢呼声", "本地，零模型费", False), ("Drex 判断事件", "8 帧 + 解说", True),
              ("Drex 读栅格", "每 0.5 秒 1 帧", True), ("运镜规划", "平滑 + 球不出框", False)]
    for i, (lab, sub, d) in enumerate(arrows):
        s.append(arrow(xs[i] + nw, xs[i + 1], ay, lab, sub, d))

    # ---- row 2 insets
    iy = 172
    # connector ticks from the two arrows to their insets
    a3 = (xs[2] + nw + xs[3]) / 2
    a4 = (xs[3] + nw + xs[4]) / 2
    s.append(f'<path d="M{a3},{ay + 32} V{iy - 14} H290 V{iy}" class="o-conn"/>')
    s.append(f'<path d="M{a4},{ay + 32} V{iy - 14} H830 V{iy}" class="o-conn"/>')
    # inset 1: one frame
    s.append(f'<rect x="20" y="{iy}" width="530" height="{VH - iy - 12}" rx="8" class="o-panel"/>')
    s.append(f'<text x="40" y="{iy + 28}" class="o-h">一帧怎么决策：把“定位”变成选择题</text>')
    fx, fy, fw, fh = 40, iy + 52, 288, 162
    s.append(f'<rect x="{fx}" y="{fy}" width="{fw}" height="{fh}" class="o-pitch"/>')
    seg = fw / 8
    for i in range(9):
        x = fx + i * seg
        s.append(f'<line x1="{x}" y1="{fy}" x2="{x}" y2="{fy + fh}" class="o-gridl"/>')
        s.append(f'<text x="{x}" y="{fy - 6}" text-anchor="middle" class="o-num">{i}</text>')
    players = [(1.2, .55), (2.1, .35), (3.3, .62), (3.8, .4), (4.6, .55), (5.4, .48), (6.7, .3), (7.3, .7), (2.6, .78)]
    for px, py in players:
        s.append(f'<circle cx="{fx + px * seg}" cy="{fy + py * fh}" r="4" class="o-player"/>')
    bx, by = fx + 4.4 * seg, fy + .5 * fh
    s.append(f'<rect x="{fx + 3 * seg}" y="{fy + .28 * fh}" width="{3 * seg}" height="{.5 * fh}" class="o-region"/>')
    cw = fh * 9 / 16
    ccx = fx + 4.5 * seg
    s.append(f'<rect x="{ccx - cw / 2}" y="{fy}" width="{cw}" height="{fh}" class="o-cropbox"/>')
    s.append(f'<circle cx="{bx}" cy="{by}" r="5" class="o-ball"/>')
    tx = fx + fw + 22
    lines = [("Drex 回答 6 道选择题", "o-t1"), ("左边界 → 3 号线", "o-t3 o-region-t"), ("右边界 → 6 号线", "o-t3 o-region-t"),
             ("球 → 4 号线附近", "o-t3 o-ball-t"), ("每个回答都有概率", "o-t2"), ("→ 当作置信度", "o-t2"),
             ("程序换算成", "o-t2"), ("最小的 9:16 框", "o-t3 o-crop-t")]
    for i, (t, c) in enumerate(lines):
        s.append(f'<text x="{tx}" y="{fy + 12 + i * 21}" class="{c}">{t}</text>')
    s.append(f'<text x="40" y="{VH - 26}" class="o-t2">Drex 没有原生的框坐标输出，所以让它在画好的编号线上做选择</text>')

    # inset 2: many frames -> one camera path
    px0 = 570
    s.append(f'<rect x="{px0}" y="{iy}" width="530" height="{VH - iy - 12}" rx="8" class="o-panel"/>')
    s.append(f'<text x="{px0 + 20}" y="{iy + 28}" class="o-h">多帧怎么连成运镜：平滑，但不丢球</text>')
    cx0, cy0, cw2, ch2 = px0 + 56, iy + 56, 450, 150
    s.append(f'<line x1="{cx0}" y1="{cy0 + ch2}" x2="{cx0 + cw2}" y2="{cy0 + ch2}" class="o-axis"/>')
    s.append(f'<line x1="{cx0}" y1="{cy0}" x2="{cx0}" y2="{cy0 + ch2}" class="o-axis"/>')
    s.append(f'<text x="{cx0 + cw2}" y="{cy0 + ch2 + 18}" text-anchor="end" class="o-t2">时间 →</text>')
    s.append(f'<text x="{cx0 - 8}" y="{cy0 + 10}" text-anchor="end" class="o-t2">右</text>')
    s.append(f'<text x="{cx0 - 8}" y="{cy0 + ch2}" text-anchor="end" class="o-t2">左</text>')
    cut_t = 0.66
    cut_x = cx0 + cut_t * cw2
    s.append(f'<line x1="{cut_x}" y1="{cy0 - 4}" x2="{cut_x}" y2="{cy0 + ch2}" class="o-cut"/>')
    s.append(f'<text x="{cut_x + 6}" y="{cy0 + ch2 - 8}" class="o-t2">镜头切换：允许跳</text>')

    def pos(t):  # smooth path, two shots
        if t < cut_t:
            return 0.35 + 0.25 * math.sin(t * 3.2) + 0.25 * t
        return 0.75 - 0.15 * (t - cut_t)
    jitter = [0.08, -0.07, 0.1, -0.05, 0.06, -0.09, 0.04, 0.07, -0.06, 0.05, -0.04, 0.06, -0.05, 0.03]
    for i, j in enumerate(jitter):
        t = 0.03 + i * 0.072
        if abs(t - cut_t) < 0.02:
            continue
        y = cy0 + (1 - (pos(t) + j)) * ch2
        s.append(f'<rect x="{cx0 + t * cw2 - 3.5}" y="{y - 3.5}" width="7" height="7" class="o-kf"/>')
    for part in ((0.0, cut_t - 0.005), (cut_t + 0.005, 1.0)):
        pts = []
        for k in range(41):
            t = part[0] + (part[1] - part[0]) * k / 40
            pts.append(f"{cx0 + t * cw2:.1f},{cy0 + (1 - pos(t)) * ch2:.1f}")
        s.append(f'<polyline points="{" ".join(pts)}" class="o-path"/>')
    # ball that would leave frame -> pinned
    tb = 0.38
    yb = cy0 + (1 - (pos(tb) + 0.2)) * ch2
    s.append(f'<circle cx="{cx0 + tb * cw2}" cy="{yb}" r="5" class="o-ball"/>')
    s.append(f'<circle cx="{cx0 + tb * cw2}" cy="{yb}" r="10" class="o-ring"/>')
    s.append(f'<line x1="{cx0 + tb * cw2 + 12}" y1="{yb - 6}" x2="{cx0 + tb * cw2 + 40}" y2="{cy0 - 2}" class="o-leader"/>')
    s.append(f'<text x="{cx0 + tb * cw2 + 44}" y="{cy0 + 2}" class="o-t3 o-ball-t">球快出框 → 钉住重算</text>')
    ly = cy0 + ch2 + 40
    s.append(f'<rect x="{px0 + 20}" y="{ly - 9}" width="8" height="8" class="o-kf"/>'
             f'<text x="{px0 + 34}" y="{ly}" class="o-t2">Drex 每帧决策（会抖）</text>')
    s.append(f'<line x1="{px0 + 200}" y1="{ly - 4}" x2="{px0 + 226}" y2="{ly - 4}" class="o-path"/>'
             f'<text x="{px0 + 232}" y="{ly}" class="o-t2">实际运镜：置信度加权平滑</text>')
    s.append(f'<text x="{px0 + 20}" y="{VH - 26}" class="o-t2">离线处理能看到未来帧，所以镜头可以在动作发生前就开始移动</text>')
    s.append('</svg>')
    return "".join(s)


if __name__ == "__main__":
    print(build())
