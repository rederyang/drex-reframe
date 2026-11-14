"""Animated scene for the top of the results page: a landscape broadcast, a 9:16 window following
the ball, and the phone screen showing what that window sees. Pure SVG + CSS classes (.sc-*);
the page's CSS holds the keyframes so prefers-reduced-motion can stop them.
"""
from __future__ import annotations

PW, PH = 320, 180          # landscape pitch frame
CW = PH * 9 / 16           # crop width (101.25)

PLAYERS = [(48, 70, "a"), (82, 120, "b"), (120, 60, "a"), (150, 128, "b"), (175, 92, "a"), (205, 58, "b"),
           (232, 118, "a"), (258, 84, "b"), (284, 102, "b"), (300, 90, "k")]


def pitch(uid: str) -> str:
    s = [f'<rect x="0" y="0" width="{PW}" height="{PH}" class="sc-grass"/>']
    for i in range(8):
        if i % 2:
            s.append(f'<rect x="{i * 40}" y="0" width="40" height="{PH}" class="sc-stripe"/>')
    s.append(f'<rect x="6" y="14" width="{PW - 12}" height="{PH - 28}" class="sc-line"/>')
    s.append(f'<line x1="{PW / 2}" y1="14" x2="{PW / 2}" y2="{PH - 14}" class="sc-line"/>')
    s.append(f'<circle cx="{PW / 2}" cy="{PH / 2}" r="22" class="sc-line"/>')
    s.append(f'<rect x="{PW - 52}" y="52" width="46" height="76" class="sc-line"/>')
    s.append(f'<rect x="{PW - 6}" y="76" width="5" height="28" class="sc-goal"/>')
    for x, y, team in PLAYERS:
        s.append(f'<circle cx="{x}" cy="{y}" r="4.2" class="sc-p sc-{team}"/>')
    # ball: drawn at origin, moved by CSS keyframes on the group
    s.append('<g class="sc-ballmove"><circle cx="0" cy="0" r="3.2" class="sc-ball"/></g>')
    s.append(f'<text x="10" y="11" class="sc-hud">RMA 1–1 BAR  72:10</text>')
    return "".join(s)


def build() -> str:
    land = (f'<svg viewBox="-2 -2 {PW + 4} {PH + 4}" class="sc-land" role="img" '
            f'aria-label="横屏转播画面，一个 9:16 的取景框跟着球从中场移动到球门">'
            f'{pitch("l")}'
            f'<g class="sc-cropmove"><rect x="0" y="0" width="{CW}" height="{PH}" class="sc-crop"/>'
            f'<rect x="0" y="0" width="{CW}" height="{PH}" class="sc-cropglow"/></g>'
            f'</svg>')
    phone = (f'<svg viewBox="0 0 {CW} {PH}" class="sc-phone" role="img" aria-label="手机竖屏上看到的取景框内画面">'
             f'<g class="sc-pan">{pitch("p")}</g>'
             f'<text x="{CW / 2}" y="{PH * 0.42}" text-anchor="middle" class="sc-goaltext">GOAL</text>'
             f'</svg>')
    return land, phone


if __name__ == "__main__":
    print(build())


# ---------------- talk scene: two people, the crop CUTS to whoever speaks ----------------
TX = {"L": 100.0, "R": 220.0}


def _person(x, shirt, cls, d=1):
    """d = +1 waves to the right of the face, -1 to the left (toward the other person)."""
    waves = "".join(f'<path d="M{x + d * (24 + i * 7)},{60 - i * 3} q{d * (6 + i * 2)},{10 + i * 3} 0,{20 + i * 6}" class="tk-wave"/>'
                    for i in range(3))
    return (f'<rect x="{x - 36}" y="94" width="72" height="70" rx="22" class="tk-shirt {shirt}"/>'
            f'<circle cx="{x}" cy="70" r="19" class="tk-skin"/>'
            f'<path d="M{x - 19},66 q19,-26 38,0 v-6 q-19,-22 -38,0 z" class="tk-hair"/>'
            f'<g class="{cls}">{waves}</g>')


def talk_scene() -> str:
    return "".join([f'<rect x="0" y="0" width="{PW}" height="{PH}" class="tk-wall"/>',
                    f'<rect x="0" y="0" width="{PW}" height="22" class="tk-shelf"/>',
                    f'<rect x="0" y="146" width="{PW}" height="34" class="tk-table"/>',
                    _person(TX["L"], "tk-a", "tk-talkL", 1), _person(TX["R"], "tk-b", "tk-talkR", -1),
                    '<rect x="62" y="150" width="18" height="24" rx="3" class="tk-glass"/>',
                    '<rect x="240" y="150" width="18" height="24" rx="3" class="tk-glass"/>'])


def build_talk():
    land = (f'<svg viewBox="-2 -2 {PW + 4} {PH + 4}" class="sc-land" role="img" '
            f'aria-label="两人同框的访谈画面，9:16 取景框在换人说话时直接切到说话人">'
            f'{talk_scene()}'
            f'<g class="tk-cropmove"><rect x="0" y="0" width="{CW}" height="{PH}" class="sc-crop"/>'
            f'<rect x="0" y="0" width="{CW}" height="{PH}" class="sc-cropglow"/></g></svg>')
    phone = (f'<svg viewBox="0 0 {CW} {PH}" class="sc-phone" role="img" aria-label="手机竖屏始终对准正在说话的人">'
             f'<g class="tk-pan">{talk_scene()}</g></svg>')
    return land, phone
