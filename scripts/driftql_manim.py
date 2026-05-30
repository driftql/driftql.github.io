"""
DriftQL — 3Blue1Brown-style explainer (sans-serif, dark bg, 16:9 widescreen).
Render:  manim -qh -r 1920,1080 driftql_manim.py DriftQL
"""
from manim import *
import numpy as np

# ---- sans-serif text + math ----
FONT = "DejaVu Sans"
SANS = TexTemplate()
SANS.add_to_preamble(r"\usepackage{sansmath}\sansmath")
SANS.add_to_preamble(r"\renewcommand{\familydefault}{\sfdefault}")


def MT(*s, **kw):
    return MathTex(*s, tex_template=SANS, **kw)


def TX(s, **kw):
    return Text(s, font=FONT, **kw)


# ---- palette ----
BG = "#050509"
POS = "#4C9BE8"
NEG = "#E5544B"
ACT = "#F2A65A"
SAMP = "#EDEDF2"
NETC = "#FFFFFF"
GREY = "#9aa0a6"
TEXT = WHITE
MUTED_TEXT = "#d6dae0"
DATA_TEXT = "#cfd6df"
DRIFT_EQ_TEXT = "#cfe0f5"
Q_EQ_TEXT = "#f6dcbf"

TAU = 0.7
TIP = 0.22
CAPTION_LEFT = -5.90
CAPTION_Y = -3.02

# Drift-field triangle lives in the left half so the frame reads wide (16:9).
PA = [
    np.array([-3.8, 1.9, 0.0]),   # a1 top
    np.array([-6.0, -2.0, 0.0]),  # a2 bottom-left (far)
    np.array([-1.1, 1.0, 0.0]),   # a3 right side of triangle
]
APLUS = np.mean(PA, axis=0)       # positive sits exactly at the centroid


def apply_theme(light=False):
    global BG, POS, NEG, ACT, SAMP, NETC, GREY
    global TEXT, MUTED_TEXT, DATA_TEXT, DRIFT_EQ_TEXT, Q_EQ_TEXT
    if light:
        BG = "#ffffff"
        POS = "#2563EB"
        NEG = "#DC2626"
        ACT = "#D97706"
        SAMP = "#111827"
        NETC = "#111827"
        GREY = "#64748B"
        TEXT = "#111827"
        MUTED_TEXT = "#334155"
        DATA_TEXT = "#334155"
        DRIFT_EQ_TEXT = "#1D4ED8"
        Q_EQ_TEXT = "#92400E"
    else:
        BG = "#050509"
        POS = "#4C9BE8"
        NEG = "#E5544B"
        ACT = "#F2A65A"
        SAMP = "#EDEDF2"
        NETC = "#FFFFFF"
        GREY = "#9aa0a6"
        TEXT = WHITE
        MUTED_TEXT = "#d6dae0"
        DATA_TEXT = "#cfd6df"
        DRIFT_EQ_TEXT = "#cfe0f5"
        Q_EQ_TEXT = "#f6dcbf"


def _tip(point, direction, color, size=TIP):
    d = np.array(direction, float)
    d = d / (np.linalg.norm(d) + 1e-9)
    perp = np.array([-d[1], d[0], 0.0])
    p = np.array(point, float)
    return Polygon(
        p,
        p - d * size + perp * size * 0.52,
        p - d * size - perp * size * 0.52,
        color=color,
        fill_color=color,
        fill_opacity=1.0,
        stroke_width=1,
        stroke_color=color,
    )


def PARROW(points, color, sw=4):
    """One continuous bent arrow with a single uniform head."""
    pts = [np.array(p, float) for p in points]
    body = VMobject().set_points_as_corners(pts).set_stroke(color, sw)
    tip = _tip(pts[-1], pts[-1] - pts[-2], color)
    return VGroup(body, tip)


def AR(a, b, color, sw=4, buff=0.16):
    return Arrow(
        a,
        b,
        buff=buff,
        color=color,
        stroke_width=sw,
        tip_length=TIP,
        max_tip_length_to_length_ratio=1.0,
    )


def forces(pts, aplus, tau=TAU):
    att, rep = [], []
    for i, x in enumerate(pts):
        att.append(aplus - x)
        idx = [j for j in range(len(pts)) if j != i]
        d = np.array([np.linalg.norm(x - pts[j]) for j in idx])
        logits = -(d * d) / (2 * tau * tau)
        logits -= logits.max()
        w = np.exp(logits)
        w /= w.sum()
        rep.append(-sum(w[k] * (pts[idx[k]] - x) for k in range(len(idx))))
    return att, rep


class DriftQL(Scene):
    light_theme = False

    def construct(self):
        apply_theme(self.light_theme)
        self.camera.background_color = BG
        self.intro()
        self.step1()
        self.step2()
        self.step3()
        self.step4()
        self.summary()
        self.outro()

    def set_step(self, n, text):
        new = VGroup(
            TX(f"{n}", weight=BOLD, color=ACT).scale(0.5),
            TX(text, color=TEXT).scale(0.42),
        ).arrange(RIGHT, buff=0.22)
        dot = Circle(radius=0.22, color=ACT, stroke_width=2).move_to(new[0])
        grp = VGroup(dot, new).to_corner(UL).shift(0.05 * DOWN)
        if not hasattr(self, "stepbar"):
            self.stepbar = grp
            self.play(FadeIn(grp, shift=0.2 * RIGHT), run_time=0.5)
            return None
        return Transform(self.stepbar, grp)

    def intro(self):
        title = TX("DriftQL", weight=BOLD, color=TEXT).scale(1.7)
        sub = TX("A new paradigm in generative offline RL", color=GREY).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.4)
        self.play(Write(title), run_time=1.45)
        self.play(FadeIn(sub, shift=0.2 * UP), run_time=0.85)
        self.wait(1.45)
        self.play(FadeOut(title, shift=0.4 * UP), FadeOut(sub), run_time=1.0)

    def step1(self):
        self.set_step(1, "Sample from the actor")

        dbox = RoundedRectangle(width=2.6, height=1.3, corner_radius=0.1, color=POS).set_fill(POS, 0.05)
        dtxt = VGroup(
            TX("Offline dataset", color=TEXT).scale(0.42),
            MT(r"(s,\,a^{+})\in\mathcal{D}", color=DATA_TEXT).scale(0.55),
        ).arrange(DOWN, buff=0.16).move_to(dbox)
        data = VGroup(dbox, dtxt).move_to([-4.8, 1.25, 0])

        # Actor box is taller while the net glyph remains normal size and centered.
        abox = RoundedRectangle(width=2.0, height=2.25, corner_radius=0.12, color=ACT).set_fill(ACT, 0.05)
        net = VGroup()
        cols = [3, 4, 3]
        xs = [-0.42, 0, 0.42]
        nodes = []
        for n, xx in zip(cols, xs):
            column = [Dot([xx, y, 0], radius=0.04, color=ACT) for y in np.linspace(0.42, -0.42, n)]
            nodes.append(column)
            net.add(*column)
        for a, b in zip(nodes[:-1], nodes[1:]):
            for u in a:
                for v in b:
                    net.add(Line(u.get_center(), v.get_center(), stroke_width=0.6, color=ACT).set_opacity(0.3))
        albl = MT(r"f_\theta(s,\epsilon)", color=TEXT).scale(0.6)
        VGroup(abox).move_to([-1.4, -0.85, 0])
        net.move_to(abox.get_center() + 0.28 * UP)
        albl.next_to(net, DOWN, buff=0.18)
        actor = VGroup(abox, net, albl)

        bell_curves = VGroup()
        for shift, opacity in [
            (0.28 * UP + 0.22 * RIGHT, 0.23),
            (0.14 * UP + 0.11 * RIGHT, 0.48),
            (ORIGIN, 1.0),
        ]:
            bell_curves.add(
                FunctionGraph(
                    lambda x: 0.5 * np.exp(-(x * 2.3) ** 2),
                    x_range=[-0.6, 0.6],
                    color=GREY,
                ).shift(shift).set_opacity(opacity)
            )
        nlab = MT(r"\epsilon_1,\epsilon_2,\epsilon_3\sim\mathcal{N}(0,I)", color=GREY).scale(0.5)
        noise = VGroup(bell_curves, nlab.next_to(bell_curves, DOWN, buff=0.12)).move_to([-4.9, -1.60, 0])

        self.play(FadeIn(data, shift=0.2 * RIGHT), run_time=0.75)
        self.play(Create(abox), FadeIn(net), Write(albl), run_time=1.05)
        self.play(FadeIn(noise, shift=0.2 * RIGHT), run_time=0.75)

        # State s: straight down from dataset, then 90-degree right into actor.
        d_bottom = data.get_bottom()
        state_y = abox.get_center()[1] + 0.38
        noise_y = abox.get_center()[1] - 0.34
        corner = np.array([d_bottom[0], state_y, 0.0])
        s_arr = PARROW([d_bottom + 0.03 * DOWN, corner, [abox.get_left()[0] - 0.04, state_y, 0]], GREY, sw=3)
        s_lbl = MT(r"\text{state } s", color=MUTED_TEXT).scale(0.55).move_to([-3.55, state_y + 0.25, 0])

        # Gaussian noise: straight line into the actor.
        n_arr = PARROW(
            [
                np.array([noise[0].get_right()[0] + 0.12, noise_y, 0]),
                np.array([abox.get_left()[0] - 0.04, noise_y, 0]),
            ],
            GREY,
            sw=3,
        )
        n_lbl = MT(r"\text{noise } \epsilon", color=MUTED_TEXT).scale(0.55).move_to([-3.55, noise_y + 0.25, 0])
        self.play(Create(s_arr), FadeIn(s_lbl), Create(n_arr), FadeIn(n_lbl), run_time=0.95)

        # Actor -> candidate actions (negatives), stacked on the right.
        cand_xy = [[1.2, 0.05, 0], [1.55, -0.85, 0], [1.2, -1.75, 0]]
        cands = VGroup(*[Dot(p, radius=0.095, color=SAMP) for p in cand_xy])
        clbl = VGroup(
            *[
                MT(rf"\hat a_{i+1}", color=SAMP).scale(0.55).next_to(cands[i], RIGHT, buff=0.1)
                for i in range(3)
            ]
        )
        out_y = abox.get_center()[1]
        o_arr = PARROW(
            [
                np.array([abox.get_right()[0] + 0.04, out_y, 0]),
                np.array([cands.get_left()[0] - 0.14, out_y, 0]),
            ],
            SAMP,
            sw=3,
        )
        self.play(Create(o_arr), run_time=0.45)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in cands], lag_ratio=0.2), FadeIn(clbl), run_time=1.15)
        neg_lbl = VGroup(
            TX("Candidate actions", color=TEXT).scale(0.4),
            TX("(Negatives)", color=NEG, weight=BOLD).scale(0.4),
        ).arrange(RIGHT, buff=0.12).next_to(cands, DOWN, buff=0.5).shift(0.45 * RIGHT)

        # Positive a+: comes out of the dataset right side, straight across.
        aplus = Star(5, outer_radius=0.18, color=POS).set_fill(POS, 1.0).move_to([1.25, 1.25, 0])
        p_arr = PARROW([data.get_right() + 0.04 * RIGHT, aplus.get_left() + 0.06 * LEFT], POS, sw=3)
        a_math = MT(r"a^{+}", color=POS).scale(0.6).next_to(aplus, RIGHT, buff=0.14)
        pos_lbl = VGroup(
            TX("Dataset action", color=TEXT).scale(0.4),
            TX("(Positive)", color=POS, weight=BOLD).scale(0.4),
        ).arrange(RIGHT, buff=0.12).next_to(aplus, UP, buff=0.22)
        self.play(Create(p_arr), run_time=0.55)
        self.play(GrowFromCenter(aplus), FadeIn(a_math), FadeIn(pos_lbl), FadeIn(neg_lbl), run_time=0.95)
        self.wait(1.45)

        self.samples = cands
        self.aplus = aplus
        self._schematic = VGroup(
            data,
            actor,
            noise,
            s_arr,
            s_lbl,
            n_arr,
            n_lbl,
            o_arr,
            clbl,
            neg_lbl,
            p_arr,
            a_math,
            pos_lbl,
        )

    def step2(self):
        cands, aplus = self.samples, self.aplus
        retitle = self.set_step(2, "Calculate the Drift Field")
        moves = [cands[i].animate.move_to(PA[i]) for i in range(3)] + [aplus.animate.move_to(APLUS)]
        self.play(retitle, FadeOut(self._schematic), *moves, run_time=1.45)

        pts = [c.get_center() for c in cands]
        ac = aplus.get_center()
        alabels = VGroup(
            *[
                MT(rf"\hat a_{i+1}", color=SAMP).scale(0.58).next_to(cands[i], UP, buff=0.14)
                for i in range(3)
            ]
        )
        aplabel = MT(r"a^{+}", color=POS).scale(0.66).next_to(aplus, DOWN, buff=0.1).shift(0.28 * RIGHT)
        self.play(FadeIn(alabels), FadeIn(aplabel), run_time=0.7)

        # Equations on the right half.
        ev = MT(r"V^{+}(\hat a_i)=a^{+}-\hat a_i").scale(0.66).set_color(POS)
        er = MT(r"V^{-}(\hat a_i)=\sum_{j\neq i} w^{-}_{ij}\,(\hat a_j-\hat a_i)").scale(0.66).set_color(NEG)
        ed = MT(r"V(\hat a_i)=", r"V^{+}(\hat a_i)", r"-", r"V^{-}(\hat a_i)").scale(0.72)
        ed.set_color(TEXT)
        ed[1].set_color(POS)
        ed[3].set_color(NEG)
        eqs = VGroup(ev, er, ed).arrange(DOWN, buff=0.45).move_to([3.5, 0.4, 0])

        # Attraction: every candidate -> a+.
        attraction = VGroup(*[AR(pts[i], ac, POS, sw=5) for i in range(3)])
        cap = TX(
            "Every candidate action is pulled toward a+, nearby candidates repel to avoid collapse.",
            color=TEXT,
        ).scale(0.44)
        if cap.width > 12.0:
            cap.scale_to_fit_width(12.0)
        cap.move_to([CAPTION_LEFT + cap.width / 2, CAPTION_Y, 0])
        self.play(
            LaggedStart(*[GrowArrow(a) for a in attraction], lag_ratio=0.2),
            Write(ev),
            run_time=1.8,
        )
        self.play(AddTextLetterByLetter(cap, time_per_char=0.018), run_time=1.55)
        self.wait(0.9)

        # Repulsion: dashed lines, thickness = closeness.
        pairs = [(0, 1), (0, 2), (1, 2)]
        dists = {p: np.linalg.norm(pts[p[0]] - pts[p[1]]) for p in pairs}
        inv = {p: 1.0 / dists[p] for p in pairs}
        mn, mx = min(inv.values()), max(inv.values())
        dashed = VGroup()
        for p in pairs:
            sw = 1.8 + (inv[p] - mn) / (mx - mn + 1e-9) * 4.2
            a, b = pts[p[0]], pts[p[1]]
            n = (b - a) / np.linalg.norm(b - a)
            dashed.add(
                DashedLine(
                    a + n * 0.18,
                    b - n * 0.18,
                    color=NEG,
                    stroke_width=sw,
                    dash_length=0.11,
                    dashed_ratio=0.55,
                )
            )
        self.play(
            LaggedStart(*[Create(d) for d in dashed], lag_ratio=0.2),
            Write(er),
            run_time=1.8,
        )
        cap2 = cap
        self.wait(1.0)

        self.play(Write(ed), run_time=1.1)
        self.wait(1.8)

        self.alabels = alabels
        self.aplabel = aplabel
        self.attraction = attraction
        self.dashed = dashed
        self.diagram = VGroup(cands, aplus, alabels, aplabel, attraction, dashed)
        self.s2eqs = eqs
        self.cap = cap2

    def step3(self):
        retitle = self.set_step(3, "Compute the loss")
        self.play(retitle, FadeOut(self.s2eqs), FadeOut(self.cap), run_time=0.8)

        ai = Dot(color=SAMP, radius=0.095).move_to([1.5, 2.5, 0])
        ai_l = MT(r"\hat a_i", color=SAMP).scale(0.55).next_to(ai, DOWN, buff=0.12)
        aip = Dot(color=NETC, radius=0.095).move_to([4.9, 2.5, 0])
        aip_l = MT(r"\hat a_i'", color=NETC).scale(0.55).next_to(aip, DOWN, buff=0.12)
        tarr = AR(ai.get_center(), aip.get_center(), NETC, sw=5, buff=0.22)
        tlab = MT(r"\Delta(\hat a_i)").scale(0.55).next_to(tarr, UP, buff=0.12)
        tlab[0].set_color(POS)
        drift_def = MT(
            r"\Delta(\hat a_i)=",
            r"V^{+}(\hat a_i)",
            r"-",
            r"V^{-}(\hat a_i)",
        ).scale(0.62)
        drift_def.set_color(TEXT)
        drift_def[1].set_color(POS)
        drift_def[3].set_color(NEG)
        drift_def.next_to(VGroup(ai, aip), DOWN, buff=0.7)
        self.play(FadeIn(ai), FadeIn(ai_l), run_time=0.6)
        self.play(GrowArrow(tarr), FadeIn(tlab), run_time=0.9)
        self.play(FadeIn(aip), FadeIn(aip_l), Write(drift_def), run_time=1.1)
        self.transport = VGroup(ai, ai_l, aip, aip_l, tarr, tlab, drift_def)

        ldrift = MT(
            r"\mathcal{L}_{\Delta}(\theta)=",
            r"\mathrm{MSE}\!\left(\hat a_i,\ ",
            r"\mathrm{sg}\!\left(\hat a_i+\Delta(\hat a_i)\right)\right)",
        ).scale(0.54)
        ldrift[0].set_color(POS)
        ldrift[1:].set_color(DRIFT_EQ_TEXT)
        qloss = MT(
            r"\mathcal{L}_{Q}(\theta)=",
            r"-\,Q^{\mathrm{avg}}(s,\hat a_i)",
        ).scale(0.56)
        qloss[0].set_color(ACT)
        qloss[1].set_color(Q_EQ_TEXT)
        lact = MT(
            r"\mathcal{L}_{\mathrm{actor}}(\theta)=",
            r"\alpha\,\mathcal{L}_{\Delta}(\theta)",
            r"+",
            r"\mathcal{L}_{Q}(\theta)",
        ).scale(0.62)
        lact.set_color(TEXT)
        lact[1].set_color(POS)
        lact[3].set_color(ACT)
        block = VGroup(VGroup(ldrift, qloss).arrange(DOWN, aligned_edge=LEFT, buff=0.26), lact)
        block.arrange(DOWN, buff=0.38).move_to([3.4, -0.55, 0])
        sgnote = TX("sg = stop-gradient on the drifted target", color=GREY).scale(0.32)
        sgnote.to_corner(DR).shift(0.18 * LEFT + 0.08 * UP)
        self.play(Write(ldrift), run_time=1.1)
        self.play(Write(qloss), run_time=0.9)
        self.play(Write(lact), run_time=1.3)
        self.play(FadeIn(sgnote, shift=0.1 * UP), run_time=0.6)
        self.wait(2.0)
        self.lossblock = VGroup(ldrift, qloss, lact, sgnote)

    def step4(self):
        retitle = self.set_step(4, "Update the actor")
        self.play(
            retitle,
            FadeOut(self.attraction),
            FadeOut(self.dashed),
            FadeOut(self.alabels),
            FadeOut(self.transport),
            run_time=0.7,
        )

        cands, aplus = self.samples, self.aplus
        ac = aplus.get_center()
        pts = [c.get_center() for c in cands]
        cands.set_z_index(5)
        aplus.set_z_index(4)
        self.aplabel.set_z_index(4)
        eta, rep_scale, box = 0.30, 0.5, 3.6
        cap = TX("Recompute drifted targets, then update the actor.", color=TEXT).scale(0.48)
        cap.move_to([CAPTION_LEFT + cap.width / 2, CAPTION_Y, 0])
        self.play(AddTextLetterByLetter(cap, time_per_char=0.02), run_time=1.15)
        for k in range(18):
            att, rep = forces(pts, ac)
            repel = rep_scale * max(0.0, 1.0 - k / 12.0)
            new = []
            for i in range(3):
                q = pts[i] + eta * (att[i] + repel * rep[i])
                q = np.clip(q, ac - box, ac + box)
                q[2] = 0
                new.append(q)
            self.add(*[Dot(pts[i], radius=0.045, color=SAMP).set_opacity(0.12) for i in range(3)])
            self.play(*[cands[i].animate.move_to(new[i]) for i in range(3)], run_time=0.18)
            pts = new
        final_offsets = [0.08 * UP, 0.07 * LEFT + 0.04 * DOWN, 0.07 * RIGHT + 0.04 * DOWN]
        final_pts = [ac + offset for offset in final_offsets]
        self.play(*[cands[i].animate.move_to(final_pts[i]) for i in range(3)], run_time=0.65)
        self.wait(1.4)

    def summary(self):
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.9)

        title = TX("DriftQL", weight=BOLD, color=TEXT).scale(1.15).to_edge(UP, buff=0.95)
        underline = Line(LEFT, RIGHT, color=ACT, stroke_width=4).scale(1.0)
        underline.next_to(title, DOWN, buff=0.16)

        bullet_texts = [
            "One-step generative policy.",
            "No denoising chains, solvers, or distillation.",
            "Robust to noisy and suboptimal data.",
            "State-of-the-art performance on OGBench and D4RL.",
        ]
        rows = VGroup()
        for text in bullet_texts:
            dot = Dot(radius=0.06, color=ACT)
            line = TX(text, color=TEXT).scale(0.47)
            row = VGroup(dot, line).arrange(RIGHT, buff=0.26)
            rows.add(row)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.38).move_to(0.35 * DOWN)

        self.play(Write(title), Create(underline), run_time=1.0)
        self.wait(0.35)
        for row in rows:
            dot, line = row
            self.play(
                GrowFromCenter(dot),
                AddTextLetterByLetter(line, time_per_char=0.014),
                run_time=1.25,
            )
            self.wait(0.2)
        self.wait(1.8)

    def outro(self):
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.35)
        self.wait(0.45)


class DriftQLWhite(DriftQL):
    light_theme = True
