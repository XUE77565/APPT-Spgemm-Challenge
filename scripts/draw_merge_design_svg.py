#!/usr/bin/env python3
# Generate the Design-2 hero figure as editable SVG, render to PNG+PDF via cairosvg.
# Before/after: prior merge (serial per row) vs ours (column-domain bucketing).
import os, cairosvg

INK   = "#1F2A33"
MUTED = "#5C6773"
GRAY  = "#8A95A1"
GRAYL = "#E9ECEF"
GRAYB = "#F4F5F6"
RED   = "#C00000"
REDBG = "#FBF0F0"
REDE  = "#E9C9C9"
TEAL  = "#1485A4"
TEALD = "#0E6E83"
TEALBG = "#F3FAFB"
TEALE = "#CFE7EC"
# bucket tints (light -> dark teal)
BAND = ["#E0F0F3", "#BDE0E7", "#97CED8"]

W, H = 1300, 700

parts = []
def emit(s): parts.append(s)

# ---------- defs ----------
emit(f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="DejaVu Sans, Arial, sans-serif">
<defs>
  <filter id="sh" x="-30%" y="-30%" width="160%" height="160%">
    <feDropShadow dx="0" dy="1.2" stdDeviation="1.6" flood-color="#1F2A33" flood-opacity="0.12"/>
  </filter>
  <marker id="ag" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker>
  <marker id="at" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="{TEAL}"/></marker>
  <marker id="ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="{RED}"/></marker>
</defs>
<rect width="{W}" height="{H}" fill="#FFFFFF"/>''')

def text(x, y, s, size=12, color=INK, weight="normal", anchor="middle", italic=False, spacing=None):
    style = f'font-size:{size}px;fill:{color};font-weight:{weight};text-anchor:{anchor}'
    if italic: style += ';font-style:italic'
    if spacing: style += f';letter-spacing:{spacing}'
    emit(f'<text x="{x:.1f}" y="{y:.1f}" style="{style}">{s}</text>')

def rrect(x, y, w, h, rx=8, fill="#fff", stroke=None, sw=1, filt=None, opacity=1):
    sopt = "" if not filt else f' filter="url(#{filt})"'
    stopt = "" if not stroke else f' stroke="{stroke}" stroke-width="{sw}"'
    emit(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
         f'fill="{fill}"{stopt}{sopt} opacity="{opacity}"/>')

def chip(cx, cy, v, fill="#FFFFFF", stroke="#C2CAD3", txt=INK, shadow=True):
    w, h = 30, 24
    rrect(cx-w/2, cy-h/2, w, h, rx=5, fill=fill, stroke=stroke, sw=1,
          filt="sh" if shadow else None)
    text(cx, cy+4, str(v), 11, txt, weight="bold")

def line(x1, y1, x2, y2, color=MUTED, w=1.5, marker=None, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#{marker})"' if marker else ""
    emit(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
         f'stroke="{color}" stroke-width="{w}"{d}{m}/>')

# ---------- title ----------
text(W/2, 44, "Column-Domain Bucketing — a fourth parallel axis for merge-based SpGEMM",
     23, INK, weight="bold")
text(W/2, 70, "Prior work parallelizes rows or K-chains, never the column range.  "
              "We split a row's columns into K disjoint buckets — merged in parallel, concatenated in order.",
     13.5, MUTED)

# ---------- panel geometry ----------
PAD, GAP = 34, 30
pw = (W - 2*PAD - GAP) / 2
py, ph = 100, 560
panels = [("L", PAD), ("R", PAD + pw + GAP)]

CHAIN_VALS = [[1, 7, 12], [3, 9, 14], [0, 5, 11]]   # 3 sorted chains over domain 0..14
SORTED = sorted(v for c in CHAIN_VALS for v in c)     # merged output
DOMAIN = 15
K = 3
BW = DOMAIN / K   # 5

def draw_panel(side, px):
    ours = (side == "R")
    accent = TEAL if ours else GRAY
    bg = TEALBG if ours else "#FBFBFC"
    edge = TEALE if ours else "#E5E7EB"
    # panel frame
    rrect(px, py, pw, ph, rx=16, fill="#FFFFFF", stroke=edge, sw=1.5, filt="sh")
    # header strip
    hx, hw = px+18, pw-36
    rrect(hx, py+16, hw, 50, rx=10, fill=bg, stroke=edge, sw=1)
    title = "Ours · column-domain bucketing" if ours else "Prior merge · Gustavson / SpArch"
    tcol = TEAL if ours else INK
    text(px+pw/2, py+40, title, 15.5, tcol, weight="bold")
    sub = "K warps — one per disjoint column bucket" if ours else "one unit scans every chain head, serially"
    text(px+pw/2, py+57, sub, 11.5, TEALD if ours else MUTED)

    # chains region
    cx0 = px + 60
    cx1 = px + pw - 30
    width = cx1 - cx0
    def vx(v): return cx0 + ((v + 0.5) / DOMAIN) * width

    ruler_y = py + 96
    chain_ys = [py + 150, py + 188, py + 226]

    # bucket bands (ours) or single serialized band (prior)
    band_top, band_bot = py + 132, py + 244
    if ours:
        for k in range(K):
            x0 = cx0 + (k*BW/DOMAIN)*width
            x1 = cx0 + ((k+1)*BW/DOMAIN)*width
            rrect(x0, band_top, x1-x0, band_bot-band_top, rx=4, fill=BAND[k], opacity=0.85)
            text((x0+x1)/2, ruler_y, f"B{k}", 11, BAND[k].replace("#", "#0") if False else TEALD,
                 weight="bold")
        # dividers
        for k in range(1, K):
            xv = cx0 + (k*BW/DOMAIN)*width
            line(xv, band_top, xv, band_bot, "#FFFFFF", 2)
    else:
        rrect(cx0, band_top, width, band_bot-band_top, rx=4, fill=GRAYB, stroke=GRAYL, sw=1)
        text(cx0+width/2, ruler_y, "whole column range — serialized", 11, RED, weight="bold")

    # chains
    for i, (vals, cy) in enumerate(zip(CHAIN_VALS, chain_ys)):
        text(px+34, cy+4, f"k{i}", 12, MUTED, weight="bold")
        for v in vals:
            chip(vx(v), cy, v, stroke=(TEAL if ours else "#C2CAD3"))

    # ---- merge stage ----
    if ours:
        unit_y = py + 296
        for k in range(K):
            x0 = cx0 + (k*BW/DOMAIN)*width
            x1 = cx0 + ((k+1)*BW/DOMAIN)*width
            cx = (x0+x1)/2
            # funnel arrow from band to warp
            line(cx, band_bot+2, cx, unit_y-22, TEAL, 1.8, marker="at")
            rrect(cx-58, unit_y-18, 116, 36, rx=10, fill=TEAL, filt="sh")
            text(cx, unit_y+4, f"warp {k}  ·  shfl-merge", 11, "#FFFFFF", weight="bold")
    else:
        unit_y = py + 296
        ccx = (cx0 + cx1) / 2
        for cy in chain_ys:                       # all chains funnel into ONE serial unit
            line(cx1, cy, ccx, unit_y - 18, MUTED, 1.3)
        rrect(ccx - 72, unit_y - 18, 144, 36, rx=10, fill=GRAY, filt="sh")
        text(ccx, unit_y + 4, "1 unit · serial merge", 11, "#FFFFFF", weight="bold")

    # ---- output bar ----
    out_y = py + 372
    out_h = 40
    ob_fill = "#EAF4F6" if ours else GRAYB
    ob_edge = TEAL if ours else GRAY
    rrect(cx0, out_y, width, out_h, rx=8, fill=ob_fill, stroke=ob_edge, sw=1.2)
    # concat arrow
    if ours:
        for k in range(K):
            x0 = cx0 + (k*BW/DOMAIN)*width
            x1 = cx0 + ((k+1)*BW/DOMAIN)*width
            line((x0+x1)/2, unit_y+18, (x0+x1)/2, out_y-2, TEAL, 1.4, marker="at")
    else:
        line((cx0+cx1)/2, unit_y+18, (cx0+cx1)/2, out_y-2, MUTED, 1.8, marker="ag")
    # output chips (sorted)
    for v in SORTED:
        chip(vx(v), out_y+out_h/2, v, fill="#FFFFFF", stroke=ob_edge, txt=ob_edge)
    text(px+34, out_y+out_h/2+4, "row i", 11, MUTED, weight="bold")
    text(cx0+width/2, out_y+out_h+16, "column-sorted CSR  (no global sort)" if ours else "column-sorted CSR",
         11.5, TEAL if ours else MUTED, weight="bold", italic=ours)

    # ---- callout ----
    cy0 = py + 446
    if ours:
        rrect(px+20, cy0, pw-40, 96, rx=10, fill=TEALBG, stroke=TEALE, sw=1)
        text(px+pw/2, cy0+24, "Disjoint, ordered buckets concatenate to a", 12.5, INK, weight="bold")
        text(px+pw/2, cy0+42, "column-sorted row — merge's no-sort dividend", 12.5, INK, weight="bold")
        text(px+pw/2, cy0+60, "survives parallelism.  Unattainable by hash", 11.5, MUTED)
        text(px+pw/2, cy0+78, "(unordered) or outer-product (loses row-merge).", 11.5, MUTED)
    else:
        rrect(px+20, cy0, pw-40, 96, rx=10, fill=REDBG, stroke=REDE, sw=1)
        text(px+pw/2, cy0+24, "A heavy row (large k, long chains) is merged", 12.5, INK, weight="bold")
        text(px+pw/2, cy0+42, "by a single unit — a straggler that gates the", 12.5, INK, weight="bold")
        text(px+pw/2, cy0+60, "whole kernel.  This is why fast GPU SpGEMM", 11.5, MUTED)
        text(px+pw/2, cy0+78, "abandoned merge for hash / ESC accumulators.", 11.5, MUTED)

for side, px in panels:
    draw_panel(side, px)

# bottom axis labels
text(PAD+pw/2, py+ph+30, "parallel axis:  rows  +  K-chains", 11, MUTED, italic=True)
text(PAD+pw+GAP+pw/2, py+ph+30, "parallel axis:  rows  +  K-chains  +  column domain  (new)", 11, TEALD, italic=True, weight="bold")

emit('</svg>')

svg = "\n".join(parts)
os.makedirs("fig", exist_ok=True)
with open("fig/merge_design.svg", "w") as f:
    f.write(svg)
cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to="fig/merge_design.png", output_width=2*W)
cairosvg.svg2pdf(bytestring=svg.encode("utf-8"), write_to="fig/merge_design.pdf")
print("wrote fig/merge_design.{svg,png,pdf}")
