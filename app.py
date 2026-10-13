"""Hybrid Edge-Color Object Segmentation - Streamlit interface.
Run:  python -m streamlit run desktop_app_v5.py
"""
import io

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

import processing as pr

st.set_page_config(page_title="Hybrid Edge-Color Object Segmentation", page_icon="🔬", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');
:root{--bg:#0b1220;--panel:#111b30;--line:#1f2f4d;--cy:#22d3ee;--bl:#3b82f6;--or:#fb923c;--gr:#34d399;--tx:#e6edf7;--mu:#8aa0c0}
html,body,[class*="css"],.stApp{font-family:'IBM Plex Sans',system-ui,sans-serif;color:var(--tx)}
.stApp{background:radial-gradient(circle at 12% 0%,rgba(34,211,238,.09),transparent 38%),radial-gradient(circle at 92% 8%,rgba(59,130,246,.10),transparent 42%),var(--bg)}
header[data-testid="stHeader"],footer,#MainMenu{display:none}
.block-container{padding:1.1rem 1.6rem 1.5rem;max-width:100%}
[data-testid="stSidebar"]{background:#0a1020;border-right:1px solid var(--line);min-width:250px;max-width:250px}
[data-testid="stSidebar"] .block-container{padding:1rem .8rem}
.brand{border:1px solid var(--line);background:var(--panel);border-radius:8px;padding:.8rem .9rem;margin-bottom:.9rem}
.brand b{font-size:.98rem;line-height:1.2;display:block}.brand span{color:var(--mu);font-size:.74rem}
[data-testid="stSidebar"] [role="radiogroup"]{gap:2px}
[data-testid="stSidebar"] label[data-baseweb="radio"]{padding:.5rem .7rem;border-radius:6px;border-left:3px solid transparent;width:100%;margin:0;transition:background .15s}
[data-testid="stSidebar"] label[data-baseweb="radio"]>div:first-child{display:none}
[data-testid="stSidebar"] label[data-baseweb="radio"]:hover{background:#13203a}
[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked){background:#13264a;border-left-color:var(--cy)}
[data-testid="stSidebar"] label[data-baseweb="radio"] p{font-size:.9rem;font-weight:500}
.status{border:1px solid var(--line);border-radius:8px;padding:.6rem .8rem;font-size:.78rem;color:var(--mu);margin:.9rem 0 .6rem}
.status b{color:var(--tx)}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.apphead{display:flex;justify-content:space-between;align-items:flex-end;border-bottom:1px solid var(--line);padding-bottom:.7rem;margin-bottom:1rem}
.apphead h1{font-size:1.5rem;font-weight:600;margin:0}
.apphead p{margin:.15rem 0 0;color:var(--cy);font-size:.88rem}
.apphead .pg{color:var(--mu);font-size:.82rem;font-family:'IBM Plex Mono',monospace}
.ct{font-size:.8rem;font-weight:600;color:var(--mu);margin:.3rem 0 .25rem}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:.8rem 1rem;margin-bottom:.8rem;font-size:.9rem}
.panel h4{margin:0 0 .4rem;font-size:.95rem}.panel p,.panel li{color:var(--mu);font-size:.88rem;line-height:1.5}
.mc{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:.65rem .85rem}
.mc small{color:var(--mu);font-size:.74rem}.mc div{font-family:'IBM Plex Mono',monospace;font-size:1.3rem;color:var(--cy);margin-top:2px}
[data-testid="stImage"] img{border:1px solid var(--line);border-radius:6px}
.flow{display:flex;flex-wrap:wrap;align-items:stretch;gap:6px;margin:.4rem 0 1rem}
.flow .st{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:.5rem .7rem;font-size:.8rem;flex:1;min-width:110px;text-align:center}
.flow .st b{display:block;color:var(--cy);font-family:'IBM Plex Mono',monospace;font-size:.72rem}
.flow .ar{color:var(--bl);align-self:center;font-size:1.1rem}
.fuse{display:flex;align-items:center;gap:10px;margin:.3rem 0 1rem}
.fuse .fin{display:flex;flex-direction:column;gap:6px}.fuse .fin div,.fuse .fb{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:.45rem .9rem;font-size:.82rem}
.fuse .fb{border-color:var(--cy);color:var(--cy);font-weight:600}.fuse .fa{color:var(--bl);font-size:1.3rem}

/* ---------- buttons ---------- */
div.stButton>button,div.stDownloadButton>button{border-radius:8px;border:1px solid var(--line);background:var(--panel);color:var(--tx);font-weight:500;padding:.6rem 1rem;transition:border-color .15s,background .15s,transform .1s,box-shadow .15s}
div.stButton>button:hover,div.stDownloadButton>button:hover{border-color:var(--cy);background:#14233f;color:var(--tx)}
div.stButton>button:active,div.stDownloadButton>button:active{transform:translateY(1px)}
div.stButton>button:focus-visible,div.stDownloadButton>button:focus-visible{outline:2px solid var(--cy);outline-offset:2px}
div.stButton>button:disabled{opacity:.4;cursor:not-allowed}
div.stButton>button[kind="primary"],div.stButton>button[data-testid="stBaseButton-primary"]{background:var(--cy);color:#04121c;font-weight:600;border:0;box-shadow:0 0 0 0 rgba(34,211,238,.0)}
div.stButton>button[kind="primary"]:hover,div.stButton>button[data-testid="stBaseButton-primary"]:hover{background:#67e8f9;color:#04121c;box-shadow:0 6px 22px rgba(34,211,238,.28)}
div.stButton>button[kind="primary"]:disabled,div.stButton>button[data-testid="stBaseButton-primary"]:disabled{background:#1b3a47;color:#6c8794;box-shadow:none}
.navsep{border-top:1px solid var(--line);margin:1.4rem 0 .8rem}

/* ---------- inputs ---------- */
[data-testid="stFileUploader"] section{background:var(--panel);border:1px dashed #2b4570;border-radius:8px;transition:border-color .15s,background .15s}
[data-testid="stFileUploader"] section:hover{border-color:var(--cy);background:#13203a}
.stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"]{font-size:.88rem;border-radius:6px 6px 0 0;padding:.5rem 1rem}
.stTabs [aria-selected="true"]{color:var(--cy)}

/* ---------- landing ---------- */
.lead h1{font-size:clamp(2rem,3.6vw,3.1rem);line-height:1.12;font-weight:600;margin:2.2rem 0 1rem;max-width:15em;letter-spacing:-.01em}
.lead p{color:var(--mu);font-size:1.05rem;line-height:1.6;max-width:34em;margin:0 0 1.6rem}
.scene{width:100%;height:auto;display:block}
.scene .mask{fill:rgba(251,146,60,.38);opacity:0;animation:dip-fade .8s .3s forwards}
.scene .edge{fill:none;stroke:#22d3ee;stroke-width:3;stroke-linejoin:round;stroke-dasharray:1;stroke-dashoffset:1;animation:dip-trace 2.2s .9s ease-in-out forwards}
.scene .box{fill:none;stroke:#34d399;stroke-width:2;stroke-dasharray:6 5;opacity:0;animation:dip-fade .6s 3.2s forwards}
.scene text{fill:#34d399;font:500 13px 'IBM Plex Mono',monospace;opacity:0;animation:dip-fade .6s 3.2s forwards}
@keyframes dip-fade{to{opacity:1}}
@keyframes dip-trace{to{stroke-dashoffset:0}}
@media (prefers-reduced-motion:reduce){.scene *{animation-duration:.01s!important;animation-delay:0s!important}}
.sigs{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin:1.6rem 0 1.2rem}
.sig{background:var(--panel);border:1px solid var(--line);border-left-width:4px;border-radius:8px;padding:.8rem 1rem}
.sig b{display:block;margin-bottom:.25rem;font-size:.95rem}.sig span{color:var(--mu);font-size:.86rem;line-height:1.5}
.sig.c{border-left-color:var(--or)}.sig.e{border-left-color:var(--cy)}.sig.h{border-left-color:var(--gr)}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

DEFAULTS = dict(max_dim=800, denoise="None", dk=3, space="HSV", edge_method="Canny", c_low=50, c_high=150,
                s_ksize=3, s_thr=60, th_method="Otsu", th_val=127, th_block=35, th_c=5, th_inv=False,
                m_op="Opening", m_k=5, m_it=1, fuse=pr.FUSIONS[0], fuse_w=0.6, edge_close=7,
                keep_largest=True, out_mode="Original background, object highlighted", obj_mode="Show largest object")
S = st.session_state
S.setdefault("P", dict(DEFAULTS, ranges={}))
S.setdefault("rgb", None); S.setdefault("info", None); S.setdefault("gt", None)
S.setdefault("started", False); S.setdefault("fkey", None)
S.setdefault("entered", False)
P = S.P

PAGE_ABOUT = "ℹ️  About Project"


# ------------------------------------------------------------------ helpers
def sel(label, name, opts):
    P[name] = st.selectbox(label, opts, index=opts.index(P[name]))

def sld(label, name, lo, hi, step=1):
    P[name] = st.slider(label, lo, hi, P[name], step)

def chk(label, name):
    P[name] = st.checkbox(label, P[name])

def compute():
    return pr.run_pipeline(S.rgb, P)

def show(img, title):
    st.markdown(f'<div class="ct">{title}</div>', unsafe_allow_html=True)
    st.image(img, use_container_width=True, clamp=True)

def mcard(label, value):
    st.markdown(f'<div class="mc"><small>{label}</small><div>{value}</div></div>', unsafe_allow_html=True)

def panel(title, body):
    st.markdown(f'<div class="panel"><h4>{title}</h4>{body}</div>', unsafe_allow_html=True)

def flow(steps):
    items = '<span class="ar">›</span>'.join(f'<div class="st"><b>{i + 1:02d}</b>{s}</div>' for i, s in enumerate(steps))
    st.markdown(f'<div class="flow">{items}</div>', unsafe_allow_html=True)

def ready():
    if S.rgb is None or not S.started:
        st.info("Upload an image in **Image Input** and press **Start segmentation** to see results here.")
        return False
    return True

def plot_style(fig, h=300):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=h, margin=dict(l=10, r=10, t=30, b=10), font=dict(family="IBM Plex Sans"))
    return fig

def png_bytes(arr):
    b = io.BytesIO(); Image.fromarray(arr).save(b, "PNG"); return b.getvalue()


# navigation callbacks (run before the next rerun, so they can safely set widget state)
def go_app(target=None):
    S.entered = True
    if target:
        S.nav = target

def go_home():
    S.entered = False

def go_to(name):
    S.nav = name

def reset_settings():
    P.clear(); P.update(dict(DEFAULTS, ranges={}))
    for sp in pr.SPACES:
        for i in range(4):
            S.pop(f"{sp}{i}", None)


# ------------------------------------------------------------------ landing
HERO_SVG = (
    '<svg class="scene" viewBox="0 0 460 340" role="img" '
    'aria-label="A shape found by its color, outlined by its edges, then kept where both agree">'
    '<defs><pattern id="px" width="20" height="20" patternUnits="userSpaceOnUse">'
    '<path d="M20 0H0V20" fill="none" stroke="#1a2a47" stroke-width="1"/></pattern></defs>'
    '<rect x="10" y="10" width="440" height="320" rx="10" fill="#0d1628" stroke="#1f2f4d"/>'
    '<rect x="10" y="10" width="440" height="320" rx="10" fill="url(#px)"/>'
    '<path class="mask" d="M232 70 C298 62 360 106 358 172 C356 214 392 232 380 262 C366 290 316 270 290 276 '
    'C240 292 150 270 110 214 C84 170 100 110 150 84 C176 72 204 72 232 70 Z"/>'
    '<path class="edge" pathLength="1" d="M230 62 C300 52 372 100 366 172 C360 242 300 286 226 280 '
    'C152 274 96 232 100 166 C104 100 160 70 230 62 Z"/>'
    '<rect class="box" x="92" y="52" width="282" height="240" rx="4"/>'
    '<text x="100" y="316">object</text>'
    '</svg>'
)


def page_landing():
    st.markdown("<style>[data-testid='stSidebar'],[data-testid='collapsedControl'],"
                "[data-testid='stSidebarCollapsedControl']{display:none}</style>", unsafe_allow_html=True)
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.markdown('<div class="lead"><h1>Segment an object by what it looks like and where it ends.</h1>'
                    '<p>Upload an image. The lab builds a color mask and an edge map, fuses them into one object mask, '
                    'and shows every step so you can tune it.</p></div>', unsafe_allow_html=True)
        b1, b2, _ = st.columns([1.1, 1, 1])
        with b1:
            st.button("Open the lab", type="primary", on_click=go_app, use_container_width=True)
        with b2:
            st.button("How it works", on_click=go_app, args=(PAGE_ABOUT,), use_container_width=True)
    with right:
        st.markdown(HERO_SVG, unsafe_allow_html=True)

    st.markdown(
        '<div class="sigs">'
        '<div class="sig c"><b>Color mask</b><span>Fills the object by its color range. Misses nothing inside, '
        'but can leak into similar colors nearby.</span></div>'
        '<div class="sig e"><b>Edge map</b><span>Follows the object boundary. Precise at the border, '
        'but outlines can have gaps and pick up background clutter.</span></div>'
        f'<div class="sig h"><b>Hybrid mask</b><span>Combines both with {len(pr.FUSIONS)} fusion strategies, '
        'then refines it into one clean object.</span></div>'
        '</div>', unsafe_allow_html=True)
    flow(["Input image", "Preprocessing", "Color space analysis", "Color segmentation", "Edge detection",
          "Thresholding", "Morphology", "Hybrid fusion", "Final object", "Result analysis"])


# ------------------------------------------------------------------ pages
def page_dashboard():
    st.markdown('<div class="panel"><p style="margin:0;color:#e6edf7">An intelligent image segmentation framework combining '
                'color information and edge information for accurate object extraction.</p></div>', unsafe_allow_html=True)
    flow(["Input image", "Preprocessing", "Color space analysis", "Color segmentation", "Edge detection",
          "Thresholding", "Morphology", "Hybrid fusion", "Final object", "Result analysis"])
    if not ready():
        if S.rgb is not None:
            show(S.rgb, "Loaded image (not processed yet)")
        return
    R = compute(); A = R["analysis"]
    cols = st.columns(4)
    for c, (img, t) in zip(cols, [(R["rgb"], "Input image"), (R["color"], "Color mask"),
                                  (R["edges"], "Edge map"), (pr.render_output(R["rgb"], R["final"], "Object on black"), "Final segmentation")]):
        with c: show(img, t)
    cols = st.columns(5)
    items = [("Object area", f"{A['pixels']:,} px"), ("Image coverage", f"{A['percent']:.1f}%"),
             ("Objects found", str(len(A["objects"]))), ("Processing time", f"{R['total_ms']:.1f} ms")]
    if S.gt:
        items.append(("IoU (vs ground truth)", f"{pr.gt_metrics(R['final'], S.gt)['IoU']:.3f}"))
    else:
        items.append(("IoU / Dice", "needs ground truth"))
    for c, (l, v) in zip(cols, items):
        with c: mcard(l, v)


def page_input():
    c1, c2 = st.columns([1, 2.2])
    with c1:
        panel("Image input", "<p>Supported: JPG, JPEG, PNG, BMP, WEBP. Nothing is processed until you press Start.</p>")
        up = st.file_uploader("Input image", ["jpg", "jpeg", "png", "bmp", "webp"], label_visibility="collapsed")
        if up is not None:
            data = up.getvalue(); key = (up.name, len(data))
            if key != S.fkey:
                S.rgb, S.info = pr.load_image(data)
                S.fkey, S.started, P["ranges"] = key, False, {}
        gt = st.file_uploader("Ground-truth mask (optional)", ["png", "jpg", "jpeg", "bmp"])
        S.gt = gt.getvalue() if gt else None
        st.markdown('<div class="ct">Preprocessing</div>', unsafe_allow_html=True)
        sld("Max image size (px)", "max_dim", 200, 1600, 50)
        sel("Noise reduction", "denoise", ["None", "Gaussian", "Median"])
        sld("Noise filter kernel", "dk", 3, 11, 2)
        go_ = st.button("Start segmentation", type="primary", use_container_width=True, disabled=S.rgb is None)
        if go_:
            S.started = True; P["ranges"] = {}
    with c2:
        if S.rgb is None:
            panel("No image loaded", "<p>Choose an image on the left to begin.</p>")
            return
        i = S.info
        cols = st.columns(4)
        for c, (l, v) in zip(cols, [("Width", f"{i['width']} px"), ("Height", f"{i['height']} px"),
                                    ("Channels", i["channels"]), ("Format", i["format"])]):
            with c: mcard(l, v)
        if not S.started:
            show(S.rgb, "Original image")
        else:
            R = compute()
            a, b, c = st.columns(3)
            with a: show(S.rgb, "Original image")
            with b: show(R["rgb"], "Preprocessed image")
            with c: show(R["gray"], "Grayscale image")
            st.success(f"Segmentation complete in {R['total_ms']:.1f} ms. Open **Hybrid Fusion** for the final result.")


def page_color():
    if not ready(): return
    c1, c2 = st.columns([1, 3])
    with c1:
        sel("Color space", "space", list(pr.SPACES))
        sp = P["space"]
        pre = pr.preprocess(S.rgb, P["max_dim"], P["denoise"], P["dk"])[0]
        if sp not in P["ranges"]:
            P["ranges"][sp] = pr.auto_range(pre, sp)
        if st.button("Auto-estimate range", use_container_width=True):
            P["ranges"][sp] = pr.auto_range(pre, sp); st.rerun()
        for i, name in enumerate(pr.SPACES[sp][1]):
            P["ranges"][sp][i] = st.slider(name, 0, pr.channel_max(sp, i), tuple(P["ranges"][sp][i]), key=f"{sp}{i}")
        st.caption("Auto range = 5th-95th percentile of the central 20% of the image (+12 margin).")
    with c2:
        R = compute()
        a, b, c = st.columns(3)
        with a: show(R["rgb"], "Original (preprocessed)")
        with b: show(pr.convert(R["rgb"], sp), f"{sp} representation")
        with c: show(R["color"], "Color mask")
        cols = st.columns(3)
        for col, (n, ch) in zip(cols, pr.channel_images(R["rgb"], sp)):
            with col: show(ch, f"{sp} - {n} channel")


def page_edge():
    if not ready(): return
    c1, c2 = st.columns([1, 3])
    with c1:
        sel("Method", "edge_method", ["Canny", "Sobel"])
        if P["edge_method"] == "Canny":
            sld("Lower threshold", "c_low", 0, 255); sld("Upper threshold", "c_high", 0, 255)
        else:
            sel("Sobel kernel size", "s_ksize", [1, 3, 5, 7]); sld("Gradient threshold", "s_thr", 1, 255)
        st.caption("Edges mark object boundaries; they are turned into filled regions in Hybrid Fusion.")
    with c2:
        R = compute()
        a, b = st.columns(2)
        with a: show(R["rgb"], "Input (edges found per RGB channel, merged)")
        with b: show(R["edges"], f"Edge map ({P['edge_method']})")


def page_threshold():
    if not ready(): return
    c1, c2 = st.columns([1, 3])
    with c1:
        sel("Method", "th_method", ["Binary", "Adaptive", "Otsu"])
        if P["th_method"] == "Binary":
            sld("Threshold value", "th_val", 0, 255)
        elif P["th_method"] == "Adaptive":
            sld("Block size", "th_block", 3, 99, 2); sld("Constant C", "th_c", -20, 30)
        chk("Invert (object is darker than background)", "th_inv")
    with c2:
        R = compute()
        st.markdown(f'<div class="panel"><h4>{P["th_method"]} thresholding</h4><p>{R["thresh_info"]}</p></div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        with a: show(R["gray"], "Grayscale input")
        with b: show(R["thresh"], "Threshold mask")


def page_morph():
    if not ready(): return
    c1, c2 = st.columns([1, 3])
    with c1:
        sel("Operation (applied to color mask)", "m_op", pr.MORPH_OPS)
        sld("Kernel size", "m_k", 3, 31, 2); sld("Iterations", "m_it", 1, 5)
        st.caption("The kernel size and iterations are also used for the final refinement of the hybrid mask.")
    with c2:
        R = compute()
        a, b = st.columns(2)
        with a: show(R["color"], "Before morphology")
        with b: show(R["morph"], f"After {P['m_op'].lower()}")
        panel("Why morphology?", "<p>Opening removes small noise, closing fills small holes and connects broken regions, "
              "and the gradient highlights object boundaries.</p>")


def page_hybrid():
    if not ready(): return
    c1, c2 = st.columns([1, 4])
    with c1:
        sel("Fusion strategy", "fuse", pr.FUSIONS)
        if P["fuse"] == "Weighted fusion":
            sld("Color weight", "fuse_w", 0.0, 1.0, 0.05)
        sld("Edge gap closing (px)", "edge_close", 3, 31, 2)
        chk("Keep only the largest object", "keep_largest")
    with c2:
        R = compute()
        st.markdown('<div class="fuse"><div class="fin"><div>Color information</div><div>Edge information</div></div>'
                    '<span class="fa">→</span><div class="fb">Hybrid fusion</div><span class="fa">→</span>'
                    '<div class="fb">Final segmentation</div></div>', unsafe_allow_html=True)
        cols = st.columns(5)
        for col, (img, t) in zip(cols, [(R["rgb"], "Original"), (R["color"], "Color mask"), (R["region"], "Edge regions"),
                                        (R["hybrid"], "Hybrid mask"), (pr.render_output(R["rgb"], R["final"], "Object on black"), "Final object")]):
            with col: show(img, t)
        st.caption("Edge map → gaps closed → outlines filled = edge regions. Fusion combines these with the cleaned color mask, then refinement "
                   "(closing, opening, hole filling) produces the final mask.")


def page_results():
    if not ready(): return
    R = compute(); A = R["analysis"]
    t1, t2, t3, t4 = st.tabs(["Final output", "Contour analysis", "Measurements", "Comparison"])
    with t1:
        c1, c2 = st.columns([1, 3])
        with c1:
            modes = ["Original background, object highlighted", "Object on black", "Object on white", "Transparent background"]
            sel("Output style", "out_mode", modes)
            out = pr.render_output(R["rgb"], R["final"], P["out_mode"].replace("Original background, object highlighted", "highlight"))
            st.download_button("Download output (PNG)", png_bytes(out), "segmentation_output.png", "image/png", use_container_width=True)
        with c2:
            a, b, c = st.columns(3)
            with a: show(out, P["out_mode"])
            with b: show(pr.boundary_overlay(R["rgb"], R["final"]), "Original + detected boundary")
            with c: show(pr.boundary_overlay(R["rgb"], R["final"], fill=True), "Original + segmentation overlay")
    with t2:
        c1, c2 = st.columns([1, 3])
        with c1:
            sel("Display", "obj_mode", ["Show largest object", "Show all objects"])
        with c2:
            a, b = st.columns([3, 2])
            with a: show(pr.draw_objects(R["rgb"], A["contours"], P["obj_mode"] == "Show largest object"), "Contours and bounding box")
            with b:
                if A["objects"]:
                    st.dataframe(A["objects"] if P["obj_mode"] == "Show all objects" else A["objects"][:1], use_container_width=True, hide_index=True)
                else:
                    st.warning("No contours found. Adjust the color range or fusion strategy.")
    with t3:
        cols = st.columns(5)
        vals = [("Segmented pixels", f"{A['pixels']:,}"), ("Image coverage", f"{A['percent']:.2f}%"),
                ("Contours", len(A["objects"])), ("Bounding box", str(A["bbox"]) if A["bbox"] else "-"),
                ("Total time", f"{R['total_ms']:.1f} ms")]
        for c, (l, v) in zip(cols, vals):
            with c: mcard(l, v)
        a, b = st.columns(2)
        with a:
            fig = go.Figure(go.Bar(x=list(R["times"].values()), y=list(R["times"].keys()), orientation="h", marker_color="#22d3ee"))
            fig.update_layout(title="Processing time per stage (ms)", yaxis=dict(autorange="reversed"))
            st.plotly_chart(plot_style(fig, 320), use_container_width=True)
        with b:
            if S.gt:
                m = pr.gt_metrics(R["final"], S.gt)
                fig = go.Figure(go.Bar(x=list(m.keys()), y=list(m.values()), marker_color="#3b82f6", text=[f"{v:.3f}" for v in m.values()]))
                fig.update_layout(title="Segmentation quality vs ground truth", yaxis=dict(range=[0, 1]))
                st.plotly_chart(plot_style(fig, 320), use_container_width=True)
            else:
                panel("Quality metrics unavailable", "<p>Ground-truth mask not provided; segmentation-quality metrics requiring "
                      "reference data are unavailable.</p>")
    with t4:
        views = {"Original": R["rgb"], "Color mask": R["color"], "Edge map": R["edges"], "Threshold mask": R["thresh"],
                 "Morphology result": R["morph"], "Hybrid mask": R["hybrid"],
                 "Final object": pr.render_output(R["rgb"], R["final"], "Object on black")}
        names = list(views)
        a, b = st.columns(2)
        with a:
            l = st.selectbox("Left image", names, index=0)
            show(views[l], l)
        with b:
            r = st.selectbox("Right image", names, index=len(names) - 1)
            show(views[r], r)


def page_about():
    a, b = st.columns(2)
    with a:
        panel("Project", "<p><b>Hybrid Edge-Color Object Segmentation</b><br>Domain: Computer Vision / Digital Image Processing<br>"
              "Purpose: improve object segmentation by combining color-based information with edge information.</p>")
        panel("Methodology", "<ol><li>Resize and optionally denoise (Gaussian/median); convert to grayscale.</li>"
              "<li>Convert to RGB, HSV, LAB or YCrCb and threshold each channel range into a color mask.</li>"
              "<li>Detect edges with Canny or Sobel, close gaps and fill outlines to get edge regions.</li>"
              "<li>Fuse the cleaned color mask with the edge regions (AND, OR or weighted).</li>"
              "<li>Refine with closing, opening and hole filling; analyse contours.</li></ol>")
        panel("Technology used", "<p>Python, Streamlit, OpenCV, NumPy, Pillow, Plotly.</p>")
    with b:
        panel("Applications", "<ul><li>Object extraction for photo editing</li><li>Industrial inspection</li>"
              "<li>Preprocessing for recognition systems</li><li>Medical and agricultural image analysis</li></ul>")
        panel("Limitations", "<ul><li>Color thresholds depend on lighting and need tuning per image.</li>"
              "<li>Cluttered backgrounds produce many edges and can merge with the object.</li>"
              "<li>Objects whose color matches the background are hard to separate.</li></ul>")
        panel("Future scope", "<ul><li>GrabCut or watershed refinement</li><li>Automatic parameter search against ground truth</li>"
              "<li>Batch processing and video segmentation</li></ul>")


# ------------------------------------------------------------------ shell
PAGES = {"🏠  Dashboard": page_dashboard, "📂  Image Input": page_input, "🎨  Color Spaces": page_color,
         "🔲  Edge Detection": page_edge, "⚫  Threshold": page_threshold, "🧱  Morphology": page_morph,
         "🔗  Hybrid Fusion": page_hybrid, "📊  Results & Analysis": page_results, PAGE_ABOUT: page_about}

# opening screen: shown until the user presses "Open the lab"
if not S.entered:
    page_landing()
    st.stop()

with st.sidebar:
    st.markdown('<div class="brand"><b>Hybrid Edge-Color<br>Object Segmentation</b><span>Computer Vision Lab</span></div>', unsafe_allow_html=True)
    page = st.radio("Navigation", list(PAGES), key="nav", label_visibility="collapsed")
    state = ("Segmented", "#22c55e") if S.started else (("Image loaded", "#facc15") if S.rgb is not None else ("No image", "#64748b"))
    st.markdown(f'<div class="status"><span class="dot" style="background:{state[1]}"></span><b>{state[0]}</b></div>', unsafe_allow_html=True)
    st.button("Reset settings", on_click=reset_settings, use_container_width=True)
    st.button("Back to home screen", on_click=go_home, use_container_width=True)

st.markdown(f'<div class="apphead"><div><h1>Hybrid edge-color object segmentation</h1>'
            f'<p>Advanced Computer Vision &amp; Image Processing System</p></div><div class="pg">{page.split("  ", 1)[1]}</div></div>',
            unsafe_allow_html=True)
PAGES[page]()

# previous / next page buttons
names = list(PAGES)
idx = names.index(page)
st.markdown('<div class="navsep"></div>', unsafe_allow_html=True)
back_col, _, next_col = st.columns([1.2, 3, 1.2])
with back_col:
    if idx > 0:
        st.button("Back: " + names[idx - 1].split("  ", 1)[1], key="nav_back", on_click=go_to,
                  args=(names[idx - 1],), use_container_width=True)
with next_col:
    if idx < len(names) - 1:
        st.button("Next: " + names[idx + 1].split("  ", 1)[1], key="nav_next", type="primary", on_click=go_to,
                  args=(names[idx + 1],), use_container_width=True)