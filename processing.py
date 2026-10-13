"""Hybrid Edge-Color Object Segmentation - processing logic (no UI code).

Pipeline: preprocess -> color mask -> edge map -> threshold -> morphology
          -> hybrid fusion -> refinement -> contour analysis.
"""
import io
import time

import cv2
import numpy as np
from PIL import Image, ImageOps

# name -> (OpenCV conversion code from RGB, channel names)
SPACES = {
    "RGB": (None, ["Red", "Green", "Blue"]),
    "HSV": (cv2.COLOR_RGB2HSV, ["Hue", "Saturation", "Value"]),
    "LAB": (cv2.COLOR_RGB2LAB, ["L", "A", "B"]),
    "YCrCb": (cv2.COLOR_RGB2YCrCb, ["Y", "Cr", "Cb"]),
}
MORPH_OPS = ["Erosion", "Dilation", "Opening", "Closing", "Morphological Gradient"]
FUSIONS = ["AND (color inside edge regions)", "OR (color or edge regions)", "Weighted fusion"]


# ---------------------------------------------------------------- input
def load_image(data: bytes):
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    info = {"format": Image.open(io.BytesIO(data)).format, "channels": len(img.getbands()),
            "width": img.width, "height": img.height}
    return np.array(img.convert("RGB")), info


def preprocess(rgb, max_dim=800, denoise="None", k=3):
    h, w = rgb.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    k = k if k % 2 else k + 1
    if denoise == "Gaussian":
        rgb = cv2.GaussianBlur(rgb, (k, k), 0)
    elif denoise == "Median":
        rgb = cv2.medianBlur(rgb, k)
    return rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


# ---------------------------------------------------------------- color
def convert(rgb, space):
    code = SPACES[space][0]
    return rgb.copy() if code is None else cv2.cvtColor(rgb, code)


def channel_max(space, i):
    return 179 if (space == "HSV" and i == 0) else 255


def channel_images(rgb, space):
    conv = convert(rgb, space)
    return [(name, conv[:, :, i]) for i, name in enumerate(SPACES[space][1])]


def auto_range(rgb, space, pad=12):
    """Estimate colour limits from the central 20% of the image (assumes the
    object is roughly centred): 5th-95th percentile of each channel, padded."""
    conv = convert(rgb, space)
    h, w = conv.shape[:2]
    c = conv[int(h * .4):int(h * .6), int(w * .4):int(w * .6)].reshape(-1, 3)
    out = []
    for i in range(3):
        lo, hi = np.percentile(c[:, i], [5, 95])
        m = channel_max(space, i)
        out.append((int(max(0, lo - pad)), int(min(m, hi + pad))))
    return out


def color_mask(rgb, space, ranges):
    conv = convert(rgb, space)
    lo = np.array([r[0] for r in ranges], np.uint8)
    hi = np.array([r[1] for r in ranges], np.uint8)
    return cv2.inRange(conv, lo, hi)


# ---------------------------------------------------------------- edges / threshold
def _edge_single(ch, method, low, high, ksize, sobel_thr):
    if method == "Canny":
        return cv2.Canny(ch, low, high)
    gx = cv2.Sobel(ch, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(ch, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = np.hypot(gx, gy)
    if mag.max() > 0:
        mag = mag / mag.max() * 255
    return ((mag >= sobel_thr) * 255).astype(np.uint8)


def edge_map(img, method="Canny", low=50, high=150, ksize=3, sobel_thr=60):
    """Edges per channel, merged with a pixel-wise maximum. A colour image is
    used so boundaries between colours of equal brightness are not missed."""
    if img.ndim == 2:
        return _edge_single(img, method, low, high, ksize, sobel_thr)
    maps = [_edge_single(img[:, :, i], method, low, high, ksize, sobel_thr) for i in range(img.shape[2])]
    return np.maximum.reduce(maps)


def threshold_mask(gray, method="Otsu", value=127, block=35, c=5, invert=False):
    t = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    if method == "Binary":
        _, m = cv2.threshold(gray, value, 255, t)
        return m, f"Fixed global threshold T = {value}"
    if method == "Otsu":
        T, m = cv2.threshold(gray, 0, 255, t + cv2.THRESH_OTSU)
        return m, f"Otsu automatically selected T = {int(T)}"
    block = block if block % 2 else block + 1
    m = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, t, max(block, 3), c)
    return m, f"Adaptive Gaussian threshold (block {block}, C = {c})"


# ---------------------------------------------------------------- morphology / fusion
def morph(mask, op="Opening", k=5, iters=1):
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    if op == "Erosion":
        return cv2.erode(mask, ker, iterations=iters)
    if op == "Dilation":
        return cv2.dilate(mask, ker, iterations=iters)
    code = {"Opening": cv2.MORPH_OPEN, "Closing": cv2.MORPH_CLOSE,
            "Morphological Gradient": cv2.MORPH_GRADIENT}[op]
    return cv2.morphologyEx(mask, code, ker, iterations=iters)


def edge_region(edges, close_k=7):
    """Close gaps in the edge map and fill the closed outlines -> edge-bounded regions."""
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, ker)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region = np.zeros_like(edges)
    cv2.drawContours(region, cnts, -1, 255, cv2.FILLED)
    return region


def fuse(color, region, strategy, w=0.6):
    c, r = color > 0, region > 0
    if strategy.startswith("AND"):
        m = c & r
    elif strategy.startswith("OR"):
        m = c | r
    else:  # weighted: score = w*color + (1-w)*edge region, keep score >= 0.5
        m = (w * c + (1 - w) * r) >= 0.5
    return (m * 255).astype(np.uint8)


def refine(mask, k=5, iters=1, keep_largest=True):
    """Closing (connect/fill) -> opening (remove noise) -> fill holes -> optional largest object."""
    m = morph(morph(mask, "Closing", k, iters), "Opening", k, iters)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if keep_largest and cnts:
        cnts = [max(cnts, key=cv2.contourArea)]
    out = np.zeros_like(m)
    cv2.drawContours(out, cnts, -1, 255, cv2.FILLED)
    return out


# ---------------------------------------------------------------- analysis / rendering
def analyse(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    objs = [{"Object": i + 1, "Area (px)": int(cv2.contourArea(c)),
             "Perimeter (px)": round(cv2.arcLength(c, True), 1),
             "Bounding box (x, y, w, h)": cv2.boundingRect(c)} for i, c in enumerate(cnts)]
    px = int((mask > 0).sum())
    return {"pixels": px, "percent": 100 * px / mask.size, "contours": cnts, "objects": objs,
            "bbox": cv2.boundingRect(cnts[0]) if cnts else None}


def draw_objects(rgb, cnts, largest_only=True):
    img = rgb.copy()
    for c in (cnts[:1] if largest_only else cnts):
        cv2.drawContours(img, [c], -1, (34, 211, 238), 2)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (250, 204, 21), 2)
    return img


def render_output(rgb, mask, mode):
    m = mask > 0
    if mode == "Object on black":
        return rgb * m[..., None]
    if mode == "Object on white":
        return np.where(m[..., None], rgb, 255).astype(np.uint8)
    if mode == "Transparent background":
        return np.dstack([rgb, mask])
    dim = (rgb * 0.25).astype(np.uint8)          # original background, object highlighted
    tint = (rgb * 0.7 + np.array([34, 211, 238]) * 0.3).astype(np.uint8)
    out = np.where(m[..., None], tint, dim)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (34, 211, 238), 2)
    return out


def boundary_overlay(rgb, mask, fill=False):
    out = rgb.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if fill:
        tint = (rgb * 0.55 + np.array([34, 211, 238]) * 0.45).astype(np.uint8)
        out = np.where((mask > 0)[..., None], tint, rgb)
    cv2.drawContours(out, cnts, -1, (250, 204, 21), 2)
    return out


def gt_metrics(mask, gt_bytes):
    gt = Image.open(io.BytesIO(gt_bytes)).convert("L").resize((mask.shape[1], mask.shape[0]), Image.NEAREST)
    g, p = np.array(gt) > 127, mask > 0
    tp, fp = int((g & p).sum()), int((~g & p).sum())
    fn, tn = int((g & ~p).sum()), int((~g & ~p).sum())
    d = lambda a, b: a / b if b else 0.0
    return {"IoU": d(tp, tp + fp + fn), "Dice": d(2 * tp, 2 * tp + fp + fn),
            "Precision": d(tp, tp + fp), "Recall": d(tp, tp + fn), "Accuracy": d(tp + tn, tp + tn + fp + fn)}


# ---------------------------------------------------------------- full pipeline
def run_pipeline(rgb0, P):
    T, R = {}, {}
    t = time.perf_counter()

    def lap(name):
        nonlocal t
        now = time.perf_counter()
        T[name] = (now - t) * 1000
        t = now

    R["rgb"], R["gray"] = preprocess(rgb0, P["max_dim"], P["denoise"], P["dk"]); lap("Preprocessing")
    sp = P["space"]
    if sp not in P["ranges"]:
        P["ranges"][sp] = auto_range(R["rgb"], sp)
    R["color"] = color_mask(R["rgb"], sp, P["ranges"][sp]); lap("Color segmentation")
    R["edges"] = edge_map(R["rgb"], P["edge_method"], P["c_low"], P["c_high"], P["s_ksize"], P["s_thr"]); lap("Edge detection")
    R["thresh"], R["thresh_info"] = threshold_mask(R["gray"], P["th_method"], P["th_val"], P["th_block"], P["th_c"], P["th_inv"]); lap("Thresholding")
    R["morph"] = morph(R["color"], P["m_op"], P["m_k"], P["m_it"]); lap("Morphology")
    # fusion uses a cleaned colour mask (opening removes speckle) + edge-bounded regions
    clean = morph(R["color"], "Opening", P["m_k"], 1)
    R["region"] = edge_region(R["edges"], P["edge_close"])
    R["hybrid"] = fuse(clean, R["region"], P["fuse"], P["fuse_w"]); lap("Hybrid fusion")
    R["final"] = refine(R["hybrid"], P["m_k"], P["m_it"], P["keep_largest"]); lap("Refinement")
    R["analysis"] = analyse(R["final"]); lap("Contour analysis")
    R["times"] = T
    R["total_ms"] = sum(T.values())
    return R