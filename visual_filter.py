#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
visual_data_auto_filter_full_copyright.py

视觉数据自动初筛工具：图片/视频低质、严重不符、疑似版权/水印风险初筛 + BAD 分级

核心功能：
1. OpenCV 基础画质检测：
   - 分辨率过低、过暗、过曝、模糊、低对比度、黑屏、白屏
   - 视频坏帧、冻结帧、重复帧比例

2. YOLO 主体检测：
   - expected：期望出现的类别
   - forbid：禁止出现的类别
   - 视频按抽帧统计目标匹配比例

3. 可选 CLIP 文本匹配：
   - 用于 caption / VLM 输出和画面严重不符初筛
   - 默认关闭，需要 --enable_clip

4. 版权/水印风险初筛：
   - 角落水印/Logo 风险：检测角落区域边缘复杂度、亮色文字/透明水印特征
   - 大面积文字/字幕风险：检测画面底部字幕条、文字密集区域
   - 可选 OCR 关键词检测：识别“版权所有、copyright、©、抖音、快手、bilibili”等
   - 可选模板匹配：用已知 Logo / 水印图片做匹配
   - 可选参考库相似度：用感知哈希检测是否和已知版权图片库高度相似

重要说明：
- “版权归属”不能靠脚本直接判断，本工具只能做“版权/水印风险初筛”。
- 版权、合规、侵权判断必须结合来源、授权信息、人工复核和业务规则。
- 本脚本输出的 copyright_risk 只表示“疑似风险”，不是法律结论。

安装基础依赖：
pip install opencv-python pandas tqdm ultralytics numpy pillow

可选 OCR：
pip install pytesseract
还需要系统安装 Tesseract OCR 程序。
Windows 示例：安装后可能需要配置 --tesseract_cmd "C:/Program Files/Tesseract-OCR/tesseract.exe"

可选 CLIP：
pip install torch transformers

基础示例：
python visual_data_auto_filter_full_copyright.py --input ./data --disable_yolo --enable_copyright

主体 + 版权风险：
python visual_data_auto_filter_full_copyright.py --input ./data --expected person "cell phone" --enable_copyright --copy_mode bad_review

启用 OCR 版权关键词：
python visual_data_auto_filter_full_copyright.py --input ./data --enable_copyright --enable_ocr --ocr_lang chi_sim+eng

启用 Logo 模板匹配：
python visual_data_auto_filter_full_copyright.py --input ./data --enable_copyright --template_dir ./logo_templates

启用参考库相似度：
python visual_data_auto_filter_full_copyright.py --input ./data --enable_copyright --reference_dir ./copyright_refs
"""

import argparse
import json
import math
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import torch
    from PIL import Image
    from transformers import CLIPProcessor, CLIPModel
except Exception:
    torch = None
    Image = None
    CLIPProcessor = None
    CLIPModel = None

try:
    import pytesseract
except Exception:
    pytesseract = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"}


@dataclass
class QualityMetric:
    width: int = 0
    height: int = 0
    brightness: float = 0.0
    blur: float = 0.0
    contrast: float = 0.0
    black_ratio: float = 0.0
    white_ratio: float = 0.0
    colorfulness: float = 0.0


@dataclass
class CopyrightMetric:
    risk_score: float = 0.0
    corner_watermark_score: float = 0.0
    bottom_text_score: float = 0.0
    center_text_score: float = 0.0
    template_match_score: float = 0.0
    reference_hash_min_distance: int = -1
    ocr_hit: bool = False
    ocr_text: str = ""
    reasons: str = ""


@dataclass
class FileResult:
    file: str
    file_type: str
    decision: str
    score: float
    reasons: str

    # BAD 分级：
    # GOOD：正常样本
    # REVIEW：可疑样本，建议人工复核
    # SOFT_BAD：自动规则判坏，但可能有主体价值，不建议直接删除
    # HARD_BAD：硬坏样本，如打不开、无有效帧、严重损坏，通常可直接剔除
    bad_level: str = "GOOD"
    bad_category: str = ""
    rescue_hint: str = ""

    width: int = 0
    height: int = 0
    duration_sec: float = 0.0
    total_frames: int = 0
    sampled_frames: int = 0

    quality_bad_ratio: float = 0.0
    freeze_ratio: float = 0.0
    read_fail_ratio: float = 0.0

    yolo_match_ratio: float = 0.0
    yolo_forbid_ratio: float = 0.0
    yolo_detected_classes: str = ""

    clip_score: float = -1.0

    copyright_risk_ratio: float = 0.0
    copyright_max_score: float = 0.0
    copyright_reasons: str = ""

    avg_brightness: float = 0.0
    avg_blur: float = 0.0
    avg_contrast: float = 0.0
    avg_black_ratio: float = 0.0
    avg_white_ratio: float = 0.0


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def collect_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path] if (is_image(input_path) or is_video(input_path)) else []

    files = []
    for p in input_path.rglob("*"):
        if p.is_file() and (is_image(p) or is_video(p)):
            files.append(p)
    return sorted(files)


def calc_colorfulness(frame: np.ndarray) -> float:
    b, g, r = cv2.split(frame.astype("float"))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    std_rg, mean_rg = np.std(rg), np.mean(rg)
    std_yb, mean_yb = np.std(yb), np.mean(yb)
    return float(np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2))


def calc_quality(frame: np.ndarray) -> QualityMetric:
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    black_ratio = float(np.mean(gray < 10))
    white_ratio = float(np.mean(gray > 245))
    colorfulness = calc_colorfulness(frame)

    return QualityMetric(
        width=w,
        height=h,
        brightness=brightness,
        blur=blur,
        contrast=contrast,
        black_ratio=black_ratio,
        white_ratio=white_ratio,
        colorfulness=colorfulness,
    )


def quality_reasons(q: QualityMetric, args) -> List[str]:
    reasons = []

    if q.width < args.min_width or q.height < args.min_height:
        reasons.append(f"分辨率过低:{q.width}x{q.height}")

    if q.blur < args.min_blur:
        reasons.append(f"画面模糊:blur={q.blur:.1f}")

    if q.brightness < args.min_brightness:
        reasons.append(f"画面过暗:brightness={q.brightness:.1f}")

    if q.brightness > args.max_brightness:
        reasons.append(f"画面过曝:brightness={q.brightness:.1f}")

    if q.contrast < args.min_contrast:
        reasons.append(f"对比度过低:contrast={q.contrast:.1f}")

    if q.black_ratio > args.max_black_ratio:
        reasons.append(f"黑屏比例过高:black={q.black_ratio:.2f}")

    if q.white_ratio > args.max_white_ratio:
        reasons.append(f"白屏比例过高:white={q.white_ratio:.2f}")

    return reasons


def frame_diff_score(prev: Optional[np.ndarray], curr: np.ndarray) -> float:
    if prev is None:
        return 9999.0

    prev_small = cv2.resize(prev, (160, 90))
    curr_small = cv2.resize(curr, (160, 90))
    prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(np.mean(diff))


def load_yolo_model(args):
    if args.disable_yolo:
        return None

    if YOLO is None:
        print("警告：未安装 ultralytics，YOLO 检测将被跳过。安装：pip install ultralytics")
        return None

    try:
        return YOLO(args.model)
    except Exception as e:
        print(f"警告：YOLO 模型加载失败，检测将被跳过：{e}")
        return None


def yolo_detect(model, frame: np.ndarray, conf: float) -> Tuple[List[str], List[Tuple[str, float, Tuple[int, int, int, int]]]]:
    if model is None:
        return [], []

    results = model.predict(frame, conf=conf, verbose=False)
    names = model.names
    classes = []
    boxes = []

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            name = names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            classes.append(name)
            boxes.append((name, score, (x1, y1, x2, y2)))

    return classes, boxes


class ClipChecker:
    def __init__(self, enabled: bool, text: str, model_name: str = "openai/clip-vit-base-patch32"):
        self.enabled = enabled
        self.text = text
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = "cpu"

        if not enabled or not text:
            self.enabled = False
            return

        if torch is None or CLIPModel is None or CLIPProcessor is None or Image is None:
            print("警告：未安装 torch/transformers/Pillow，CLIP 检测将被跳过。")
            self.enabled = False
            return

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.enabled = True
        except Exception as e:
            print(f"警告：CLIP 模型加载失败，CLIP 检测将被跳过：{e}")
            self.enabled = False

    def score_frame(self, frame_bgr: np.ndarray) -> float:
        if not self.enabled:
            return -1.0

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        inputs = self.processor(
            text=[self.text],
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            score = float(logits_per_image[0][0].detach().cpu())

        return score


def ahash(frame: np.ndarray, hash_size: int = 8) -> int:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (hash_size, hash_size))
    avg = small.mean()
    bits = (small > avg).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def load_reference_hashes(reference_dir: Optional[str]) -> List[Tuple[str, int]]:
    refs = []
    if not reference_dir:
        return refs

    ref_dir = Path(reference_dir)
    if not ref_dir.exists():
        print(f"警告：参考库目录不存在：{ref_dir}")
        return refs

    for p in ref_dir.rglob("*"):
        if not p.is_file() or not is_image(p):
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        refs.append((str(p), ahash(img)))

    print(f"已加载参考库图片哈希数量：{len(refs)}")
    return refs


def load_template_images(template_dir: Optional[str]) -> List[Tuple[str, np.ndarray]]:
    templates = []
    if not template_dir:
        return templates

    tdir = Path(template_dir)
    if not tdir.exists():
        print(f"警告：模板目录不存在：{tdir}")
        return templates

    for p in tdir.rglob("*"):
        if not p.is_file() or not is_image(p):
            continue

        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # 模板太大时缩小，避免过慢
        h, w = img.shape[:2]
        if max(h, w) > 220:
            scale = 220 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        templates.append((str(p), img))

    print(f"已加载 Logo/水印模板数量：{len(templates)}")
    return templates


class CopyrightChecker:
    def __init__(self, args):
        self.enabled = args.enable_copyright
        self.args = args
        self.templates = load_template_images(args.template_dir) if self.enabled else []
        self.reference_hashes = load_reference_hashes(args.reference_dir) if self.enabled else []

        if args.enable_ocr and pytesseract is None:
            print("警告：未安装 pytesseract，OCR 检测将跳过。安装：pip install pytesseract")

        if args.tesseract_cmd and pytesseract is not None:
            pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

        default_keywords = [
            "版权", "版权所有", "未经授权", "侵权", "转载", "来源", "水印",
            "copyright", "all rights reserved", "©",
            "抖音", "douyin", "快手", "kuaishou", "bilibili", "哔哩哔哩",
            "youtube", "youtu", "instagram", "tiktok", "微博", "小红书",
            "腾讯视频", "优酷", "爱奇艺", "芒果tv", "西瓜视频",
        ]

        keywords = list(default_keywords)
        if args.copyright_keywords:
            keywords.extend(args.copyright_keywords)

        self.keywords = [k.lower() for k in keywords]

    def edge_text_score(self, roi: np.ndarray) -> float:
        """
        用边缘密度 + 亮色比例估计文字/水印风险。
        不是精准 OCR，只是便宜的初筛。
        """
        if roi.size == 0:
            return 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)

        edge_ratio = float(np.mean(edges > 0))
        bright_ratio = float(np.mean(gray > 180))
        dark_ratio = float(np.mean(gray < 60))
        contrast = float(np.std(gray))

        # 文字/Logo 常见特征：边缘较密、局部对比明显、亮/暗像素比例较突出
        score = 0.0
        score += min(edge_ratio / 0.12, 1.0) * 45
        score += min(contrast / 70.0, 1.0) * 35
        score += min(max(bright_ratio, dark_ratio) / 0.45, 1.0) * 20
        return float(min(score, 100.0))

    def corner_watermark_score(self, frame: np.ndarray) -> float:
        h, w = frame.shape[:2]
        ch, cw = int(h * 0.22), int(w * 0.28)

        corners = [
            frame[0:ch, 0:cw],
            frame[0:ch, w - cw:w],
            frame[h - ch:h, 0:cw],
            frame[h - ch:h, w - cw:w],
        ]

        scores = [self.edge_text_score(c) for c in corners]
        return float(max(scores)) if scores else 0.0

    def bottom_text_score(self, frame: np.ndarray) -> float:
        h, w = frame.shape[:2]
        roi = frame[int(h * 0.76):h, 0:w]
        return self.edge_text_score(roi)

    def center_text_score(self, frame: np.ndarray) -> float:
        h, w = frame.shape[:2]
        y1, y2 = int(h * 0.35), int(h * 0.65)
        x1, x2 = int(w * 0.15), int(w * 0.85)
        roi = frame[y1:y2, x1:x2]
        return self.edge_text_score(roi)

    def template_match_score(self, frame: np.ndarray) -> float:
        if not self.templates:
            return 0.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        best = 0.0

        # 优先在角落和底部查 Logo/水印，减少误判和耗时
        rois = []
        ch, cw = int(h * 0.30), int(w * 0.35)
        rois.append(gray[0:ch, 0:cw])
        rois.append(gray[0:ch, w - cw:w])
        rois.append(gray[h - ch:h, 0:cw])
        rois.append(gray[h - ch:h, w - cw:w])
        rois.append(gray[int(h * 0.70):h, 0:w])

        for _, templ in self.templates:
            th, tw = templ.shape[:2]
            for roi in rois:
                rh, rw = roi.shape[:2]
                if th >= rh or tw >= rw:
                    continue
                try:
                    res = cv2.matchTemplate(roi, templ, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    best = max(best, float(max_val))
                except Exception:
                    continue

        return best

    def reference_hash_distance(self, frame: np.ndarray) -> int:
        if not self.reference_hashes:
            return -1

        h = ahash(frame)
        distances = [hamming_distance(h, ref_h) for _, ref_h in self.reference_hashes]
        return int(min(distances)) if distances else -1

    def run_ocr(self, frame: np.ndarray) -> Tuple[bool, str]:
        if not self.args.enable_ocr or pytesseract is None:
            return False, ""

        # OCR 太慢，这里缩小图像并只做全图粗识别
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        max_side = max(h, w)
        if max_side > 1000:
            scale = 1000 / max_side
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))

        try:
            text = pytesseract.image_to_string(rgb, lang=self.args.ocr_lang)
        except Exception as e:
            return False, f"OCR失败:{e}"

        text_norm = text.lower().replace(" ", "")
        hit = any(k.replace(" ", "") in text_norm for k in self.keywords)
        return hit, text.strip().replace("\n", " ")[:300]

    def check_frame(self, frame: np.ndarray) -> CopyrightMetric:
        if not self.enabled:
            return CopyrightMetric()

        reasons = []

        corner_score = self.corner_watermark_score(frame)
        bottom_score = self.bottom_text_score(frame)
        center_score = self.center_text_score(frame)

        template_score = self.template_match_score(frame)
        ref_dist = self.reference_hash_distance(frame)

        ocr_hit, ocr_text = self.run_ocr(frame)

        risk_score = 0.0

        if corner_score >= self.args.corner_watermark_threshold:
            reasons.append(f"角落疑似水印/Logo:{corner_score:.1f}")
            risk_score += 30

        if bottom_score >= self.args.bottom_text_threshold:
            reasons.append(f"底部疑似字幕/水印文字:{bottom_score:.1f}")
            risk_score += 25

        if center_score >= self.args.center_text_threshold:
            reasons.append(f"中心区域疑似大面积文字/水印:{center_score:.1f}")
            risk_score += 20

        if template_score >= self.args.template_match_threshold:
            reasons.append(f"命中Logo/水印模板:{template_score:.2f}")
            risk_score += 45

        if ref_dist >= 0 and ref_dist <= self.args.reference_hash_threshold:
            reasons.append(f"与版权参考库高度相似:hash_distance={ref_dist}")
            risk_score += 55

        if ocr_hit:
            reasons.append(f"OCR命中版权/平台关键词:{ocr_text[:120]}")
            risk_score += 45

        risk_score = float(min(risk_score, 100.0))

        return CopyrightMetric(
            risk_score=risk_score,
            corner_watermark_score=round(corner_score, 3),
            bottom_text_score=round(bottom_score, 3),
            center_text_score=round(center_score, 3),
            template_match_score=round(template_score, 4),
            reference_hash_min_distance=ref_dist,
            ocr_hit=ocr_hit,
            ocr_text=ocr_text,
            reasons="；".join(reasons),
        )


def decide_from_reasons(
    hard_reasons: List[str],
    review_reasons: List[str],
    base_score: float = 100.0,
    subject_present: bool = False,
    fatal_reasons: Optional[List[str]] = None,
    enable_bad_grading: bool = True,
    protect_subject: bool = True,
) -> Tuple[str, float, str, str, str, str]:
    """
    返回：
    decision, score, reasons, bad_level, bad_category, rescue_hint

    BAD 分级逻辑：
    1. fatal_reasons：文件打不开、视频无有效帧等不可挽救问题 -> HARD_BAD
    2. hard_reasons + 没有主体保护 -> BAD / SOFT_BAD
    3. 如果检测到主体目标，且问题主要是画质/水印/轻微风险 -> 降级为 REVIEW，避免误删有价值主体样本
    """
    fatal_reasons = fatal_reasons or []

    all_reasons = fatal_reasons + hard_reasons + review_reasons

    score = base_score
    score -= 50 * len(fatal_reasons)
    score -= 30 * len(hard_reasons)
    score -= 15 * len(review_reasons)
    score = max(0.0, min(100.0, score))

    reasons = "；".join(all_reasons)

    if not enable_bad_grading:
        if fatal_reasons or hard_reasons:
            return "BAD", score, reasons, "BAD", "未分级BAD", ""
        if review_reasons:
            return "REVIEW", score, reasons, "REVIEW", "可疑样本", "建议人工复核"
        return "GOOD", score, reasons, "GOOD", "", ""

    if fatal_reasons:
        return (
            "BAD",
            score,
            reasons,
            "HARD_BAD",
            "文件级硬错误/不可挽救样本",
            "文件打不开、视频无有效帧或严重损坏，通常可直接剔除。"
        )

    if hard_reasons:
        # 有主体时保护：避免把主体可用样本直接丢掉
        if protect_subject and subject_present:
            return (
                "REVIEW",
                score,
                reasons,
                "REVIEW",
                "有主体但存在质量/版权/规则风险",
                "检测到目标主体，虽然存在问题，但可能仍有训练价值；建议人工复核后决定保留、修正或剔除。"
            )

        return (
            "BAD",
            score,
            reasons,
            "SOFT_BAD",
            "规则判坏/可能可挽救样本",
            "自动规则判为BAD，但不是文件级硬错误；建议抽检，若主体清晰可改为REVIEW或保留为困难样本。"
        )

    if review_reasons:
        return (
            "REVIEW",
            score,
            reasons,
            "REVIEW",
            "可疑样本",
            "建议人工复核，确认是否进入标注或训练。"
        )

    return "GOOD", score, reasons, "GOOD", "", ""


def draw_preview(frames: List[np.ndarray], out_path: Path, title: str, max_cols: int = 4) -> None:
    if not frames:
        return

    thumbs = []
    for frame in frames:
        thumb = cv2.resize(frame, (240, 135))
        thumbs.append(thumb)

    cols = min(max_cols, len(thumbs))
    rows = math.ceil(len(thumbs) / cols)
    canvas = np.full((rows * 135 + 40, cols * 240, 3), 255, dtype=np.uint8)

    cv2.putText(
        canvas,
        title[:80],
        (5, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)
        y = 40 + r * 135
        x = c * 240
        canvas[y:y + 135, x:x + 240] = thumb
        cv2.putText(
            canvas,
            f"#{i + 1}",
            (x + 5, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    safe_mkdir(out_path.parent)
    cv2.imwrite(str(out_path), canvas)


def apply_copyright_decision(
    metric: CopyrightMetric,
    args,
    hard_reasons: List[str],
    review_reasons: List[str],
) -> None:
    if not args.enable_copyright:
        return

    if metric.risk_score <= 0:
        return

    reason = f"疑似版权/水印风险:score={metric.risk_score:.1f}"
    if metric.reasons:
        reason += f"({metric.reasons})"

    # 默认版权只进 REVIEW，避免误杀。
    # 如果业务要求严格，可以加 --copyright_as_bad。
    if args.copyright_as_bad and metric.risk_score >= args.copyright_bad_score:
        hard_reasons.append(reason)
    elif metric.risk_score >= args.copyright_review_score:
        review_reasons.append(reason)


def analyze_image(path: Path, yolo_model, clip_checker: ClipChecker, copyright_checker: CopyrightChecker, args) -> FileResult:
    frame = cv2.imread(str(path))
    if frame is None:
        return FileResult(
            file=str(path),
            file_type="image",
            decision="BAD",
            score=0,
            reasons="图片读取失败或文件损坏",
            bad_level="HARD_BAD",
            bad_category="文件级硬错误/不可挽救样本",
            rescue_hint="图片无法读取，通常可直接剔除。",
        )

    q = calc_quality(frame)
    q_reasons = quality_reasons(q, args)

    expected_set = set(args.expected or [])
    forbid_set = set(args.forbid or [])

    detected_classes, _ = yolo_detect(yolo_model, frame, args.conf)
    detected_set = set(detected_classes)

    yolo_match_ratio = 1.0
    yolo_forbid_ratio = 0.0

    hard_reasons = []
    review_reasons = []
    fatal_reasons = []

    for r in q_reasons:
        if "黑屏比例过高" in r or "白屏比例过高" in r or "分辨率过低" in r:
            hard_reasons.append(r)
        else:
            review_reasons.append(r)

    if expected_set:
        if not (expected_set & detected_set):
            yolo_match_ratio = 0.0
            hard_reasons.append(f"目标严重不符:未检测到期望类别{sorted(expected_set)}")
        else:
            yolo_match_ratio = 1.0

    if forbid_set:
        hit = sorted(forbid_set & detected_set)
        if hit:
            yolo_forbid_ratio = 1.0
            hard_reasons.append(f"检测到禁止类别{hit}")

    clip_score = -1.0
    if clip_checker.enabled:
        clip_score = clip_checker.score_frame(frame)
        if clip_score < args.clip_bad_threshold:
            hard_reasons.append(f"CLIP文本匹配分过低:{clip_score:.2f}")
        elif clip_score < args.clip_review_threshold:
            review_reasons.append(f"CLIP文本匹配分偏低:{clip_score:.2f}")

    c_metric = copyright_checker.check_frame(frame)
    apply_copyright_decision(c_metric, args, hard_reasons, review_reasons)

    subject_present = False
    if expected_set:
        subject_present = yolo_match_ratio >= args.subject_protect_ratio
    else:
        subject_present = len(detected_set) > 0

    decision, score, reasons, bad_level, bad_category, rescue_hint = decide_from_reasons(
        hard_reasons,
        review_reasons,
        fatal_reasons=fatal_reasons,
        subject_present=subject_present,
        enable_bad_grading=not args.disable_bad_grading,
        protect_subject=not args.disable_subject_protection,
    )

    return FileResult(
        file=str(path),
        file_type="image",
        decision=decision,
        score=score,
        reasons=reasons,
        bad_level=bad_level,
        bad_category=bad_category,
        rescue_hint=rescue_hint,
        width=q.width,
        height=q.height,
        sampled_frames=1,
        quality_bad_ratio=1.0 if q_reasons else 0.0,
        yolo_match_ratio=yolo_match_ratio,
        yolo_forbid_ratio=yolo_forbid_ratio,
        yolo_detected_classes=",".join(sorted(detected_set)),
        clip_score=round(clip_score, 4) if clip_score >= 0 else -1,
        copyright_risk_ratio=1.0 if c_metric.risk_score >= args.copyright_review_score else 0.0,
        copyright_max_score=round(c_metric.risk_score, 3),
        copyright_reasons=c_metric.reasons,
        avg_brightness=round(q.brightness, 3),
        avg_blur=round(q.blur, 3),
        avg_contrast=round(q.contrast, 3),
        avg_black_ratio=round(q.black_ratio, 3),
        avg_white_ratio=round(q.white_ratio, 3),
    )


def analyze_video(
    path: Path,
    yolo_model,
    clip_checker: ClipChecker,
    copyright_checker: CopyrightChecker,
    args,
    preview_dir: Path,
) -> FileResult:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return FileResult(
            file=str(path),
            file_type="video",
            decision="BAD",
            score=0,
            reasons="视频打开失败或文件损坏",
            bad_level="HARD_BAD",
            bad_category="文件级硬错误/不可挽救样本",
            rescue_hint="视频无法打开，通常可直接剔除。",
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0
    sample_interval = max(int(fps / args.sample_fps), 1)

    sampled_frames = 0
    read_fail_frames = 0
    quality_bad_frames = 0
    freeze_frames = 0
    yolo_match_frames = 0
    yolo_forbid_frames = 0
    copyright_risk_frames = 0
    copyright_max_score = 0.0
    copyright_reason_list = []

    detected_all = set()
    q_values: List[QualityMetric] = []
    clip_scores: List[float] = []
    preview_frames: List[np.ndarray] = []

    expected_set = set(args.expected or [])
    forbid_set = set(args.forbid or [])

    prev_sampled_frame = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % sample_interval != 0:
            frame_idx += 1
            continue

        sampled_frames += 1

        if frame is None:
            read_fail_frames += 1
            frame_idx += 1
            continue

        if args.save_preview and len(preview_frames) < args.preview_frames:
            preview_frames.append(frame.copy())

        q = calc_quality(frame)
        q_values.append(q)

        if quality_reasons(q, args):
            quality_bad_frames += 1

        diff_score = frame_diff_score(prev_sampled_frame, frame)
        if prev_sampled_frame is not None and diff_score < args.freeze_diff_threshold:
            freeze_frames += 1
        prev_sampled_frame = frame.copy()

        detected_classes, _ = yolo_detect(yolo_model, frame, args.conf)
        detected_set = set(detected_classes)
        detected_all.update(detected_set)

        if expected_set:
            if expected_set & detected_set:
                yolo_match_frames += 1
        else:
            yolo_match_frames += 1

        if forbid_set and (forbid_set & detected_set):
            yolo_forbid_frames += 1

        if clip_checker.enabled:
            try:
                clip_scores.append(clip_checker.score_frame(frame))
            except Exception:
                pass

        c_metric = copyright_checker.check_frame(frame)
        if args.enable_copyright and c_metric.risk_score >= args.copyright_review_score:
            copyright_risk_frames += 1
            copyright_max_score = max(copyright_max_score, c_metric.risk_score)
            if c_metric.reasons:
                copyright_reason_list.append(c_metric.reasons)

        if sampled_frames >= args.max_frames:
            break

        frame_idx += 1

    cap.release()

    if sampled_frames == 0:
        return FileResult(
            file=str(path),
            file_type="video",
            decision="BAD",
            score=0,
            reasons="视频未采样到有效帧",
            bad_level="HARD_BAD",
            bad_category="文件级硬错误/不可挽救样本",
            rescue_hint="视频无有效帧，通常可直接剔除。",
            total_frames=total_frames,
            duration_sec=round(duration_sec, 3),
        )

    quality_bad_ratio = quality_bad_frames / sampled_frames
    freeze_ratio = freeze_frames / max(sampled_frames - 1, 1)
    read_fail_ratio = read_fail_frames / sampled_frames
    yolo_match_ratio = yolo_match_frames / sampled_frames
    yolo_forbid_ratio = yolo_forbid_frames / sampled_frames
    copyright_risk_ratio = copyright_risk_frames / sampled_frames

    avg = lambda arr: float(np.mean(arr)) if arr else 0.0

    avg_brightness = avg([q.brightness for q in q_values])
    avg_blur = avg([q.blur for q in q_values])
    avg_contrast = avg([q.contrast for q in q_values])
    avg_black_ratio = avg([q.black_ratio for q in q_values])
    avg_white_ratio = avg([q.white_ratio for q in q_values])

    hard_reasons = []
    review_reasons = []
    fatal_reasons = []

    if read_fail_ratio > args.max_read_fail_ratio:
        hard_reasons.append(f"坏帧/读取失败比例过高:{read_fail_ratio:.2f}")

    if quality_bad_ratio >= args.max_bad_quality_ratio:
        hard_reasons.append(f"低质帧比例过高:{quality_bad_ratio:.2f}")
    elif quality_bad_ratio >= args.review_bad_quality_ratio:
        review_reasons.append(f"低质帧比例偏高:{quality_bad_ratio:.2f}")

    if freeze_ratio >= args.max_freeze_ratio:
        hard_reasons.append(f"疑似冻结/重复帧比例过高:{freeze_ratio:.2f}")
    elif freeze_ratio >= args.review_freeze_ratio:
        review_reasons.append(f"疑似冻结/重复帧比例偏高:{freeze_ratio:.2f}")

    if expected_set:
        if yolo_match_ratio < args.min_match_ratio:
            hard_reasons.append(f"目标严重不符:期望{sorted(expected_set)},匹配帧比例{yolo_match_ratio:.2f}")
        elif yolo_match_ratio < args.review_match_ratio:
            review_reasons.append(f"目标匹配比例偏低:期望{sorted(expected_set)},匹配帧比例{yolo_match_ratio:.2f}")

    if forbid_set:
        if yolo_forbid_ratio >= args.forbid_bad_ratio:
            hard_reasons.append(f"禁止类别出现比例过高:{sorted(forbid_set)},比例{yolo_forbid_ratio:.2f}")
        elif yolo_forbid_ratio >= args.forbid_review_ratio:
            review_reasons.append(f"禁止类别偶发出现:{sorted(forbid_set)},比例{yolo_forbid_ratio:.2f}")

    clip_score = -1.0
    if clip_scores:
        clip_score = float(np.mean(clip_scores))
        if clip_score < args.clip_bad_threshold:
            hard_reasons.append(f"CLIP文本匹配分过低:{clip_score:.2f}")
        elif clip_score < args.clip_review_threshold:
            review_reasons.append(f"CLIP文本匹配分偏低:{clip_score:.2f}")

    copyright_reasons = "；".join(list(dict.fromkeys(copyright_reason_list))[:5])
    if args.enable_copyright and copyright_risk_ratio >= args.copyright_video_review_ratio:
        reason = (
            f"疑似版权/水印风险帧比例:{copyright_risk_ratio:.2f},"
            f"最高风险分:{copyright_max_score:.1f}"
        )
        if copyright_reasons:
            reason += f"({copyright_reasons})"

        if args.copyright_as_bad and (
            copyright_risk_ratio >= args.copyright_video_bad_ratio
            or copyright_max_score >= args.copyright_bad_score
        ):
            hard_reasons.append(reason)
        else:
            review_reasons.append(reason)

    subject_present = False
    if expected_set:
        subject_present = yolo_match_ratio >= args.subject_protect_ratio
    else:
        subject_present = len(detected_all) > 0

    decision, score, reasons, bad_level, bad_category, rescue_hint = decide_from_reasons(
        hard_reasons,
        review_reasons,
        fatal_reasons=fatal_reasons,
        subject_present=subject_present,
        enable_bad_grading=not args.disable_bad_grading,
        protect_subject=not args.disable_subject_protection,
    )

    if args.save_preview:
        preview_name = path.stem + "_preview.jpg"
        draw_preview(preview_frames, preview_dir / preview_name, f"{decision} | {path.name}")

    first_q = q_values[0] if q_values else QualityMetric()

    return FileResult(
        file=str(path),
        file_type="video",
        decision=decision,
        score=score,
        reasons=reasons,
        bad_level=bad_level,
        bad_category=bad_category,
        rescue_hint=rescue_hint,
        width=first_q.width,
        height=first_q.height,
        duration_sec=round(duration_sec, 3),
        total_frames=total_frames,
        sampled_frames=sampled_frames,
        quality_bad_ratio=round(quality_bad_ratio, 4),
        freeze_ratio=round(freeze_ratio, 4),
        read_fail_ratio=round(read_fail_ratio, 4),
        yolo_match_ratio=round(yolo_match_ratio, 4),
        yolo_forbid_ratio=round(yolo_forbid_ratio, 4),
        yolo_detected_classes=",".join(sorted(detected_all)),
        clip_score=round(clip_score, 4) if clip_score >= 0 else -1,
        copyright_risk_ratio=round(copyright_risk_ratio, 4),
        copyright_max_score=round(copyright_max_score, 3),
        copyright_reasons=copyright_reasons,
        avg_brightness=round(avg_brightness, 3),
        avg_blur=round(avg_blur, 3),
        avg_contrast=round(avg_contrast, 3),
        avg_black_ratio=round(avg_black_ratio, 3),
        avg_white_ratio=round(avg_white_ratio, 3),
    )


def copy_by_decision(results: List[FileResult], output_dir: Path, copy_mode: str) -> None:
    if copy_mode == "none":
        return

    if copy_mode == "bad":
        allowed = {"BAD"}
    elif copy_mode == "bad_review":
        allowed = {"BAD", "REVIEW"}
    else:
        allowed = {"GOOD", "BAD", "REVIEW"}

    for r in results:
        if r.decision not in allowed:
            continue

        src = Path(r.file)
        if not src.exists():
            continue

        dst_dir = output_dir / r.decision.lower()
        safe_mkdir(dst_dir)

        dst = dst_dir / src.name
        if dst.exists():
            dst = dst_dir / f"{src.stem}_copy{src.suffix}"

        shutil.copy2(src, dst)


def write_summary(results: List[FileResult], out_json: Path, args) -> None:
    total = len(results)
    good = sum(1 for r in results if r.decision == "GOOD")
    review = sum(1 for r in results if r.decision == "REVIEW")
    bad = sum(1 for r in results if r.decision == "BAD")
    hard_bad = sum(1 for r in results if r.bad_level == "HARD_BAD")
    soft_bad = sum(1 for r in results if r.bad_level == "SOFT_BAD")

    reason_counter: Dict[str, int] = {}
    for r in results:
        if not r.reasons:
            continue
        for reason in r.reasons.split("；"):
            key = reason.split(":")[0]
            reason_counter[key] = reason_counter.get(key, 0) + 1

    summary = {
        "total": total,
        "GOOD": good,
        "REVIEW": review,
        "BAD": bad,
        "HARD_BAD": hard_bad,
        "SOFT_BAD": soft_bad,
        "bad_rate": round(bad / total, 4) if total else 0,
        "review_rate": round(review / total, 4) if total else 0,
        "top_reasons": sorted(reason_counter.items(), key=lambda x: x[1], reverse=True),
        "args": vars(args),
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="视觉数据自动初筛工具：低质、严重不符、疑似版权/水印风险过滤"
    )

    parser.add_argument("--input", required=True, help="输入文件或文件夹")
    parser.add_argument("--output_dir", default="visual_filter_output", help="输出目录")

    # YOLO
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO 模型路径，如 yolov8n.pt 或 best.pt")
    parser.add_argument("--disable_yolo", action="store_true", help="禁用 YOLO，仅做画质/版权风险检测")
    parser.add_argument("--expected", nargs="*", default=[], help="期望出现的 YOLO 类别，如 person car 'cell phone'")
    parser.add_argument("--forbid", nargs="*", default=[], help="禁止出现的 YOLO 类别")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO 置信度阈值")

    # CLIP
    parser.add_argument("--enable_clip", action="store_true", help="启用 CLIP 文本匹配")
    parser.add_argument("--clip_text", default="", help="CLIP 对比文本，如 'a person is driving a car'")
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32", help="CLIP 模型名")
    parser.add_argument("--clip_bad_threshold", type=float, default=18.0, help="CLIP 严重不符阈值，越高越严格")
    parser.add_argument("--clip_review_threshold", type=float, default=21.0, help="CLIP 可疑阈值，越高越严格")

    # copyright
    parser.add_argument("--enable_copyright", action="store_true", help="启用疑似版权/水印风险初筛")
    parser.add_argument("--copyright_as_bad", action="store_true", help="将高版权风险直接判 BAD；默认只判 REVIEW")
    parser.add_argument("--copyright_review_score", type=float, default=35.0, help="图片版权风险 REVIEW 分数")
    parser.add_argument("--copyright_bad_score", type=float, default=70.0, help="图片版权风险 BAD 分数")
    parser.add_argument("--copyright_video_review_ratio", type=float, default=0.15, help="视频版权风险 REVIEW 帧比例")
    parser.add_argument("--copyright_video_bad_ratio", type=float, default=0.40, help="视频版权风险 BAD 帧比例")

    parser.add_argument("--corner_watermark_threshold", type=float, default=65.0, help="角落水印/Logo 风险阈值")
    parser.add_argument("--bottom_text_threshold", type=float, default=72.0, help="底部字幕/水印文字风险阈值")
    parser.add_argument("--center_text_threshold", type=float, default=78.0, help="中心大面积文字风险阈值")

    parser.add_argument("--template_dir", default="", help="Logo/水印模板图片目录，可选")
    parser.add_argument("--template_match_threshold", type=float, default=0.72, help="模板匹配阈值")

    parser.add_argument("--reference_dir", default="", help="已知版权图片参考库目录，可选")
    parser.add_argument("--reference_hash_threshold", type=int, default=6, help="感知哈希距离阈值，越小越严格")

    parser.add_argument("--enable_ocr", action="store_true", help="启用 OCR 关键词检测")
    parser.add_argument("--ocr_lang", default="chi_sim+eng", help="OCR 语言，如 chi_sim+eng")
    parser.add_argument("--tesseract_cmd", default="", help="Tesseract 程序路径，Windows 可填写")
    parser.add_argument("--copyright_keywords", nargs="*", default=[], help="额外版权/平台关键词")

    # BAD 分级与主体保护
    parser.add_argument("--disable_bad_grading", action="store_true", help="关闭 BAD 分级，退回普通 GOOD/REVIEW/BAD")
    parser.add_argument("--disable_subject_protection", action="store_true", help="关闭主体保护；检测到主体时也可能直接 BAD")
    parser.add_argument("--subject_protect_ratio", type=float, default=0.20, help="主体保护阈值：图片检测到目标或视频目标匹配帧比例达到该值时，质量/版权类硬问题优先降为 REVIEW")

    # 视频采样
    parser.add_argument("--sample_fps", type=float, default=1.0, help="视频每秒采样帧数")
    parser.add_argument("--max_frames", type=int, default=120, help="每个视频最多采样帧数")

    # 质量阈值
    parser.add_argument("--min_width", type=int, default=240, help="最小宽度")
    parser.add_argument("--min_height", type=int, default=240, help="最小高度")
    parser.add_argument("--min_blur", type=float, default=35.0, help="模糊阈值，越大越严格")
    parser.add_argument("--min_brightness", type=float, default=25.0, help="最低亮度")
    parser.add_argument("--max_brightness", type=float, default=235.0, help="最高亮度")
    parser.add_argument("--min_contrast", type=float, default=12.0, help="最低对比度")
    parser.add_argument("--max_black_ratio", type=float, default=0.85, help="黑屏像素比例阈值")
    parser.add_argument("--max_white_ratio", type=float, default=0.85, help="白屏像素比例阈值")

    # 视频比例阈值
    parser.add_argument("--review_bad_quality_ratio", type=float, default=0.25, help="低质帧 REVIEW 比例")
    parser.add_argument("--max_bad_quality_ratio", type=float, default=0.50, help="低质帧 BAD 比例")
    parser.add_argument("--review_freeze_ratio", type=float, default=0.35, help="冻结帧 REVIEW 比例")
    parser.add_argument("--max_freeze_ratio", type=float, default=0.65, help="冻结帧 BAD 比例")
    parser.add_argument("--freeze_diff_threshold", type=float, default=1.2, help="相邻采样帧差异小于该值视为冻结/重复")
    parser.add_argument("--max_read_fail_ratio", type=float, default=0.10, help="坏帧/读取失败 BAD 比例")

    # YOLO 匹配比例
    parser.add_argument("--min_match_ratio", type=float, default=0.20, help="期望类别 BAD 最低匹配帧比例")
    parser.add_argument("--review_match_ratio", type=float, default=0.35, help="期望类别 REVIEW 最低匹配帧比例")
    parser.add_argument("--forbid_review_ratio", type=float, default=0.05, help="禁止类别 REVIEW 出现比例")
    parser.add_argument("--forbid_bad_ratio", type=float, default=0.15, help="禁止类别 BAD 出现比例")

    # 输出
    parser.add_argument("--copy_mode", choices=["none", "bad", "bad_review", "all"], default="bad", help="复制哪些文件")
    parser.add_argument("--save_preview", action="store_true", help="保存视频抽帧预览图")
    parser.add_argument("--preview_frames", type=int, default=8, help="每个视频保存多少张预览帧")

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    safe_mkdir(output_dir)

    files = collect_files(input_path)
    if not files:
        print("未找到图片或视频文件。")
        return

    print(f"待处理文件数：{len(files)}")
    print(f"输出目录：{output_dir}")

    yolo_model = load_yolo_model(args)
    clip_checker = ClipChecker(args.enable_clip, args.clip_text, args.clip_model)
    copyright_checker = CopyrightChecker(args)

    preview_dir = output_dir / "previews"
    results: List[FileResult] = []

    for path in tqdm(files, desc="自动初筛"):
        try:
            if is_image(path):
                result = analyze_image(path, yolo_model, clip_checker, copyright_checker, args)
            elif is_video(path):
                result = analyze_video(path, yolo_model, clip_checker, copyright_checker, args, preview_dir)
            else:
                continue
        except Exception as e:
            result = FileResult(
                file=str(path),
                file_type="unknown",
                decision="BAD",
                score=0,
                reasons=f"处理异常:{e}",
                bad_level="HARD_BAD",
                bad_category="处理异常",
                rescue_hint="脚本处理异常，建议检查文件格式或单独复核。",
            )

        results.append(result)

    report_csv = output_dir / "visual_filter_report.csv"
    summary_json = output_dir / "visual_filter_summary.json"

    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(report_csv, index=False, encoding="utf-8-sig")

    write_summary(results, summary_json, args)
    copy_by_decision(results, output_dir, args.copy_mode)

    good = sum(1 for r in results if r.decision == "GOOD")
    review = sum(1 for r in results if r.decision == "REVIEW")
    bad = sum(1 for r in results if r.decision == "BAD")

    print("\n处理完成")
    print(f"GOOD  : {good}")
    print(f"REVIEW: {review}")
    hard_bad = sum(1 for r in results if r.bad_level == "HARD_BAD")
    soft_bad = sum(1 for r in results if r.bad_level == "SOFT_BAD")
    print(f"BAD   : {bad}")
    print(f"HARD_BAD: {hard_bad}")
    print(f"SOFT_BAD: {soft_bad}")
    print(f"CSV 报告 : {report_csv}")
    print(f"JSON 汇总: {summary_json}")

    if args.copy_mode != "none":
        print(f"样本复制目录: {output_dir}")


if __name__ == "__main__":
    main()
