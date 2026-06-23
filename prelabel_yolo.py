#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prelabel_yolo.py

YOLO 云端预标注脚本：
- 连接第一阶段筛选结果：默认只处理 good / review，不处理 bad
- 支持普通图片目录
- 支持可选视频抽帧后预标注
- 输出 YOLO txt 预标注框
- 输出 images / labels / previews / report.csv / data.yaml
- 结果用于 LabelImg / Label Studio 人工修正，不建议直接训练
"""

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from tqdm import tqdm

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("未安装 ultralytics，请先安装：pip install ultralytics") from e

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"}


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def is_video(p: Path) -> bool:
    return p.suffix.lower() in VIDEO_EXTS


def sanitize_name(rel: Path) -> str:
    parts = list(rel.parts)
    name = "__".join(parts)
    name = name.replace(":", "").replace("\\", "__").replace("/", "__")
    return name


def collect_sources(input_path: Path, from_decisions: List[str]) -> List[Path]:
    """
    如果 input_path 下有 good/review/bad 目录，默认只取 from_decisions。
    否则递归取 input_path 里的图片/视频。
    """
    decision_dirs = [d for d in ["good", "review", "bad"] if (input_path / d).exists()]
    roots = []

    if decision_dirs:
        for d in from_decisions:
            p = input_path / d
            if p.exists():
                roots.append(p)
    else:
        roots.append(input_path)

    files = []
    for root in roots:
        if root.is_file() and (is_image(root) or is_video(root)):
            files.append(root)
        elif root.is_dir():
            for p in root.rglob("*"):
                if p.is_file() and (is_image(p) or is_video(p)):
                    files.append(p)

    return sorted(files)


def yolo_line_from_xyxy(cls_id: int, xyxy, img_w: int, img_h: int, conf: Optional[float] = None, save_conf: bool = False) -> str:
    x1, y1, x2, y2 = map(float, xyxy)
    x1 = max(0.0, min(x1, img_w - 1))
    y1 = max(0.0, min(y1, img_h - 1))
    x2 = max(0.0, min(x2, img_w - 1))
    y2 = max(0.0, min(y2, img_h - 1))

    x_c = ((x1 + x2) / 2.0) / img_w
    y_c = ((y1 + y2) / 2.0) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h

    if save_conf and conf is not None:
        return f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f} {conf:.6f}"
    return f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}"


def draw_boxes(img, boxes, names: Dict[int, str]):
    out = img.copy()
    for cls_id, conf, xyxy in boxes:
        x1, y1, x2, y2 = map(int, xyxy)
        label = f"{names.get(cls_id, str(cls_id))} {conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, label, (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def write_data_yaml(out_dir: Path, names: Dict[int, str]):
    text = "path: .\n"
    text += "train: images\n"
    text += "val: images\n"
    text += "names:\n"
    for k in sorted(names.keys()):
        text += f"  {k}: {names[k]}\n"
    (out_dir / "data.yaml").write_text(text, encoding="utf-8")


def extract_video_frames(video_path: Path, frame_dir: Path, sample_fps: float, max_frames: int) -> List[Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    interval = max(int(fps / sample_fps), 1)
    frames = []
    idx = 0
    saved = 0
    safe_mkdir(frame_dir)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % interval == 0:
            out_name = f"{video_path.stem}_frame_{idx:06d}.jpg"
            out_path = frame_dir / out_name
            cv2.imwrite(str(out_path), frame)
            frames.append(out_path)
            saved += 1
            if saved >= max_frames:
                break
        idx += 1

    cap.release()
    return frames


def predict_one_image(model, img_path: Path, args, out_dirs, root_for_rel: Path, keep_ids: Optional[set], names: Dict[int, str]):
    images_dir, labels_dir, previews_dir, empty_dir = out_dirs

    img = cv2.imread(str(img_path))
    if img is None:
        return {
            "file": str(img_path),
            "status": "READ_FAIL",
            "num_boxes": 0,
            "classes": "",
            "label_file": "",
            "image_file": "",
            "preview_file": "",
            "reason": "图片读取失败",
        }

    h, w = img.shape[:2]

    try:
        rel = img_path.relative_to(root_for_rel)
    except Exception:
        rel = Path(img_path.name)

    safe_base = Path(sanitize_name(rel)).with_suffix("")
    out_img = images_dir / f"{safe_base}{img_path.suffix.lower()}"
    out_label = labels_dir / f"{safe_base}.txt"
    out_preview = previews_dir / f"{safe_base}.jpg"

    results = model.predict(
        source=str(img_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )

    boxes_for_draw = []
    lines = []
    cls_names = []

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if keep_ids is not None and cls_id not in keep_ids:
                continue

            xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
            line = yolo_line_from_xyxy(cls_id, xyxy, w, h, conf=conf, save_conf=args.save_conf)
            lines.append(line)
            boxes_for_draw.append((cls_id, conf, xyxy))
            cls_names.append(names.get(cls_id, str(cls_id)))

    if args.copy_images:
        safe_mkdir(out_img.parent)
        shutil.copy2(img_path, out_img)

    if args.write_empty_label or lines:
        out_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    if not lines and args.copy_empty_to_review:
        safe_mkdir(empty_dir)
        if args.copy_images:
            shutil.copy2(img_path, empty_dir / img_path.name)

    if args.save_preview:
        safe_mkdir(previews_dir)
        preview = draw_boxes(img, boxes_for_draw, names)
        cv2.imwrite(str(out_preview), preview)

    status = "PRELABELED" if lines else "NO_BOX"
    reason = "" if lines else "未检测到目标框，建议人工确认是否作为负样本或补标"

    return {
        "file": str(img_path),
        "status": status,
        "num_boxes": len(lines),
        "classes": ",".join(sorted(set(cls_names))),
        "label_file": str(out_label) if (args.write_empty_label or lines) else "",
        "image_file": str(out_img) if args.copy_images else "",
        "preview_file": str(out_preview) if args.save_preview else "",
        "reason": reason,
    }


def main():
    parser = argparse.ArgumentParser(description="YOLO 云端预标注脚本，连接 GOOD/REVIEW 筛选结果生成 labels")
    parser.add_argument("--input", required=True, help="输入目录：可以是第一阶段 cloud_output_xxx，也可以是普通图片/视频目录")
    parser.add_argument("--output_dir", default="prelabel_output", help="输出目录")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO 权重，如 yolov8n.pt 或 /model/best.pt")
    parser.add_argument("--from_decisions", nargs="*", default=["good", "review"], help="如果输入是筛选结果目录，默认只处理 good review")
    parser.add_argument("--keep_names", nargs="*", default=[], help="只保留这些类别名，如 face phone smoke；为空则保留全部检测类别")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    parser.add_argument("--device", default="", help="设备：空=自动，0=第一张GPU，cpu=CPU")
    parser.add_argument("--copy_images", action="store_true", default=True, help="复制图片到输出 images 目录")
    parser.add_argument("--no_copy_images", action="store_true", help="不复制图片，只输出 labels/report")
    parser.add_argument("--save_conf", action="store_true", help="txt 标签中保存置信度；正式 YOLO 训练通常不需要 conf")
    parser.add_argument("--write_empty_label", action="store_true", default=True, help="无检测框时也写空 txt，方便作为负样本")
    parser.add_argument("--copy_empty_to_review", action="store_true", help="无框样本额外复制到 empty_review")
    parser.add_argument("--save_preview", action="store_true", help="保存带框预览图")
    parser.add_argument("--extract_video_frames", action="store_true", help="遇到视频时先抽帧，再对帧预标注")
    parser.add_argument("--sample_fps", type=float, default=1.0, help="视频抽帧 FPS")
    parser.add_argument("--max_video_frames", type=int, default=120, help="每个视频最多抽帧数量")

    args = parser.parse_args()
    if args.no_copy_images:
        args.copy_images = False

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    previews_dir = out_dir / "previews"
    empty_dir = out_dir / "empty_review"
    frame_dir = out_dir / "_video_frames"

    for d in [out_dir, images_dir, labels_dir]:
        safe_mkdir(d)
    if args.save_preview:
        safe_mkdir(previews_dir)

    print("=" * 70)
    print("YOLO 云端预标注启动")
    print("input     :", input_path)
    print("output    :", out_dir)
    print("model     :", args.model)
    print("decisions :", args.from_decisions)
    print("keep_names:", args.keep_names if args.keep_names else "全部类别")
    print("=" * 70)

    model = YOLO(args.model)
    names = model.names
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    names = {int(k): str(v) for k, v in names.items()}

    name_to_id = {v: k for k, v in names.items()}
    keep_ids = None
    if args.keep_names:
        missing = [n for n in args.keep_names if n not in name_to_id]
        if missing:
            print("警告：模型不包含这些类别名，将被忽略：", missing)
        keep_ids = {name_to_id[n] for n in args.keep_names if n in name_to_id}
        if not keep_ids:
            print("错误：keep_names 没有任何类别能在模型中找到。")
            print("当前模型类别：", names)
            return

    sources = collect_sources(input_path, args.from_decisions)
    if not sources:
        print("未找到可处理的图片/视频。")
        return

    image_files = []
    skipped_videos = []
    for p in sources:
        if is_image(p):
            image_files.append(p)
        elif is_video(p):
            if args.extract_video_frames:
                frames = extract_video_frames(p, frame_dir / p.stem, args.sample_fps, args.max_video_frames)
                image_files.extend(frames)
            else:
                skipped_videos.append(str(p))

    if skipped_videos:
        print(f"提示：发现 {len(skipped_videos)} 个视频，但未启用 --extract_video_frames，已跳过视频。")

    if not image_files:
        print("没有可预标注的图片。")
        return

    records = []
    out_dirs = (images_dir, labels_dir, previews_dir, empty_dir)

    for img_path in tqdm(image_files, desc="预标注"):
        try:
            rec = predict_one_image(model, img_path, args, out_dirs, input_path, keep_ids, names)
        except Exception as e:
            rec = {
                "file": str(img_path),
                "status": "ERROR",
                "num_boxes": 0,
                "classes": "",
                "label_file": "",
                "image_file": "",
                "preview_file": "",
                "reason": f"处理异常:{e}",
            }
        records.append(rec)

    report_csv = out_dir / "prelabel_report.csv"
    with report_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "status", "num_boxes", "classes", "label_file", "image_file", "preview_file", "reason"])
        writer.writeheader()
        writer.writerows(records)

    if keep_ids is not None:
        out_names = {i: names[i] for i in sorted(keep_ids)}
    else:
        out_names = names
    write_data_yaml(out_dir, out_names)

    summary = {
        "input": str(input_path),
        "output_dir": str(out_dir),
        "model": args.model,
        "total_images": len(image_files),
        "prelabeled": sum(1 for r in records if r["status"] == "PRELABELED"),
        "no_box": sum(1 for r in records if r["status"] == "NO_BOX"),
        "errors": sum(1 for r in records if r["status"] in ["READ_FAIL", "ERROR"]),
        "skipped_videos": len(skipped_videos),
        "keep_names": args.keep_names,
        "model_names": out_names,
    }
    (out_dir / "prelabel_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("预标注完成")
    print("总图片数 :", summary["total_images"])
    print("已出框   :", summary["prelabeled"])
    print("无框     :", summary["no_box"])
    print("异常     :", summary["errors"])
    print("报告     :", report_csv)
    print("输出目录 :", out_dir)
    print("=" * 70)
    print("下一步：下载输出目录后，用 LabelImg / Label Studio 人工修正 labels。")


if __name__ == "__main__":
    main()
