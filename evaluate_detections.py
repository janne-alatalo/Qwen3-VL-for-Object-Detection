#!/usr/bin/env python3
"""Evaluate detection/classification quality against YOLO-style ground truth boxes."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, UnidentifiedImageError

from query_bbox import normalize_bbox


EMPTY_LABEL_KEY = "<none>"


def warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare detections.jsonl outputs against YOLO ground-truth boxes. "
            "Computes per-class precision/recall/F1 plus IoU statistics."
        )
    )
    parser.add_argument("results_path", type=Path, help="JSONL detections from batch_detect.py.")
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Root directory that the detections' relative image paths are based on.",
    )
    parser.add_argument(
        "labels_root",
        type=Path,
        help="Root directory containing YOLO label files (mirrors dataset structure or flat).",
    )
    parser.add_argument(
        "--label-suffix",
        default=".txt",
        help="Extension for YOLO label files (default: %(default)s).",
    )
    parser.add_argument(
        "--label-path-template",
        help=(
            "Optional format string to resolve label paths (before suffix replacement). "
            "Placeholders: {rel_path}, {stem}, {name}, {parent}. "
            "Relative templates are interpreted under labels_root."
        ),
    )
    parser.add_argument(
        "--strip-prefix",
        action="append",
        default=[],
        metavar="PREFIX",
        help="Strip this leading path (relative) before mirroring under labels_root. Repeatable.",
    )
    parser.add_argument(
        "--missing-label-policy",
        choices=["skip", "empty"],
        default="skip",
        help="If 'empty', treat missing label files as images with zero objects (default: skip).",
    )
    parser.add_argument(
        "--class-names",
        type=Path,
        help="Optional file listing class names line-by-line (YOLO-style).",
    )
    parser.add_argument(
        "--class-map",
        action="append",
        default=[],
        metavar="ID=NAME",
        help="Override or supply class name for a YOLO ID. Example: --class-map 2=knot_with_crack",
    )
    parser.add_argument(
        "--ignore-case",
        action="store_true",
        help="Compare labels case-insensitively (default: case-sensitive).",
    )
    parser.add_argument(
        "--lowercase-labels",
        action="store_true",
        help="Convert all ground-truth and predicted labels to lowercase before evaluation.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for a localization match (default: %(default)s).",
    )
    return parser.parse_args()


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


@dataclass
class Annotation:
    box: Tuple[float, float, float, float]
    label: str
    key: str


@dataclass
class ClassStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    iou_sum: float = 0.0
    iou_count: int = 0


class LabelRegistry:
    def __init__(self, ignore_case: bool, lowercase_labels: bool) -> None:
        self.ignore_case = ignore_case
        self.lowercase_labels = lowercase_labels
        self.display: Dict[str, str] = {}

    def canonical(self, raw_label: str) -> str:
        raw = (raw_label or "").strip()
        if self.lowercase_labels:
            raw = raw.lower()
        if not raw:
            key = EMPTY_LABEL_KEY
            if key not in self.display:
                self.display[key] = "<none>"
            return key

        key = raw.casefold() if self.ignore_case else raw
        self.display.setdefault(key, raw)
        return key

    def display_name(self, key: str) -> str:
        return self.display.get(key, key)


def normalize_rel_path(rel_path: str) -> PurePosixPath:
    return PurePosixPath(rel_path.replace("\\", "/"))


def load_class_lookup(names_path: Optional[Path], overrides: Sequence[str]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    if names_path:
        if not names_path.is_file():
            raise FileNotFoundError(f"Class names file not found: {names_path}")
        with names_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                name = line.strip()
                if not name:
                    continue
                lookup[str(idx)] = name

    for spec in overrides:
        if "=" not in spec:
            raise ValueError(f"Invalid --class-map entry '{spec}'. Use ID=NAME.")
        idx_part, label = spec.split("=", 1)
        idx = idx_part.strip()
        label = label.strip()
        if not idx or not label:
            raise ValueError(f"Invalid --class-map entry '{spec}'. Use ID=NAME.")
        lookup[idx] = label
    return lookup


def get_image_size(image_path: Path, cache: Dict[Path, Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if image_path in cache:
        return cache[image_path]
    try:
        with Image.open(image_path) as img:
            cache[image_path] = img.size
            return img.size
    except (FileNotFoundError, UnidentifiedImageError) as exc:
        warn(f"Could not open image '{image_path}': {exc}")
        return None


def compute_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    return inter_area / denom if denom > 0 else 0.0


def validate_box(box: Sequence[float]) -> Optional[Tuple[float, float, float, float]]:
    if len(box) != 4:
        return None
    x1, y1, x2, y2 = box
    x1, x2 = sorted((float(x1), float(x2)))
    y1, y2 = sorted((float(y1), float(y2)))
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return x1, y1, x2, y2


def resolve_label_path(
    rel_path: PurePosixPath,
    labels_root: Path,
    args: argparse.Namespace,
) -> Optional[Path]:
    candidates: List[Path] = []

    def append_candidate(raw_path: str | Path) -> None:
        path = Path(raw_path)
        if not path.is_absolute():
            path = labels_root / path
        path = path.with_suffix(args.label_suffix)
        if path not in candidates:
            candidates.append(path)

    if args.label_path_template:
        parent = "" if rel_path.parent == PurePosixPath(".") else rel_path.parent.as_posix()
        rendered = args.label_path_template.format(
            rel_path=rel_path.as_posix(),
            stem=rel_path.stem,
            name=rel_path.name,
            parent=parent,
        )
        append_candidate(rendered)

    mirror = Path(*rel_path.parts)
    mirror = mirror.with_suffix(args.label_suffix)
    candidates.append(labels_root / mirror)

    for prefix in args.strip_prefix:
        prefix_path = normalize_rel_path(prefix)
        prefix_len = len(prefix_path.parts)
        if rel_path.parts[:prefix_len] == prefix_path.parts:
            trimmed = PurePosixPath(*rel_path.parts[prefix_len:])
            if trimmed.parts:
                candidate = labels_root / Path(*trimmed.parts)
            else:
                candidate = labels_root / Path(rel_path.name)
            candidates.append(candidate.with_suffix(args.label_suffix))

    filename_only = labels_root / f"{rel_path.stem}{args.label_suffix}"
    candidates.append(filename_only)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def load_yolo_boxes(
    label_path: Path,
    image_size: Tuple[int, int],
    class_lookup: Dict[str, str],
    registry: LabelRegistry,
) -> Optional[List[Annotation]]:
    width, height = image_size
    boxes: List[Annotation] = []
    try:
        with label_path.open("r", encoding="utf-8") as handle:
            lines = list(handle)
    except FileNotFoundError:
        return None

    for line_number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 5:
            warn(f"Skipping malformed label in '{label_path}' line {line_number}: {line.strip()}")
            continue
        class_token = parts[0]
        coords = parts[1:5]
        try:
            class_index = int(float(class_token))
        except ValueError:
            class_index = class_token

        try:
            x_center = float(coords[0]) * width
            y_center = float(coords[1]) * height
            w = float(coords[2]) * width
            h = float(coords[3]) * height
        except ValueError:
            warn(f"Non-numeric bbox values in '{label_path}' line {line_number}: {line.strip()}")
            continue

        x1 = max(0.0, x_center - w / 2)
        y1 = max(0.0, y_center - h / 2)
        x2 = min(width, x_center + w / 2)
        y2 = min(height, y_center + h / 2)
        box = validate_box((x1, y1, x2, y2))
        if not box:
            warn(f"Degenerate bbox in '{label_path}' line {line_number}: {line.strip()}")
            continue

        key = str(class_index)
        label = class_lookup.get(key, key)
        canonical = registry.canonical(label)
        boxes.append(Annotation(box=box, label=label, key=canonical))
    return boxes


def load_predictions(
    detections: Sequence[Dict[str, object]],
    image_size: Tuple[int, int],
    registry: LabelRegistry,
) -> List[Annotation]:
    width, height = image_size
    results: List[Annotation] = []
    for index, detection in enumerate(detections):
        if not isinstance(detection, dict):
            warn(f"Ignoring detection #{index} (not an object).")
            continue
        bbox = detection.get("bbox_2d")
        label_raw = detection.get("label", "")
        if bbox is None:
            warn(f"Ignoring detection #{index} (missing bbox_2d).")
            continue
        try:
            scaled = normalize_bbox(bbox, width, height)
        except Exception as exc:  # noqa: BLE001 - surface parsing issue
            warn(f"Ignoring detection #{index} due to invalid bbox: {exc}")
            continue
        box = validate_box(scaled)
        if not box:
            warn(f"Ignoring detection #{index} due to zero-area bbox: {bbox}")
            continue
        label = str(label_raw).strip()
        canonical = registry.canonical(label)
        results.append(Annotation(box=box, label=label or "<none>", key=canonical))
    return results


def match_boxes(
    preds: Sequence[Annotation],
    gts: Sequence[Annotation],
    iou_threshold: float,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    candidates: List[Tuple[float, int, int]] = []
    for p_idx, pred in enumerate(preds):
        for g_idx, gt in enumerate(gts):
            iou = compute_iou(pred.box, gt.box)
            if iou >= iou_threshold:
                candidates.append((iou, p_idx, g_idx))

    candidates.sort(reverse=True)
    matched_preds: set[int] = set()
    matched_gts: set[int] = set()
    matches: List[Tuple[int, int, float]] = []

    for iou, p_idx, g_idx in candidates:
        if p_idx in matched_preds or g_idx in matched_gts:
            continue
        matched_preds.add(p_idx)
        matched_gts.add(g_idx)
        matches.append((p_idx, g_idx, iou))

    unmatched_preds = [idx for idx in range(len(preds)) if idx not in matched_preds]
    unmatched_gts = [idx for idx in range(len(gts)) if idx not in matched_gts]
    return matches, unmatched_preds, unmatched_gts


def evaluate(args: argparse.Namespace) -> None:
    if not args.results_path.is_file():
        raise FileNotFoundError(f"Results file not found: {args.results_path}")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")
    if not args.labels_root.is_dir():
        raise FileNotFoundError(f"Labels root not found: {args.labels_root}")

    class_lookup = load_class_lookup(args.class_names, args.class_map)
    registry = LabelRegistry(ignore_case=args.ignore_case, lowercase_labels=args.lowercase_labels)
    for label in class_lookup.values():
        registry.canonical(label)

    image_cache: Dict[Path, Tuple[int, int]] = {}
    class_stats: defaultdict[str, ClassStats] = defaultdict(ClassStats)
    label_confusions: Counter[Tuple[str, str]] = Counter()
    skip_reasons: Counter[str] = Counter()

    total_images = 0
    used_images = 0
    total_preds = 0
    total_gts = 0
    localization_iou_sum = 0.0
    localization_iou_count = 0
    correct_label_iou_sum = 0.0
    correct_label_iou_count = 0
    correct_label_matches = 0
    total_matches = 0

    with args.results_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                warn(f"Skipping malformed JSON at line {line_number}: {exc}")
                continue

            rel_path_raw = record.get("image")
            detections = record.get("detections", [])
            if not isinstance(rel_path_raw, str):
                warn(f"Skipping record at line {line_number}: missing 'image' path.")
                continue
            if not isinstance(detections, list):
                warn(f"Skipping record for '{rel_path_raw}': detections is not a list.")
                continue

            total_images += 1
            rel_path = normalize_rel_path(rel_path_raw)
            image_path = args.dataset_root / Path(*rel_path.parts)
            image_size = get_image_size(image_path, image_cache)
            if not image_size:
                skip_reasons["missing_image"] += 1
                continue

            label_path = resolve_label_path(rel_path, args.labels_root, args)
            if label_path is None:
                if args.missing_label_policy == "skip":
                    skip_reasons["missing_label"] += 1
                    continue
                ground_truths: List[Annotation] = []
            else:
                ground_truths = load_yolo_boxes(label_path, image_size, class_lookup, registry)
                if ground_truths is None:
                    if args.missing_label_policy == "skip":
                        skip_reasons["missing_label"] += 1
                        continue
                    ground_truths = []

            predictions = load_predictions(detections, image_size, registry)
            total_preds += len(predictions)
            total_gts += len(ground_truths)

            matches, unmatched_preds, unmatched_gts = match_boxes(
                predictions, ground_truths, args.iou_threshold
            )

            used_images += 1
            total_matches += len(matches)

            for p_idx, g_idx, iou in matches:
                pred = predictions[p_idx]
                gt = ground_truths[g_idx]
                localization_iou_sum += iou
                localization_iou_count += 1
                if pred.key == gt.key:
                    stats = class_stats[pred.key]
                    stats.tp += 1
                    stats.iou_sum += iou
                    stats.iou_count += 1
                    correct_label_matches += 1
                    correct_label_iou_sum += iou
                    correct_label_iou_count += 1
                else:
                    label_confusions[(gt.key, pred.key)] += 1
                    class_stats[gt.key].fn += 1
                    class_stats[pred.key].fp += 1

            for idx in unmatched_preds:
                class_stats[predictions[idx].key].fp += 1
            for idx in unmatched_gts:
                class_stats[ground_truths[idx].key].fn += 1

    if used_images == 0:
        print("No overlapping samples between detections and labels were evaluated.")
        if skip_reasons:
            for reason, count in skip_reasons.items():
                print(f"Skipped ({reason}): {count}")
        return

    print(f"Images processed: {total_images}")
    print(f"Images evaluated: {used_images}")
    if skip_reasons:
        for reason, count in sorted(skip_reasons.items()):
            print(f"Skipped ({reason}): {count}")

    print(f"\nIoU threshold: {args.iou_threshold:.2f}")
    print(f"Ground-truth boxes: {total_gts}")
    print(f"Predicted boxes : {total_preds}")
    print(f"Matches (IoU >= threshold): {total_matches}")

    mean_iou_all = safe_divide(localization_iou_sum, localization_iou_count)
    mean_iou_correct = safe_divide(correct_label_iou_sum, correct_label_iou_count)
    print(f"Mean IoU (all matches): {mean_iou_all:.4f}")
    print(f"Mean IoU (label-correct matches): {mean_iou_correct:.4f}")

    label_accuracy = safe_divide(correct_label_matches, total_matches)
    print(f"Label accuracy (within matched boxes): {label_accuracy:.4f}")

    micro_tp = sum(stat.tp for stat in class_stats.values())
    micro_fp = sum(stat.fp for stat in class_stats.values())
    micro_fn = sum(stat.fn for stat in class_stats.values())
    micro_precision = safe_divide(micro_tp, micro_tp + micro_fp)
    micro_recall = safe_divide(micro_tp, micro_tp + micro_fn)
    micro_f1 = safe_divide(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    print("\nPer-class metrics:")
    header = f"{'Class':<22} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Mean IoU':>9}"
    print(header)
    print("-" * len(header))
    for key in sorted(class_stats.keys()):
        stats = class_stats[key]
        precision = safe_divide(stats.tp, stats.tp + stats.fp)
        recall = safe_divide(stats.tp, stats.tp + stats.fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        mean_iou = safe_divide(stats.iou_sum, stats.iou_count)
        label = registry.display_name(key)
        print(
            f"{label:<22} {stats.tp:>5} {stats.fp:>5} {stats.fn:>5} "
            f"{precision:>8.3f} {recall:>8.3f} {f1:>8.3f} {mean_iou:>9.3f}"
        )

    print("\nMicro-averaged metrics:")
    print(f"Precision: {micro_precision:.4f}")
    print(f"Recall   : {micro_recall:.4f}")
    print(f"F1       : {micro_f1:.4f}")

    if label_confusions:
        print("\nLabel confusions (IoU ok, label mismatch):")
        for (gt_key, pred_key), count in label_confusions.most_common():
            gt = registry.display_name(gt_key)
            pred = registry.display_name(pred_key)
            print(f"Actual {gt} predicted as {pred}: {count}")


def main() -> None:
    try:
        args = parse_args()
        evaluate(args)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
