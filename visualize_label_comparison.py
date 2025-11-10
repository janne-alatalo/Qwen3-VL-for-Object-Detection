#!/usr/bin/env python3
"""Create side-by-side visualizations of YOLO ground truth vs. model predictions."""

from __future__ import annotations

import argparse
import json
from json import JSONDecodeError
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from query_bbox import get_label_font, render_bounding_boxes, sanitize_detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create side-by-side images showing YOLO ground-truth boxes on the left and "
            "model predictions on the right."
        )
    )
    parser.add_argument("dataset_root", type=Path, help="Root directory of the dataset images.")
    parser.add_argument("results_path", type=Path, help="JSONL detections from batch_detect.py.")
    parser.add_argument("labels_root", type=Path, help="Root directory containing YOLO label files.")
    parser.add_argument("output_root", type=Path, help="Destination directory for visualizations.")
    parser.add_argument(
        "--label-suffix",
        default=".txt",
        help="Extension for YOLO label files (default: %(default)s).",
    )
    parser.add_argument(
        "--label-path-template",
        help=(
            "Optional format string used to locate label files. Placeholders: "
            "{rel_path}, {stem}, {name}, {parent}. Relative templates are resolved under labels_root."
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
        help="When 'skip', images without labels are skipped. 'empty' treats missing labels as zero boxes.",
    )
    parser.add_argument(
        "--class-names",
        type=Path,
        help="Optional file with YOLO class names (one per line).",
    )
    parser.add_argument(
        "--class-map",
        action="append",
        default=[],
        metavar="ID=NAME",
        help="Override or define class names for YOLO ids. Example: --class-map 0=Crack",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing comparison images (default: skip existing).",
    )
    return parser.parse_args()


def warn(message: str) -> None:
    print(f"Warning: {message}")


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
        raw_idx, label = spec.split("=", 1)
        idx = raw_idx.strip()
        label = label.strip()
        if not idx or not label:
            raise ValueError(f"Invalid --class-map entry '{spec}'. Use ID=NAME.")
        lookup[idx] = label
    return lookup


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

    mirror = Path(*rel_path.parts).with_suffix(args.label_suffix)
    candidates.append(labels_root / mirror)

    for prefix in args.strip_prefix:
        prefix_path = normalize_rel_path(prefix)
        prefix_len = len(prefix_path.parts)
        if rel_path.parts[:prefix_len] == prefix_path.parts:
            trimmed = PurePosixPath(*rel_path.parts[prefix_len:])
            target = labels_root / Path(*trimmed.parts) if trimmed.parts else labels_root / rel_path.name
            candidates.append(target.with_suffix(args.label_suffix))

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


def convert_yolo_box(
    parts: Sequence[str],
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    if len(parts) < 4:
        return None
    try:
        x_center = float(parts[0]) * width
        y_center = float(parts[1]) * height
        w = float(parts[2]) * width
        h = float(parts[3]) * height
    except ValueError:
        return None
    x1 = max(0.0, x_center - w / 2)
    y1 = max(0.0, y_center - h / 2)
    x2 = min(width, x_center + w / 2)
    y2 = min(height, y_center + h / 2)
    if x2 <= x1 or y2 <= y1:
        return None
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def load_yolo_boxes(
    label_path: Path,
    image_size: Tuple[int, int],
    class_lookup: Dict[str, str],
) -> Optional[List[Dict[str, Any]]]:
    width, height = image_size
    boxes: List[Dict[str, Any]] = []
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
        class_id = parts[0]
        box = convert_yolo_box(parts[1:5], width, height)
        if not box:
            warn(f"Degenerate bbox in '{label_path}' line {line_number}: {line.strip()}")
            continue
        label = class_lookup.get(class_id, class_id)
        boxes.append({"bbox": box, "label": label})
    return boxes


def load_results(results_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    detections: Dict[str, List[Dict[str, Any]]] = {}
    with results_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
                image_key = record["image"]
                dets = record.get("detections", [])
                sanitized = sanitize_detections(dets)
                detections[image_key] = sanitized if sanitized else dets
            except (JSONDecodeError, KeyError) as exc:
                warn(f"Skipping malformed record at line {line_number}: {exc}")
    return detections


def draw_ground_truth(image: Image.Image, boxes: Sequence[Dict[str, Any]]) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    font = get_label_font(18)
    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        label = box.get("label", "")
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        if label:
            text_pos = (x1, max(0, y1 - 12))
            draw.text(text_pos, label, fill="green", font=font)
    return output


def build_output_path(output_root: Path, rel_path: Path) -> Path:
    rel_parent = rel_path.parent
    base_name = rel_path.stem + "_gt_vs_pred"
    extension = rel_path.suffix or ".png"
    target_dir = output_root if rel_parent == Path(".") else output_root / rel_parent
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{base_name}{extension}"


def create_side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    width, height = left.size
    right = right.resize((width, height))
    combined = Image.new("RGB", (width * 2, height))
    combined.paste(left, (0, 0))
    combined.paste(right, (width, 0))
    return combined


def process_images(
    dataset_root: Path,
    labels_root: Path,
    output_root: Path,
    results: Dict[str, List[Dict[str, Any]]],
    args: argparse.Namespace,
) -> Tuple[int, int]:
    processed = 0
    skipped = 0
    class_lookup = load_class_lookup(args.class_names, args.class_map)

    for rel_path_str, detections in results.items():
        rel_path = Path(rel_path_str)
        source_image = dataset_root / rel_path
        if not source_image.is_file():
            warn(f"Skipping missing image {rel_path_str}")
            skipped += 1
            continue

        output_path = build_output_path(output_root, rel_path)
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            original = Image.open(source_image).convert("RGB")
        except OSError as exc:
            warn(f"Skipping unreadable image '{source_image}': {exc}")
            skipped += 1
            continue

        rel_posix = normalize_rel_path(rel_path_str)
        label_path = resolve_label_path(rel_posix, labels_root, args)
        if label_path is None:
            if args.missing_label_policy == "skip":
                warn(f"No label file found for {rel_path_str}; skipping.")
                skipped += 1
                continue
            ground_truths: List[Dict[str, Any]] = []
        else:
            ground_truths = load_yolo_boxes(label_path, original.size, class_lookup) or []

        left = draw_ground_truth(original, ground_truths)
        right = render_bounding_boxes(original, detections)
        combined = create_side_by_side(left, right)
        combined.save(output_path)
        processed += 1
        print(f"Wrote {output_path}")

    return processed, skipped


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    labels_root = args.labels_root.resolve()
    results_path = args.results_path.resolve()
    output_root = args.output_root.resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not labels_root.is_dir():
        raise FileNotFoundError(f"Labels root not found: {labels_root}")
    if not results_path.is_file():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    output_root.mkdir(parents=True, exist_ok=True)

    results = load_results(results_path)
    if not results:
        print("No detections found in the results file.")
        return

    processed, skipped = process_images(
        dataset_root,
        labels_root,
        output_root,
        results,
        args,
    )
    print(f"Visualization complete. Processed: {processed}, skipped: {skipped}")


if __name__ == "__main__":
    main()
