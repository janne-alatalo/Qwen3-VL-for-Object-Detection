#!/usr/bin/env python3
"""Iteratively refine object detections by asking the model to review its own boxes."""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from query_bbox import (
    DetectionError,
    SYSTEM_PROMPT,
    build_payload,
    encode_image,
    extract_detections,
    get_label_font,
    normalize_bbox,
    request_completion,
    sanitize_detections,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Query the detector, then iteratively ask it to correct/confirm its own bounding boxes."
        )
    )
    parser.add_argument("image_path", type=Path, help="Path to the image file.")
    parser.add_argument("prompt", help="Detection prompt for the initial call.")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum correction rounds to run after the initial prediction (default: %(default)s).",
    )
    parser.add_argument(
        "--api-base",
        default="http://10.88.0.1:8000/v1",
        help="OpenAI-compatible base URL (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        default="qwen3-VL",
        help="Model name to request from the server (default: %(default)s).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for all rounds (default: %(default)s).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2000,
        help="Maximum response tokens to request (default: %(default)s).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to store per-iteration visualizations. Defaults to <image>_iterations next to the image.",
    )
    return parser.parse_args()


def warn(message: str) -> None:
    print(f"Warning: {message}")


def encode_pil_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def ensure_output_dir(image_path: Path, requested: Optional[Path]) -> Path:
    if requested:
        requested.mkdir(parents=True, exist_ok=True)
        return requested
    default_dir = image_path.parent / f"{image_path.stem}_iterations"
    default_dir.mkdir(parents=True, exist_ok=True)
    return default_dir


def assign_detection_ids(detections: List[Dict[str, Any]]) -> None:
    """Ensure every detection has an integer id, adding new ones sequentially."""
    next_id = 1
    seen: set[int] = set()
    for det in detections:
        det_id = det.get("id")
        if isinstance(det_id, int):
            seen.add(det_id)
            next_id = max(next_id, det_id + 1)

    for det in detections:
        det_id = det.get("id")
        if not isinstance(det_id, int):
            while next_id in seen:
                next_id += 1
            det["id"] = next_id
            seen.add(next_id)
            next_id += 1


def detections_to_prompt_json(detections: List[Dict[str, Any]]) -> str:
    prompt_entries = []
    for det in sorted(detections, key=lambda d: d.get("id", 0)):
        entry = {
            "id": det.get("id"),
            "label": det.get("label", ""),
            "bbox": det.get("bbox_2d", []),
        }
        prompt_entries.append(entry)
    return json.dumps(prompt_entries, ensure_ascii=False, separators=(",", ":"))


def canonical_signature(detections: List[Dict[str, Any]]) -> List[Tuple[int, str, Tuple[int, int, int, int]]]:
    signature: List[Tuple[int, str, Tuple[int, int, int, int]]] = []
    for det in detections:
        det_id = det.get("id") or 0
        label = det.get("label", "")
        bbox = tuple(det.get("bbox_2d", ()))
        signature.append((int(det_id), label, tuple(int(coord) for coord in bbox)))
    signature.sort()
    return signature


def render_with_ids(image: Image.Image, detections: Sequence[Dict[str, Any]]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = get_label_font(18)
    width, height = annotated.size

    for det in detections:
        bbox = det.get("bbox_2d")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = normalize_bbox(bbox, width, height)
        except ValueError:
            continue
        label = det.get("label", "")
        det_id = det.get("id")
        descriptor = f"#{det_id} {label}".strip()
        draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)
        text_pos = (x1, max(0, y1 - 18))
        draw.text(text_pos, descriptor, fill="yellow", font=font)
    return annotated


def save_iteration_image(output_dir: Path, image_stem: str, iteration: int, image: Image.Image) -> Path:
    path = output_dir / f"{image_stem}_iter_{iteration:02d}.png"
    image.save(path)
    return path


def initial_detection(
    args: argparse.Namespace,
    image_data: str,
) -> Tuple[List[Dict[str, Any]], str]:
    payload = build_payload(
        args.prompt,
        image_data,
        args.model,
        args.temperature,
        args.max_tokens,
    )
    body = request_completion(args.api_base, payload, args.timeout)
    detections, raw_text, raw_body = extract_detections(body)
    detections = list(detections)
    sanitized = sanitize_detections(detections)
    if detections and not sanitized:
        raise DetectionError(
            "Model produced detections but none were usable; inspect stderr warnings."
        )
    usable = sanitized if sanitized else detections
    assign_detection_ids(usable)
    return usable, json.dumps(usable, ensure_ascii=False, indent=2)


def build_refinement_messages(
    prompt: str,
    original_image_data: str,
    previous_json: str,
    overlay_image_data: str,
) -> List[Dict[str, Any]]:
    instruction_text = (
        "This is your predicted bbox, which is visualized on the original image. "
        f"Your predictions were:\n{previous_json}\n"
        "Are you happy with the prediction? If you are happy, repeat the JSON above. "
        "If you want to fix some of the bboxes, do so now. Remember to add also the other bboxes "
        "to the prediction that you do not want to change exactly as they were! If you remove them, "
        "they are removed from the prediction. You can also add new predictions. "
        "If you add new predictions, just increment the id by every new prediction."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "detail": "auto", "image_url": original_image_data},
            ],
        },
        {"role": "assistant", "content": [{"type": "output_text", "text": previous_json}]},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": instruction_text},
                {"type": "input_image", "detail": "auto", "image_url": overlay_image_data},
            ],
        },
    ]


def request_refinement(
    args: argparse.Namespace,
    prompt: str,
    original_image_data: str,
    current_detections: List[Dict[str, Any]],
    overlay_image_data: str,
) -> List[Dict[str, Any]]:
    previous_json = detections_to_prompt_json(current_detections)
    messages = build_refinement_messages(prompt, original_image_data, previous_json, overlay_image_data)
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    body = request_completion(args.api_base, payload, args.timeout)
    detections, raw_text, raw_body = extract_detections(body)
    detections = list(detections)
    sanitized = sanitize_detections(detections)
    if detections and not sanitized:
        raise DetectionError(
            "Model produced detections but none were usable; inspect stderr warnings."
        )
    usable = sanitized if sanitized else detections
    assign_detection_ids(usable)
    return usable


def iterative_refinement(args: argparse.Namespace) -> None:
    if not args.image_path.is_file():
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    original_image = Image.open(args.image_path).convert("RGB")
    original_image_data = encode_image(args.image_path)
    output_dir = ensure_output_dir(args.image_path, args.output_dir)
    image_stem = args.image_path.stem

    detections, pretty_json = initial_detection(args, original_image_data)
    signature = canonical_signature(detections)

    print("Initial detections:")
    print(pretty_json)

    overlay = render_with_ids(original_image, detections)
    saved_path = save_iteration_image(output_dir, image_stem, 0, overlay)
    print(f"Iteration 0 visualization saved to {saved_path}")

    for iteration in range(1, args.max_iterations + 1):
        overlay_data = encode_pil_image(overlay)
        refined = request_refinement(
            args, args.prompt, original_image_data, detections, overlay_data
        )
        new_signature = canonical_signature(refined)
        if new_signature == signature:
            print(f"No changes detected on iteration {iteration}; stopping.")
            break

        detections = refined
        signature = new_signature
        overlay = render_with_ids(original_image, detections)
        saved_path = save_iteration_image(output_dir, image_stem, iteration, overlay)
        print(f"Iteration {iteration} visualization saved to {saved_path}")

    print("Final detections:")
    print(json.dumps(detections, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    iterative_refinement(args)


if __name__ == "__main__":
    main()
