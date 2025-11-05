#!/usr/bin/env python3
import argparse
import base64
import json
import mimetypes
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests
from PIL import Image, ImageDraw, ImageFont


SYSTEM_PROMPT = (
    "You are an assistant that performs object detection. "
    "Return only a JSON array where each item is an object with keys "
    '`"bbox_2d"` and `"label"`. `"bbox_2d"` must be a list of four integers '
    "representing x1, y1, x2, y2 coordinates within [0, 1000]. "
    "Do not include any explanations, code fences, or additional text—return the raw JSON only."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit an image and prompt to a qwen3-VL model via an OpenAI-compatible API."
    )
    parser.add_argument("image_path", type=Path, help="Path to the image file to analyze.")
    parser.add_argument("prompt", help="User prompt describing the detection task.")
    parser.add_argument(
        "--api-base",
        default="http://10.88.0.1:8000/v1",
        help="Base URL for the OpenAI-compatible endpoint (default: %(default)s).",
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
        help="Sampling temperature; defaults to 0 for deterministic outputs.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum response tokens to request from the model.",
    )
    return parser.parse_args()


def encode_image(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if not mime_type:
        raise ValueError(f"Unable to determine MIME type for {image_path}")

    with image_path.open("rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def build_payload(
    prompt: str, image_data: str, model: str, temperature: float, max_tokens: int
) -> Dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "detail": "auto", "image_url": image_data},
                ],
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def extract_text_content(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, Iterable):
        fragments: List[str] = []
        for item in message_content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    fragments.append(text)
        return "".join(fragments)
    raise ValueError("Unexpected message content format.")


def parse_detection_json(raw_text: str) -> List[Dict[str, Any]]:
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        # Some models wrap JSON in a code fence; attempt to strip it.
        stripped = stripped.strip("`")
        if "\n" in stripped:
            stripped = "\n".join(
                line for line in stripped.splitlines() if not line.strip().startswith("json")
            )
    return json.loads(stripped)


def normalize_bbox(
    bbox: List[float], image_width: int, image_height: int
) -> List[int]:
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox of length 4, got {bbox}")

    def clamp(value: float) -> float:
        return max(0.0, min(1000.0, value))

    x1, y1, x2, y2 = (clamp(coord) for coord in bbox)
    scale_x = image_width / 1000.0
    scale_y = image_height / 1000.0
    return [
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
        int(round(x2 * scale_x)),
        int(round(y2 * scale_y)),
    ]


def draw_bounding_boxes(
    image_path: Path, detections: List[Dict[str, Any]]
) -> Path:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size

    for detection in detections:
        bbox = detection.get("bbox_2d")
        label = detection.get("label", "")
        if not isinstance(bbox, list):
            continue

        try:
            x1, y1, x2, y2 = normalize_bbox(bbox, width, height)
        except ValueError:
            continue

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        if label:
            text_pos = (x1, max(0, y1 - 12))
            draw.text(text_pos, label, fill="red", font=font)

    output_path = image_path.with_stem(f"{image_path.stem}_bbox")
    image.save(output_path)
    return output_path


def open_image_viewer(image_path: Path) -> None:
    try:
        if sys.platform.startswith("darwin"):
            subprocess.run(["open", str(image_path)], check=False)
        elif os.name == "nt":
            os.startfile(image_path)  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(image_path)], check=False)
    except Exception as exc:
        print(f"Failed to open image viewer automatically: {exc}")


def main() -> None:
    args = parse_args()
    if not args.image_path.is_file():
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    image_data = encode_image(args.image_path)
    payload = build_payload(args.prompt, image_data, args.model, args.temperature, args.max_tokens)

    response = requests.post(
        f"{args.api_base.rstrip('/')}/chat/completions",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()

    body = response.json()
    try:
        raw_content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as error:
        raise ValueError(f"Unexpected response structure: {body}") from error

    raw_text = extract_text_content(raw_content)
    detections = parse_detection_json(raw_text)

    if not isinstance(detections, list):
        raise ValueError("Model output is not a JSON array.")

    output_image = draw_bounding_boxes(args.image_path, detections)

    print(json.dumps(detections, indent=2))
    print(f"Saved annotated image to: {output_image}")
    open_image_viewer(output_image)


if __name__ == "__main__":
    main()
