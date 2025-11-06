#!/usr/bin/env python3
import argparse
import base64
import json
import mimetypes
import sys
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import requests
from PIL import Image, ImageDraw, ImageFont


SYSTEM_PROMPT = (
    "You are an assistant that performs object detection. "
    "Return only a JSON array where each item is an object with keys "
    '`"bbox_2d"` and `"label"`. `"bbox_2d"` must be a list of four integers '
    "representing x1, y1, x2, y2 coordinates within [0, 1000]. "
    "Do not include any explanations, code fences, or additional text—return the raw JSON only."
)


REQUEST_TIMEOUT_DEFAULT = 120.0


class DetectionError(Exception):
    """Raised when the model call fails or returns unusable data."""


def warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


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
        default="/storage/proj/llm/hf-transformers-models/Qwen3-VL-30B-A3B-Thinking",
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
        default=10000,
        help="Maximum response tokens to request from the model.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=REQUEST_TIMEOUT_DEFAULT,
        help="Request timeout in seconds for the model API.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        help="Optional path to save the annotated image. Treats directories as output folders.",
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


def render_bounding_boxes(
    image_path: Path, detections: List[Dict[str, Any]]
) -> Image.Image:
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

    return image


def request_completion(api_base: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}/chat/completions"
    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.Timeout as exc:
        raise DetectionError(
            f"Request to {url} timed out after {timeout:.0f} seconds. "
            "Increase --timeout or check server load."
        ) from exc
    except requests.RequestException as exc:
        raise DetectionError(f"Failed to reach the model endpoint: {exc}") from exc

    if not response.ok:
        snippet = response.text[:500]
        raise DetectionError(
            f"Model endpoint returned HTTP {response.status_code}. Response snippet: {snippet}"
        )

    try:
        body = response.json()
    except JSONDecodeError as exc:
        snippet = response.text[:500]
        raise DetectionError(f"Could not decode JSON response: {exc}. Snippet: {snippet}") from exc

    if isinstance(body, dict) and "error" in body:
        error_detail = body["error"]
        if isinstance(error_detail, dict):
            message = error_detail.get("message") or json.dumps(error_detail)
        else:
            message = str(error_detail)
        raise DetectionError(f"Model returned an error payload: {message}")

    return body


def extract_detections(body: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
    try:
        choices = body["choices"]
        if not choices:
            raise KeyError
    except (KeyError, TypeError):
        raise DetectionError("Model response did not include any choices.")

    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    if finish_reason == "length":
        raise DetectionError(
            "Generation stopped because it reached the max token limit. "
            "Increase --max-tokens or reduce the prompt length."
        )
    if finish_reason and finish_reason not in ("stop", None):
        warn(f"Model finish_reason: {finish_reason}")

    try:
        raw_content = choice["message"]["content"]
    except KeyError as exc:
        raise DetectionError("Model response missing message content.") from exc

    raw_text = extract_text_content(raw_content)
    if not raw_text.strip():
        raise DetectionError("Model produced an empty response.")

    try:
        detections = parse_detection_json(raw_text)
    except (ValueError, TypeError) as exc:
        raise DetectionError(f"Failed to parse model output as JSON: {exc}") from exc

    if not isinstance(detections, list):
        raise DetectionError("Model output is not a JSON array.")

    return detections


def sanitize_detections(detections: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for index, detection in enumerate(detections):
        if not isinstance(detection, dict):
            warn(f"Ignoring detection #{index} because it is not an object.")
            continue

        bbox_raw = detection.get("bbox_2d")
        if not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
            warn(f"Ignoring detection #{index} with invalid bbox_2d: {bbox_raw}")
            continue

        bbox: List[float] = []
        valid_bbox = True
        for coord in bbox_raw:
            if isinstance(coord, (int, float)):
                bbox.append(float(coord))
            elif isinstance(coord, str):
                try:
                    bbox.append(float(coord))
                except ValueError:
                    valid_bbox = False
                    break
            else:
                valid_bbox = False
                break

        if not valid_bbox:
            warn(f"Ignoring detection #{index} with non-numeric bbox entries: {bbox_raw}")
            continue

        if any(coord < 0.0 or coord > 1000.0 for coord in bbox):
            warn(f"Clamping detection #{index} bbox coordinates to [0, 1000].")
        bbox_ints = [
            int(round(max(0.0, min(1000.0, value))))
            for value in bbox
        ]

        label = detection.get("label", "")
        if label is None:
            label = ""
        elif not isinstance(label, str):
            label = str(label)
            warn(f"Coercing non-string label on detection #{index} to string.")

        cleaned.append({"bbox_2d": bbox_ints, "label": label})

    if not cleaned and detections:
        warn("All detections from the model were discarded due to formatting issues.")

    return cleaned


def resolve_save_path(requested: Path, source_image: Path, suffix: str) -> Path:
    extension = source_image.suffix or ".png"
    if requested.exists() and requested.is_dir():
        return requested / f"{source_image.stem}{suffix}{extension}"
    if requested.suffix:
        return requested
    return requested / f"{source_image.stem}{suffix}{extension}"


def show_image(image: Image.Image, title: str) -> None:
    try:
        image.show(title=title)
    except Exception as exc:  # pragma: no cover - best effort utility
        warn(f"Failed to open image viewer automatically: {exc}")


def main() -> None:
    args = parse_args()
    if not args.image_path.is_file():
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    image_data = encode_image(args.image_path)
    payload = build_payload(args.prompt, image_data, args.model, args.temperature, args.max_tokens)

    body = request_completion(args.api_base, payload, args.timeout)
    detections = list(extract_detections(body))
    sanitized_detections = sanitize_detections(detections)
    if detections and not sanitized_detections:
        raise DetectionError(
            "Model returned detections but none were usable; review the warnings above."
        )

    detections_to_draw = sanitized_detections if sanitized_detections else detections

    annotated_image = render_bounding_boxes(args.image_path, detections_to_draw)

    print(json.dumps(detections_to_draw, indent=2))

    if args.save_path is not None:
        save_target = resolve_save_path(args.save_path, args.image_path, "_bbox")
        save_target.parent.mkdir(parents=True, exist_ok=True)
        annotated_image.save(save_target)
        print(f"Saved annotated image to: {save_target}")

    show_image(annotated_image, title=f"{args.image_path.name} detections")


if __name__ == "__main__":
    try:
        main()
    except DetectionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        sys.exit(130)
