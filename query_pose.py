#!/usr/bin/env python3
import argparse
import base64
import json
import mimetypes
import os
import subprocess
import sys
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import requests
from PIL import Image, ImageDraw, ImageFont


SYSTEM_PROMPT = (
    "You are an assistant that performs human pose estimation. "
    "Return only a JSON array where each item is an object with keys "
    '`"point_2d"` and `"label"`. `"point_2d"` must be a list of two integers '
    "representing x, y coordinates within [0, 1000]. "
    "Do not include any explanations, code fences, or additional text—return the raw JSON only."
)


REQUEST_TIMEOUT_DEFAULT = 120.0


class PoseEstimationError(Exception):
    """Raised when the model call fails or returns unusable keypoints."""


def warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit an image and prompt to a qwen3-VL model for pose estimation via an OpenAI-compatible API."
    )
    parser.add_argument("image_path", type=Path, help="Path to the image file to analyze.")
    parser.add_argument("prompt", help="User prompt describing the pose estimation task.")
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


def parse_keypoint_json(raw_text: str) -> List[Dict[str, Any]]:
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if "\n" in stripped:
            stripped = "\n".join(
                line for line in stripped.splitlines() if not line.strip().startswith("json")
            )
    return json.loads(stripped)


def request_completion(api_base: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}/chat/completions"
    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.Timeout as exc:
        raise PoseEstimationError(
            f"Request to {url} timed out after {timeout:.0f} seconds. "
            "Increase --timeout or check server load."
        ) from exc
    except requests.RequestException as exc:
        raise PoseEstimationError(f"Failed to reach the model endpoint: {exc}") from exc

    if not response.ok:
        snippet = response.text[:500]
        raise PoseEstimationError(
            f"Model endpoint returned HTTP {response.status_code}. Response snippet: {snippet}"
        )

    try:
        body = response.json()
    except JSONDecodeError as exc:
        snippet = response.text[:500]
        raise PoseEstimationError(f"Could not decode JSON response: {exc}. Snippet: {snippet}") from exc

    if isinstance(body, dict) and "error" in body:
        error_detail = body["error"]
        if isinstance(error_detail, dict):
            message = error_detail.get("message") or json.dumps(error_detail)
        else:
            message = str(error_detail)
        raise PoseEstimationError(f"Model returned an error payload: {message}")

    return body


def extract_keypoints(body: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
    try:
        choices = body["choices"]
        if not choices:
            raise KeyError
    except (KeyError, TypeError):
        raise PoseEstimationError("Model response did not include any choices.")

    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    if finish_reason == "length":
        raise PoseEstimationError(
            "Generation stopped because it reached the max token limit. "
            "Increase --max-tokens or reduce the prompt length."
        )
    if finish_reason and finish_reason not in ("stop", None):
        warn(f"Model finish_reason: {finish_reason}")

    try:
        raw_content = choice["message"]["content"]
    except KeyError as exc:
        raise PoseEstimationError("Model response missing message content.") from exc

    raw_text = extract_text_content(raw_content)
    if not raw_text.strip():
        raise PoseEstimationError("Model produced an empty response.")

    try:
        keypoints = parse_keypoint_json(raw_text)
    except (ValueError, TypeError) as exc:
        raise PoseEstimationError(f"Failed to parse model output as JSON: {exc}") from exc

    if not isinstance(keypoints, list):
        raise PoseEstimationError("Model output is not a JSON array.")

    return keypoints


def sanitize_keypoints(keypoints: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for index, keypoint in enumerate(keypoints):
        if not isinstance(keypoint, dict):
            warn(f"Ignoring keypoint #{index} because it is not an object.")
            continue

        point_raw = keypoint.get("point_2d")
        if not isinstance(point_raw, list) or len(point_raw) != 2:
            warn(f"Ignoring keypoint #{index} with invalid point_2d: {point_raw}")
            continue

        numeric_point: List[float] = []
        valid_point = True
        for coord in point_raw:
            if isinstance(coord, (int, float)):
                numeric_point.append(float(coord))
            elif isinstance(coord, str):
                try:
                    numeric_point.append(float(coord))
                except ValueError:
                    valid_point = False
                    break
            else:
                valid_point = False
                break

        if not valid_point:
            warn(f"Ignoring keypoint #{index} with non-numeric coordinates: {point_raw}")
            continue

        if any(coord < 0.0 or coord > 1000.0 for coord in numeric_point):
            warn(f"Clamping keypoint #{index} coordinates to [0, 1000].")

        point_ints = [
            int(round(max(0.0, min(1000.0, value))))
            for value in numeric_point
        ]

        label = keypoint.get("label", "")
        if label is None:
            label = ""
        elif not isinstance(label, str):
            label = str(label)
            warn(f"Coercing non-string label on keypoint #{index} to string.")

        cleaned.append({"point_2d": point_ints, "label": label})

    if not cleaned and keypoints:
        warn("All keypoints from the model were discarded due to formatting issues.")

    return cleaned


def normalize_point(point: List[int], width: int, height: int) -> List[int]:
    if len(point) != 2:
        raise ValueError(f"Expected point of length 2, got {point}")

    scale_x = width / 1000.0
    scale_y = height / 1000.0
    return [
        int(round(point[0] * scale_x)),
        int(round(point[1] * scale_y)),
    ]


def draw_keypoints(image_path: Path, keypoints: List[Dict[str, Any]]) -> Path:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size

    radius = max(3, int(round(min(width, height) * 0.005)))

    for keypoint in keypoints:
        point = keypoint.get("point_2d")
        label = keypoint.get("label", "")
        if not isinstance(point, list):
            continue

        try:
            x, y = normalize_point(point, width, height)
        except ValueError:
            continue

        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill="red", outline="white")
        if label:
            draw.text((x + radius + 2, y - radius - 2), label, fill="red", font=font)

    output_path = image_path.with_stem(f"{image_path.stem}_pose")
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
    except Exception as exc:  # pragma: no cover - best effort utility
        warn(f"Failed to open image viewer automatically: {exc}")


def main() -> None:
    args = parse_args()
    if not args.image_path.is_file():
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    image_data = encode_image(args.image_path)
    payload = build_payload(args.prompt, image_data, args.model, args.temperature, args.max_tokens)

    body = request_completion(args.api_base, payload, args.timeout)
    keypoints = list(extract_keypoints(body))
    sanitized_keypoints = sanitize_keypoints(keypoints)
    if keypoints and not sanitized_keypoints:
        raise PoseEstimationError(
            "Model returned keypoints but none were usable; review the warnings above."
        )

    output_keypoints = sanitized_keypoints if sanitized_keypoints else keypoints
    output_image = draw_keypoints(args.image_path, output_keypoints)

    print(json.dumps(output_keypoints, indent=2))
    print(f"Saved annotated image to: {output_image}")
    open_image_viewer(output_image)


if __name__ == "__main__":
    try:
        main()
    except PoseEstimationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        sys.exit(130)
