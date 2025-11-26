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


POSE_LABELS: List[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

SKELETON_EDGES = [
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

POSE_LABEL_SET = set(POSE_LABELS)

SYSTEM_PROMPT = (
    "You are an assistant that performs human pose estimation. "
    "Return only a JSON array where each element is an object with keys "
    '`"point_2d"` and `"label"`. `"point_2d"` must be a list of two integers '
    "representing x, y coordinates within [0, 1000]. "
    "Produce exactly one entry for each of the following labels in this order: "
    f"{', '.join(POSE_LABELS)}. "
    "If a joint is partially occluded, infer its most likely location. "
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
        default="http://127.0.0.1:8000/v1",
        help="Base URL for the OpenAI-compatible endpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        default="qwen3-vl",
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
        help="Optional path to save the annotated pose image. Treats directories as output folders.",
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
    seen_labels: set[str] = set()
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

        if not label:
            warn(f"Ignoring keypoint #{index} without a label.")
            continue

        if label in seen_labels:
            warn(f"Ignoring duplicate keypoint label '{label}' at index {index}.")
            continue

        if label not in POSE_LABEL_SET:
            warn(f"Keypoint '{label}' is not part of the expected skeleton.")

        seen_labels.add(label)
        cleaned.append({"point_2d": point_ints, "label": label})

    if not cleaned and keypoints:
        warn("All keypoints from the model were discarded due to formatting issues.")

    return cleaned


def resolve_save_path(requested: Path, source_image: Path, suffix: str) -> Path:
    extension = source_image.suffix or ".png"
    if requested.exists() and requested.is_dir():
        return requested / f"{source_image.stem}{suffix}{extension}"
    if requested.suffix:
        return requested
    return requested / f"{source_image.stem}{suffix}{extension}"


def normalize_point(point: List[int], width: int, height: int) -> List[int]:
    if len(point) != 2:
        raise ValueError(f"Expected point of length 2, got {point}")

    scale_x = width / 1000.0
    scale_y = height / 1000.0
    return [
        int(round(point[0] * scale_x)),
        int(round(point[1] * scale_y)),
    ]


def render_keypoints(image: Image.Image, keypoints: List[Dict[str, Any]]) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    width, height = output.size

    radius = max(3, int(round(min(width, height) * 0.005)))

    pixel_points: Dict[str, tuple[int, int]] = {}
    for keypoint in keypoints:
        point = keypoint.get("point_2d")
        label = keypoint.get("label", "")
        if not isinstance(point, list) or not label:
            continue

        try:
            x, y = normalize_point(point, width, height)
        except ValueError:
            continue

        pixel_points[label] = (x, y)

    line_color = "cyan"
    line_width = max(2, radius)
    for joint_a, joint_b in SKELETON_EDGES:
        if joint_a in pixel_points and joint_b in pixel_points:
            draw.line([pixel_points[joint_a], pixel_points[joint_b]], fill=line_color, width=line_width)

    for label, (x, y) in pixel_points.items():
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill="red", outline="white")
        draw.text((x + radius + 2, y - radius - 2), label, fill="red", font=font)

    return output


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
    keypoints = list(extract_keypoints(body))
    sanitized_keypoints = sanitize_keypoints(keypoints)
    if keypoints and not sanitized_keypoints:
        raise PoseEstimationError(
            "Model returned keypoints but none were usable; review the warnings above."
        )

    present_labels = {kp["label"] for kp in sanitized_keypoints if kp.get("label")}
    missing_labels = [label for label in POSE_LABELS if label not in present_labels]
    if missing_labels:
        warn(
            "Model response is missing joints for: "
            + ", ".join(missing_labels)
        )

    output_keypoints = sanitized_keypoints if sanitized_keypoints else keypoints
    original_image = Image.open(args.image_path).convert("RGB")
    annotated_image = render_keypoints(original_image, output_keypoints)

    print(json.dumps(output_keypoints, indent=2))

    if args.save_path is not None:
        save_target = resolve_save_path(args.save_path, args.image_path, "_pose")
        save_target.parent.mkdir(parents=True, exist_ok=True)
        annotated_image.save(save_target)
        print(f"Saved annotated image to: {save_target}")

    show_image(annotated_image, title=f"{args.image_path.name} pose")


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
