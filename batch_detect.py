#!/usr/bin/env python3
import argparse
import json
import signal
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from json import JSONDecodeError
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from query_bbox import (
    DetectionError,
    encode_image,
    extract_detections,
    request_completion,
    sanitize_detections,
    build_payload,
    load_example_pairs,
    load_context_images,
    format_debug_info,
)

DEFAULT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batched object detection over a dataset tree using the Qwen3-VL model."
    )
    parser.add_argument("dataset_root", type=Path, help="Root directory of the dataset.")
    parser.add_argument("prompt", help="User prompt describing the detection task.")
    parser.add_argument("results_path", type=Path, help="Path to the JSONL results file.")
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
        default=10000,
        help="Maximum response tokens to request from the model.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds for the model API.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of concurrent requests (default: %(default)s).",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=sorted(DEFAULT_EXTENSIONS),
        help="Image file extensions to include (default: common formats).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit on the number of images to process.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images already present in the results file (default: %(default)s).",
    )
    parser.add_argument(
        "--example",
        nargs=2,
        metavar=("IMAGE", "ANNOTATION"),
        action="append",
        help="Provide few-shot examples (image path followed by JSON annotations). Can repeat.",
    )
    parser.add_argument(
        "--context-image",
        action="append",
        metavar="IMAGE",
        help="Reference image to include before the target image. Can repeat.",
    )
    return parser.parse_args()


def normalize_extensions(exts: Iterable[str]) -> Set[str]:
    return {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in exts}


def find_images(root: Path, extensions: Set[str]) -> List[Path]:
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in extensions
    ]


def load_existing_results(results_path: Path) -> Dict[str, Dict[str, Any]]:
    processed: Dict[str, Dict[str, Any]] = {}
    if not results_path.exists():
        return processed

    with results_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
                rel_path = record["image"]
                processed[rel_path] = record
            except (JSONDecodeError, KeyError) as exc:
                warn(f"Skipping malformed record at line {line_number}: {exc}")
    return processed


def ensure_output_directory(results_path: Path) -> None:
    parent = results_path.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def append_result(results_path: Path, lock: Lock, record: Dict[str, Any]) -> None:
    payload = json.dumps(record, separators=(",", ":"))
    with lock:
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")


def detect_single_image(
    image_path: Path,
    rel_path: str,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    api_base: str,
    timeout: float,
    examples: Optional[Sequence[Tuple[str, str]]] = None,
    context_images: Optional[Sequence[str]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    image_data = encode_image(image_path)
    payload = build_payload(
        prompt,
        image_data,
        model,
        temperature,
        max_tokens,
        examples=examples,
        context_images=context_images,
    )
    body = request_completion(api_base, payload, timeout)
    detections, raw_text, raw_body = extract_detections(body)
    detections = list(detections)
    sanitized = sanitize_detections(detections)
    if detections and not sanitized:
        raise DetectionError(
            f"Image '{rel_path}' produced detections but none were usable."
            + format_debug_info(raw_text, raw_body)
        )
    return rel_path, sanitized if sanitized else detections


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found or not a directory: {dataset_root}")

    extensions = normalize_extensions(args.extensions or DEFAULT_EXTENSIONS)
    ensure_output_directory(args.results_path)

    processed_records = load_existing_results(args.results_path) if args.resume else {}
    processed_paths = set(processed_records.keys())

    images = find_images(dataset_root, extensions)
    if args.limit is not None:
        images = images[: args.limit]

    if not images:
        print("No images found to process.", file=sys.stderr)
        return

    print(f"Discovered {len(images)} images. Processed entries loaded: {len(processed_paths)}")

    lock = Lock()
    stop_requested = False
    example_specs = [
        (Path(image_path), Path(annotation_path))
        for image_path, annotation_path in (args.example or [])
    ]
    context_specs = [Path(path) for path in (args.context_image or [])]
    example_payloads = load_example_pairs(example_specs) if example_specs else None
    context_payloads = None
    if context_specs:
        context_payloads = load_context_images(context_specs)
    if example_payloads and context_payloads:
        raise ValueError("Specify either --example or --context-image, not both.")

    def signal_handler(signum: int, frame: Any) -> None:  # pragma: no cover - system integration
        nonlocal stop_requested
        stop_requested = True
        warn(f"Received signal {signum}; finishing in-progress tasks gracefully...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures: Dict[Future[Tuple[str, List[Dict[str, Any]]]], str] = {}
        pending = deque(images)

        def submit_next() -> None:
            while pending and len(futures) < args.max_workers and not stop_requested:
                image_path = pending.popleft()
                rel_path = image_path.relative_to(dataset_root).as_posix()
                if rel_path in processed_paths:
                    continue
                futures[executor.submit(
                    detect_single_image,
                    image_path,
                    rel_path,
                    args.prompt,
                    args.model,
                    args.temperature,
                    args.max_tokens,
                    args.api_base,
                    args.timeout,
                    example_payloads,
                    context_payloads,
                )] = rel_path

        submit_next()

        try:
            while futures:
                for future in as_completed(list(futures.keys())):
                    rel_path = futures.pop(future)
                    try:
                        image_key, detections = future.result()
                        record = {
                            "image": image_key,
                            "detections": detections,
                        }
                        append_result(args.results_path, lock, record)
                        processed_paths.add(image_key)
                        print(f"Processed {image_key} ({len(detections)} detections)")
                    except DetectionError as exc:
                        warn(f"Detection error for {rel_path}: {exc}")
                    except Exception as exc:
                        warn(f"Unexpected error for {rel_path}: {exc}")
                    if stop_requested:
                        warn("Stopping submission of new tasks due to signal.")
                    submit_next()
                    if stop_requested and not futures:
                        warn("All in-flight tasks completed after stop signal.")
                        break
                if stop_requested and not futures:
                    break
        finally:
            for future in futures:
                future.cancel()

    print(f"Completed. Total processed images: {len(processed_paths)}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        sys.exit(130)
