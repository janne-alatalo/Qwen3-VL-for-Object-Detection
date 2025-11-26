# Multimodal Object Detection Tools

This repository contains lightweight helpers for exercising the Qwen3-VL model (served via vLLM’s OpenAI-compatible API) for 2D object detection and human pose estimation.

## Prerequisites

- Python 3.9+
- Packages: `requests`, `Pillow`

Install dependencies into your environment:

```bash
python -m pip install --upgrade pip
python -m pip install requests pillow
```

## Environment Variables

Both scripts default to the local vLLM endpoint at `http://10.88.0.1:8000/v1`. Override as needed with the `--api-base` flag.

## Object Detection (`query_bbox.py`)

Submit an image and prompt for bounding box predictions:

```bash
python query_bbox.py path/to/image.jpg "Find every bicycle and rider"
```

Key flags:

- `--api-base`: OpenAI-compatible base URL (default `http://10.88.0.1:8000/v1`)
- `--model`: Model name served by vLLM (default `qwen3-vl`)
- `--temperature`, `--top-p`, `--top-k`, `--repetition-penalty`, `--presence-penalty`, `--seed`: Decoding controls (defaults match script help)
- `--max-tokens`: Increase if the model truncates responses (default `10000`)
- `--timeout`: Request timeout in seconds (default `120`)
- `--save-path`: Optional file or directory to persist the annotated image
- `--save-generation-details`: Persist request/response metadata plus CoT reasoning as JSON
- `--example IMAGE JSON`: Add few-shot examples (image plus JSON annotation). Repeat as needed.
- `--context-image IMAGE`: Attach additional reference images (without JSON). The final positional
  image argument is always processed. Repeatable.

Output:

- Prints normalized detections (labels plus `[x1, y1, x2, y2]` in full image pixels)
- Shows the annotated image using the OS default viewer (best effort)
- Saves only when `--save-path` is supplied; directories are treated as output folders

If the call fails or the response is malformed, the script surfaces the error on stderr with actionable hints.

## Pose Estimation (`query_pose.py`)

Request COCO-style keypoints for the same image:

```bash
python query_pose.py path/to/image.jpg "Detect the full body pose"
```

The system prompt asks the model for the canonical 17 joints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles). After successful parsing the script:

- Warns about missing or duplicate joints
- Prints the sanitized keypoint list
- Renders a skeleton overlay (`*_pose.*`) with cyan bones and red joint markers
- Opens the annotated image automatically

You can tune `--max-tokens`, `--timeout`, or request a different model with `--model`. Use `--save-path` to persist the rendered skeleton to a specific directory or file; otherwise only the viewer opens.

## Batch Dataset Processing (`batch_detect.py`)

Run inference over an entire dataset tree and store detections incrementally:

```bash
python batch_detect.py /data/dataset "Detect every pedestrian" detections.jsonl --max-workers 6
```

- Recursively scans the dataset root for common image types (override with `--extensions`)
- Keeps multiple requests in flight with `--max-workers` (vLLM handles batching server-side)
- Writes JSON Lines records as soon as each image finishes: `{"image": "relative/path.jpg", "detections": [...]}`  
  (paths are stored relative to the dataset root so they can be re-used later)
- `--resume` skips images already present in the output file, enabling crash-safe restarts
- `--limit` caps how many images are processed in one run
- `--generation-details-path` writes per-image generation metadata JSONL; defaults next to results
- `--generation-arguments-path` snapshots the run configuration to a JSON file
- `--example IMAGE JSON` can inject the same few-shot examples used by `query_bbox.py`
- `--context-image IMAGE` attaches extra reference images before each target image. Do not
  combine with `--example`.
- Shares the same API tuning flags (`--api-base`, `--model`, decoding penalties, `--max-tokens`, `--timeout`, `--temperature`, `--seed`); defaults to model `qwen3-VL` and the same API base as `query_bbox.py`

No annotated images are saved during batch jobs—the JSONL file is meant for downstream visualization or evaluation scripts.

## Iterative Detection (`query_bbox_iterative.py`)

Experimental loop where the model critiques and revises its own boxes:

1) Run a normal detection call (same system prompt as `query_bbox.py`).  
2) Render boxes with numeric ids on the image.  
3) Send the original prompt/image plus the overlaid image back to the model, along with its previous JSON, asking it to keep ids and adjust boxes/labels as needed. Repeat until boxes stop changing or `--max-iterations` is reached.

Usage:

```bash
python query_bbox_iterative.py path/to/image.jpg "Find all defects" --max-iterations 5
```

Key options:

- `--max-iterations`: Correction rounds after the initial pass (default `5`)
- `--api-base`, `--model`, `--temperature`, `--max-tokens`, `--timeout`: Apply to all rounds (defaults: API base `http://10.88.0.1:8000/v1`, model `qwen3-VL`, temperature `0.0`, max tokens `2000`, timeout `120s`)
- `--output-dir`: If set, saves per-iteration overlays (`*_iter_XX.png`); otherwise displays them at the end

Stops early when the refined detections match the previous round exactly.

## Visualization (`visualize_results.py`)

Convert detection logs back into images with side-by-side comparisons:

```bash
python visualize_results.py /data/dataset detections.jsonl labeled_outputs/
```

- Reads the JSON Lines file produced by `batch_detect.py`
- Recreates directory structure under the chosen output root, appending `_labeled` to filenames
- Produces a composite image with the original on the left and the annotated version on the right
- Skip existing outputs unless `--overwrite` is provided

This makes it easy to inspect detections manually without modifying the source dataset.

## Classification Metrics (`evaluate_classification.py`)

When a dataset only indicates whether an image contains zero vs. one-or-more target objects, you can still
score the detector by collapsing the JSONL output into binary predictions:

```bash
python evaluate_classification.py detections.jsonl \
  --positive-regex "positives/" \
  --negative-regex "negatives/" \
  --unmatched-policy skip
```

- `--positive-regex` is required and should match every positive sample's relative path.
- `--negative-regex` is optional; if omitted, unmatched samples follow `--unmatched-policy` (defaults to skip).
- Metrics include accuracy, precision, recall, F1, specificity, and a confusion matrix.

## Detection Metrics (`evaluate_detections.py`)

Score JSONL detections against YOLO-style ground truth boxes:

```bash
python evaluate_detections.py detections.jsonl /data/dataset /data/labels --iou-threshold 0.5
```

Key options:

- `--label-suffix`: Extension for YOLO label files (default `.txt`)
- `--label-path-template`: Custom format to locate labels (placeholders: `{rel_path}`, `{stem}`, `{name}`, `{parent}`)
- `--strip-prefix`: Remove a leading path segment before mirroring under the labels root (repeatable)
- `--missing-label-policy`: `skip` (default) or `empty` to treat missing labels as zero objects
- `--class-names` / `--class-map`: Supply or override YOLO class IDs to names
- `--ignore-case`: Match labels case-insensitively
- `--lowercase-labels`: Force all predicted and ground-truth labels to lowercase before scoring
- `--iou-threshold`: IoU cutoff for a match (default `0.5`)

Outputs include per-class precision/recall/F1, mean IoU, micro-averaged metrics, and a label-confusion summary for IoU-matched boxes.

## Troubleshooting

- `Error: Generation stopped because it reached the max token limit`  
  Increase `--max-tokens` (thinking models count hidden chain-of-thought tokens).

- `Request timed out`  
  Raise `--timeout` or verify the vLLM server is responsive.

- Missing joints / malformed boxes  
  Inspect stderr warnings. Adjust the user prompt or temperature, or manually post-process the JSON.

## Development

Standard Git workflow applies:

```bash
git status
git add <files>
git commit -m "Your message"
```

Feel free to extend these utilities with batch processing, evaluation metrics, or alternative visualization styles to suit your experiments.
