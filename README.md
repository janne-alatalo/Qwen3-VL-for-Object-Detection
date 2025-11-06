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
- `--model`: Model name served by vLLM (default `qwen3-VL`)
- `--max-tokens`: Increase if the model truncates responses (default `10000`)
- `--timeout`: Request timeout in seconds (default `120`)
- `--save-path`: Optional file or directory to persist the annotated image

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

You can tune `--max-tokens`, `--timeout`, or request a different model with `--model`.
- Use `--save-path` to persist the rendered skeleton to a specific directory or file; otherwise only the viewer opens.

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
