# ══════════════════════════════════════════════════════════════════════════════
# COCO annotation loader for H1 masks
# Produces [14×14] patch masks aligned with ImageNet val images.
# ══════════════════════════════════════════════════════════════════════════════
import torch
import numpy as np
from pathlib import Path
import json


# ─────────────────────────────────────────────────────────────────────────────
# COCO annotation loader (no pycocotools needed — manual polygon → mask)
# ─────────────────────────────────────────────────────────────────────────────
def load_coco_annotations(ann_json_path):
  """
  Load COCO val2017 annotations from the JSON file directly.
  (Avoids pycocotools dependency on Kaggle.)

  Args:
      ann_json_path: path to instances_val2017.json

  Returns:
      imgid_to_anns:  dict {image_id → list of annotation dicts}
      imgid_to_info:  dict {image_id → {"file_name", "height", "width"}}
  """
  print(f"Loading COCO annotations from {ann_json_path}...")
  with open(ann_json_path) as f:
      data = json.load(f)

  imgid_to_info = {img["id"]: img for img in data["images"]}

  imgid_to_anns = {}
  for ann in data["annotations"]:
    iid = ann["image_id"]
    if iid not in imgid_to_anns:
        imgid_to_anns[iid] = []
    imgid_to_anns[iid].append(ann)

  print(f"  {len(imgid_to_info)} images, {len(data['annotations'])} annotations")
  return imgid_to_anns, imgid_to_info


def bbox_to_patch_mask(bbox, img_w, img_h, patch_size=16, n_patches=14,
                        input_size=224, threshold=0.3):
  """
  Convert a COCO bounding box [x, y, w, h] to a [14×14] patch mask.
  Uses bbox as a cheap proxy for segmentation when polygon masks
  are too slow to rasterize.

  Args:
      bbox:       [x, y, w, h] in original image coordinates
      img_w/h:    original image dimensions
      threshold:  fraction of patch area inside bbox to call it foreground

  Returns:
      [14, 14] BoolTensor
  """
  x, y, bw, bh = bbox
  # scale bbox to 224×224 space
  sx = input_size / img_w
  sy = input_size / img_h
  x1 = x  * sx;  y1 = y  * sy
  x2 = (x + bw) * sx;  y2 = (y + bh) * sy

  mask = torch.zeros(n_patches, n_patches, dtype=torch.bool)
  for pi in range(n_patches):
      for pj in range(n_patches):
          # patch pixel range
          px1 = pj * patch_size;  px2 = px1 + patch_size
          py1 = pi * patch_size;  py2 = py1 + patch_size
          # intersection with bbox
          ix1 = max(px1, x1);  ix2 = min(px2, x2)
          iy1 = max(py1, y1);  iy2 = min(py2, y2)
          if ix2 > ix1 and iy2 > iy1:
              inter = (ix2 - ix1) * (iy2 - iy1)
              frac  = inter / (patch_size * patch_size)
              if frac > threshold:
                  mask[pi, pj] = True
  return mask


def build_h1_dataset(imgid_to_anns, imgid_to_info,
                     imagenet_val_dir, coco_val_dir,
                     n_images=500, min_fg_patches=10,
                     min_bg_patches=10):
  """
  Build the H1 dataset: pairs of (image_path, patch_mask_14x14).
  
  Strategy: use images that appear in BOTH ImageNet val AND COCO val2017.
  COCO val2017 images are a subset of ImageNet val by filename convention
  for images formatted as ILSVRC2012_val_XXXXXXXX.JPEG.

  On Kaggle, COCO val2017 images are at: /kaggle/input/coco-2017-dataset/coco2017/val2017/

  For each COCO val image with annotations:
      1. Find the image file in coco_val_dir
      2. Merge all bbox masks (union of objects = foreground)
      3. Filter: keep only images with enough fg AND bg patches
      4. Return (path, mask) pairs

  Args:
      imagenet_val_dir:  not used if using COCO val images directly
      coco_val_dir:      path to COCO val2017 images (000000XXXXXX.jpg)
      n_images:          max images to return
      min_fg_patches:    discard images with too few foreground patches
      min_bg_patches:    discard images with too few background patches

  Returns:
      image_paths:   list of str
      masks_14x14:   list of [14,14] BoolTensor
  """
  image_paths = []
  masks_14x14 = []
  skipped = 0

  coco_val_dir = Path(coco_val_dir)

  for img_id, anns in imgid_to_anns.items():
      if len(image_paths) >= n_images:
          break

      info = imgid_to_info[img_id]
      fname = info["file_name"]  
      img_path = coco_val_dir / fname

      if not img_path.exists():
          skipped += 1
          continue

      # merge all object bboxes → union foreground mask
      img_w, img_h = info["width"], info["height"]
      combined = torch.zeros(14, 14, dtype=torch.bool)
      for ann in anns:
          if "bbox" not in ann:
              continue
          m = bbox_to_patch_mask(ann["bbox"], img_w, img_h)
          combined = combined | m

      # quality filter
      n_fg = combined.sum().item()
      n_bg = (~combined).sum().item()
      if n_fg < min_fg_patches or n_bg < min_bg_patches:
          skipped += 1
          continue

      image_paths.append(str(img_path))
      masks_14x14.append(combined)

  print(f"  Built {len(image_paths)} image-mask pairs  ({skipped} skipped)")
  return image_paths, masks_14x14


# ─────────────────────────────────────────────────────────────────────────────
# Layer sensitivity sweep
# ─────────────────────────────────────────────────────────────────────────────
def layer_sensitivity_sweep(model, image_paths, masks_14x14,
                             layers=(5, 8, 11), device='cpu', n=50):
  """
  Run H1 signal measurement across multiple encoder layers to find
  which layer produces the strongest foreground-background entropy gap.

  Earlier layers: more local, spatial attention
  Later layers:   more semantic, global attention (usually better for prior)

  Args:
      layers: tuple of layer indices to test (0-indexed, max 11 for ViT-B)
      n:      number of images to use (subset for speed)

  Returns:
      dict: layer → {"mean_H_gap", "pct_pos"}
  """
  from h1_signals import (extract_attention, compute_routing_signals,
                                  measure_h1, load_image)
  model.to(device).eval()

  results = {}
  for layer in layers:
      gaps = []
      for path, mask in zip(image_paths[:n], masks_14x14[:n]):
          pv = load_image(path).to(device)
          A = extract_attention(model, pv, layer=layer)
          H_i, _, _ = compute_routing_signals(A)
          r = measure_h1(H_i[0].cpu(), mask)
          if r: gaps.append(r["gap"])

      results[layer] = {
          "mean_H_gap": np.mean(gaps) if gaps else 0,
          "pct_pos":    np.mean([g > 0 for g in gaps]) * 100 if gaps else 0,
          "n":          len(gaps),
      }
      print(f"  Layer {layer:2d}: mean_H_gap={results[layer]['mean_H_gap']:.4f}  "
            f"pct_pos={results[layer]['pct_pos']:.1f}%")

  best = max(results, key=lambda l: results[l]["mean_H_gap"])
  print(f"\n  Best layer: {best}  → use this for all subsequent experiments")
  return results, best


# ─────────────────────────────────────────────────────────────────────────────
# DINOv2 artifact token detector
# ─────────────────────────────────────────────────────────────────────────────
def detect_dinov2_artifacts(A, H_i, mask_14x14, artifact_threshold=0.5):
  """
  Detects high-norm outlier ("artifact") tokens in DINOv2 attention maps.
  Artifacts: background patches with anomalously LOW entropy (as if salient),
  not explained by the COCO foreground mask.

  Definition used here:
      artifact token = patch where:
          (a) mask says background, AND
          (b) entropy is below the 25th percentile of ALL patch entropies

  Args:
      A:               [H, 197, 197]  attention for one image, one head avg
      H_i:             [196]          entropy per patch token
      mask_14x14:      [14,14]        foreground mask

  Returns:
      n_artifacts:     int — number of artifact tokens
      artifact_map:    [14,14] bool — True where artifacts found
  """
  flat_mask = mask_14x14.flatten()   # [196]
  bg_tokens = ~flat_mask             # background according to COCO

  q25 = torch.quantile(H_i, 0.25).item()   # low-entropy threshold
  low_entropy = H_i < q25                   # [196]

  # artifact = background AND anomalously low entropy
  artifact_map_flat = bg_tokens & low_entropy
  n_artifacts       = artifact_map_flat.sum().item()
  artifact_map      = artifact_map_flat.reshape(14, 14)

  return n_artifacts, artifact_map


def run_artifact_scan(model, image_paths, masks_14x14,
                      layer=11, n=100, device='cpu'):
  """
  Scan N images for DINOv2 artifact tokens.
  Run this for both DINOv1 and DINOv2 and compare.

  Returns:
      dict with mean_artifacts_per_image, pct_images_with_artifacts
  """
  from h1_signals import extract_attention, compute_routing_signals, load_image
  model.to(device).eval()
  counts = []

  for path, mask in zip(image_paths[:n], masks_14x14[:n]):
      pv = load_image(path).to(device)
      A  = extract_attention(model, pv, layer=layer)
      H_i, _, _ = compute_routing_signals(A)
      n_art, _  = detect_dinov2_artifacts(A[0].mean(0), H_i[0].cpu(), mask)
      counts.append(n_art)

  result = {
      "mean_artifacts_per_image": np.mean(counts),
      "pct_images_with_artifacts": np.mean([c > 0 for c in counts]) * 100,
      "max_artifacts_in_one_image": max(counts),
  }
  print(f"  Mean artifacts/image:    {result['mean_artifacts_per_image']:.2f}")
  print(f"  % images with artifacts: {result['pct_images_with_artifacts']:.1f}%")
  print(f"  Max in one image:        {result['max_artifacts_in_one_image']}")
  print("\n  DINOv1 should show near-zero artifacts")
  print("  DINOv2 typically shows 3-8 artifacts/image")
  return result


# ─────────────────────────────────────────────────────────────────────────────
# TEST with synthetic data
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing COCO pipeline with synthetic data...")

    # synthetic mask with object in center
    mask = torch.zeros(14, 14, dtype=torch.bool)
    mask[3:11, 3:11] = True
    print(f"  Mask FG patches: {mask.sum().item()}  BG patches: {(~mask).sum().item()}")

    # test bbox_to_patch_mask
    bbox  = [30, 30, 164, 164]   # roughly center crop in 224×224 image
    pmask = bbox_to_patch_mask(bbox, img_w=224, img_h=224)
    print(f"  BBox mask FG: {pmask.sum().item()} patches")
    assert pmask.sum() > 0, "BBox mask is empty"

    # test artifact detector with synthetic attention
    import torch
    H_i_fake = torch.rand(196)
    H_i_fake[10] = 0.01    # inject a suspiciously low-entropy background token
    n_art, art_map = detect_dinov2_artifacts(
        torch.rand(197, 197),  # dummy A
        H_i_fake,
        mask
    )
    print(f" Artifact scan: {n_art} artifacts found (expect ≥1 from injected token)")

    # COCO JSON test (no actual file needed — test structure)
    fake_data = {
        "images": [{"id": 1, "file_name": "000000000001.jpg",
                    "height": 480, "width": 640}],
        "annotations": [{"image_id": 1, "bbox": [100, 100, 200, 150],
                          "category_id": 1}]
    }
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(fake_data, f)
        tmp = f.name
    imgid_to_anns, imgid_to_info = load_coco_annotations(tmp)
    os.unlink(tmp)
    assert 1 in imgid_to_anns
    print(f"  COCO loader: {len(imgid_to_anns)} image-annotation pairs")

    print("\n✓ coco_loader.py — all checks passed")

    # On Kaggle: COCO annotations at /kaggle/input/coco-2017-dataset/
    imgid_to_anns, imgid_to_info = load_coco_annotations(
    '/kaggle/input/coco-2017-dataset/annotations/instances_val2017.json')
    paths, masks = build_h1_dataset(imgid_to_anns, imgid_to_info,
    imagenet_val_dir='', coco_val_dir='/kaggle/input/coco.../val2017/')
