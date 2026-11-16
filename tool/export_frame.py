"""Export original and reconstructed yuv420p videos to aligned PNG folders."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess


def output_layout(root, split, dataset, scene, codec=None, qp=None):
    if split == 'train':
        if codec is None:
            return root / 'gt' / scene, 'frame_%04d.png'
        return root / 'comp' / f'{codec}_{scene}_QP{qp}' / 'frame', f'frame_%d_QP{qp}.png'
    if codec is None:
        return root / 'GT' / dataset / scene, '%08d.png'
    return root / codec / f'QP{qp}' / dataset / scene, '%08d.png'


def export_frames(source, destination, width, height, frames, ffmpeg, fps, filename_pattern='%08d.png', start_number=1):
    expected = width * height * 3 // 2 * frames
    if width % 2 or height % 2 or source.stat().st_size != expected:
        raise ValueError(f"Invalid yuv420p dimensions or file size: {source} (expected {expected} bytes)")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {destination}. Use a new output folder.")
    destination.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, '-nostdin', '-n', '-hide_banner', '-loglevel', 'error',
               '-f', 'rawvideo', '-pixel_format', 'yuv420p',
               '-video_size', f'{width}x{height}', '-framerate', str(fps),
               '-i', str(source), '-frames:v', str(frames),
               '-fps_mode', 'passthrough', '-start_number', str(start_number),
               '-pix_fmt', 'rgb24', str(destination / filename_pattern)]
    subprocess.run(command, check=True)
    actual = len(list(destination.glob('*.png')))
    if actual != frames:
        raise RuntimeError(f"Frame count mismatch: {destination}: {actual} != {frames}")
    print(f"Saved {frames} PNGs: {destination}")


def rgb_frames(root, scene, width, height, frames):
    from PIL import Image
    folder = root / scene
    if not folder.is_dir():
        raise FileNotFoundError(f'Missing RGB scene folder: {folder}')
    def natural_key(path):
        return [int(part) if part.isdigit() else part.lower()
                for part in re.split(r'(\d+)', path.name)]
    images = sorted((p for p in folder.iterdir() if p.is_file()
                     and p.suffix.lower() in {'.png', '.jpg', '.jpeg'}), key=natural_key)
    if len(images) != frames:
        raise ValueError(f'RGB frame count mismatch in {folder}: {len(images)} != {frames}')
    for path in images:
        with Image.open(path) as image:
            if image.size != (width, height):
                raise ValueError(f'RGB size mismatch: {path}: {image.size} != {(width, height)}')
            image.verify()
    return images


def copy_rgb_frames(images, destination, filename_pattern):
    from PIL import Image
    destination.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(images, start=1):
        target = destination / (filename_pattern % index)
        if source.suffix.lower() == '.png':
            shutil.copy2(source, target)
        else:
            with Image.open(source) as image:
                image.convert('RGB').save(target, format='PNG')
    print(f'Saved {len(images)} original RGB GT frames: {destination}')


def discover_rgb_frames(input_root, output_root, scene, width, height, frames):
    """Look for scene RGB folders beside the input directory, never in outputs."""
    bases = [input_root, input_root.parent, input_root.parent / 'original_rgb',
             input_root.parent / 'rgb', input_root.parent / 'images',
             input_root.parent / 'gt', input_root.parent / 'GT',
             input_root.parent / 'original_rgb' / input_root.name,
             input_root.parent / 'rgb' / input_root.name,
             input_root.parent / 'gt' / input_root.name,
             input_root.parent / 'GT' / input_root.name]
    candidates = []
    seen = set()
    output = output_root.resolve()
    for base in bases:
        for scene_root in (base / scene, base / scene / 'images', base / scene / 'frame'):
            resolved = scene_root.resolve()
            if resolved in seen or resolved == output or output in resolved.parents:
                continue
            seen.add(resolved)
            if scene_root.is_dir() and any(p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'} for p in scene_root.iterdir()):
                candidates.append(scene_root)
    if len(candidates) > 1:
        raise ValueError(f'Multiple RGB candidates for {scene}; keep only the intended GT folder in the search locations: {candidates}')
    if not candidates:
        print(f'No original RGB folder found for {scene}; extracting GT from original YUV.')
        return None
    folder = candidates[0]
    print(f'Found original RGB GT: {folder}')
    return rgb_frames(folder.parent, folder.name, width, height, frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Original YUV folder used by compress.py')
    parser.add_argument('--compressed', type=Path, required=True, help='compress.py output folder')
    parser.add_argument('--output', type=Path, required=True, help='PNG output root, e.g. testset/final_test')
    parser.add_argument('--split', choices=('train', 'test'), default='test', help='Output folder and filename layout (default: test)')
    parser.add_argument('--dataset', help='Dataset folder name required for test, e.g. CO3Dv2')
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--ffmpeg', default=os.environ.get('FFMPEG_PATH', 'ffmpeg'))
    args = parser.parse_args()
    if args.split == 'test' and not args.dataset:
        parser.error('--dataset is required for --split test')
    if args.dataset and (args.dataset in {'.', '..'} or '/' in args.dataset or '\\' in args.dataset):
        parser.error('--dataset must be a single folder name')
    pattern = re.compile(r'(.+)_(\d+)x(\d+)_(\d+)\.yuv$')
    jobs = []
    rgb_jobs = {}
    destinations = set()
    for source in sorted(args.input.glob('*.yuv')):
        match = pattern.fullmatch(source.name)
        if not match:
            print(f'Skipping filename without dimensions: {source.name}')
            continue
        scene, width, height, frames = match.groups()
        width, height, frames = int(width), int(height), int(frames)
        reconstructions = []