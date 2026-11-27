#!/usr/bin/env python3
"""Task 1: calibrate the chessboard clip, replay it with a cube, and measure.

Run from the repository root:

    python3 task1/run.py calibrate
    python3 task1/run.py replay --start 168 --stop 432
    python3 task1/run.py measure --checks 180,420 --scale-frame 420

The calibration segment is source frames 0-167 (the first 7 s).  The replay
default is source frames 168-431 (7.0 s up to, but not including, 18.0 s),
renumbered from 0.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "assets" / "chessboard_104435_preview.mp4"
OUT = ROOT / "results" / "task1"

PATTERN = (9, 6)  # inner corners: 9 columns, 6 rows
SQUARE_MM = 20.0
CALIBRATION_STOP = 168

# Every 6th frame of the calibration segment plus the last frame.  Frames are
# chosen before fitting; see calibration_selection.md for the resulting
# coverage of distance, image position and viewing obliquity.
CALIBRATION_FRAMES = (0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78,
                      84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144,
                      150, 156, 162, 167)

# Detector order: bottom row first, right to left, then upwards.  The board is
# printed with the text below it, so this order maps to object coordinates with
# x to the right, y upwards and +z out of the printed side (towards camera).
CUBE_ROWS = (2, 4)
CUBE_COLS = (4, 6)
CUBE_EDGE_MM = 2 * SQUARE_MM


@dataclass
class Pose:
    rvec: np.ndarray
    tvec: np.ndarray
    rotation: np.ndarray
    center: np.ndarray  # board-pattern centre in camera coordinates, mm
    rmse_px: float
    valid: bool


def object_grid(square_mm: float) -> np.ndarray:
    """Inner corners in detector order, millimetres, board plane z=0."""
    return np.array(
        [((8 - j) * square_mm, r * square_mm, 0.0)
         for r in range(6) for j in range(9)],
        dtype=np.float32,
    )


def board_center(square_mm: float) -> np.ndarray:
    return np.array([4.0 * square_mm, 2.5 * square_mm, 0.0])


def cube_corners(square_mm: float) -> np.ndarray:
    base = [
        ((c - 1) * square_mm, (6 - r) * square_mm, 0.0)
        for r, c in ((CUBE_ROWS[0], CUBE_COLS[0]),
                     (CUBE_ROWS[0], CUBE_COLS[1]),
                     (CUBE_ROWS[1], CUBE_COLS[1]),
                     (CUBE_ROWS[1], CUBE_COLS[0]))
    ]
    top = [(x, y, CUBE_EDGE_MM) for x, y, _ in base]
    return np.array(base + top, dtype=np.float32)


CUBE_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0),
              (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7))


def detect_corners(image: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(gray, PATTERN)
    if not found:
        return None
    return corners.reshape(-1, 2).astype(np.float32)


def read_selected(start: int, stop: int, wanted: set[int] | None = None):
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {VIDEO}")
    try:
        for index in range(stop):
            ok, image = cap.read()
            if not ok:
                raise SystemExit(f"cannot decode source frame {index}")
            if index < start:
                continue
            if wanted is None or index in wanted:
                yield index, image
    finally:
        cap.release()


def solve_pose(corners: np.ndarray, matrix: np.ndarray, distortion: np.ndarray,
               square_mm: float, max_rmse_px: float = 2.0) -> Pose | None:
    objp = object_grid(square_mm)
    ok, rvec, tvec = cv2.solvePnP(
        objp, corners, matrix, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    center = rotation @ board_center(square_mm) + tvec.ravel()
    projected, _ = cv2.projectPoints(objp, rvec, tvec, matrix, distortion)
    rmse = float(np.sqrt(np.mean(np.sum(
        (projected.reshape(-1, 2) - corners) ** 2, axis=1))))
    normal = rotation[:, 2]
    valid = bool(
        np.isfinite(center).all()
        and center[2] > 0.0
        and np.dot(normal, -center) > 0.0  # printed side faces the camera
        and rmse <= max_rmse_px
    )
    return Pose(rvec, tvec.ravel(), rotation, center, rmse, valid)


def load_calibration(path: Path):
    with np.load(path, allow_pickle=False) as data:
        matrix = data["K"]
        distortion = data["dist"]
        width, height = (int(v) for v in data["image_size"])
        square_mm = float(data["square_size_m"]) * 1000.0
    return matrix, distortion, (width, height), square_mm


def save_calibration(path: Path, matrix, distortion, size, rmse, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        K=matrix.astype(np.float64),
        dist=distortion.astype(np.float64).reshape(1, -1),
        image_size=np.array(size, dtype=np.int32),
        square_size_m=np.array(SQUARE_MM / 1000.0),
        rms=np.array(rmse),
        frames=np.array(frames, dtype=np.int32),
    )


def role(distance_mm: float, tilt_deg: float, cx: float) -> str:
    parts = []
    if distance_mm < 500:
        parts.append("近距/强透视")
    elif distance_mm < 800:
        parts.append("中距")
    elif distance_mm < 1100:
        parts.append("远距")
    else:
        parts.append("回放工作距")
    if tilt_deg >= 30:
        parts.append("大倾角")
    elif tilt_deg <= 8:
        parts.append("近正视")
    if cx < 300:
        parts.append("偏左")
    elif cx > 335:
        parts.append("偏右")
    else:
        parts.append("居中")
    return "、".join(parts)


def calibrate(args) -> None:
    wanted = set(CALIBRATION_FRAMES)
    views = []
    for index, image in read_selected(0, CALIBRATION_STOP, wanted):
        corners = detect_corners(image)
        if corners is None:
            raise SystemExit(f"calibration frame {index}: chessboard not found")
        views.append((index, image, corners))
    if [i for i, _, _ in views] != list(CALIBRATION_FRAMES):
        raise SystemExit("calibration frames are not all readable")

    objp = object_grid(SQUARE_MM)
    object_points = [objp] * len(views)
    image_points = [corners for _, _, corners in views]
    rmse, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, (views[0][1].shape[1],
                                      views[0][1].shape[0]),
        None, None, flags=cv2.CALIB_FIX_K3)

    entries = []
    for (index, image, corners), rvec, tvec in zip(views, rvecs, tvecs):
        rotation, _ = cv2.Rodrigues(rvec)
        center = rotation @ board_center(SQUARE_MM) + tvec.ravel()
        normal = rotation[:, 2]
        tilt = math.degrees(math.acos(np.clip(
            np.dot(normal, -center) / np.linalg.norm(center), -1.0, 1.0)))
        projected, _ = cv2.projectPoints(
            objp, rvec, tvec, matrix, distortion)
        frame_rmse = float(np.sqrt(np.mean(np.sum(
            (projected.reshape(-1, 2) - corners) ** 2, axis=1))))
        # Physical top row: detected row 5.  Width: leftmost/rightmost corner.
        width = float(np.linalg.norm(corners[5 * 9 + 8] - corners[5 * 9]))
        cx, cy = corners.mean(axis=0)
        entries.append({
            "source_frame": index,
            "time_s": index / 24.0,
            "center_px": [round(float(cx), 1), round(float(cy), 1)],
            "inner_width_px": round(width, 1),
            "optical_depth_mm": round(float(center[2]), 1),
            "distance_mm": round(float(np.linalg.norm(center)), 1),
            "tilt_deg": round(tilt, 1),
            "reprojection_rmse_px": round(frame_rmse, 3),
            "coverage": role(float(np.linalg.norm(center)), tilt, float(cx)),
        })

    report = {
        "source_video": str(VIDEO.relative_to(ROOT)),
        "pattern_inner_corners": list(PATTERN),
        "square_size_mm": SQUARE_MM,
        "calibration_frames": list(CALIBRATION_FRAMES),
        "distortion_model": "k1,k2,p1,p2, k3 fixed to 0 (CALIB_FIX_K3)",
        "rms_reprojection_px": round(float(rmse), 4),
        "camera_matrix": matrix.tolist(),
        "distortion": distortion.ravel().tolist(),
        "views": entries,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calibration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    save_calibration(OUT / "calibration.npz", matrix, distortion,
                     (int(views[0][1].shape[1]), int(views[0][1].shape[0])),
                     rmse, CALIBRATION_FRAMES)

    # Contact sheet: 6 x 5 at quarter resolution.
    tw, th, cols = 320, 180, 6
    tiles = []
    for entry, (_, image, _) in zip(entries, views):
        tile = cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (tw, 22), (0, 0, 0), -1)
        cv2.putText(tile, f"f{entry['source_frame']}  {entry['distance_mm']:.0f}mm  "
                          f"tilt {entry['tilt_deg']:.0f}deg",
                    (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for n, tile in enumerate(tiles):
        y, x = divmod(n, cols)
        sheet[y * th:(y + 1) * th, x * tw:(x + 1) * tw] = tile
    cv2.imwrite(str(OUT / "calibration_selection.png"), sheet,
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"calibration: {len(views)} views, RMS {rmse:.4f} px")
    print(f"  fx={matrix[0, 0]:.2f} fy={matrix[1, 1]:.2f} "
          f"cx={matrix[0, 2]:.2f} cy={matrix[1, 2]:.2f}")
    print(f"  dist={np.round(distortion.ravel(), 5)}")
    print(f"  saved {OUT / 'calibration.npz'}")


def draw_text_box(image, lines, origin=(6, 6), color=(255, 255, 255),
                  scale=0.45, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    widths, heights = [], []
    for line in lines:
        (w, h), base = cv2.getTextSize(line, font, scale, thickness)
        widths.append(w)
        heights.append(h + base)
    box_w = max(widths) + 10
    box_h = sum(heights) + 6 * (len(lines) + 1)
    x0, y0 = origin
    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
    y = y0 + 6
    for line, h in zip(lines, heights):
        y += h
        cv2.putText(image, line, (x0 + 5, y), font, scale, color,
                    thickness, cv2.LINE_AA)


def draw_cube(image, pose: Pose, matrix, distortion, square_mm,
              thickness: int = 2):
    projected, _ = cv2.projectPoints(
        cube_corners(square_mm), pose.rvec, pose.tvec, matrix, distortion)
    p = projected.reshape(-1, 2)
    for n, (a, b) in enumerate(CUBE_EDGES):
        if n < 4:
            color = (0, 255, 0)      # base: green
        elif n < 8:
            color = (0, 255, 255)    # top: yellow
        else:
            color = (255, 255, 0)    # vertical edges: cyan
        cv2.line(image, tuple(np.round(p[a]).astype(int)),
                 tuple(np.round(p[b]).astype(int)), color, thickness,
                 cv2.LINE_AA)
    marker, _ = cv2.projectPoints(
        board_center(square_mm).reshape(1, 3), pose.rvec, pose.tvec,
        matrix, distortion)
    cv2.drawMarker(image, tuple(np.round(marker.ravel()).astype(int)),
                   (255, 0, 255), cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)


class VideoWriter:
    def __init__(self, path: Path, size: tuple[int, int], fps: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.size = size
        self.fps = fps
        self.use_ffmpeg = shutil.which("ffmpeg") is not None
        if self.use_ffmpeg:
            self.proc = subprocess.Popen([
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-",
                "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(path),
            ], stdin=subprocess.PIPE)
        else:
            self.writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            if not self.writer.isOpened():
                raise SystemExit("no ffmpeg and OpenCV cannot write video")

    def write(self, image):
        if self.use_ffmpeg:
            self.proc.stdin.write(image.tobytes())
        else:
            self.writer.write(image)

    def close(self):
        if self.use_ffmpeg:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise SystemExit("ffmpeg failed")
        else:
            self.writer.release()


def replay(args) -> None:
    matrix, distortion, size, square_mm = load_calibration(
        OUT / "calibration.npz")
    out = Path(args.output)
    writer = VideoWriter(out, size, 24.0)
    total = args.stop - args.start
    valid = invalid = 0
    try:
        for index, image in read_selected(args.start, args.stop):
            corners = detect_corners(image)
            pose = None
            if corners is not None:
                pose = solve_pose(corners, matrix, distortion, square_mm)
            if pose is not None and pose.valid:
                valid += 1
                draw_cube(image, pose, matrix, distortion, square_mm)
                distance = float(np.linalg.norm(pose.center))
                label = f"Board-centre distance: {distance:.1f} mm"
                color = (255, 255, 255)
            else:
                invalid += 1
                label = "POSE INVALID: cube hidden"
                color = (0, 0, 255)
                cv2.rectangle(image, (0, 0),
                              (image.shape[1] - 1, image.shape[0] - 1),
                              (0, 0, 255), 2)
            play_index = index - args.start
            draw_text_box(image, [
                label,
                f"frame {play_index} / {total - 1}   t={index / 24.0:.2f} s",
            ], color=color)
            writer.write(image)
    finally:
        writer.close()
    print(f"replay: {out} ({total} frames, {valid} valid, {invalid} invalid)")


def screenshot(index: int, corners: np.ndarray, pose: Pose, matrix,
               distortion, square_mm: float, output: Path, scale: int = 3):
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, image = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read frame {index}")
    big = cv2.resize(image, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_CUBIC)
    matrix_big = matrix.copy()
    matrix_big[0, 0] *= scale
    matrix_big[0, 2] *= scale
    matrix_big[1, 1] *= scale
    matrix_big[1, 2] *= scale
    draw_cube(big, pose, matrix_big, distortion, square_mm, thickness=4)
    distance = float(np.linalg.norm(pose.center))
    depth = float(pose.center[2])
    draw_text_box(big, [
        f"source frame {index}  t={index / 24.0:.1f} s",
        f"Z={depth:.1f} mm   D={distance:.1f} mm   "
        f"D-Z={distance - depth:.1f} mm",
        f"reprojection RMSE={pose.rmse_px:.2f} px",
    ], origin=(12, 12), color=(0, 255, 255), scale=0.85, thickness=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), big)


def measure_frame(index: int, matrix, distortion, square_mm: float):
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, image = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read frame {index}")
    corners = detect_corners(image)
    if corners is None:
        raise SystemExit(f"frame {index}: chessboard not found")
    pose = solve_pose(corners, matrix, distortion, square_mm)
    if pose is None or not pose.valid:
        raise SystemExit(f"frame {index}: invalid pose")
    return image, corners, pose


def measure(args) -> None:
    matrix, distortion, size, square_mm = load_calibration(
        OUT / "calibration.npz")
    checks = [int(v) for v in args.checks.split(",") if v.strip()]
    scale_frame = args.scale_frame
    rows = []
    for index in checks:
        image, corners, pose = measure_frame(
            index, matrix, distortion, square_mm)
        distance = float(np.linalg.norm(pose.center))
        depth = float(pose.center[2])
        rows.append((index, square_mm, depth, distance, distance - depth,
                     pose.rmse_px, "20 mm"))
        screenshot(index, corners, pose, matrix, distortion, square_mm,
                   OUT / f"acceptance_{index}_{index / 24.0:.1f}s.png")
        print(f"frame {index}: square={square_mm:.0f} mm "
              f"Z={depth:.3f} mm D={distance:.3f} mm "
              f"diff={distance - depth:.3f} mm rmse={pose.rmse_px:.3f} px")
    if scale_frame is not None:
        _, _, pose40 = measure_frame(
            scale_frame, matrix, distortion, 2 * square_mm)
        distance = float(np.linalg.norm(pose40.center))
        depth = float(pose40.center[2])
        rows.append((scale_frame, 2 * square_mm, depth, distance,
                     distance - depth, pose40.rmse_px, "40 mm"))
        print(f"frame {scale_frame}: square={2 * square_mm:.0f} mm "
              f"Z={depth:.3f} mm D={distance:.3f} mm "
              f"diff={distance - depth:.3f} mm rmse={pose40.rmse_px:.3f} px")

    table = Path(args.csv)
    with table.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["source_frame", "time_s", "square_size_mm",
                         "optical_depth_m", "straight_distance_m",
                         "distance_minus_depth_m", "reprojection_rmse_px"])
        for index, square, depth, distance, diff, rmse, _ in rows:
            writer.writerow([index, f"{index / 24.0:g}", f"{square:g}",
                             f"{depth / 1000:.6f}", f"{distance / 1000:.6f}",
                             f"{diff / 1000:.6f}", f"{rmse:.3f}"])
    print(f"measurements: {table}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("calibrate", help="select views and solve intrinsics")
    p.set_defaults(func=calibrate)

    p = sub.add_parser("replay", help="render a cube replay segment")
    p.add_argument("--start", type=int, default=168)
    p.add_argument("--stop", type=int, default=432,
                   help="exclusive source-frame upper bound")
    p.add_argument("--output", default=str(OUT / "replay.mp4"))
    p.set_defaults(func=replay)

    p = sub.add_parser("measure", help="acceptance frames and scale control")
    p.add_argument("--checks", default="180,420")
    p.add_argument("--scale-frame", type=int, default=420,
                   help="frame for the 40 mm square-size control; -1 disables")
    p.add_argument("--csv", default=str(ROOT / "task1" / "measurements.csv"))
    p.set_defaults(func=measure)

    args = parser.parse_args()
    if args.command == "measure" and args.scale_frame is not None \
            and args.scale_frame < 0:
        args.scale_frame = None
    args.func(args)


if __name__ == "__main__":
    main()
