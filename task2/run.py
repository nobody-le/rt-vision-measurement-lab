#!/usr/bin/env python3
"""Task 2: track A/B vehicles through blackout and compare estimators.

Estimators share one selected-observation sequence:

* direct: observation, missing during blackout;
* exponential smoothing: alpha=0.7, holds the last value when no observation;
* linear Kalman: constant-velocity state [x, y, vx, vy], predicts during
  blackout and updates on observations.

Two measurement-noise settings are run on case A (sigma_r = 4 px and 8 px,
i.e. R differs by 4x), the better recovery is selected, and the same setting
is used unchanged on case B.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.inputs import vehicle_case_files, vehicle_frames  # noqa: E402

OUT = ROOT / "results" / "task2"

EMA_ALPHA = 0.7
SIGMA_A = 800.0          # px/s^2, constant-velocity process noise
SIGMA_R = {1: 4.0, 2: 8.0}  # px; setting 2 has 2x std, 4x covariance
P0_POS = 5.0             # px
P0_VEL = 100.0           # px/s

SEED_INDEX = {"a": 3, "b": 2}

# Manual centres marked on the aligned clean video, in clip frame numbers.
MANUAL = {
    "a": {14: (864.0, 279.0), 18: (842.0, 238.0), 20: (834.0, 224.0)},
    "b": {12: (678.0, 262.0), 16: (677.0, 227.0), 18: (675.0, 205.0)},
}
CROPS = {
    "a": {14: (780, 220, 950, 340), 18: (770, 190, 910, 300),
          20: (770, 180, 900, 280)},
    "b": {12: (625, 220, 730, 310), 16: (625, 190, 730, 270),
          18: (625, 165, 725, 245)},
}


@dataclass
class Observation:
    frame: int
    time: float
    center: np.ndarray
    box: tuple[float, float, float, float]
    confidence: float
    index: int


def area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def select_observations(case: str, frames) -> dict[int, Observation]:
    """Pick the same vehicle before and after blackout.

    Candidates are gated by area continuity against the last accepted box;
    the remaining candidate closest to a constant-velocity prediction wins.
    This uses only current/past data and is independent of the R setting.
    """
    seed = SEED_INDEX[case]
    obs: dict[int, Observation] = {}
    last: Observation | None = None
    velocity = np.zeros(2)
    for tf in frames:
        if not tf.candidates:
            continue
        if last is None:
            index = min(seed, len(tf.candidates) - 1)
        else:
            gap = max(1, tf.index - last.frame)
            predicted = last.center + velocity * gap
            ratios = [area(d.xyxy) / area(last.box) for d in tf.candidates]
            valid = [i for i, ratio in enumerate(ratios)
                     if 0.18 <= ratio <= 2.0]
            if not valid:
                valid = list(range(len(tf.candidates)))
            distances = [np.linalg.norm(
                np.asarray(tf.candidates[i].center) - predicted) for i in valid]
            index = valid[int(np.argmin(distances))]
        detection = tf.candidates[index]
        current = Observation(
            tf.index, tf.time_s, np.asarray(detection.center, dtype=float),
            tuple(float(v) for v in detection.xyxy),
            float(detection.confidence), index)
        if last is not None:
            step = (current.center - last.center) / max(1, tf.index - last.frame)
            velocity = 0.5 * velocity + 0.5 * step
        obs[tf.index] = current
        last = current
    return obs


def ema_estimates(frames, obs: dict[int, Observation], alpha: float):
    out: dict[int, np.ndarray | None] = {}
    previous = None
    for tf in frames:
        if tf.index in obs:
            value = obs[tf.index].center
            previous = value.copy() if previous is None else (
                alpha * value + (1.0 - alpha) * previous)
        out[tf.index] = None if previous is None else previous.copy()
    return out


def kalman_estimates(frames, obs: dict[int, Observation], sigma_r: float):
    """Constant-velocity linear KF; returns prior, posterior and mode."""
    f0 = min(obs)
    z0 = obs[f0].center
    # Causal start: no future observation is used to set the initial velocity.
    # The second observation, when it actually arrives, updates the state and
    # therefore the velocity through the normal Kalman update.
    x = np.array([z0[0], z0[1], 0.0, 0.0], dtype=float)
    P = np.diag([P0_POS ** 2, P0_POS ** 2, P0_VEL ** 2, P0_VEL ** 2])
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    R = np.eye(2) * sigma_r ** 2

    prior: dict[int, np.ndarray] = {}
    post: dict[int, np.ndarray] = {}
    mode: dict[int, str] = {}
    prior[f0] = x[:2].copy()
    post[f0] = x[:2].copy()
    mode[f0] = "init"

    for previous, current in zip(frames, frames[1:]):
        if current.index <= f0:
            continue
        dt = current.time_s - previous.time_s
        F = np.array([[1.0, 0.0, dt, 0.0],
                      [0.0, 1.0, 0.0, dt],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
        Q = SIGMA_A ** 2 * np.array([
            [dt ** 4 / 4, 0.0, dt ** 3 / 2, 0.0],
            [0.0, dt ** 4 / 4, 0.0, dt ** 3 / 2],
            [dt ** 3 / 2, 0.0, dt ** 2, 0.0],
            [0.0, dt ** 3 / 2, 0.0, dt ** 2],
        ])
        x = F @ x
        P = F @ P @ F.T + Q
        prior[current.index] = x[:2].copy()
        if current.index in obs:
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ (obs[current.index].center - H @ x)
            P = (np.eye(4) - K @ H) @ P
            mode[current.index] = "update"
        else:
            mode[current.index] = "predict"
        post[current.index] = x[:2].copy()
    return prior, post, mode


def counts(frames, obs):
    # Segment by the contiguous blackout flags in the input.
    black = [tf.index for tf in frames if tf.blackout]
    if black:
        first, last = min(black), max(black)
        pre_n = sum(1 for tf in frames if tf.index < first)
        post_n = sum(1 for tf in frames if tf.index > last)
    else:
        pre_n = len(frames)
        post_n = 0
    return {
        "pre": pre_n,
        "blackout": len(black),
        "post": post_n,
        "direct_valid": len(obs),
        "ema_valid": len(frames),
        "kalman_valid": len(frames),
    }


def recovery_frame(frames, obs) -> int:
    """First observed frame after the last input blackout frame."""
    black = [tf.index for tf in frames if tf.blackout]
    if black:
        last_blackout = max(black)
        candidates = [tf.index for tf in frames
                      if tf.index > last_blackout and not tf.blackout
                      and tf.index in obs]
    else:
        candidates = [tf.index for tf in frames if tf.index in obs]
    if not candidates:
        raise SystemExit(
            "no observed frame after the blackout; cannot export recovery frame")
    return min(candidates)


def write_frame_table(path: Path, frames, obs, ema, post, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow([
            "frame", "time_s", "blackout",
            "observation_x_px", "observation_y_px",
            "direct_x_px", "direct_y_px",
            "ema_x_px", "ema_y_px",
            "kalman_x_px", "kalman_y_px", "kalman_mode",
        ])
        for tf in frames:
            f = tf.index
            o = obs.get(f)
            e = ema.get(f)
            k = post.get(f)
            writer.writerow([
                f, f"{tf.time_s:.6f}", int(tf.blackout),
                "" if o is None else f"{o.center[0]:.3f}",
                "" if o is None else f"{o.center[1]:.3f}",
                "" if o is None else f"{o.center[0]:.3f}",
                "" if o is None else f"{o.center[1]:.3f}",
                "" if e is None else f"{e[0]:.3f}",
                "" if e is None else f"{e[1]:.3f}",
                "" if k is None else f"{k[0]:.3f}",
                "" if k is None else f"{k[1]:.3f}",
                mode.get(f, ""),
            ])


def plot_case(case: str, frames, obs, ema, post, manual, setting: int,
              path: Path):
    fs = [tf.index for tf in frames]
    black = np.array([tf.blackout for tf in frames], dtype=bool)
    obs_x = np.array([obs[f].center[0] if f in obs else np.nan for f in fs])
    obs_y = np.array([obs[f].center[1] if f in obs else np.nan for f in fs])
    kx = np.array([post[f][0] for f in fs])
    ky = np.array([post[f][1] for f in fs])
    ex = np.array([ema[f][0] if ema[f] is not None else np.nan for f in fs])
    ey = np.array([ema[f][1] if ema[f] is not None else np.nan for f in fs])
    mx = np.array([manual[f][0] for f in manual])
    my = np.array([manual[f][1] for f in manual])
    black_idx = np.where(black)[0]
    first_black, last_black = black_idx[0], black_idx[-1]
    hold_idx = first_black - 1
    recovery = recovery_frame(frames, obs)

    zoom_ids = sorted(set(range(max(0, first_black - 2),
                                min(len(fs), last_black + 3))) | set(manual)
                      | {recovery})
    zx_values = list(kx[zoom_ids]) + list(mx) + [
        obs_x[recovery], ema[hold_idx][0]]
    zy_values = list(ky[zoom_ids]) + list(my) + [
        obs_y[recovery], ema[hold_idx][1]]
    if case == "b":
        zx0, zx1 = min(zx_values) - 40, max(zx_values) + 70
    else:
        zx0, zx1 = min(zx_values) - 12, max(zx_values) + 30
    zy0, zy1 = min(zy_values) - 12, max(zy_values) + 12

    fig, (ax_full, ax_zoom) = plt.subplots(
        1, 2, figsize=(13.5, 6.0), gridspec_kw={"width_ratios": [1.02, 1.0]})
    label_offsets = {
        "a": {14: (7, 12), 18: (-34, -12), 20: (8, -16)},
        "b": {12: (7, -12), 16: (7, -12), 18: (7, 12)},
    }

    def draw(ax, zoom: bool):
        ax.plot(obs_x, obs_y, "-o", color="tab:blue", lw=1.3, ms=4.2,
                label="direct (= selected observation)")
        ax.plot(ex, ey, "--", color="tab:orange", lw=2.0,
                label=f"EMA (alpha={EMA_ALPHA:g}, hold when missing)")
        ax.plot(ex, ey, "^", color="tab:orange", ms=3.0, alpha=0.55)
        ax.plot(kx, ky, "-", color="tab:green", lw=2.2,
                label=f"Kalman (setting {setting})")
        idx = np.where(black)[0]
        lo, hi = max(0, idx[0] - 1), min(len(fs) - 1, idx[-1] + 1)
        seg = np.arange(lo, hi + 1)
        ax.plot(kx[seg], ky[seg], "--", color="tab:green", lw=2.5)
        ax.plot(kx[black], ky[black], "s", ms=6.5, mfc="white",
                mec="tab:green", mew=1.7, label="Kalman pure prediction")
        ax.plot(mx, my, "D", color="magenta", ms=8, mfc="none", mew=1.8,
                label="manual reference centre")
        ax.plot(obs_x[0], obs_y[0], "k*", ms=12)
        ax.plot(obs_x[recovery], obs_y[recovery], "k^", ms=9)
        ax.invert_yaxis()
        ax.grid(True, ls=":", alpha=0.45)
        if zoom:
            for f, (x, y) in manual.items():
                dx, dy = label_offsets[case][f]
                label = f"f{f} ({x:.0f},{y:.0f})"
                if not frames[f].blackout:
                    label += " recovery"
                ax.annotate(label, (x, y),
                            xytext=(dx, dy), textcoords="offset points",
                            color="magenta", fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.15",
                                      fc="white", ec="none", alpha=0.75))
            if recovery not in manual:
                ax.annotate(f"recovery f{recovery}",
                            (obs_x[recovery], obs_y[recovery]),
                            xytext=(8, -16), textcoords="offset points",
                            fontsize=8, color="black")
            hold = ema[hold_idx]
            ax.plot(hold[0], hold[1], "s", ms=11, mfc="none",
                    mec="tab:orange", mew=2.2)
            ema_offset = (25, 18) if case == "b" else (-110, 34)
            ax.annotate(f"EMA holds f{hold_idx} value",
                        (hold[0], hold[1]), xytext=ema_offset,
                        textcoords="offset points", color="tab:orange",
                        fontsize=8,
                        arrowprops=dict(arrowstyle="->", color="tab:orange",
                                        lw=1))
            ax.text(0.03, 0.04, f"direct: missing\nf{first_black}-f{last_black}",
                    transform=ax.transAxes, fontsize=8, color="tab:blue",
                    va="bottom", bbox=dict(boxstyle="round,pad=0.2",
                                           fc="white", ec="none", alpha=0.8))
            mid = len(black_idx) // 2
            if case == "b":
                ax.annotate("KF pure prediction",
                            (kx[black][mid], ky[black][mid]),
                            xytext=(0.02, 0.80), textcoords="axes fraction",
                            color="tab:green", fontsize=8, ha="left",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                      ec="none", alpha=0.8),
                            arrowprops=dict(arrowstyle="->",
                                            color="tab:green", lw=1))
            else:
                ax.annotate("KF pure prediction",
                            (kx[black][mid], ky[black][mid]),
                            xytext=(-85, 36), textcoords="offset points",
                            color="tab:green", fontsize=8,
                            arrowprops=dict(arrowstyle="->",
                                            color="tab:green", lw=1))
        else:
            ax.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0,
                                   fill=False, ec="0.4", ls="--", lw=1))
            ax.annotate("zoom", (zx0, zy0), xytext=(4, -12),
                        textcoords="offset points", fontsize=8, color="0.3")
            ax.annotate("start f0", (obs_x[0], obs_y[0]), xytext=(8, 6),
                        textcoords="offset points", fontsize=8)

    draw(ax_full, False)
    draw(ax_zoom, True)
    for ax in (ax_full, ax_zoom):
        ax.set_xlabel("x / px")
        ax.set_ylabel("y / px (downwards)")
    ax_full.set_aspect("equal", adjustable="datalim")
    ax_full.set_title("Full trajectory")
    ax_zoom.set_xlim(zx0, zx1)
    ax_zoom.set_ylim(zy1, zy0)
    ax_zoom.set_aspect("equal", adjustable="box")
    ax_zoom.set_title(f"Blackout f{first_black}-f{last_black} and "
                      f"recovery f{recovery}")
    handles, labels = ax_zoom.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               framealpha=0.95, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"Case {case.upper()}: selected trajectory and estimators",
                 y=0.98)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.88, bottom=0.24,
                        wspace=0.18)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)

def recovery_screenshot(case: str, frame: int, observation: Observation,
                        prior: np.ndarray, posterior: np.ndarray,
                        setting: int, path: Path):
    folder = vehicle_case_files(case)
    cap = cv2.VideoCapture(str(folder / f"{case}_clean.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read {case} clean frame {frame}")
    scale = 2
    image = cv2.resize(image, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_CUBIC)

    x1, y1, x2, y2 = (int(round(v * scale)) for v in observation.box)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 4)
    cv2.putText(image, f"reconnected candidate #{observation.index} "
                       f"conf={observation.confidence:.2f}",
                (x1, max(24, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 255), 2, cv2.LINE_AA)

    lines = [
        f"case {case.upper()}   clip frame {frame}   "
        f"t={observation.time:.3f} s",
        f"setting {setting}:  pred=({prior[0]:.1f},{prior[1]:.1f})  "
        f"obs=({observation.center[0]:.1f},{observation.center[1]:.1f})  "
        f"post=({posterior[0]:.1f},{posterior[1]:.1f})",
        f"selected candidate box: "
        f"({observation.box[0]:.1f},{observation.box[1]:.1f})-"
        f"({observation.box[2]:.1f},{observation.box[3]:.1f})",
    ]
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (1250, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
    for n, line in enumerate(lines):
        cv2.putText(image, line, (20, 40 + 30 * n),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2,
                    cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def reference_montage(path: Path):
    tiles = []
    for case in ("a", "b"):
        folder = vehicle_case_files(case)
        cap = cv2.VideoCapture(str(folder / f"{case}_clean.mp4"))
        for frame in sorted(MANUAL[case]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            if not ok:
                raise SystemExit(f"cannot read {case} frame {frame}")
            x, y = MANUAL[case][frame]
            cv2.drawMarker(image, (int(x), int(y)), (0, 255, 255),
                           cv2.MARKER_CROSS, 18, 2)
            x0, y0, x1, y1 = CROPS[case][frame]
            crop = image[y0:y1, x0:x1].copy()
            cv2.rectangle(crop, (0, 0), (crop.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(crop,
                        f"{case.upper()} f{frame} ({x:.0f},{y:.0f})",
                        (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                        (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(cv2.resize(crop, None, fx=2, fy=2,
                                    interpolation=cv2.INTER_CUBIC))
        cap.release()
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    sheet = np.zeros((2 * h, 3 * w, 3), np.uint8)
    for n, tile in enumerate(tiles):
        row, col = divmod(n, 3)
        sheet[row * h:row * h + tile.shape[0],
              col * w:col * w + tile.shape[1]] = tile
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("a", "b", "all"), default="all")
    args = parser.parse_args()
    cases = ("a", "b") if args.case == "all" else (args.case,)

    data = {}
    errors = {}
    results = {}
    for case in cases:
        folder = vehicle_case_files(case)
        frames = vehicle_frames(case, folder)
        obs = select_observations(case, frames)
        ema = ema_estimates(frames, obs, EMA_ALPHA)
        data[case] = (frames, obs, ema)

    # Run both R settings on A, then apply the chosen one to B.  A B-only
    # rerun keeps the setting selected from A.
    if "a" in data:
        frames_a, obs_a, ema_a = data["a"]
        kf_a = {s: kalman_estimates(frames_a, obs_a, SIGMA_R[s])
                for s in (1, 2)}
        recovery_a = recovery_frame(frames_a, obs_a)
        if recovery_a in MANUAL["a"]:
            manual_a = MANUAL["a"][recovery_a]
            err_a = {s: float(np.linalg.norm(
                kf_a[s][1][recovery_a] - manual_a)) for s in (1, 2)}
            selected = min(err_a, key=err_a.get)
            print(f"A recovery f{recovery_a}: corrected error "
                  f"setting1={err_a[1]:.2f} px, setting2={err_a[2]:.2f} px "
                  f"-> setting {selected}")
        else:
            manual_a = None
            err_a = {}
            selected = 1
            print(f"A recovery f{recovery_a} is not a fixed manual centre; "
                  f"keeping setting 1")
    else:
        frames_a = obs_a = ema_a = kf_a = None
        recovery_a = None
        err_a = {}
        selected = 1

    jobs = []
    if "a" in data:
        jobs.append(("a", frames_a, obs_a, ema_a,
                     {selected: kf_a[selected]}))
    if "b" in data:
        frames_b, obs_b, ema_b = data["b"]
        jobs.append(("b", frames_b, obs_b, ema_b, {
            selected: kalman_estimates(frames_b, obs_b,
                                       SIGMA_R[selected])}))

    OUT.mkdir(parents=True, exist_ok=True)
    for case, frames, obs, ema, kf in jobs:
        prior, post, mode = kf[selected]
        write_frame_table(OUT / f"{case}_frames.csv", frames, obs, ema,
                          post, mode)
        plot_case(case, frames, obs, ema, post, MANUAL[case], selected,
                  OUT / f"{case}_trajectory.png")
        rec = recovery_frame(frames, obs)
        recovery_screenshot(case, rec, obs[rec], prior[rec], post[rec],
                            selected, OUT / f"{case}_recovery.png")
        errors[case] = {}
        for f, manual in MANUAL[case].items():
            o = obs.get(f)
            errors[case][f] = {
                "manual": manual,
                "stage": "blackout" if frames[f].blackout else "recovery",
                "direct": None if o is None else float(
                    np.linalg.norm(o.center - np.array(manual))),
                "ema": float(np.linalg.norm(ema[f] - np.array(manual))),
                "kalman": float(np.linalg.norm(post[f] - np.array(manual))),
            }
        results[case] = {
            "counts": counts(frames, obs),
            "setting": selected,
        }

    # Fill the two templates.
    ref = ROOT / "task2" / "reference-errors.csv"
    with ref.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["case", "frame", "stage", "manual_x_px",
                         "manual_y_px", "direct_error_px", "ema_error_px",
                         "kalman_error_px"])
        for case in cases:
            for f in sorted(MANUAL[case]):
                e = errors[case][f]
                writer.writerow([
                    case, f, e["stage"], f"{e['manual'][0]:.1f}",
                    f"{e['manual'][1]:.1f}", fmt(e["direct"]),
                    f"{e['ema']:.3f}", f"{e['kalman']:.3f}",
                ])

    if "a" in data and recovery_a in MANUAL["a"]:
        rec = ROOT / "task2" / "recovery-comparison.csv"
        with rec.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(["setting", "case", "frame", "pred_x_px",
                             "pred_y_px", "obs_x_px", "obs_y_px",
                             "corrected_x_px", "corrected_y_px",
                             "selected_for_b"])
            for setting in (1, 2):
                prior, post, _ = kf_a[setting]
                o = obs_a[recovery_a]
                writer.writerow([
                    setting, "a", recovery_a,
                    f"{prior[recovery_a][0]:.3f}",
                    f"{prior[recovery_a][1]:.3f}",
                    f"{o.center[0]:.3f}", f"{o.center[1]:.3f}",
                    f"{post[recovery_a][0]:.3f}",
                    f"{post[recovery_a][1]:.3f}",
                    1 if setting == selected else 0,
                ])

    reference_montage(OUT / "reference_centers.png")

    summary = {
        "ema_alpha": EMA_ALPHA,
        "sigma_a_px_s2": SIGMA_A,
        "sigma_r_px": SIGMA_R,
        "p0": {"pos_px": P0_POS, "vel_px_s": P0_VEL},
        "selected_setting": selected,
        "a_recovery_errors_px": err_a,
        "cases": results,
        "reference_errors": errors,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    for case in results:
        print(f"{case}: {results[case]['counts']}")
        for f, e in errors[case].items():
            print(f"  f{f} {e['stage']:8s} direct={fmt(e['direct']) or '-':>7s} "
                  f"ema={e['ema']:.2f} kalman={e['kalman']:.2f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
