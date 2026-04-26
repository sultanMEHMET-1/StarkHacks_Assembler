# Milestone 2: Cube Pose Fusion via solvePnP

## Context

Milestone 1 is complete. The system detects individual `tag16h5` AprilTags, computes per-tag 6DOF poses, and displays them as overlays on a live 1080p 60fps webcam feed. Camera calibration is working.

**This milestone's goal:** Fuse detections from multiple AprilTags on a rigid cube into a single 6DOF pose for the cube's center. This is the hand's position and orientation as far as the rest of the system is concerned.

**Approach:** Instead of averaging per-tag poses (lossy, tricky with rotations), we bypass the per-tag pose entirely. We take the raw 2D corner pixel positions from all detected tags, map them to known 3D positions in the cube's local frame, and run a single `cv2.solvePnP` call. This gives us the cube's pose directly from the most accurate data available (the corner detections).

**What stays the same:** The existing AprilTag detection pipeline is untouched. `pupil-apriltags` still detects tags and returns per-tag poses. The per-tag pose overlays remain for debugging. The cube pose is a new layer added on top of existing detections.

---

## What's Being Added

```
lib/
├── cube_model.py           # NEW: Cube geometry definition + 3D corner lookup
├── cube_pose.py            # NEW: solvePnP fusion logic
```

`detect.py` gets updated to call cube pose fusion after tag detection, and to draw the fused cube axes on the video feed in a distinct color.

No other existing files are modified structurally. `visualization.py` gets one new function for drawing the cube's fused pose axes.

---

## Cube Geometry: How This Works

The cube is a rigid body. Each face has a `tag16h5` AprilTag at a known position. Each tag has 4 corners. For a 6-faced cube, that's 24 corner points total, all at fixed 3D positions relative to the cube's center.

The key insight: when `pupil-apriltags` detects a tag, it gives you the 2D pixel coordinates of those 4 corners. You already know where those corners are in 3D (from your cube model). Feed all visible corners into `cv2.solvePnP` as a single batch, and it returns the cube's pose.

More visible tags = more corner points = more constrained solution = better accuracy.

### Defining the cube's local coordinate system

The cube's origin is its geometric center. Axes:

```
Cube-local frame (right-handed):

        ^ +Y (top face)
        |
        |
        +-------> +X (right face)
       /
      /
     v +Z (front face)
```

Each face is a square at a known offset from the center. For a cube with face size `S` (in meters), the 6 face centers are at:

- Front  (+Z): center at (0, 0, +S/2), normal pointing +Z
- Back   (-Z): center at (0, 0, -S/2), normal pointing -Z
- Right  (+X): center at (+S/2, 0, 0), normal pointing +X
- Left   (-X): center at (-S/2, 0, 0), normal pointing -X
- Top    (+Y): center at (0, +S/2, 0), normal pointing +Y
- Bottom (-Y): center at (0, -S/2, 0), normal pointing -Y

Each tag's 4 corners are positioned on that face, offset from the face center by half the tag size. The tag may be smaller than the face (it doesn't fill the entire face), so `tag_size` and `face_size` are independent parameters.

---

## Cube Model Definition (the part that must be easy to change)

File: `lib/cube_model.py`

This file defines the physical geometry of the cube. **All measurements go here and only here.** The user measures the real cube, edits the constants at the top of this file, and everything else adapts.

```python
"""
Cube geometry definition.

MODIFY THESE VALUES to match your physical cube.
Measure with calipers if possible. All units are meters.
"""

# === MEASURE AND UPDATE THESE ===

# Outer dimension of the cube (edge length), in meters.
# This is the distance from one face of the cube to the opposite face.
CUBE_FACE_SIZE = 0.030  # 30mm = ~1.2 inches

# Size of the AprilTag on each face, in meters.
# This is the outer dimension of the black border of the tag,
# NOT the size of the full cube face.
# The tag is typically smaller than the face. Measure the printed tag.
TAG_SIZE = 0.025  # 25mm — PLACEHOLDER, measure your actual tags

# Mapping of tag IDs to cube faces.
# Keys are the tag IDs printed on each face.
# Values are face names used internally.
# Update the IDs to match which physical tag you stuck on which face.
TAG_ID_TO_FACE = {
    0: "front",   # +Z face
    1: "back",    # -Z face
    2: "right",   # +X face
    3: "left",    # -X face
    4: "top",     # +Y face
    5: "bottom",  # -Y face
}

# === END OF USER-EDITABLE SECTION ===
```

Below the constants, the file has one function:

```python
def get_tag_corners_3d(tag_id: int) -> np.ndarray | None:
    """
    Return the 4 corner positions of the given tag in cube-local 3D coordinates.

    Returns a (4, 3) numpy array of [x, y, z] positions in meters,
    or None if the tag_id is not part of the cube.

    Corner order matches pupil-apriltags output:
    [bottom-left, bottom-right, top-right, top-left] when viewing the tag face-on.
    """
```

The function computes the 3D corner positions on the fly from `CUBE_FACE_SIZE`, `TAG_SIZE`, and the face assignment. It does NOT hardcode 24 individual corner coordinates. The logic is:

1. Look up which face the tag belongs to from `TAG_ID_TO_FACE`.
2. Compute the 4 corner positions as if the tag were on the front face (centered at the origin, lying in the XY plane).
3. Apply a rotation to move those corners to the correct face of the cube.
4. Apply a translation to offset them to the face's position (half of `CUBE_FACE_SIZE` along the face normal).

This way, changing `CUBE_FACE_SIZE` or `TAG_SIZE` automatically updates all 24 corner positions. No manual coordinate editing.

### Critical detail: corner ordering

`pupil-apriltags` returns tag corners in a specific order. The 3D model must use the same order or `solvePnP` will produce garbage. The convention is:

```
Viewing the tag face-on:

  corners[3] -------- corners[2]
      |                    |
      |     TAG FACE       |
      |                    |
  corners[0] -------- corners[1]

  [0] = bottom-left
  [1] = bottom-right
  [2] = top-right
  [3] = top-left
```

Verify this against the actual `pupil-apriltags` output for YOUR version of the library. Print the corner pixel positions for a tag held upright and confirm the ordering. If it's wrong, the solvePnP output will be wildly incorrect and hard to debug. This is a one-time check but it's essential.

---

## Cube Pose Estimation

File: `lib/cube_pose.py`

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class CubePose:
    translation: np.ndarray       # [x, y, z] meters, cube center in camera frame
    rotation_matrix: np.ndarray   # 3x3 rotation matrix
    num_tags_used: int            # how many tags contributed to this estimate
    reprojection_error: float     # RMS reprojection error in pixels
    tag_ids_used: list[int]       # which tags were visible


def estimate_cube_pose(
    detections: list,             # list of TagDetection from the existing detector
    camera_matrix: np.ndarray,    # 3x3 intrinsic matrix
    dist_coeffs: np.ndarray,      # distortion coefficients
) -> CubePose | None:
    """
    Fuse all detected AprilTags into a single cube pose.

    Returns None if no known cube tags are detected.
    """
```

The logic inside `estimate_cube_pose`:

1. For each detection, check if its `tag_id` is in `TAG_ID_TO_FACE`. Skip unknown IDs.
2. For each known tag, call `get_tag_corners_3d(tag_id)` to get the 3D points (4x3), and grab the 2D corners from the detection (4x2).
3. Stack all 3D points into one array, stack all 2D points into one array.
4. Call `cv2.solvePnP`:

```python
success, rvec, tvec = cv2.solvePnP(
    object_points_3d,    # (N*4, 3) float64
    image_points_2d,     # (N*4, 2) float64
    camera_matrix,
    dist_coeffs,
    flags=cv2.SOLVEPNP_ITERATIVE
)
```

5. If `success` is False, return None.
6. Convert `rvec` to a 3x3 rotation matrix via `cv2.Rodrigues(rvec)`.
7. Compute reprojection error: project the 3D points back using `cv2.projectPoints`, compute RMS distance to the original 2D points.
8. Return `CubePose`.

### Edge cases

**Only 1 tag visible:** `solvePnP` with 4 coplanar points from a single tag. This works but gives you two possible solutions (the pose ambiguity problem). `SOLVEPNP_ITERATIVE` usually picks the right one, but you should be aware this is the weakest case. The per-tag pose from `pupil-apriltags` is essentially the same solve, so you're not gaining anything with just 1 tag, but you're not losing anything either. Consistency matters: even with 1 tag, the output should go through the same path so the rest of the pipeline sees a uniform `CubePose` interface.

**2-3 tags visible:** This is where solvePnP shines. Non-coplanar points from different faces heavily constrain the solution. The pose ambiguity disappears and accuracy improves significantly.

**0 tags visible:** Return None. The caller handles this (skip the cube overlay, don't print cube pose).

**Unknown tag IDs detected:** Skip them. Could be a false positive or a stray tag in the scene.

---

## Changes to detect.py

The detection loop becomes:

```
capture frame
  -> convert to grayscale
  -> detect individual tags (existing)
  -> draw per-tag overlays (existing)
  -> estimate cube pose (NEW)
  -> draw cube pose overlay (NEW)
  -> print cube pose to terminal (NEW)
  -> display frame
```

### Terminal output

Add a cube pose line after the per-tag lines:

```
[Tag 02] Pos: (x=+0.052, y=-0.031, z=+0.347) m | Rot: (R:+2.1, P:-5.3, Y:+12.7) deg
[Tag 04] Pos: (x=+0.048, y=-0.029, z=+0.351) m | Rot: (R:+1.8, P:-4.9, Y:+13.1) deg
[CUBE]   Pos: (x=+0.050, y=-0.030, z=+0.349) m | Rot: (R:+2.0, P:-5.1, Y:+12.9) deg | tags: 2 | err: 0.31px
```

The `[CUBE]` line shows the fused pose, the number of tags used, and the reprojection error. This lets you immediately see whether the fused pose is reasonable compared to the individual tags.

### Visualization

Add to `lib/visualization.py`:

```python
def draw_cube_axes(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_length=0.03):
    """
    Draw thick RGB axes at the cube's center. Same convention as tag axes
    (X=red, Y=green, Z=blue) but thicker lines and longer axes so they're
    visually distinct from the per-tag axes.
    """
```

Use thicker lines (3-4px vs 2px for per-tag axes) and a longer axis length so the cube pose is visually distinct from the individual tag poses.

### CLI changes

Add to `detect.py`:

```
--no-cube          Disable cube pose fusion (show only per-tag poses)
```

No other new CLI args. The cube geometry is edited in `lib/cube_model.py`, not via CLI.

---

## Implementation Order

### Phase 1: Cube model

File: `lib/cube_model.py`

Write the geometry definition with the editable constants at the top. Implement `get_tag_corners_3d()`. Verify by printing the output for each tag ID and sanity-checking the coordinates by hand. For a 30mm cube with 25mm tags:

- Tag 0 (front, +Z face): corners should be at z = +0.015, x and y within ±0.0125
- Tag 2 (right, +X face): corners should be at x = +0.015, y and z within ±0.0125

If these numbers don't make sense for your cube, the geometry is wrong.

### Phase 2: Cube pose estimation

File: `lib/cube_pose.py`

Implement `estimate_cube_pose()`. Wire it into `detect.py` after the existing detection. Print the cube pose to terminal alongside the per-tag poses. At this point you can verify the numbers make sense even without the overlay.

### Phase 3: Visualization

Add `draw_cube_axes()` to `visualization.py`. Call it from `detect.py` when a cube pose is available. Verify visually: the cube axes should appear at the cube's center, and they should be stable when you hold the cube still. They should also roughly agree with the per-tag axes.

### Phase 4: Validation

This is the important part. With the cube in front of the camera:

- Rotate the cube slowly. The cube pose should track smoothly.
- Show 1 tag, then rotate to show 2, then 3. The pose should become more stable (less jitter) as more tags become visible. If it jumps when a new tag appears, the cube geometry or corner ordering is wrong.
- Check the reprojection error. With good calibration and correct geometry, it should be under 1.0 pixels. If it's consistently above 2-3 pixels, something is off (wrong tag size, wrong face size, wrong corner order).
- Hold the cube still and log 100 frames of cube pose. Compute the standard deviation of x, y, z. This is your noise floor. You'll need this number for later milestones.

---

## Verification Checklist

- [ ] `cube_model.py` constants are at the top, clearly labeled, with comments about units
- [ ] Changing `CUBE_FACE_SIZE` or `TAG_SIZE` updates all corner positions automatically
- [ ] `TAG_ID_TO_FACE` mapping matches the physical cube
- [ ] `get_tag_corners_3d()` corner order matches `pupil-apriltags` corner order
- [ ] `estimate_cube_pose()` returns None when 0 known tags are detected
- [ ] `estimate_cube_pose()` works with 1, 2, or 3 visible tags
- [ ] Unknown tag IDs are silently skipped
- [ ] Terminal shows both per-tag poses and fused cube pose
- [ ] Cube axes are visually distinct from per-tag axes (thicker, longer)
- [ ] Reprojection error is displayed and is < 1.0px with correct geometry
- [ ] Pose doesn't jump when a new tag rotates into view
- [ ] `--no-cube` flag disables cube fusion

---

## What Is NOT In Scope for This Milestone

- Filtering or smoothing the cube pose (Kalman filter, etc.)
- Workspace calibration (camera-to-robot transform)
- IMU fusion
- Recording pose data
- Any robot arm communication