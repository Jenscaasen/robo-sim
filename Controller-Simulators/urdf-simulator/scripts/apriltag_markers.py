import os
import sys
import json
from typing import List, Tuple, Optional

try:
    import pybullet as p
except ImportError as e:
    print("PyBullet is not installed. Activate your environment and pip install pybullet.", file=sys.stderr)
    raise

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Registry of placed AprilTags for calibration export
_TAGS_REGISTRY: List[dict] = []


def _parse_tag_id_from_filename(path: str) -> Optional[int]:
    """
    Extract an integer ID from a tag image filename like '002.jpg' -> 2.
    Returns None if no integer can be parsed.
    """
    base = os.path.splitext(os.path.basename(path))[0]
    digits = ''.join(ch for ch in base if ch.isdigit())
    try:
        return int(digits) if digits else None
    except Exception:
        return None


def _quat_to_rot_matrix(q: Tuple[float, float, float, float]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Convert quaternion to 3x3 rotation matrix using PyBullet helper.
    Returns columns (ex, ey, ez) as 3-tuples.
    """
    m = p.getMatrixFromQuaternion(q)
    # PyBullet returns row-major 3x3 as list of 9
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = m
    # Columns are the local axes in world frame
    ex = (r00, r10, r20)
    ey = (r01, r11, r21)
    ez = (r02, r12, r22)
    return ex, ey, ez


def _register_tag(tag_id: int,
                  center_pos: Tuple[float, float, float],
                  quat_wxyz: Tuple[float, float, float, float],
                  tag_size: float,
                  plane: str) -> None:
    """
    Register a tag in the global registry with precomputed 4 world corners
    in canonical order [top-left, top-right, bottom-right, bottom-left]
    with respect to the tag's local +x (to the right) and +y (down) axes.

    Local tag frame:
      - Origin at tag center
      - +z is the tag normal
      - +x to tag's right, +y to tag's down (for corner ordering consistency)
    """
    # Convert to rotation axes (world-frame unit vectors of tag's local axes)
    ex, ey, ez = _quat_to_rot_matrix(quat_wxyz)  # local axes in world
    cx, cy, cz = center_pos
    s = float(tag_size)
    hs = 0.5 * s

    # Define local corner offsets in canonical order (top-left, top-right, bottom-right, bottom-left)
    # Using +y down => top has -y, bottom has +y in local coordinates.
    local_offsets = [
        (-hs, -(-hs), 0.0),  # top-left -> (-hs, +hs, 0) keeping +y down convention
        ( hs, -(-hs), 0.0),  # top-right
        ( hs, -( hs), 0.0),  # bottom-right
        (-hs, -( hs), 0.0),  # bottom-left
    ]
    # Simplify the above numeric signs explicitly
    local_offsets = [(-hs, hs, 0.0), (hs, hs, 0.0), (hs, -hs, 0.0), (-hs, -hs, 0.0)]

    def add_scaled(v, a):
        return (v[0] * a, v[1] * a, v[2] * a)

    def vec_add(a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    corners_world: List[Tuple[float, float, float]] = []
    for ox, oy, oz in local_offsets:
        pw = vec_add(
            (cx, cy, cz),
            vec_add(
                vec_add(add_scaled(ex, ox), add_scaled(ey, oy)),
                add_scaled(ez, oz),
            ),
        )
        corners_world.append([float(pw[0]), float(pw[1]), float(pw[2])])

    _TAGS_REGISTRY.append({
        "id": int(tag_id),
        "center": [float(cx), float(cy), float(cz)],
        "quat_wxyz": [float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])],
        "size": float(tag_size),
        "plane": str(plane),
        "corners_world": corners_world,  # [tl, tr, br, bl]
    })


def clear_tags_registry() -> None:
    """Clear the in-memory tags registry."""
    _TAGS_REGISTRY.clear()


def write_tags_registry_json(out_path: str) -> None:
    """
    Persist the current tags registry to JSON for offline calibration.
    Schema:
    {
      "notes": "...",
      "count": N,
      "tags": [
        { "id": 180, "center": [x,y,z], "quat_wxyz": [x,y,z,w], "size": s, "plane": "table"|"wall",
          "corners_world": [[x,y,z] * 4 order [tl,tr,br,bl]] }
      ]
    }
    """
    payload = {
        "notes": "Auto-generated AprilTags world mapping for PnP calibration. corners_world order: [top-left, top-right, bottom-right, bottom-left].",
        "count": len(_TAGS_REGISTRY),
        "tags": _TAGS_REGISTRY,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def _find_default_tags_dir() -> str:
    candidates = [
        os.path.normpath(os.path.join(CURRENT_DIR, "..", "..", "..", "April-Tags", "April-Tags")),
        os.path.normpath(os.path.join(CURRENT_DIR, "..", "..", "April-Tags", "April-Tags")),
        os.path.normpath(os.path.join(CURRENT_DIR, "April-Tags", "April-Tags")),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None
def _list_image_files(directory: str) -> List[str]:
    if not directory or not os.path.isdir(directory):
        return []
    # ensure sorted order for determinism
    names = [n for n in os.listdir(directory) if n.lower().endswith((".png", ".jpg", ".jpeg"))]
    names.sort()
    return [os.path.join(directory, n) for n in names]


def list_apriltag_images(image_dir: Optional[str] = None) -> List[str]:
    """
    Public helper to list AprilTag image paths (sorted) from a directory.
    If image_dir is None, attempts to auto-detect the default tag folder.
    """
    if image_dir is None:
        image_dir = _find_default_tags_dir()
    return _list_image_files(image_dir) if image_dir else []


def _compute_positions_from_aabb(aabb_min: Tuple[float, float, float],
                                 aabb_max: Tuple[float, float, float],
                                 tag_size: float,
                                 margin: float) -> List[Tuple[float, float]]:
    x0, y0, _ = aabb_min
    x1, y1, _ = aabb_max
    half = tag_size / 2.0
    m = max(margin, half + 0.005)
    xc = 0.5 * (x0 + x1)
    yc = 0.5 * (y0 + y1)
    positions = [
        (x0 + m, y0 + m),
        (x0 + m, y1 - m),
        (x1 - m, y0 + m),
        (x1 - m, y1 - m),
        (xc, y0 + m),
        (xc, y1 - m),
        (x0 + m, yc),
        (x1 - m, yc),
        (xc, yc),
    ]
    return positions


def spawn_apriltags_on_table(table_body_id: int,
                              image_dir: str = None,
                              image_files: Optional[List[str]] = None,
                              tag_size: float = 0.07,
                              thickness: float = 0.005,
                              z_offset: float = 0.01,
                              margin: float = 0.02) -> List[int]:
    """
    Spawn 9 AprilTag markers as thin, visual-only squares on the table surface:
      - 4 corners
      - 4 edge midpoints
      - 1 center

    Also registers each tag (id parsed from filename) with world-space 3D corners.

    Returns a list of created body IDs.
    """
    if table_body_id is None or table_body_id < 0:
        raise ValueError("Invalid table body id")

    aabb_min, aabb_max = p.getAABB(table_body_id)
    top_z = aabb_max[2] + float(z_offset)

    # choose image set
    if image_files is not None:
        images = list(image_files)
    else:
        if image_dir is None:
            image_dir = _find_default_tags_dir()
        images = _list_image_files(image_dir) if image_dir else []
    if not images:
        raise FileNotFoundError(f"No AprilTag images found in: {image_dir}")

    # prepare geometry
    half_extents = [0.5 * tag_size, 0.5 * tag_size, 0.5 * thickness]

    positions_xy = _compute_positions_from_aabb(aabb_min, aabb_max, tag_size, margin)
    created_ids: List[int] = []

    # Ensure UNIQUE tags on the table: do not cycle textures if fewer images than positions
    count = min(len(images), len(positions_xy))
    for i in range(count):
        x, y = positions_xy[i]
        pos = [float(x), float(y), float(top_z)]
        quat = (0.0, 0.0, 0.0, 1.0)  # identity

        # Create a unique visual shape per marker
        vis_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[1.0, 1.0, 1.0, 1.0],
            specularColor=[0.0, 0.0, 0.0],
        )

        body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis_shape,
            basePosition=pos,
            baseOrientation=quat,
        )

        img_path = images[i]
        try:
            tex_id = p.loadTexture(img_path)
            p.changeVisualShape(body_id, -1, textureUniqueId=tex_id)
        except Exception as e:
            print(f"Warning: failed to apply texture {img_path}: {e}", file=sys.stderr)

        # Register this tag for calibration export
        tag_id = _parse_tag_id_from_filename(img_path)
        if tag_id is not None:
            _register_tag(tag_id=tag_id, center_pos=tuple(pos), quat_wxyz=tuple(quat), tag_size=float(tag_size), plane="table")
        else:
            print(f"Warning: could not parse tag id from filename {img_path}", file=sys.stderr)

        # Add a visible debug label above the tag to confirm presence in GUI
        try:
            label = f"Tag {tag_id}" if tag_id is not None else "Tag"
            label_pos = [pos[0], pos[1], pos[2] + max(tag_size * 0.6, 0.01)]
            p.addUserDebugText(label, label_pos, textColorRGB=[0, 0, 0], textSize=1.2, lifeTime=0)
        except Exception:
            pass

        # Console feedback
        try:
            print(f"Spawned table tag {tag_id} at {pos} using {os.path.basename(img_path)}")
        except Exception:
            pass

        created_ids.append(body_id)

    return created_ids


def _compute_wall_positions_from_table(aabb_min: Tuple[float, float, float],
                                       aabb_max: Tuple[float, float, float],
                                       tag_size: float,
                                       margin: float) -> List[Tuple[float, float]]:
    """
    Compute a 3x2 grid (6 positions) on the wall, aligned with the table's Y span and above the tabletop.
    Returns list of (y, z) pairs on the wall plane (x fixed).
    """
    _, y0, _ = aabb_min
    _, y1, top_z = aabb_max
    m = max(margin, tag_size / 2.0 + 0.005)
    yc = 0.5 * (y0 + y1)
    z1 = top_z + 0.15
    z2 = top_z + 0.35
    positions_yz = [
        (y0 + m, z1), (yc, z1), (y1 - m, z1),
        (y0 + m, z2), (yc, z2), (y1 - m, z2),
    ]
    return positions_yz


def spawn_apriltags_on_wall(wall_x: float,
                             table_body_id: int,
                             image_dir: str = None,
                             image_files: Optional[List[str]] = None,
                             tag_size: float = 0.07,
                             thickness: float = 0.005,
                             x_offset: float = 0.02,
                             margin: float = 0.02) -> List[int]:
    """
    Spawn AprilTag markers as thin squares on the wall plane at x=wall_x + x_offset.
    Positions are derived from the table AABB (y-span and height), forming a 3x2 grid.

    Also registers each tag (id parsed from filename) with world-space 3D corners.

    Returns list of created body IDs.
    """
    if table_body_id is None or table_body_id < 0:
        raise ValueError("Invalid table body id")

    aabb_min, aabb_max = p.getAABB(table_body_id)
    positions_yz = _compute_wall_positions_from_table(aabb_min, aabb_max, tag_size, margin)

    # choose image set
    if image_files is not None:
        images = list(image_files)
    else:
        if image_dir is None:
            image_dir = _find_default_tags_dir()
        images = _list_image_files(image_dir) if image_dir else []
    if not images:
        raise FileNotFoundError(f"No AprilTag images found in: {image_dir}")

    half_extents = [0.5 * tag_size, 0.5 * tag_size, 0.5 * thickness]
    face_out_q = p.getQuaternionFromEuler([0.0, 1.0 * 3.141592653589793 / 2.0, 0.0])  # rotate +90deg around Y => normal +X (faces into room)

    created_ids: List[int] = []
    count = min(len(images), len(positions_yz))
    x = float(wall_x) + float(x_offset)
    for i in range(count):
        y, z = positions_yz[i]
        pos = [x, float(y), float(z)]

        vis_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[1.0, 1.0, 1.0, 1.0],
            specularColor=[0.0, 0.0, 0.0],
        )

        body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis_shape,
            basePosition=pos,
            baseOrientation=face_out_q,
        )

        img_path = images[i]
        try:
            tex_id = p.loadTexture(img_path)
            p.changeVisualShape(body_id, -1, textureUniqueId=tex_id)
        except Exception as e:
            print(f"Warning: failed to apply texture {img_path}: {e}", file=sys.stderr)

        # Register this tag for calibration export
        tag_id = _parse_tag_id_from_filename(img_path)
        if tag_id is not None:
            _register_tag(tag_id=tag_id, center_pos=tuple(pos), quat_wxyz=tuple(face_out_q), tag_size=float(tag_size), plane="wall")
        else:
            print(f"Warning: could not parse tag id from filename {img_path}", file=sys.stderr)

        # Add a visible debug label to confirm presence
        try:
            label = f"Tag {tag_id}" if tag_id is not None else "Tag"
            label_pos = [pos[0] + max(tag_size * 0.6, 0.01), pos[1], pos[2]]
            p.addUserDebugText(label, label_pos, textColorRGB=[0, 0, 0], textSize=1.2, lifeTime=0)
        except Exception:
            pass

        # Console feedback
        try:
            print(f"Spawned wall tag {tag_id} at {pos} using {os.path.basename(img_path)}")
        except Exception:
            pass

        created_ids.append(body_id)

    return created_ids