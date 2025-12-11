import cv2
import numpy as np


def pad_with_border(img, pad_ratio=0.08, color=(255, 255, 255)):
    """Add a constant border around the image."""
    h, w = img.shape[:2]
    pad_y = int(h * pad_ratio)
    pad_x = int(w * pad_ratio)
    padded = cv2.copyMakeBorder(
        img,
        pad_y, pad_y, pad_x, pad_x,
        borderType=cv2.BORDER_CONSTANT,
        value=color
    )
    return padded


def cyl_curl_vertical(
    img,
    k=0.3,
    p=2.5,
    border_value=(255, 255, 255),
    interpolation=cv2.INTER_LINEAR
):
    """Vertical curl: top and bottom curl inward."""
    h, w = img.shape[:2]
    cx = w / 2.0
    cy = h / 2.0

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    v = (y - cy) / cy
    if cy == 0:
        v = np.zeros_like(y)

    s = 1.0 - k * (np.abs(v) ** p)
    map_x = cx + (x - cx) * s
    map_y = y

    warped = cv2.remap(
        img,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )
    return warped


def cyl_curl_horizontal(
    img,
    k=0.3,
    p=2.5,
    border_value=(255, 255, 255),
    interpolation=cv2.INTER_LINEAR
):
    """Horizontal curl: left and right curl inward."""
    h, w = img.shape[:2]
    cx = w / 2.0
    cy = h / 2.0

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    u = (x - cx) / cx
    if cx == 0:
        u = np.zeros_like(x)

    s = 1.0 - k * (np.abs(u) ** p)
    map_x = x
    map_y = cy + (y - cy) * s

    warped = cv2.remap(
        img,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )
    return warped


def add_folds_shading(img, num_folds=2, max_strength=0.1, width_range=(10, 25), border_softness=1.0):
    h, w = img.shape[:2]
    result = img.astype(np.float32) / 255.0

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)

    light_dir = np.array([-1.0, -1.2])
    light_dir = light_dir / np.linalg.norm(light_dir)

    fold_mask = np.zeros((h, w), dtype=np.float32)

    for _ in range(num_folds):
        x1, y1 = np.random.uniform(0, w), np.random.uniform(0, h)
        x2, y2 = np.random.uniform(0, w), np.random.uniform(0, h)
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy) + 1e-6
        nx, ny = -dy / length, dx / length
        dist = (xx - x1) * nx + (yy - y1) * ny
        width = np.random.uniform(*width_range)
        profile = np.exp(-(np.abs(dist) ** border_softness) / (2 * width * width))
        normal = np.array([nx, ny])
        ndotl = np.dot(normal, light_dir)
        strength = max_strength * ndotl
        fold_mask += (strength * profile).astype(np.float32)

    fold_mask = np.clip(fold_mask, -max_strength, max_strength)
    fold_mask = fold_mask[..., None]
    result = result * (1.0 + fold_mask)
    result = np.clip(result, 0.0, 1.0)
    return (result * 255).astype(np.uint8)


def halftone_like(img, block_size=2):
    if block_size <= 1:
        return img
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // block_size), max(1, h // block_size)), interpolation=cv2.INTER_AREA)
    halftoned = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return halftoned


def add_paper_texture(img, strength=0.1):
    h, w = img.shape[:2]
    noise = np.random.rand(h, w).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=15, sigmaY=15)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
    texture = 1.0 + (noise - 0.5) * 2.0 * strength
    texture = texture[..., None]
    out = img.astype(np.float32) / 255.0
    out = out * texture
    out = np.clip(out, 0.0, 1.0)
    return (out * 255).astype(np.uint8)


def slight_color_shift(img):
    """
    HSV color shift with saturation/brightness-coupled darkening and RGB noise.
    f(S) = 0.8 - X*S where X ~ Uniform(0.2, 0.6), applied to V: V' = V * f(S)
    H: shift in (-8°, 2°) => OpenCV H units (-4, 1)
    S': S' = S * (Y + (1-Y)*(1-S)) where Y ~ Uniform(0.55, 0.85)
    After HSV->BGR, add per-channel uniform noise in (-10, 10)
    """
    if img.ndim != 3 or img.shape[2] != 3:
        return img

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    h_shift = np.random.uniform(-4.0, 1.0)
    h = (h + h_shift) % 180.0

    s_norm = s / 255.0
    s_norm_orig = s_norm.copy()

    Y = np.random.uniform(0.55, 0.85)
    s_scale = Y + (1.0 - Y) * (1.0 - s_norm_orig)
    s_norm_new = s_norm_orig * s_scale
    s = np.clip(s_norm_new * 255.0, 0, 255)

    X = np.random.uniform(0.2, 0.6)
    f_s = 0.8 - X * s_norm_orig
    f_s = np.clip(f_s, 0.0, 1.0)
    v = np.clip(v * f_s, 0, 255)

    hsv_merged = cv2.merge([h, s, v])
    bgr = cv2.cvtColor(hsv_merged.astype(np.uint8), cv2.COLOR_HSV2BGR)

    bgr_f = bgr.astype(np.float32)
    noise = np.random.uniform(-10.0, 10.0, size=bgr.shape).astype(np.float32)
    bgr_noisy = np.clip(bgr_f + noise, 0, 255).astype(np.uint8)

    return bgr_noisy


def slight_blur(img, ksize=3):
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    if ksize == 1:
        return img
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def rotate_image(img, angle_deg, border_value=(255, 255, 255), interpolation=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value
    )
    return rotated, M


def transform_points_geometric(points, img_shape, pad_ratio, curl_mode, curl_amount, curl_flatness, rot_angle):
    """Apply geometric transforms to a list of (x,y) coordinates with numerical stability checks."""
    if len(points) == 0:
        return points

    pts = points.astype(np.float64)

    if not np.isfinite(pts).all():
        pts[~np.isfinite(pts)] = -10000.0

    h, w = img_shape
    pad_y = int(h * pad_ratio)
    pad_x = int(w * pad_ratio)

    pts[:, 0] += pad_x
    pts[:, 1] += pad_y

    h_pad = h + 2 * pad_y
    w_pad = w + 2 * pad_x
    cx = w_pad / 2.0
    cy = h_pad / 2.0

    # Clamp normalized coordinates to [-1, 1] to prevent singularity
    if curl_mode == "vertical":
        v = (pts[:, 1] - cy) / cy
        v = np.clip(v, -1.0, 1.0)
        s = 1.0 - curl_amount * (np.abs(v) ** curl_flatness)
        pts[:, 0] = cx + (pts[:, 0] - cx) / s
    elif curl_mode == "horizontal":
        u = (pts[:, 0] - cx) / cx
        u = np.clip(u, -1.0, 1.0)
        s = 1.0 - curl_amount * (np.abs(u) ** curl_flatness)
        pts[:, 1] = cy + (pts[:, 1] - cy) / s

    mask_bad = ~np.isfinite(pts).all(axis=1) | (np.abs(pts) > 1e7).any(axis=1)
    valid_indices = np.where(~mask_bad)[0]

    if len(valid_indices) == 0:
        return pts

    pts_valid = pts[valid_indices]

    if rot_angle != 0.0:
        M = cv2.getRotationMatrix2D((cx, cy), rot_angle, 1.0)
        pts_reshaped = pts_valid.reshape(-1, 1, 2)
        pts_transformed = cv2.transform(pts_reshaped, M)
        pts_valid = pts_transformed.reshape(-1, 2)

    pts[valid_indices] = pts_valid

    if mask_bad.any():
        pts[mask_bad] = -10000.0

    return pts.astype(np.float32)


def random_paper_augmentation_for_unet(
    img,
    mask,
    grid_points,
    pad_ratio=0.08,
    max_rotation_deg=3.0
):
    """Apply augmentation to Image, Mask (3-channel), and Grid Points."""
    orig_h, orig_w = img.shape[:2]

    img_out = pad_with_border(img, pad_ratio=pad_ratio, color=(255, 255, 255))
    mask_out = pad_with_border(mask, pad_ratio=pad_ratio, color=(0, 0, 0))

    pad_h, pad_w = img_out.shape[:2]

    curl_mode = np.random.choice(["vertical", "horizontal", None])
    curl_amount = np.random.uniform(0.05, 0.15)
    curl_flatness = 3.0

    if curl_mode == "vertical":
        img_out = cyl_curl_vertical(img_out, k=curl_amount, p=curl_flatness, border_value=(255, 255, 255))
        mask_out = cyl_curl_vertical(mask_out, k=curl_amount, p=curl_flatness, border_value=(0, 0, 0),
                                     interpolation=cv2.INTER_NEAREST)
    elif curl_mode == "horizontal":
        img_out = cyl_curl_horizontal(img_out, k=curl_amount, p=curl_flatness, border_value=(255, 255, 255))
        mask_out = cyl_curl_horizontal(mask_out, k=curl_amount, p=curl_flatness, border_value=(0, 0, 0),
                                       interpolation=cv2.INTER_NEAREST)

    angle = np.random.uniform(-max_rotation_deg, max_rotation_deg)

    img_out, _ = rotate_image(img_out, angle_deg=angle, border_value=(255, 255, 255))
    mask_out, _ = rotate_image(mask_out, angle_deg=angle, border_value=(0, 0, 0), interpolation=cv2.INTER_NEAREST)

    transformed_pts = transform_points_geometric(
        grid_points,
        (orig_h, orig_w),
        pad_ratio,
        curl_mode,
        curl_amount,
        curl_flatness,
        angle
    )

    aug_grid_mask = np.zeros((pad_h, pad_w), dtype=np.uint8)

    valid_mask = (
        (transformed_pts[:, 0] >= 0) &
        (transformed_pts[:, 0] < pad_w) &
        (transformed_pts[:, 1] >= 0) &
        (transformed_pts[:, 1] < pad_h)
    )
    valid_pts = transformed_pts[valid_mask]

    if len(valid_pts) > 0:
        pts_int = np.round(valid_pts).astype(np.int32)
        aug_grid_mask[pts_int[:, 1], pts_int[:, 0]] = 255

    texture_strength = np.random.uniform(0.0, 0.35)
    if texture_strength > 0.0:
        img_out = add_paper_texture(img_out, strength=texture_strength)

    img_out = slight_color_shift(img_out)

    blur_ksize = np.random.choice([0, 1, 2])
    img_out = slight_blur(img_out, ksize=blur_ksize)

    return img_out, mask_out, aug_grid_mask
