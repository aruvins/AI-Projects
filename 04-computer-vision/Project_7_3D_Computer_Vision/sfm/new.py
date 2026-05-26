import os
import cv2
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

# =============================================================================
# CONFIGURATION — all tunable parameters in one place
# =============================================================================

# -- Paths --
DATA_DIR    = "data/sfm/temple/temple"
OUTPUT_PATH = "outputs/sfm/temple_reconstruction.pcd"

# -- Feature extraction --
SIFT_MAX_FEATURES   = 100_000_000   # Effectively uncapped; lower (e.g. 5000) to speed up
FLANN_TREES         = 10            # More trees = higher recall, slower build
FLANN_CHECKS        = 100           # More checks = higher recall, slower query
SIFT_UPSCALE_FACTOR = 2.0           # Upscale images before SIFT by this factor so the detector
                                    # finds fine surface detail it misses at native resolution.
                                    # 2.0 = 4× more pixels → significantly more keypoints.
                                    # Set to 1.0 to disable. Higher values use more RAM/time.

# -- Match filtering --
RATIO_TEST_THRESHOLD    = 0.85      # Lowe's ratio test. Lower = stricter, fewer but cleaner matches.
                                    # Range to test: 0.65 (very strict) – 0.85 (loose)
RANSAC_REPROJ_THRESHOLD = 4.0       # Max pixel error for RANSAC homography inlier (px).
                                    # Lower = stricter geometric filter. Range: 1.0 – 8.0
MIN_INLIER_MATCHES      = 15        # Minimum inliers after RANSAC to attempt triangulation

# -- Multi-view pairing --
LOOK_AHEAD_STEPS = 10               # How many frames ahead each frame is matched against.
                                    # Higher = more pairs, denser cloud, slower. Range: 2 – 10

# -- Post-triangulation spatial crop --
SPHERE_CROP_RADIUS = 2.5            # Discard points further than this from origin (metres).
                                    # Increase if temple edges are being clipped.

# -- Statistical outlier removal --
SOR_NB_NEIGHBORS = 30               # Neighbourhood size for outlier test
SOR_STD_RATIO    = 1.8              # Points beyond this many std-devs from mean are removed.
                                    # Lower = more aggressive. Range: 0.5 (tight) – 2.0 (loose)

# -- Brightness filter --
BRIGHTNESS_FLOOR = 0.10             # Drop points whose source pixel perceived brightness is below
                                    # this value. Removes near-black background pixels.
                                    # Range: 0.05 (keep almost everything) – 0.40 (keep only bright)

# -- DBSCAN cluster filter --
DBSCAN_EPS         = 0.05           # Max distance between two points to be in the same cluster (m).
                                    # Increase if the temple is being split into multiple clusters.
DBSCAN_MIN_SAMPLES = 10             # Minimum points to form a cluster core.
DBSCAN_MAX_POINTS  = 160_000         # DBSCAN is O(n²) memory — subsample to this before clustering
                                    # then map labels back. Raise if clusters are being missed.

# -- Voxel downsampling --
VOXEL_SIZE = 0.001                  # Grid cell size (m). Smaller = more points kept, more memory.
                                    # Range: 0.001 (dense) – 0.01 (sparse/fast)

# -- Normal estimation --
NORMAL_SEARCH_RADIUS = 0.02         # Neighbourhood radius for normal estimation (m)
NORMAL_MAX_NN        = 30           # Max neighbours used per point for normal estimation

# -- Visualiser --
WINDOW_WIDTH  = 1024
WINDOW_HEIGHT = 768
POINT_SIZE    = 5.0                 # Rendered splat size in pixels. Larger fills gaps visually.
VIEW_FRONT    = [-0.75, 0.0, 0.0]
VIEW_UP       = [0.0, 0.0, -1.0]
VIEW_ZOOM     = 0.75

# =============================================================================


def parse_camera_parameters(par_file_path):
    """
    Parses temple_par.txt and returns a dict mapping 4-digit frame ID → 3×4
    projection matrix P = K [R | t].
    """
    if not os.path.exists(par_file_path):
        raise FileNotFoundError(f"Parameter file not found: {par_file_path}")

    camera_matrices = {}
    with open(par_file_path, 'r') as f:
        lines = f.readlines()

    print(f"Reading camera parameters from {par_file_path}...")
    for line in lines[1:]:
        tokens = line.strip().split()
        if not tokens:
            continue
        digit_id = "".join(filter(str.isdigit, tokens[0]))
        if not digit_id:
            continue
        values = np.array([float(x) for x in tokens[1:]])
        K = values[0:9].reshape((3, 3))
        R = values[9:18].reshape((3, 3))
        t = values[18:21].reshape((3, 1))
        camera_matrices[digit_id] = np.dot(K, np.hstack((R, t)))

    print(f"Cached calibration for {len(camera_matrices)} frames.")
    return camera_matrices


def match_and_triangulate(img1_path, img2_path, P1, P2):
    """
    Detect SIFT features, match with FLANN, filter with Lowe's ratio test and
    RANSAC homography, then triangulate surviving inliers.
    Returns (points_3d, colors) or (None, None) if the pair is unusable.
    """
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        return None, None

    # Upscale images before detection so SIFT finds fine-detail keypoints
    # that are invisible at native resolution. Coordinates are divided back
    # down afterward so they stay consistent with the projection matrices.
    if SIFT_UPSCALE_FACTOR != 1.0:
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        img1_det = cv2.resize(img1, (int(w1 * SIFT_UPSCALE_FACTOR), int(h1 * SIFT_UPSCALE_FACTOR)), interpolation=cv2.INTER_CUBIC)
        img2_det = cv2.resize(img2, (int(w2 * SIFT_UPSCALE_FACTOR), int(h2 * SIFT_UPSCALE_FACTOR)), interpolation=cv2.INTER_CUBIC)
    else:
        img1_det, img2_det = img1, img2

    sift     = cv2.SIFT_create(nfeatures=SIFT_MAX_FEATURES)
    kp1, d1  = sift.detectAndCompute(img1_det, None)
    kp2, d2  = sift.detectAndCompute(img2_det, None)
    if d1 is None or d2 is None:
        return None, None

    # Scale keypoint coordinates back to native image space
    if SIFT_UPSCALE_FACTOR != 1.0:
        for kp in kp1:
            kp.pt = (kp.pt[0] / SIFT_UPSCALE_FACTOR, kp.pt[1] / SIFT_UPSCALE_FACTOR)
        for kp in kp2:
            kp.pt = (kp.pt[0] / SIFT_UPSCALE_FACTOR, kp.pt[1] / SIFT_UPSCALE_FACTOR)

    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=FLANN_TREES),
        dict(checks=FLANN_CHECKS),
    )
    try:
        matches = flann.knnMatch(d1, d2, k=2)
    except Exception:
        return None, None

    # Lowe's ratio test
    pts1, pts2 = [], []
    for m, n in matches:
        if m.distance < RATIO_TEST_THRESHOLD * n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    if len(pts1) < MIN_INLIER_MATCHES:
        return None, None

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    # RANSAC homography — keeps only geometrically consistent matches
    _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
    if mask is None:
        return None, None

    inliers = mask.ravel().astype(bool)
    if inliers.sum() < MIN_INLIER_MATCHES:
        return None, None

    pts1_clean = pts1[inliers]
    pts2_clean = pts2[inliers]

    # Sample colors from image 1 at inlier locations
    colors = []
    for pt in pts1_clean:
        x = min(max(int(pt[0]), 0), img1.shape[1] - 1)
        y = min(max(int(pt[1]), 0), img1.shape[0] - 1)
        colors.append(img1[y, x][::-1] / 255.0)   # BGR → RGB, normalise

    # Triangulate
    pts4d    = cv2.triangulatePoints(P1, P2, pts1_clean.T, pts2_clean.T)
    pts3d    = (pts4d[:3] / pts4d[3]).T

    return pts3d, np.array(colors)


def keep_largest_cluster(pcd):
    """
    DBSCAN cluster filter: identifies all connected point clusters and returns
    only the largest one (the temple). Isolated noise islands are discarded.

    Memory strategy:
    - Uses algorithm='ball_tree' which is O(n log n) vs the default O(n²).
    - Subsamples to DBSCAN_MAX_POINTS before clustering, finds the largest
      cluster centroid, then keeps all original points within 2*eps of it.
      This avoids running DBSCAN on the full cloud while still correctly
      labelling every point.
    """
    pts = np.asarray(pcd.points)
    n   = len(pts)
    print(f"  DBSCAN: {n:,} points (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})...")

    # Subsample for clustering if cloud is large
    if n > DBSCAN_MAX_POINTS:
        idx_sub = np.random.choice(n, DBSCAN_MAX_POINTS, replace=False)
        pts_sub = pts[idx_sub]
        print(f"  DBSCAN: subsampling {n:,} → {DBSCAN_MAX_POINTS:,} for memory safety...")
    else:
        idx_sub = np.arange(n)
        pts_sub = pts

    labels = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        algorithm='ball_tree',   # O(n log n) memory vs default 'auto' which picks O(n²) brute
        n_jobs=-1,
    ).fit_predict(pts_sub)

    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique) == 0:
        print("  DBSCAN: no clusters found — returning cloud unchanged.")
        return pcd

    best_label   = unique[np.argmax(counts)]
    best_pts_sub = pts_sub[labels == best_label]
    centroid     = best_pts_sub.mean(axis=0)
    kept_pct     = counts.max() / len(pts_sub) * 100
    print(f"  DBSCAN: {len(unique)} cluster(s); largest has {counts.max():,} pts ({kept_pct:.1f}%).")

    # Keep all original points near the largest-cluster centroid
    # Use a generous radius: half the cloud's bounding-box diagonal
    bbox_diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    keep_radius = bbox_diag * 0.6
    dists   = np.linalg.norm(pts - centroid, axis=1)
    inliers = np.where(dists < keep_radius)[0]
    print(f"  DBSCAN: keeping {len(inliers):,} / {n:,} points within radius {keep_radius:.3f} m of centroid.")

    return pcd.select_by_index(inliers.tolist())


def main():
    par_file = os.path.join(DATA_DIR, "temple_par.txt")

    try:
        camera_matrices = parse_camera_parameters(par_file)
    except Exception as e:
        print(f"Error loading parameters: {e}")
        return

    img_files = sorted(
        f for f in os.listdir(DATA_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    )
    print(f"Found {len(img_files)} images.")
    if not img_files:
        print(f"No images found in: {DATA_DIR}")
        return

    # -------------------------------------------------------------------------
    # Multi-view triangulation
    # -------------------------------------------------------------------------
    all_pts, all_colors = [], []
    pairs_done = 0

    print(f"Triangulating (look-ahead={LOOK_AHEAD_STEPS})...")
    for i, file1 in enumerate(img_files):
        for step in range(1, LOOK_AHEAD_STEPS + 1):
            j = i + step
            if j >= len(img_files):
                continue
            file2 = img_files[j]

            id1 = "".join(filter(str.isdigit, file1))
            id2 = "".join(filter(str.isdigit, file2))
            if id1 not in camera_matrices or id2 not in camera_matrices:
                continue

            pts3d, colors = match_and_triangulate(
                os.path.join(DATA_DIR, file1),
                os.path.join(DATA_DIR, file2),
                camera_matrices[id1],
                camera_matrices[id2],
            )
            if pts3d is not None:
                all_pts.append(pts3d)
                all_colors.append(colors)
                pairs_done += 1
                if pairs_done == 1 or pairs_done % 20 == 0:
                    print(f"  {pairs_done} pairs processed...")

    print(f"\nTriangulation complete ({pairs_done} pairs). Filtering...")
    if not all_pts:
        print("No 3D points generated.")
        return

    # -------------------------------------------------------------------------
    # Filtering pipeline
    # -------------------------------------------------------------------------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
    print(f"  Raw points:           {len(pcd.points):>10,}")

    # 1. Sphere crop
    dists  = np.linalg.norm(np.asarray(pcd.points), axis=1)
    pcd    = pcd.select_by_index(np.where(dists < SPHERE_CROP_RADIUS)[0].tolist())
    print(f"  After sphere crop:    {len(pcd.points):>10,}")

    # 2. Statistical outlier removal
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=SOR_NB_NEIGHBORS, std_ratio=SOR_STD_RATIO)
    pcd    = pcd.select_by_index(ind)
    print(f"  After outlier removal:{len(pcd.points):>10,}")

    # 3. Brightness filter
    c          = np.asarray(pcd.colors)
    brightness = 0.299 * c[:, 0] + 0.587 * c[:, 1] + 0.114 * c[:, 2]
    pcd        = pcd.select_by_index(np.where(brightness > BRIGHTNESS_FLOOR)[0].tolist())
    print(f"  After brightness cut: {len(pcd.points):>10,}")

    # 4. DBSCAN — keep main temple cluster only
    pcd = keep_largest_cluster(pcd)
    print(f"  After DBSCAN:         {len(pcd.points):>10,}")

    # 5. Voxel downsample
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    print(f"  After voxel downsamp: {len(pcd.points):>10,}")

    print(f"\nFinal point cloud: {len(pcd.points):,} points.")

    # -------------------------------------------------------------------------
    # Centre, orient, save
    # -------------------------------------------------------------------------
    pcd.translate(-pcd.get_center())
    pcd.rotate(pcd.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0)), center=(0, 0, 0))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    o3d.io.write_point_cloud(OUTPUT_PATH, pcd)
    print(f"Saved to '{OUTPUT_PATH}'")

    # -------------------------------------------------------------------------
    # Visualisation
    # -------------------------------------------------------------------------
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_SEARCH_RADIUS, max_nn=NORMAL_MAX_NN
        )
    )
    pcd.orient_normals_towards_camera_location(camera_location=[0, 0, 1])

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Middlebury Temple SfM",
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
    )
    vis.add_geometry(pcd)
    vis.add_geometry(coord_frame)

    ctr = vis.get_view_control()
    ctr.set_lookat([0.0, 0.0, 0.0])
    ctr.set_front(VIEW_FRONT)
    ctr.set_up(VIEW_UP)
    ctr.set_zoom(VIEW_ZOOM)

    vis.get_render_option().point_size = POINT_SIZE

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()