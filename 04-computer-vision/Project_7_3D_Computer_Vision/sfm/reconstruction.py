import os
import cv2
import numpy as np
import open3d as o3d

def parse_camera_and_angles(data_dir):
    """
    Parses temple_par.txt for matrices and temple_ang.txt to map the
    precise angular tracking of each frame for baseline optimization.
    """
    par_file = os.path.join(data_dir, "temple_par.txt")
    ang_file = os.path.join(data_dir, "temple_ang.txt")
    
    camera_data = {}
    
    # 1. Parse Parameters
    with open(par_file, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        tokens = line.strip().split()
        if not tokens: continue
        filename = tokens[0]
        digit_id = "".join(filter(str.isdigit, filename))
        if not digit_id: continue
        
        values = np.array([float(x) for x in tokens[1:]])
        K = values[0:9].reshape((3, 3))
        R = values[9:18].reshape((3, 3))
        t = values[18:21].reshape((3, 1))
        Rt = np.hstack((R, t))
        P = np.dot(K, Rt)
        
        # Calculate optical center of camera: C = -R^T * t
        camera_center = -np.dot(R.T, t).ravel()
        
        camera_data[digit_id] = {"P": P, "K": K, "center": camera_center, "angle": 0.0}

    # 2. Parse Angles for baseline screening
    if os.path.exists(ang_file):
        with open(ang_file, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) >= 2:
                    filename = tokens[-1]
                    digit_id = "".join(filter(str.isdigit, filename))
                    if digit_id in camera_data:
                        camera_data[digit_id]["angle"] = float(tokens[1])
                        
    return camera_data

def match_and_triangulate_exquisite(img1_path, img2_path, cam1, cam2):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None: return None, None

    sift = cv2.SIFT_create(nfeatures=20000, contrastThreshold=0.03, edgeThreshold=10)
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)
    if desc1 is None or desc2 is None: return None, None

    # Sub-pixel coordinate tracking
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    pts1_raw = np.float32([kp.pt for kp in kp1]).reshape(-1, 1, 2)
    pts2_raw = np.float32([kp.pt for kp in kp2]).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    pts1_refined = cv2.cornerSubPix(gray1, pts1_raw, (3, 3), (-1, -1), criteria).squeeze()
    pts2_refined = cv2.cornerSubPix(gray2, pts2_raw, (3, 3), (-1, -1), criteria).squeeze()

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    pts1_matched, pts2_matched, colors_matched = [], [], []
    for m, n in matches:
        if m.distance < 0.80 * n.distance:
            pt1 = pts1_refined[m.queryIdx]
            pts1_matched.append(pt1)
            pts2_matched.append(pts2_refined[m.trainIdx])
            x, y = int(pt1[0]), int(pt1[1])
            x, y = min(max(x, 0), img1.shape[1] - 1), min(max(y, 0), img1.shape[0] - 1)
            colors_matched.append(img1[y, x][::-1] / 255.0)

    if len(pts1_matched) < 40: return None, None

    pts1_arr, pts2_arr = np.array(pts1_matched), np.array(pts2_matched)
    
    # Fundamental RANSAC verification
    _, mask = cv2.findFundamentalMat(pts1_arr, pts2_arr, cv2.FM_RANSAC, 0.8, 0.999)
    if mask is None: return None, None
    mask = mask.ravel() == 1

    pts1_final = pts1_arr[mask].T
    pts2_final = pts2_arr[mask].T
    colors_final = np.array(colors_matched)[mask]

    points_4d = cv2.triangulatePoints(cam1["P"], cam2["P"], pts1_final, pts2_final)
    points_3d = points_4d[:3, :] / points_4d[3, :]
    
    return points_3d.T, colors_final

def main():
    data_dir = "data/sfm/temple/temple"
    camera_matrices = parse_camera_and_angles(data_dir)
    img_files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    all_points, all_colors = [], []
    
    print("⏳ Running Baseline-Filtered Triangulation...")
    for i in range(len(img_files)):
        for j in range(i + 1, min(i + 5, len(img_files))):
            id1 = "".join(filter(str.isdigit, img_files[i]))
            id2 = "".join(filter(str.isdigit, img_files[j]))
            if id1 not in camera_matrices or id2 not in camera_matrices: continue
            
            cam1, cam2 = camera_matrices[id1], camera_matrices[id2]
            
            angle_diff = abs(cam1["angle"] - cam2["angle"])
            if angle_diff < 4.0 or angle_diff > 25.0:
                continue
                
            pts_3d, colors = match_and_triangulate_exquisite(
                os.path.join(data_dir, img_files[i]), os.path.join(data_dir, img_files[j]), cam1, cam2
            )
            if pts_3d is not None:
                all_points.append(pts_3d)
                all_colors.append(colors)

    if not all_points:
        print("❌ Error: No 3D points were generated from triangulation.")
        return

    final_points = np.vstack(all_points)
    final_colors = np.vstack(all_colors)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_points)
    pcd.colors = o3d.utility.Vector3dVector(final_colors)
    
    print(f"Raw Point Count: {len(pcd.points)}")
    pcd = pcd.voxel_down_sample(voxel_size=0.0015) 
    print(f"Post-Voxel Homogenized Count: {len(pcd.points)}")
    
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=45, std_ratio=0.6)
    pcd = pcd.select_by_index(ind)
   
    # Calculate geometric anchor centers
    mesh_center = pcd.get_center()
    
    # 1. High-Precision Normal Estimation BEFORE transformations
    print("Estimating high-precision surface normals from camera viewpoints...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.010, max_nn=30))
    
    all_cam_centers = np.array([cam["center"] for cam in camera_matrices.values()])
    mean_cam_center = np.mean(all_cam_centers, axis=0)
    pcd.orient_normals_towards_camera_location(camera_location=mean_cam_center)

    # Perform geometry transformations safely now
    pcd.translate(-mesh_center)
    R_align = pcd.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
    pcd.rotate(R_align, center=(0, 0, 0))
    
    # 2. Compute High-Resolution Poisson Mesh
    print("Computing high-fidelity Poisson Surface Reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
    print(f"Poisson Base Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles.")

    # 3. HIGH-PRECISION DENSITY TRIMMING
    print("Trimming low-density boundary bubbles...")
    densities = np.asarray(densities)
    # The true corrected normals allow us to aggressively clear the ghost walls
    trim_threshold = np.percentile(densities, 24) 
    vertices_to_remove = densities < trim_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # 4. Topology Correction
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    
    # 5. Non-Destructive Taubin Smoothing
    print("Polishing flat surfaces with Taubin filter...")
    if len(mesh.triangles) > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=6, lambda_filter=0.5, mu=-0.53)
    
    # 6. AGGRESSIVE FLOATING ISLAND PURGE
    print("Purging disconnected floating mesh fragments...")
    if len(mesh.triangles) > 0:
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        
        large_clusters = np.where(cluster_n_triangles > (len(mesh.triangles) * 0.12))[0]
        triangles_to_remove = ~np.isin(triangle_clusters, large_clusters)
        mesh.remove_triangles_by_mask(triangles_to_remove)
        mesh.remove_unreferenced_vertices()

    # 7. FIXED PHOTO-REALISTIC MULTI-VIEW GEOMETRIC TEXTURING
    print("Projecting photo-realistic textures onto clean geometry...")
    mesh_vertices = np.asarray(mesh.vertices)
    vertex_colors = np.zeros_like(mesh_vertices)
    vertex_weights = np.zeros((len(mesh_vertices), 1))
    
    # FIXED: Multiply by R_align.T to project correctly back into original camera space
    orig_vertices = np.dot(mesh_vertices, R_align.T) + mesh_center

    for filename in img_files:
        digit_id = "".join(filter(str.isdigit, filename))
        if digit_id not in camera_matrices: continue
        
        cam = camera_matrices[digit_id]
        img = cv2.imread(os.path.join(data_dir, filename))
        if img is None: continue
        h, w, _ = img.shape
        
        pts_3d_homo = np.hstack((orig_vertices, np.ones((len(orig_vertices), 1))))
        pts_2d_homo = np.dot(cam["P"], pts_3d_homo.T).T
        
        valid_depth = pts_2d_homo[:, 2] > 0.1
        pts_2d = pts_2d_homo[:, :2] / pts_2d_homo[:, 2:3]
        
        x_img = np.round(pts_2d[:, 0]).astype(int)
        y_img = np.round(pts_2d[:, 1]).astype(int)
        
        in_frame = (x_img >= 0) & (x_img < w) & (y_img >= 0) & (y_img < h) & valid_depth
        
        if np.any(in_frame):
            colors_sampled = img[y_img[in_frame], x_img[in_frame]][:, ::-1] / 255.0
            cam_vector = cam["center"] - orig_vertices[in_frame]
            cam_dist = np.linalg.norm(cam_vector, axis=1, keepdims=True)
            
            # Distance-weighted priority mapping
            weight = 1.0 / (cam_dist + 1e-6)
            vertex_colors[in_frame] += colors_sampled * weight
            vertex_weights[in_frame] += weight

    valid_weights = (vertex_weights > 0).squeeze()
    vertex_colors[valid_weights] /= vertex_weights[valid_weights]
    vertex_colors[~valid_weights] = [0.70, 0.67, 0.62] # Warm architectural stone fallback
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    mesh.compute_vertex_normals()

    # --- ADVANCED PRESENTATION ENGINE ---
    print("Launching final clean architectural viewport...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Middlebury Temple - Complete Textured Surface", width=1280, height=960)
    
    vis.add_geometry(mesh)
    
    # Render Styling Controls
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0.05, 0.05, 0.05])  
    render_opt.light_on = True                                   
    render_opt.mesh_show_back_face = True                        
    
    ctr = vis.get_view_control()
    ctr.set_lookat([0.0, 0.0, 0.0])
    ctr.set_front([0.25, 0.15, 0.95]) 
    ctr.set_up([0.0, 1.0, 0.0])
    ctr.set_zoom(0.8)
    
    print("Rendering complete! Enjoy your clean, sharp, photo-textured temple model.")
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()