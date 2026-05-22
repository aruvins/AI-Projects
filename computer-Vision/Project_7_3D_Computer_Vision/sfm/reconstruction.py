import os
import cv2
import numpy as np
import open3d as o3d

def parse_camera_parameters(par_file_path):
    """
    Parses temple_par.txt to map the 4-digit frame ID string (e.g., '0001') 
    to its 3x4 Projection Matrix.
    """
    camera_matrices = {}
    if not os.path.exists(par_file_path):
        raise FileNotFoundError(f"Could not find parameter file at: {par_file_path}")
        
    with open(par_file_path, 'r') as f:
        lines = f.readlines()
    
    print(f"Reading camera parameters from {par_file_path}...")
    for line in lines[1:]:
        tokens = line.strip().split()
        if not tokens:
            continue
        
        filename = tokens[0]  # e.g., "temple0001.png"
        
        # Extract just the 4-digit number (e.g., "0001") to be extension-agnostic
        # Finds digits in the filename
        digit_id = "".join(filter(str.isdigit, filename))
        if not digit_id:
            continue
            
        values = np.array([float(x) for x in tokens[1:]])
        
        K = values[0:9].reshape((3, 3))
        R = values[9:18].reshape((3, 3))
        t = values[18:21].reshape((3, 1))
        
        # Formulate Projection Matrix P = K * [R | t]
        Rt = np.hstack((R, t))
        P = np.dot(K, Rt)
        
        camera_matrices[digit_id] = P
        
    print(f"Successfully cached calibration for {len(camera_matrices)} frames.")
    return camera_matrices

def match_and_triangulate(img1_path, img2_path, P1, P2):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        return None, None

    # Use SIFT to extract keypoints
    sift = cv2.SIFT_create(nfeatures=2000)
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)
    
    if desc1 is None or desc2 is None:
        return None, None

    # FLANN Matching
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        matches = flann.knnMatch(desc1, desc2, k=2)
    except Exception:
        return None, None
    
    good_matches = []
    pts1 = []
    pts2 = []
    colors = []
    
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
            pt1 = kp1[m.queryIdx].pt
            pts1.append(pt1)
            pts2.append(kp2[m.trainIdx].pt)
            
            x, y = int(pt1[0]), int(pt1[1])
            x = min(max(x, 0), img1.shape[1] - 1)
            y = min(max(y, 0), img1.shape[0] - 1)
            colors.append(img1[y, x][::-1] / 255.0)

    # We need at least a few solid matches to triangulate reliably
    if len(good_matches) < 15:
        return None, None

    pts1 = np.float32(pts1).T
    pts2 = np.float32(pts2).T

    # Triangulate
    points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    points_3d = points_4d[:3, :] / points_4d[3, :]
    
    return points_3d.T, np.array(colors)

def main():
    # Adjusted path dynamically relative to script location or execution root
    data_dir = "data/sfm/temple/temple"
    par_file = os.path.join(data_dir, "temple_par.txt")
    
    try:
        camera_matrices = parse_camera_parameters(par_file)
    except Exception as e:
        print(f"Error loading parameter file: {e}")
        return
    
    # Gather image files
    all_files = sorted(os.listdir(data_dir))
    img_extensions = ('.png', '.jpg', '.jpeg')
    img_files = [f for f in all_files if f.lower().endswith(img_extensions)]
    
    print(f"Found {len(img_files)} images in directory structural scan.")
    if len(img_files) == 0:
        print(f"Check your data directory path. Read 0 image assets from: {data_dir}")
        return

    all_points_3d = []
    all_colors = []
    
    # Process sequential pairs
    pairs_computed = 0
    for i in range(0, len(img_files) - 1):
        file1 = img_files[i]
        file2 = img_files[i + 1]
        
        # Extract digits purely (e.g. "0001" from "temple0001.jpg")
        id1 = "".join(filter(str.isdigit, file1))
        id2 = "".join(filter(str.isdigit, file2))
        
        if id1 not in camera_matrices or id2 not in camera_matrices:
            # Diagnostic flag to catch mismatches early
            if i == 0:
                print(f"Diagnostic: Frame mapping missed! Couldn't match numeric ID '{id1}' or '{id2}' to matrices list.")
            continue
            
        P1 = camera_matrices[id1]
        P2 = camera_matrices[id2]
        
        img1_path = os.path.join(data_dir, file1)
        img2_path = os.path.join(data_dir, file2)
        
        pts_3d, colors = match_and_triangulate(img1_path, img2_path, P1, P2)
        
        if pts_3d is not None:
            all_points_3d.append(pts_3d)
            all_colors.append(colors)
            pairs_computed += 1
            if pairs_computed % 10 == 0 or pairs_computed == 1:
                print(f"Successfully processed {pairs_computed} image pairs...")

    print("\nReconstruction passes complete! Finalizing point cloud filtering...")
    
    if not all_points_3d:
        print("Error: Still no 3D points were successfully triangulated.")
        print("Please verify that SIFT and FLANN are generating valid features on your images.")
        return
        
    final_points = np.vstack(all_points_3d)
    final_colors = np.vstack(all_colors)
    
    # Point cloud assembly
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_points)
    pcd.colors = o3d.utility.Vector3dVector(final_colors)
    
    # Filter noisy artifact points
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    cleaned_pcd = pcd.select_by_index(ind)
    
    print(f"Success! Generated sparse cloud with {len(cleaned_pcd.points)} points.")
    
    # --- CHANGED: ORIENT AROUND THE FIXED CENTROID ---
    print("Centering and orienting point cloud upright...")
    
    # 1. Get the current bounding box center (fixed origin point of the object)
    mesh_center = cleaned_pcd.get_center()
    
    # 2. Translate the cloud so its center sits exactly at (0, 0, 0)
    cleaned_pcd.translate(-mesh_center)
    
    # 3. Create the rotation matrix (-90 degrees around X-axis)
    R_correct = cleaned_pcd.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
    
    # 4. Rotate the cloud locally around its new (0, 0, 0) center
    cleaned_pcd.rotate(R_correct, center=(0, 0, 0))
    
    # NOTE: If you want the temple back at its original location in world space, 
    # you could run: cleaned_pcd.translate(mesh_center)
    # Keeping it at (0,0,0) makes it much easier to inspect and rotate in Open3D.
    # -------------------------------------------------

    output_mesh_path = "outputs/sfm/temple_reconstruction.pcd"
    o3d.io.write_point_cloud(output_mesh_path, cleaned_pcd)
    print(f"Saved locally centered point cloud to '{output_mesh_path}'")
    
    # Launch viewer with a coordinate frame helper sitting directly inside the temple base
    print("Launching 3D visualization window...")
    o3d.io.write_point_cloud(output_mesh_path, cleaned_pcd)
    print(f"Saved locally centered point cloud to '{output_mesh_path}'")
    
    # Create a coordinate frame sitting at the center (0,0,0)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    
    # --- NEW: ADVANCED INITIAL VIEWPORT CONTROL ---
    print("Launching 3D visualization window at specific coordinates...")
    
    # 1. Initialize the advanced visualizer class
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Middlebury Temple SfM (Custom View)", width=1024, height=768)
    
    # 2. Add your geometries to the scene
    vis.add_geometry(cleaned_pcd)
    vis.add_geometry(coord_frame)
    
    # 3. Access the camera view control properties
    ctr = vis.get_view_control()
    
    # Set the point the camera targets (looking right at our centered temple origin)
    ctr.set_lookat([0.0, 0.0, 0.0])
    
    # Set the camera position vector relative to lookat [X, Y, Z]
    # Adjust these numbers to move your starting camera position closer, further, or to the side
    ctr.set_front([0.0, 0.0, 1.0]) 
    
    # Set the orientation vector defining which way is "up" [X, Y, Z]
    ctr.set_up([0.0, 1.0, 0.0])
    
    # Adjust zoom level (smaller numbers mean closer zoom)
    ctr.set_zoom(0.5)
    
    # 4. Render the scene and open interactive loops
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()