import numpy as np

def project_points(points, cam_matrix, intrinsics):
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    points_camera = (cam_matrix @ points_h.T).T
    x, y, z = points_camera[:, 0], points_camera[:, 1], points_camera[:, 2]
    z[z == 0] = 1e-6
    x_proj = (-intrinsics[0, 0] * x / z) + intrinsics[0, 2]
    y_proj = (intrinsics[1, 1] * y / z) + intrinsics[1, 2]
    projected = np.vstack((x_proj, y_proj)).T
    depth = z.reshape(-1, 1)
    return projected, depth