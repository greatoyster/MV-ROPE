import torch
from torch.linalg import svd

# please refer to
# https://github.com/sunghoon031/RobustSingleRotationAveraging/blob/master/RobustSingleRotationAveraging_ReleaseCode.m


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    """
    Computes the exponential map of so(3), which maps a 3D rotation vector to a rotation matrix in SO(3)
    :param w: 3D rotation vector in the tangent space of SO(3)
    :return: 3x3 rotation matrix in SO(3)
    """

    theta = torch.norm(w)

    if theta < 1e-7:
        return torch.eye(3, dtype=w.dtype, device=w.device) + torch.cross(w, w)

    wx = torch.zeros((3, 3), dtype=w.dtype, device=w.device)
    wx[0][1] = -w[2]
    wx[0][2] = w[1]
    wx[1][0] = w[2]
    wx[1][2] = -w[0]
    wx[2][0] = -w[1]
    wx[2][1] = w[0]

    exp_w = (
        torch.eye(3, dtype=w.dtype, device=w.device)
        + (torch.sin(theta) / theta) * wx
        + ((1 - torch.cos(theta)) / (theta**2)) * torch.matmul(wx, wx)
    )

    return exp_w


def so3_log(rot_mat: torch.Tensor) -> torch.Tensor:
    """
    Compute the logarithm map of SO(3), which maps a rotation matrix to a rotation vector
    :param rot_mat: 3x3 rotation matrix
    :return: 3D rotation vector in the tangent space of SO(3)
    """
    trace = rot_mat.trace()
    theta = torch.acos((trace - 1) / 2.0)

    if theta < 1e-7:
        return torch.zeros(3, dtype=rot_mat.dtype, device=rot_mat.device)

    ln_rot_mat = (theta / (2 * torch.sin(theta))) * (rot_mat - rot_mat.transpose(0, 1))
    log_vector = torch.tensor(
        [ln_rot_mat[2, 1], ln_rot_mat[0, 2], ln_rot_mat[1, 0]],
        dtype=rot_mat.dtype,
        device=rot_mat.device,
    )

    return log_vector


def random_rotation(max_angle_rad):
    unit_axis = torch.randn(3)
    unit_axis /= torch.norm(unit_axis)
    angle = torch.rand(1) * max_angle_rad
    return rotation_from_unit_axis_angle(unit_axis, angle)


def rotation_from_unit_axis_angle(unit_axis, angle):
    R = so3_exp(unit_axis * angle)
    return R


def project_onto_so3(M):
    U, _, Vh = svd(M)
    V = Vh.t()
    R = torch.mm(U, V.t())
    if torch.det(R) < 0:
        V[:, 2] *= -1
        R = torch.mm(U, V.t())
    return R


def so3_geodesic_l1_mean(R_input, b_outlier_rejection, n_iterations, thr_convergence):
    # 1. Initialize
    if isinstance(R_input, list):
        R_input = torch.stack(R_input, dim=0)
    n_samples = len(R_input)
    vectors_total = torch.zeros((9, n_samples), device=R_input.device)

    for i in range(n_samples):
        vectors_total[:, i] = R_input[i].reshape(-1)

    s = torch.median(vectors_total, dim=1).values

    [U, _, Vh] = svd(torch.reshape(s, (3, 3)))
    V = Vh.t()
    R = torch.mm(U, V.t())
    if torch.det(R) < 0:
        V[:, 2] *= -1
        R = torch.mm(U, V.t())

    # 2. Optimize
    for j in range(n_iterations):
        vs = torch.zeros((3, n_samples), device=R_input.device)
        v_norms = torch.zeros(n_samples)
        for i in range(n_samples):
            v = so3_log(torch.mm(R_input[i], R.t()))
            v_norm = torch.norm(v)
            vs[:, i] = v
            v_norms[i] = v_norm

        # Compute the inlier threshold (if we reject outliers).
        thr = float("inf")
        if b_outlier_rejection:
            sorted_v_norms = torch.sort(v_norms)[0]
            v_norm_firstQ = sorted_v_norms[
                torch.ceil(torch.tensor(n_samples / 4)).int()
            ]
            if n_samples <= 50:
                thr = max(v_norm_firstQ, 1)
            else:
                thr = max(v_norm_firstQ, 0.5)

        step_num = torch.zeros(3, device=R_input.device)
        step_den = 0

        for i in range(n_samples):
            v = vs[:, i]
            v_norm = v_norms[i]
            if v_norm > thr:
                continue
            step_num += v / v_norm
            step_den += 1 / v_norm

        delta = step_num / step_den
        delta_angle = torch.norm(delta)
        delta_axis = delta / delta_angle

        R_delta = rotation_from_unit_axis_angle(delta_axis, delta_angle)
        R = torch.mm(R_delta, R)
        if delta_angle < thr_convergence:
            break

    return R


def so3_chordal_l1_mean(R_input, b_outlier_rejection, n_iterations, thr_convergence):
    # 1. Initialize
    n_samples = len(R_input)
    vectors_total = torch.zeros((9, n_samples))
    for i in range(n_samples):
        vectors_total[:, i] = R_input[i].reshape(-1)
    s = torch.median(vectors_total, dim=1).values

    # 2. Optimize
    for j in range(n_iterations):
        if torch.sum(torch.abs(vectors_total - s.view(-1, 1)) == 0) != 0:
            s += torch.rand(s.shape) * 0.001

        v_norms = torch.zeros(n_samples)
        for i in range(n_samples):
            v = vectors_total[:, i] - s
            v_norm = torch.norm(v)
            v_norms[i] = v_norm

        # Compute the inlier threshold (if we reject outliers).
        thr = float("inf")
        if b_outlier_rejection:
            sorted_v_norms = torch.sort(v_norms)[0]
            v_norm_firstQ = sorted_v_norms[
                torch.ceil(torch.tensor(n_samples / 4)).int()
            ]

            if n_samples <= 50:
                thr = max(v_norm_firstQ, 1.356)
            else:
                thr = max(v_norm_firstQ, 0.7)

        step_num = torch.zeros(9)
        step_den = 0

        for i in range(n_samples):
            v_norm = v_norms[i]
            if v_norm > thr:
                continue
            step_num += vectors_total[:, i] / v_norm
            step_den += 1 / v_norm

        s_prev = s
        s = step_num / step_den
        update_medvec = s - s_prev
        if torch.norm(update_medvec) < thr_convergence:
            break

    return project_onto_so3(torch.reshape(s, (3, 3)))


def fast_so3_chordal_l1_mean(
    R_input, b_outlier_rejection, n_iterations, thr_convergence
):
    # 1. Initialize
    n_samples = len(R_input)
    vectors_total = torch.stack(R_input).reshape(n_samples, -1)

    s = torch.median(vectors_total, dim=0).values

    # 2. Optimize
    for j in range(n_iterations):
        equal_mask = torch.all(vectors_total == s, dim=1)
        if torch.any(equal_mask):
            s += torch.rand_like(s) * 0.001

        v_norms = torch.norm(vectors_total - s.view(1, -1), dim=1)

        # Compute the inlier threshold (if we reject outliers).
        thr = float("inf")
        if b_outlier_rejection:
            sorted_v_norms = torch.sort(v_norms)[0]
            v_norm_firstQ = sorted_v_norms[
                torch.ceil(torch.tensor(n_samples / 4)).int()
            ]

            if n_samples <= 50:
                thr = max(v_norm_firstQ, 1.356)
            else:
                thr = max(v_norm_firstQ, 0.7)

        step_num = torch.zeros_like(s)
        step_den = 0

        v_norm_mask = v_norms <= thr
        v_norm_masked = v_norms.masked_select(v_norm_mask).view(-1, 1)
        vectors_masked = vectors_total[v_norm_mask]

        step_num = torch.sum(vectors_masked / v_norm_masked, dim=0)
        step_den = torch.sum(1 / v_norm_masked)

        s_prev = s
        s = step_num / step_den
        update_medvec = s - s_prev
        if torch.norm(update_medvec) < thr_convergence:
            break

    return project_onto_so3(s.view(3, 3))


if __name__ == "__main__":
    # random_points = (torch.rand(100,3)-0.5)*2
    # real_point = torch.randn(1,3)

    # # Add Gaussian noise of 0.05
    # noise = 0.05 * torch.randn(100, 3)
    # points = real_point + noise

    # robust_median = geodesic_median(points)
    # print(real_point)
    # print(robust_median)
    # exit(0)
    # Set parameters
    n_inliers = 8
    n_outliers = 2
    inlier_noise_level = 5
    R_true = random_rotation(torch.tensor([3.141592653589793]))
    print("R_true:", R_true)

    # 1. Create input rotations
    n_samples = n_inliers + n_outliers
    R_samples = []
    for i in range(n_samples):
        if i < n_inliers:
            axis_perturb = torch.randn(3)
            axis_perturb /= torch.norm(axis_perturb)
            angle_perturb = torch.normal(
                mean=0, std=torch.tensor(inlier_noise_level * torch.pi / 180)
            )
            R_perturb = rotation_from_unit_axis_angle(axis_perturb, angle_perturb)
            R_samples.append(torch.mm(R_perturb, R_true))
        else:
            R_samples.append(random_rotation(torch.tensor([3.141592653589793])))

    # 2-a. Average them using Hartley's L1 geodesic method (with our initialization and outlier rejection scheme)
    b_outlier_rejection = True
    n_iterations = 10
    thr_convergence = 0.001
    R_geodesic = so3_geodesic_l1_mean(
        R_samples, b_outlier_rejection, n_iterations, thr_convergence
    )

    # 2-b. Average them using our approximate L1 chordal method (with our initialization and outlier rejection scheme)
    R_chordal = so3_chordal_l1_mean(
        R_samples, b_outlier_rejection, n_iterations, thr_convergence
    )

    R_chordal_batch = fast_so3_chordal_l1_mean(
        R_samples, b_outlier_rejection, n_iterations, thr_convergence
    )
    # 3. Evaluate the rotation error (deg)
    error_GeodesicL1Mean = (
        torch.abs(
            torch.acos((torch.trace(torch.mm(R_true, torch.t(R_geodesic))) - 1) / 2)
        )
        * 180
        / torch.pi
    )
    error_ChordalL1Mean = (
        torch.abs(
            torch.acos((torch.trace(torch.mm(R_true, torch.t(R_chordal))) - 1) / 2)
        )
        * 180
        / torch.pi
    )
    error_ChordalL1Mean_batch = (
        torch.abs(
            torch.acos(
                (torch.trace(torch.mm(R_true, torch.t(R_chordal_batch))) - 1) / 2
            )
        )
        * 180
        / torch.pi
    )

    print(f"Error (geodesic L1 mean) = {error_GeodesicL1Mean} deg")
    print(f"Error (chordal L1 mean) = {error_ChordalL1Mean} deg")
    print(f"Error (chordal L1 mean) batch = {error_ChordalL1Mean_batch} deg")
