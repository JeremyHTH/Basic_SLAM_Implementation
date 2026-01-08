import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Kalman_Filter_Package import Extended_Kalman_Filter_SLAM


def wrap_to_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def load_data(dataset_dir):
    poses = pd.read_csv(f"{dataset_dir}/poses_controls.csv")
    landmarks = pd.read_csv(f"{dataset_dir}/landmarks.csv")
    measurements = pd.read_csv(f"{dataset_dir}/measurements.csv")
    return poses, landmarks, measurements


def main():
    dataset_dir = "Dataset"
    poses, landmarks, measurements = load_data(dataset_dir)

    # Initial state from the first pose in the dataset.
    init_state = np.zeros((3, 1))
    init_state[0, 0] = poses.loc[0, "x"]
    init_state[1, 0] = poses.loc[0, "y"]
    init_state[2, 0] = poses.loc[0, "theta"]

    init_cov = np.diag([0.1, 0.1, 0.05])

    process_noise = np.diag([0.02, 0.02, 0.01])

    sigma_r = 0.05
    sigma_b = math.radians(1)
    meas_noise = np.diag([sigma_r * sigma_r, sigma_b * sigma_b])

    current_landmark_id = {"value": None}
    landmark_indices = {}
    next_landmark_index = {"value": 0}
    initial_landmark_uncertainty = 1.0

    def expand_state(vec, new_size):
        if vec.shape[0] == new_size:
            return vec
        expanded = np.zeros((new_size, 1))
        expanded[: vec.shape[0], 0] = vec[:, 0]
        return expanded

    def expand_covariance(cov, new_size, initial_uncertainty):
        if cov.shape[0] == new_size:
            return cov
        old_size = cov.shape[0]
        expanded = np.zeros((new_size, new_size))
        expanded[:old_size, :old_size] = cov
        expanded[old_size:, old_size:] = np.eye(new_size - old_size) * initial_uncertainty
        return expanded

    def expand_process_noise(q, new_size):
        if q.shape[0] == new_size:
            return q
        old_size = q.shape[0]
        expanded = np.zeros((new_size, new_size))
        expanded[:old_size, :old_size] = q
        return expanded

    def f_func(state, control):
        v = control[0, 0]
        omega = control[1, 0]
        dt = control[2, 0]

        x = state[0, 0]
        y = state[1, 0]
        theta = state[2, 0]

        x_new = x + v * dt * math.cos(theta)
        y_new = y + v * dt * math.sin(theta)
        theta_new = wrap_to_pi(theta + omega * dt)

        next_state = state.copy()
        next_state[0, 0] = x_new
        next_state[1, 0] = y_new
        next_state[2, 0] = theta_new
        return next_state

    def A_func(state, control):
        v = control[0, 0]
        dt = control[2, 0]
        theta = state[2, 0]

        state_dim = state.shape[0]
        A = np.eye(state_dim)
        A[0, 2] = -v * dt * math.sin(theta)
        A[1, 2] = v * dt * math.cos(theta)
        return A

    def h_func(state):
        lm_id = current_landmark_id["value"]
        if lm_id is None:
            raise ValueError("landmark index not set")
        if lm_id not in landmark_indices:
            raise ValueError("landmark index missing for measurement")

        base = 3 + 2 * landmark_indices[lm_id]
        lx = state[base, 0]
        ly = state[base + 1, 0]

        x = state[0, 0]
        y = state[1, 0]
        theta = state[2, 0]

        dx = lx - x
        dy = ly - y
        r = math.hypot(dx, dy)
        bearing = wrap_to_pi(math.atan2(dy, dx) - theta)

        return np.array([[r], [bearing]])

    def C_func(state):
        lm_id = current_landmark_id["value"]
        if lm_id is None:
            raise ValueError("landmark index not set")
        if lm_id not in landmark_indices:
            raise ValueError("landmark index missing for measurement")

        base = 3 + 2 * landmark_indices[lm_id]
        lx = state[base, 0]
        ly = state[base + 1, 0]

        x = state[0, 0]
        y = state[1, 0]
        theta = state[2, 0]

        dx = lx - x
        dy = ly - y
        q = dx * dx + dy * dy
        r = math.sqrt(max(q, 1e-12))

        state_dim = state.shape[0]
        C = np.zeros((2, state_dim))
        C[0, 0] = -dx / r
        C[0, 1] = -dy / r
        C[0, 2] = 0.0
        C[1, 0] = dy / max(q, 1e-12)
        C[1, 1] = -dx / max(q, 1e-12)
        C[1, 2] = -1.0

        C[0, base] = dx / r
        C[0, base + 1] = dy / r
        C[1, base] = -dy / max(q, 1e-12)
        C[1, base + 1] = dx / max(q, 1e-12)
        return C

    ekf = Extended_Kalman_Filter_SLAM(
        f_func,
        h_func,
        A_func,
        C_func,
        number_of_states=3,
        number_of_measurements=2,
        init_state=init_state,
        init_covariance=init_cov,
        process_noise=process_noise,
        measurement_noise=meas_noise,
    )

    measurements_by_time = {
        t: df for t, df in measurements.groupby("t", sort=True)
    }

    estimated_states = []
    has_measurements_log = []

    ekf.predicted_state = ekf.state.copy()
    ekf.predicted_covariance = ekf.covariance.copy()

    times = poses["t"].values

    for k, t in enumerate(times):
        has_measurements = t in measurements_by_time
        if has_measurements:
            for _, row in measurements_by_time[t].iterrows():
                lm_id = int(row["landmark_id"])
                current_landmark_id["value"] = lm_id

                if lm_id not in landmark_indices:
                    r_meas = float(row["r"])
                    b_meas = float(row["bearing"])
                    theta = ekf.predicted_state[2, 0]
                    x = ekf.predicted_state[0, 0]
                    y = ekf.predicted_state[1, 0]
                    lx = x + r_meas * math.cos(b_meas + theta)
                    ly = y + r_meas * math.sin(b_meas + theta)

                    landmark_indices[lm_id] = next_landmark_index["value"]
                    next_landmark_index["value"] += 1

                    ekf.add_landmark(np.array([lx, ly]), initial_uncertainty=initial_landmark_uncertainty)
                    ekf.process_noise = expand_process_noise(
                        ekf.process_noise,
                        ekf.number_of_states,
                    )
                    ekf.predicted_state = expand_state(ekf.predicted_state, ekf.number_of_states)
                    ekf.predicted_state[-2:, 0] = np.array([lx, ly])
                    ekf.predicted_covariance = expand_covariance(
                        ekf.predicted_covariance,
                        ekf.number_of_states,
                        initial_landmark_uncertainty,
                    )

                z = np.array([[row["r"]], [row["bearing"]]])
                predicted = h_func(ekf.predicted_state)
                z[1, 0] = predicted[1, 0] + wrap_to_pi(z[1, 0] - predicted[1, 0])

                ekf.update(z)

        if not has_measurements:
            ekf.state = ekf.predicted_state.copy()
            ekf.covariance = ekf.predicted_covariance.copy()

        estimated_states.append(ekf.state.copy())
        has_measurements_log.append(has_measurements)

        if k < len(times) - 1:
            dt = times[k + 1] - times[k]
            v = poses.loc[k, "v_exec"]
            omega = poses.loc[k, "omega_exec"]
            control = np.array([[v], [omega], [dt]])
            ekf.predict(control)

    est_xy = np.array([[state[0, 0], state[1, 0]] for state in estimated_states])
    true_xy = poses[["x", "y"]].values
    has_measurements_log = np.array(has_measurements_log, dtype=bool)

    rmse = np.sqrt(np.mean(np.sum((est_xy - true_xy) ** 2, axis=1)))
    print(f"Position RMSE (x,y): {rmse:.3f} m")

    plt.figure(figsize=(10, 7))
    plt.plot(true_xy[:, 0], true_xy[:, 1], label="Ground Truth")
    plt.plot(est_xy[:, 0], est_xy[:, 1], label="EKF-SLAM Estimate")
    if len(est_xy) > 0:
        pred_only = est_xy[~has_measurements_log]
        if len(pred_only) > 0:
            plt.scatter(
                pred_only[:, 0],
                pred_only[:, 1],
                color="tab:green",
                s=18,
                label="Prediction Only",
            )
    plt.scatter(landmarks["x"], landmarks["y"], marker="x", label="Landmarks GT")

    est_landmarks = []
    for lm_id in sorted(observed_landmarks):
        base = 3 + 2 * lm_id
        est_landmarks.append([ekf.state[base, 0], ekf.state[base + 1, 0]])
    if est_landmarks:
        est_landmarks = np.array(est_landmarks)
        plt.scatter(
            est_landmarks[:, 0],
            est_landmarks[:, 1],
            marker="o",
            facecolors="none",
            edgecolors="tab:red",
            label="Landmarks Estimated",
        )

    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("EKF-SLAM on Sample Dataset")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
