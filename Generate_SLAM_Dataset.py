import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# -----------------------------
# Utility functions
# -----------------------------
def wrap_to_pi(a):
    return math.atan2(np.sin(a), np.cos(a))

def unicycle_step(x, y, th, v, w, dt):
    x += v * np.cos(th) * dt
    y += v * np.sin(th) * dt
    th = wrap_to_pi(th + w * dt)
    return x, y, th

# -----------------------------
# Control model
# -----------------------------
def desired_control(t):
    """Desired (noise-free) commands"""
    v_d = 1.0 + 0.2 * np.sin(0.1 * t)
    w_d = 0.3 * np.sin(0.07 * t) + 0.15 * np.sin(0.17 * t)
    return v_d, w_d

# -----------------------------
# Main simulation
# -----------------------------
def main():
    np.random.seed(4)

    # Time
    dt = 0.1
    T = 120.0
    t = np.arange(0, T + dt, dt)

    # Noise parameters
    sigma_v = 0.05          # m/s
    sigma_w = np.deg2rad(2)  # rad/s

    sigma_r = 0.05           # m
    sigma_b = np.deg2rad(1)  # rad

    # LiDAR limits
    max_range = 18.0
    fov = np.deg2rad(240)

    # Landmarks
    n_lm = 25
    lm_xy = np.random.uniform(-20, 20, size=(n_lm, 2))

    # Initial pose
    x, y, th = 0.0, 0.0, np.deg2rad(0)

    pose_log = []
    meas_log = []

    # -----------------------------
    # Simulation loop
    # -----------------------------
    for k, tk in enumerate(t):
        # Desired control
        v_d, w_d = desired_control(tk)

        # Executed control (with noise)
        v = v_d + np.random.randn() * sigma_v
        w = w_d + np.random.randn() * sigma_w

        # Log state and executed control
        pose_log.append({
            "t": tk,
            "x": x,
            "y": y,
            "theta": th,
            "v_cmd": v_d,
            "omega_cmd": w_d,
            "v_exec": v,
            "omega_exec": w
        })

        # LiDAR measurements
        for lm_id, (lx, ly) in enumerate(lm_xy):
            dx = lx - x
            dy = ly - y
            r_true = np.hypot(dx, dy)
            if r_true > max_range:
                continue

            bearing_true = wrap_to_pi(np.arctan2(dy, dx) - th)
            if abs(bearing_true) > fov / 2:
                continue

            # Add sensor noise
            r_meas = r_true + np.random.randn() * sigma_r
            b_meas = wrap_to_pi(bearing_true + np.random.randn() * sigma_b)

            meas_log.append({
                "t": tk,
                "landmark_id": lm_id,
                "r": r_meas,
                "bearing": b_meas
            })

        # Propagate state
        if k < len(t) - 1:
            x, y, th = unicycle_step(x, y, th, v, w, dt)

    # -----------------------------
    # DataFrames
    # -----------------------------
    df_pose = pd.DataFrame(pose_log)
    df_lm = pd.DataFrame({
        "landmark_id": np.arange(n_lm),
        "x": lm_xy[:, 0],
        "y": lm_xy[:, 1]
    })
    df_meas = pd.DataFrame(meas_log)

    # -----------------------------
    # Save dataset
    # -----------------------------
    df_pose.to_csv("Dataset/poses_controls.csv", index=False)
    df_lm.to_csv("Dataset/landmarks.csv", index=False)
    df_meas.to_csv("Dataset/measurements.csv", index=False)

    print("Saved datasets:")
    print(" - Dataset/poses_controls.csv")
    print(" - Dataset/landmarks.csv")
    print(" - Dataset/measurements.csv")

    # -----------------------------
    # Plots
    # -----------------------------
    # Ground-truth path + landmarks
    plt.figure()
    plt.plot(df_pose["x"], df_pose["y"], label="Robot path")
    plt.scatter(df_lm["x"], df_lm["y"], marker="x", label="Landmarks")
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Ground Truth Robot Path & Landmarks")
    plt.legend()
    plt.grid()

    # Control commands vs time
    plt.figure()
    plt.plot(df_pose["t"], df_pose["v_cmd"], "--", label="v desired")
    plt.plot(df_pose["t"], df_pose["v_exec"], label="v executed")
    plt.plot(df_pose["t"], df_pose["omega_cmd"], "--", label="ω desired")
    plt.plot(df_pose["t"], df_pose["omega_exec"], label="ω executed")
    plt.xlabel("time [s]")
    plt.ylabel("control")
    plt.title("Control Commands (Desired vs Executed)")
    plt.legend()
    plt.grid()

    plt.show()

if __name__ == "__main__":
    main()
