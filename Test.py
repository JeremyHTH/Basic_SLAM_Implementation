import matplotlib.pyplot as plt
from Kalman_Filter_Package import Extended_Kalman_Filter
import numpy as np


np.random.seed(0)

dt = 0.1
steps = 100

# State: [position, velocity]
def f_func(state, control):
    a = control[0, 0]
    x = state[0, 0]
    v = state[1, 0]
    return np.array([[x + v * dt + 0.5 * a * dt * dt],
                        [v + a * dt]])

def h_func(state):
    return np.array([[state[0, 0]]])

def A_func(state, control):
    return np.array([[1.0, dt],
                        [0.0, 1.0]])

def C_func(state):
    return np.array([[1.0, 0.0]])

init_state = np.array([[0.0],
                        [1.0]])

ekf = Extended_Kalman_Filter(
    f_func,
    h_func,
    A_func,
    C_func,
    number_of_states=2,
    number_of_measurements=1,
    init_state=init_state,
    init_covariance=np.eye(2) * 0.1,
    process_noise=np.eye(2) * 0.01,
    measurement_noise=np.eye(1) * 0.05,
)

true_states = [init_state]
measurements = []
controls = []

for _ in range(steps):
    control = np.array([[0.2]])
    next_state = f_func(true_states[-1], control)
    measurement = h_func(next_state) + np.random.normal(
        0.0,
        np.sqrt(ekf.measurement_noise[0, 0]),
        (1, 1),
    )
    true_states.append(next_state)
    measurements.append(measurement)
    controls.append(control)

estimates = []
for idx in range(steps):
    ekf.predict(controls[idx])
    ekf.update(measurements[idx])
    estimates.append(ekf.state.copy())

t = np.arange(steps + 1) * dt
true_positions = np.array([state[0, 0] for state in true_states])
estimated_positions = np.array([state[0, 0] for state in estimates])

plt.figure(figsize=(10, 6))
plt.plot(t, true_positions, label="Ground Truth")
plt.plot(t[1:], estimated_positions, label="EKF Prediction")
plt.xlabel("Time (s)")
plt.ylabel("Position")
plt.title("EKF Position Estimate vs Ground Truth")
plt.legend()
plt.tight_layout()
plt.show()
