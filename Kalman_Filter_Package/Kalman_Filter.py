import numpy as np

class Kalman_Filter:
    def __init__(self, number_of_states, number_of_measurements, init_state=None, init_covariance=None, process_noise=None, measurement_noise=None):
        self.number_of_states = number_of_states
        self.number_of_measurements = number_of_measurements

        if number_of_states <= 0:
            raise ValueError("number_of_states must be a positive integer.")
        if number_of_measurements <= 0:
            raise ValueError("number_of_measurements must be a positive integer.")

        # Check and set initial state and covariance

        if init_state is None:
            self.state = np.array([[0.0] for _ in range(number_of_states)])
        else:
            if len(init_state.shape) == 1:
                self.state = init_state.reshape(-1, 1)
            elif len(init_state.shape) > 2:
                raise ValueError("Initial state must be a 1D or 2D array.")
            
            self.state = init_state
            
            if init_state.shape[0] != number_of_states:
                raise ValueError("Initial state dimension does not match number_of_states.")
            

        if init_covariance is None:
            self.covariance = np.eye(number_of_states)
        else:
            if init_covariance.shape != (number_of_states, number_of_states):
                raise ValueError("Initial covariance dimension does not match number_of_states.")
            self.covariance = init_covariance

        if process_noise is None:
            self.process_noise = np.eye(number_of_states) * 0.01
        else:
            if process_noise.shape != (number_of_states, number_of_states):
                raise ValueError("Process noise dimension does not match number_of_states.")
            self.process_noise = process_noise

        if measurement_noise is None:
            self.measurement_noise = np.eye(number_of_measurements) * 0.01
        else:
            if measurement_noise.shape != (number_of_measurements, number_of_measurements):
                raise ValueError("Measurement noise dimension does not match number_of_measurements.")
            self.measurement_noise = measurement_noise

        self.predicted_state = self.state.copy()
        self.predicted_covariance = self.covariance.copy()

    @property
    def x(self):
        return self.state
    
    @property
    def P(self):
        return self.covariance
    
    @property
    def Q(self):
        return self.process_noise
    
    @property
    def R(self):
        return self.measurement_noise


    def predict(self):
        raise Exception("This is a abstract method (predict) and should be implemented in a subclass.")

    def update(self):
        raise Exception("This is a abstract method (update) and should be implemented in a subclass.")
    
    # State transition function
    def f(self, state, control):
        raise Exception("This is a abstract method (f) and should be implemented in a subclass.")
    
    # Measurement function
    def h(self, state):
        raise Exception("This is a abstract method (h) and should be implemented in a subclass.")
    
    # Jacobian of the state transition function
    def A(self, state, control):
        raise Exception("This is a abstract method (A) and should be implemented in a subclass.")
    
    # Jacobian of the measurement function
    def C(self, state):
        raise Exception("This is a abstract method (C) and should be implemented in a subclass.")

class Extended_Kalman_Filter(Kalman_Filter):
    def __init__(self,f_func, h_func, A_func, C_func, number_of_states, number_of_measurements, init_state=None, init_covariance=None, process_noise=None, measurement_noise=None, ):
        super().__init__(number_of_states, number_of_measurements, init_state, init_covariance, process_noise, measurement_noise)
        self.f_func = f_func
        self.h_func = h_func
        self.A_func = A_func
        self.C_func = C_func
        
    def predict(self, control):
        A_k =  self.A(self.state, control)
        self.predicted_state = self.f(self.state, control)
        self.predicted_covariance = A_k @ self.covariance @ A_k.T + self.process_noise

    def update(self, measurement):
        C = self.C(self.predicted_state)
        K_gain = self.predicted_covariance @ C.T @ np.linalg.inv(C @ self.predicted_covariance @ C.T + self.measurement_noise)
        self.state = self.predicted_state + K_gain @ (measurement - self.h(self.predicted_state))
        self.covariance = (np.eye(self.number_of_states) - K_gain @ C) @ self.predicted_covariance

    # State transition function
    def f(self, state, control):
        return self.f_func(state, control)
    
    # Measurement function
    def h(self, state):
        return self.h_func(state)
    
    # Jacobian of the state transition function
    def A(self, state, control):
        return self.A_func(state, control)
    
    # Jacobian of the measurement function
    def C(self, state):
        return self.C_func(state)


class Extended_Kalman_Filter_SLAM(Extended_Kalman_Filter):
    def __init__(self, f_func, h_func, A_func, C_func, number_of_states=3, number_of_measurements=3, init_state=None, init_covariance=None, process_noise=None, measurement_noise=None):
        
        if number_of_states < 3:
            raise ValueError("number_of_states must be at least 3 for SLAM.")
        elif (number_of_states - 3) % 2 != 0: # Ensure landmarks are in pairs (x, y)
            raise ValueError("number_of_states minus 3 must be an even integer for SLAM (to account for landmark positions).")
        
        super().__init__(f_func, h_func, A_func, C_func, number_of_states, number_of_measurements, init_state, init_covariance, process_noise, measurement_noise)

    # def predict(self, control):
    #     F_x = np.zeros((3, self.number_of_states))
    #     F_x[:, :3] = np.eye(3)


    # def update(self, measurement, landmark_index):
    #     super().update(measurement)

    
    def add_landmark(self, landmark_position, initial_uncertainty=1.0):

        if landmark_position.shape != (2,) and landmark_position.shape != (2, 1):
            raise ValueError("landmark_position must be a 2D vector (x, y).")
        
        # Expand state vector
        self.state = np.vstack((self.state, landmark_position.reshape(-1, 1)))
        
        # Expand covariance matrix
        old_size = self.covariance.shape[0]
        new_size = old_size + 2  # Assuming landmark position is 2D (x, y)
        
        new_covariance = np.zeros((new_size, new_size))
        new_covariance[:old_size, :old_size] = self.covariance
        new_covariance[old_size:, old_size:] = np.eye(2) * initial_uncertainty
        
        self.covariance = new_covariance
        
        # Update number of states
        self.number_of_states = new_size
