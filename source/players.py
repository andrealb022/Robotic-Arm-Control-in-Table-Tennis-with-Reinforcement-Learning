"""
Course: Machine Learning 2023/2024
Students :
Alberti Andrea	0622702370	a.alberti2@studenti.unisa.it
Attianese Carmine 0622702355 c.attianese13@studenti.unisa.it
Capaldo Vincenzo 0622702347 v.capaldo7@studenti.unisa.it
Esposito Paolo 0622702292 p.esposito57@studenti.unisa.it
 """
import math
import joblib
import numpy as np
import torch
from server import PlayerInterface, JOINTS
from utils.nets import SupervisedNeuralNetwork


def predict_joints(input_data, model, scaler_X, scaler_y):
    # Scale the input data using the input scaler
    scaled_input = scaler_X.transform(input_data)
    # Convert the scaled input to a PyTorch tensor
    scaled_input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
    # Disable gradient calculation for efficiency
    with torch.no_grad():
        # Make a prediction with the neural network model
        prediction_scaled = model(scaled_input_tensor)
    # Inverse scale the prediction to get it back to the original scale
    prediction = scaler_y.inverse_transform(prediction_scaled.numpy())
    # Flatten the prediction and return it
    return prediction.flatten()



class HighPlayerInterface(PlayerInterface):
    def __init__(self, agent=None):
        self.agent = agent
        # Load the agent checkpoint if an agent is provided
        if agent is not None:
            agent.load_checkpoint()
        # Set the threshold distance for deciding proximity actions
        self.threshold_distance = 0.3
        # Initialize and load the neural network model
        self.model = SupervisedNeuralNetwork(1, 2)
        self.model.load_state_dict(torch.load('saved_models/supervised/model_high.pth'))
        self.model.eval()  # Set the model to evaluation mode

        # Load the input and output scalers
        self.scaler_X = joblib.load('saved_models/supervised/high_scaler_X.pkl')
        self.scaler_y = joblib.load('saved_models/supervised/high_scaler_y.pkl')

    def update(self, state):
        # Extract relevant state variables
        px, py, pz = state[11:14]
        bx, by, bz = state[17:20]
        vx, vy, vz = state[20:23]

        jp = self.get_neutral_joint_position()

        # Check if the game is playing, the ball touch player's court and the ball is coming
        if state[28] and vy <= 0 and state[30]:
            # Choose the z-coordinate for the joint prediction
            z = self.choose_z(state)
            if z is not None:
                # Predict the joint positions based on the chosen z-coordinate
                joints = predict_joints(np.array([[z]]), self.model, self.scaler_X, self.scaler_y)
                jp[5] = joints[0]
                jp[7] = joints[1]
                jp[9] = - (jp[5] + jp[7])
                # Choose the final position for the joints
                self.choose_position(state, jp)
        else:
            # If conditions are not met, set joint position based on ball's x-coordinate
            jp[1] = bx

        # If an agent is provided, calculate additional actions
        if self.agent is not None:
            # Calculate the distance from the ball
            distance_from_ball = math.sqrt((bx - px) ** 2 + (by - py) ** 2 + (bz - pz) ** 2)
            # If the ball is close and is coming
            if distance_from_ball <= self.threshold_distance and vy < 0:
                # Convert state to a tensor for the agent
                state_np = state[9:23]
                state_tensor = torch.Tensor(np.array([state_np]))
                # Calculate the actions using the agent
                actions_tensor = self.agent.calc_action(state_tensor)
                # Adjust joint positions based on the agent's actions, paddle's pitch and roll
                jp[9] = state_np[0] + actions_tensor[0][0].item()
                jp[10] = state_np[1] + actions_tensor[0][1].item()
        return jp

    def choose_position(self, state, jp):
        # Extract relevant state variables
        px, py, pz = state[11:14]
        bx, by, bz = state[17:20]
        vx, vy, vz = state[20:23]

        # Calculate the distance between player and ball
        dist = math.hypot(px - bx, py - by, pz - bz)
        # Calculate the ball's velocity
        vel = math.hypot(vx, vy, vz)

        # Set joint position based on ball's x-coordinate
        if state[27] or vel < 0.05:
            jp[1] = bx
            return

        # Adjust joint position based on proximity to ball
        extra_y = 0.0
        if dist < vel * 1.5 * 0.02:
            extra_y = 0.3

        # Simulation step size
        d = 0.05
        # Gravitational constant
        g = 9.81

        # Simulate ball trajectory to predict future positions
        while vz > 0 or bz + d * vz >= pz:
            bx += d * vx
            by += d * vy
            bz += d * vz
            vz -= d * g

        # Set the joint positions based on the predicted ball positions
        jp[1] = bx
        dy = py - state[0]
        jp[0] = by - dy + extra_y

    def choose_z(self, state):
        # Extract relevant state variables
        px, py, pz = state[11:14]
        bx, by, bz = state[17:20]
        vx, vy, vz = state[20:23]

        # Calculate the distance between player and ball
        dist = math.hypot(px - bx, py - by, pz - bz)

        # If the distance is too small, return None
        if dist < 0.1:
            return None

        # Gravitational constant and simulation step size
        g = 9.81
        d = 0.05

        # Initialize minimum z-coordinate
        min_z = bz

        # Simulate ball trajectory
        while vz > 0 or bz + d * vz >= pz:
            by += vy * d
            bz += vz * d
            vz -= g * d
            min_z = min(min_z, bz)

        # Return the z-coordinate at the end of simulate
        return min_z

    def get_neutral_joint_position(self):
        jp = np.zeros((JOINTS,))
        jp[0] = -0.3
        jp[2] = math.pi
        jp[5] = math.pi / 4
        jp[7] = math.pi / 4
        jp[9] = - (jp[5] + jp[7])
        jp[10] = math.pi / 2
        return jp


class LowPlayerInterface(PlayerInterface):
    def __init__(self):
        # Initialize and load the neural network model for low player interface
        self.model = SupervisedNeuralNetwork(2, 3)
        self.model.load_state_dict(torch.load('saved_models/supervised/model_low.pth'))
        self.model.eval()  # Set the model to evaluation mode
        self.mode = 'low'  # Set the initial mode to 'low'

        # Load the input and output scalers
        self.scaler_X = joblib.load('saved_models/supervised/low_scaler_X.pkl')
        self.scaler_y = joblib.load('saved_models/supervised/low_scaler_y.pkl')

    def update(self, state):
        # Extract relevant state variables
        bx, by, bz = state[17:20]
        vx, vy, vz = state[20:23]

        # Determine the mode based on if the player is waiting for opponent service, otherwise the function choose_mod
        if state[27]:
            self.mode = 'low'
        else:
            self.choose_mod(state)

        # Get the neutral joint positions
        jp = self.get_neutral_joint_position()

        # Check if the game is playing, the ball touch player's court and the ball is coming
        if state[28] and vy <= 0 and state[30]:
            y, z = self.choose_y_z(state)
            if y is not None and z is not None:
                # Predict the joint positions based on chosen y and z coordinates
                joints = predict_joints(np.array([[y, z]]), self.model, self.scaler_X, self.scaler_y)
                jp[3] = joints[0]
                jp[5] = joints[1]
                jp[7] = joints[2]
                jp[9] = - (jp[3] + jp[5] + jp[7]) + (math.pi - math.pi / 3.5)

        # Choose the final position for the joints
        self.choose_position(state, jp)
        return jp

    def choose_position(self, state, jp):
        # Extract relevant state variables
        px, py, pz = state[11:14]
        bx, by, bz = state[17:20]
        vx, vy, vz = state[20:23]

        # Calculate the distance between player and ball
        dist = math.hypot(px - bx, py - by, pz - bz)
        # Calculate the ball's velocity
        vel = math.hypot(vx, vy, vz)

        # Set joint position based on ball's x-coordinate
        if state[27] or vel < 0.05:
            jp[1] = bx
            return

        # Adjust joint position based on proximity to ball
        extra_y = 0.0
        if dist < vel * 1.5 * 0.02:
            extra_y = 0.3

        # Simulation step size
        d = 0.05
        # Gravitational constant
        g = 9.81

        # Simulate ball trajectory to predict future positions
        while vz > 0 or bz + d * vz >= pz:
            bx += d * vx
            by += d * vy
            bz += d * vz
            vz -= d * g

        # Set the joint positions based on the predicted ball positions
        jp[1] = bx
        dy = py - state[0]
        jp[0] = by - dy + extra_y

    def choose_y_z(self, state):
        # Extract relevant state variables
        px, py, pz = state[11:14]
        bx, by, bz = state[17:20]
        vx, vy, vz = state[20:23]

        # Calculate the distance between player and ball
        dist = math.hypot(px - bx, py - by, pz - bz)

        # If the distance is too small, return None
        if dist < 0.1:
            return None, None

        # Gravitational constant and simulation step size
        g = 9.81
        d = 0.05

        # Initialize minimum z-coordinate
        min_z = bz

        # Simulate ball trajectory
        while vz > 0 or bz + d * vz >= pz:
            by += vy * d
            bz += vz * d
            vz -= g * d
            min_z = min(min_z, bz)

        # Return the y and z-coordinates at the end of simulate
        return by, min_z

    def get_neutral_joint_position(self):
        jp = np.zeros((JOINTS,))
        jp[2] = math.pi
        jp[10] = math.pi / 2
        if self.mode == 'medium':
            jp[5] = math.pi / 3.8
            jp[7] = math.pi / 3.8
            jp[9] = math.pi / 3.5
        elif self.mode == 'low':
            jp[3] = math.pi / 4
            jp[5] = math.pi / 6
            jp[7] = math.pi / 12
            jp[9] = math.pi / 3.5
        return jp

    def choose_mod(self, next_state):
        # Extract relevant state variables
        px, py, pz = next_state[11:14]
        bx, by, bz = next_state[17:20]
        vx, vy, vz = next_state[20:23]

        # Calculate the distance between player and ball
        dist = math.hypot(px - bx, py - by, pz - bz)

        # Return without changing mode
        if not next_state[28] or dist < 0.1:
            return

        # If the ball is going to the other court and close to the net, set mode to 'low'
        if vy > 0 and 0.95 < by < 1.05:
            self.mode = 'low'

        # If the ball is coming and close to the net
        if vy < 0 and 0.95 < by < 1.05:
            # If the ball is tight, set mode to 'low'
            if 0.04 < bz <= 0.4 and -0.3 < vz < 0.3:
                self.mode = 'low'
            else:
                # Otherwise, set mode to 'medium'
                self.mode = 'medium'


class FinalPlayer(PlayerInterface):
    def __init__(self, agent_high=None):
        # Initialize chosen player index
        self.chosen_player = 0

        # Initialize low and high player interfaces
        self.players = [LowPlayerInterface(), HighPlayerInterface(agent_high)]

        # Initialize the state with zeros
        self.state = np.zeros((37,))

    def update(self, state):
        # Extract player coordinates
        px, py, pz = state[11:14]
        # Extract ball coordinates
        bx, by, bz = state[17:20]
        # Extract ball velocity components
        vy, vz = state[21:23]

        # Choose the appropriate player (low or high) based on the current state
        self.choose_player(state)

        # Update the chosen player and get the joint positions
        jp = self.players[self.chosen_player].update(state)

        # If the ball has not hit my table (state[30] == 0.0), is coming (vy < 0),
        # and either the ball is behind the table (by <= -0.2) or has passed my paddle (by < py),
        # or if the ball cannot touch my field in the final part (by <= 0.4 and vz >= 0),
        # then move the paddle away
        if state[30] == 0.0 and vy < 0 and (by <= -0.2 or by < py or (by <= 0.4 and vz >= 0)):
            # Move the paddle to a safe position based on ball's x-coordinate
            jp[1] = 0.8 if bx <= 0 else -0.8
        else:
            # Otherwise, set the joint position based on ball's x-coordinate
            jp[1] = bx

        # Return the final joint positions
        return jp

    def choose_player(self, next_state):
        # Extract relevant state variables
        px, py, pz = next_state[11:14]
        bx, by, bz = next_state[17:20]
        vy = next_state[21]

        # Calculate the distance between player and ball
        dist = math.hypot(px - bx, py - by, pz - bz)

        # If the distance is too small, return without changing the player
        if dist < 0.1:
            return

        # If waiting for the opponent's serve, choose the high player
        if next_state[27]:
            self.chosen_player = 1

        # If the ball is moving to the other side and is in the other court, choose the low player
        if vy > 0 and self.state[29] and next_state[29] == 0.0:
            self.chosen_player = 0

        # Update the state
        self.state = next_state
