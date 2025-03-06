import math
import time
import joblib
import numpy as np
import sys
import torch
from client import Client, JOINTS, DEFAULT_PORT
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

def run(cli):
    # Initialize joint positions
    jp = np.zeros((JOINTS,))
    jp[0] = 0.0
    jp[2] = math.pi
    jp[10] = math.pi / 2

    # Send initial joint positions to the client
    cli.send_joints(jp)

    # Initialize the neural network model
    model = SupervisedNeuralNetwork(1, 2)
    # Load the pretrained model state
    model.load_state_dict(torch.load('saved_models/supervised/model_high.pth'))
    # Load the input and output scalers
    scaler_X = joblib.load('saved_models/supervised/high_scaler_X.pkl')
    scaler_y = joblib.load('saved_models/supervised/high_scaler_y.pkl')

    # Set the target z coordinate
    z_target = 1.0

    # Predict the joint positions for the target z coordinate
    joints = predict_joints(np.array([[z_target]]), model, scaler_X, scaler_y)
    # Update the joint positions with the predicted values
    jp[5] = joints[0]
    jp[7] = joints[1]
    jp[9] = - (jp[5] + jp[7])

    # Send the updated joint positions to the client
    cli.send_joints(jp)

    # Wait for 1 second
    time.sleep(1)

    # Get the current state from the client
    state = cli.get_state()

    # Calculate the error between the current z coordinate and the target z coordinate
    error_z = abs(state[13] - z_target)
    # Print the error
    print("error_z = ", error_z)


def main():
    name='Gruppo06 Client'
    if len(sys.argv)>1:
        name=sys.argv[1]

    port=DEFAULT_PORT
    if len(sys.argv)>2:
        port=sys.argv[2]

    host='localhost'
    if len(sys.argv)>3:
        host=sys.argv[3]
    cli=Client(name, host, port)

    run(cli)

if __name__ == '__main__':

    main()