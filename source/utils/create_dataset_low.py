"""
Course: Machine Learning 2023/2024
Students :
Alberti Andrea	0622702370	a.alberti2@studenti.unisa.it
Attianese Carmine 0622702355 c.attianese13@studenti.unisa.it
Capaldo Vincenzo 0622702347 v.capaldo7@studenti.unisa.it
Esposito Paolo 0622702292 p.esposito57@studenti.unisa.it
 """
import numpy as np
import sys
import math
import time
from client import Client, JOINTS, DEFAULT_PORT

def run(cli):
    # Open the file to save the dataset
    with open("../saved_dataset/dataset_low.csv", "w") as f:

        # Initialize joint positions
        jp = [0.0] * JOINTS
        jp[2] = math.pi
        jp[9] = math.pi / 3.5
        jp[10] = math.pi / 2
        cli.send_joints(jp)

        # Wait for 3 seconds to ensure the robot is in the correct position
        time.sleep(3)

        # Define the step size for joint movement
        step = math.pi / 90

        # Define the range for jp3, jp5, and jp7
        range_jp3 = np.arange(0, math.pi / 4 + step, step)
        range_jp5 = np.arange(0, math.pi / 2 + step, step)
        range_jp7 = np.arange(0, math.pi / 2 + step, step)

        # Iterate over the range of jp3
        for jp3 in range_jp3:
            # Iterate over the range of jp5
            for jp5 in range_jp5:
                # Iterate over the range of jp7
                for jp7 in range_jp7:
                    jp[3] = jp3  # Assign the value of joint 3 (pitch of the first arm link)
                    jp[5] = jp5  # Assign the value of joint 5 (pitch of the second arm link)
                    jp[7] = jp7  # Assign the value of joint 7 (pitch of the third arm link)

                    # Calculate jp9 to keep the paddle angle constant (downward)
                    jp[9] = - (jp3 + jp5 + jp7) + (math.pi - math.pi / 3.5)

                    # Send the joint values to the server
                    cli.send_joints(jp)

                    # Wait for 1 second to ensure the joints are in the correct position
                    time.sleep(1)

                    # Get the current state from the server
                    state = cli.get_state()

                    # Check if z is positive and jp9 is within the valid range
                    if state[13] >= 0.0 and (-3 / 4) * math.pi < jp[9] < (3 / 4) * math.pi:
                        # Write coordinates (y, z) and joint angles (jp3, jp5, jp7) to the file
                        f.write(f"{state[12]},{state[13]},{state[3]},{state[5]},{state[7]}\n")
                        f.flush()

def main():

    name = 'Dataset low'
    if len(sys.argv) > 1:
        name = sys.argv[1]

    port = DEFAULT_PORT
    if len(sys.argv) > 2:
        port = sys.argv[2]

    host = 'localhost'
    if len(sys.argv) > 3:
        host = sys.argv[3]

    cli = Client(name, host, port)
    run(cli)

if __name__ == '__main__':

    main()
