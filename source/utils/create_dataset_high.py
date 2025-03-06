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
    with (open("../saved_dataset/dataset_high.csv", "w") as f):

        # Initialize joint positions
        jp = np.zeros((JOINTS,))
        jp[0] = 0.0
        jp[2] = math.pi
        jp[10] = math.pi / 2
        cli.send_joints(jp)

        # Wait for 3 seconds to ensure the robot is in the correct position
        time.sleep(3)

        # Define the step size for joint movement (1 degree)
        step = math.pi / 180

        # Define the range for jp5 and jp7
        range_jp5 = np.arange(0, math.pi / 2 + step, step)
        range_jp7 = np.arange(0, math.pi / 2 + step, step)

        # Iterate over the range of jp5
        for jp5 in range_jp5:
            # Iterate over the range of jp7
            for jp7 in range_jp7:
                jp[5] = jp5  # Assign the value of joint 5 (pitch of the second arm link)
                jp[7] = jp7  # Assign the value of joint 7 (pitch of the third arm link)

                # Calculate jp9 to keep the paddle angle constant (upward)
                jp[9] = - (jp[5] + jp[7])

                # Send joint positions to the robot
                cli.send_joints(jp)

                # Wait for 1 second to ensure the joints are in the correct position
                time.sleep(1)

                # Get the current state of the robot
                state = cli.get_state()

                # Check if jp9 is within the valid range
                if (-3 / 4) * math.pi < jp[9] < (3 / 4) * math.pi:
                    # Write coordinates (z) and joint angles (jp5, jp7) to the file
                    f.write(f"{state[13]},{state[5]},{state[7]}\n")
                    f.flush()

def main():

    name = 'Dataset high'
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
