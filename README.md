# Robotic-Arm-Control-in-Table-Tennis-with-Reinforcement-Learning
This project report outlines the development and training of neural networks to control a robotic arm for playing table tennis.  The project employs supervised learning for inverse kinematics and reinforcement learning to optimize the robot’s movements.

## How to Run:
\
To run the server:
`python .\src\server.py`
\
All the possible options for the server are avaiable at interface.txt.

To play with the trained model:
\
`python .\src\test.py`
\

### Supervised Learning:
To start a **Supervised learning** session:
- High Player: `python .\src\train_supervised_high.py`;
- Low Player: `python .\src\train_supervised_low.py`;

#### Reinforcement Learning:
To start a **reinforcement learning** session, you need to start the server and connect two players:
- 1 player: `python .\src\server.py -auto`;
- Player to train: `python .\src\train_reinforcement.py`;
