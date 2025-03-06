"""
Course: Machine Learning 2023/2024
Students :
Alberti Andrea	0622702370	a.alberti2@studenti.unisa.it
Attianese Carmine 0622702355 c.attianese13@studenti.unisa.it
Capaldo Vincenzo 0622702347 v.capaldo7@studenti.unisa.it
Esposito Paolo 0622702292 p.esposito57@studenti.unisa.it
 """
from agent import Agent
from client import Client, DEFAULT_PORT
import sys
from players import FinalPlayer


def run(client, player):
    while True:
        state = client.get_state()
        jp = player.update(state)
        client.send_joints(jp)


def main():
    name = 'Gruppo06'
    if len(sys.argv) > 1:
        name = sys.argv[1]

    port = DEFAULT_PORT
    if len(sys.argv) > 2:
        port = sys.argv[2]

    host = 'localhost'
    if len(sys.argv) > 3:
        host = sys.argv[3]

    agent_high = Agent(0.99, 0.001, [128, 128, 64], 14, 2, 32, checkpoint_dir="./saved_models/high")
    player = FinalPlayer(agent_high)
    client = Client(name, host, port)
    run(client, player)


if __name__ == '__main__':
    '''
    python test_supervised_high.py name port host
    Default parameters:
     name: 'Example Client'
     port: client.DEFAULT_PORT
     host: 'localhost'

    To run the one simulation on the server, run this in 3 separate command shells:
    > python test_supervised_high.py player_A
    > python test_supervised_high.py player_B
    > python server.py

    To run a second simulation, select a different PORT on the server:
    > python test_supervised_high.py player_A 9544
    > python test_supervised_high.py player_B 9544
    > python server.py -port 9544    
    '''

    main()
