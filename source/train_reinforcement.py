"""
Course: Machine Learning 2023/2024
Students :
Alberti Andrea	0622702370	a.alberti2@studenti.unisa.it
Attianese Carmine 0622702355 c.attianese13@studenti.unisa.it
Capaldo Vincenzo 0622702347 v.capaldo7@studenti.unisa.it
Esposito Paolo 0622702292 p.esposito57@studenti.unisa.it
 """
import argparse
import logging
import math
import os
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from agent import Agent
from players import HighPlayerInterface
from utils.noise import OrnsteinUhlenbeckActionNoise
from utils.replay_memory import ReplayMemory, Transition
from client import Client, DEFAULT_PORT

# Definition of input and output dimensions
INPUT_DIM = 14  # state variables used as input (9 to 22)
OUTPUT_DIM = 2  # Actions to predict (j9,j10)


# Function to calculate the reward
def get_reward(state, next_state, environment, hit=False):
    reward = 0

    if hit:
        old_score = state[34] + state[35]  # sum of the previous score
        reward += 5 # Positive reward for hitting the ball
        new_state1 = environment.get_state()
        new_score = new_state1[34] + new_state1[35] # sum of the new score
        while old_score == new_score:
            new_state1 = environment.get_state()
            new_score = new_state1[34] + new_state1[35]

        if new_state1[33] == 0.0: # If the ball is out of bounds
            reward += -20  # If the ball goes out of bounds gives a negative reward
        else:
            reward += 20  # If the ball goes into the opponent's court then gives a positive reward.
    else:
        py = next_state[18] # y position of the ball
        ry = next_state[12] # y position of the paddle

        # Not impressed
        if py < ry: # If the ball is after the paddle, it has not been hit
            reward += -5  # If the ball is not hit at all gives a negative reward

    return reward


# Create logger
log_directory = "./logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
# Creates a file handler that writes log messages to a log file
log_file_path = os.path.join(log_directory, 'training_log.log')
file_handler = logging.FileHandler(log_file_path)
# Add the file handler to the logger
logger.addHandler(file_handler)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Parse given arguments
parser = argparse.ArgumentParser()
parser.add_argument("--load_model", default=False, type=bool,
                    help="Load a pretrained model (default: False)")
parser.add_argument("--save_dir", default="./saved_models/high/",
                    help="Dir. path to save and load a model (default: ./saved_models/)")
parser.add_argument("--timesteps", default=1e4, type=int,
                    help="Num. of total timesteps of training (default: 1e6)")
parser.add_argument("--batch_size", default=32, type=int,
                    help="Batch size (default: 64; OpenAI: 128)")
parser.add_argument("--replay_size", default=1e6, type=int,
                    help="Size of the replay buffer (default: 1e6; OpenAI: 1e5)")
parser.add_argument("--gamma", default=0.99,
                    help="Discount factor (default: 0.99)")
parser.add_argument("--tau", default=0.001,
                    help="Update factor for the soft update of the target networks (default: 0.001)")
parser.add_argument("--noise_stddev", default=0.2, type=int,
                    help="Standard deviation of the OU-Noise (default: 0.2)")
parser.add_argument("--hidden_size", nargs=3, default=[128, 128, 64], type=tuple,
                    help="Num. of units of the hidden layers (default: [128, 128, 64]; OpenAI: [64, 64])")
parser.add_argument("--n_test_cycles", default=10, type=int,
                    help="Num. of episodes in the evaluation phases (default: 10; OpenAI: 20)")
args = parser.parse_args()

# Device Setting (GPU or CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger.info("Using {}".format(device))

if __name__ == "__main__":

    # Define the directory where to save and load models
    checkpoint_dir = args.save_dir
    writer = SummaryWriter('runs/run_1')

    # Define the reward threshold when the task is solved (if existing) for model saving
    reward_threshold = np.inf

    # Initialize OU-Noise, JOINTS=output_dim
    ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(OUTPUT_DIM),
                                            sigma=float(args.noise_stddev) * np.ones(OUTPUT_DIM))

    # Define and build DDPG agent STATE_DIMENSION=input_dim
    hidden_size = tuple(args.hidden_size)
    agent = Agent(args.gamma, args.tau, hidden_size, INPUT_DIM, OUTPUT_DIM, args.batch_size,
                  checkpoint_dir=checkpoint_dir)

    # Initialize replay memory
    memory = ReplayMemory(int(args.replay_size))

    # Define counters and other variables
    start_step = 0
    # timestep = start_step
    if args.load_model:
        # Load agent if necessary
        start_step, memory = agent.load_checkpoint()

    timestep = start_step // 2 + 1
    rewards, policy_losses, value_losses, mean_train_rewards = [], [], [], []
    max_reward = -np.inf
    epoch = 0
    t = 1
    time_last_checkpoint = time.time()

    # Setting Environment
    env = Client('Train Client', 'localhost', DEFAULT_PORT)
    player = HighPlayerInterface()
    threshold_distance = 0.3  # Defining the distance to determine if the ball is close to the paddle
    # Start training
    logger.info('Train agent')
    logger.info('Doing {} timesteps'.format(args.timesteps))
    logger.info('Start at timestep {0} with t = {1}'.format(timestep, t))
    logger.info('Start training at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))

    while timestep <= args.timesteps:
        ou_noise.reset()
        epoch_return = 0

        initial_state = env.get_state()
        while initial_state[26] == 1.0:  # Waiting for the game to start
            initial_state = env.get_state()

        print(f"episodio num {timestep}")

        while True:
            px, py, pz = initial_state[11:14]  # paddle position
            bx, by, bz = initial_state[17:20]  # ball position
            vy = initial_state[21]             # speed of the ball
            done = False
            distance_from_ball = math.sqrt((bx - px) ** 2 + (by - py) ** 2 + (bz - pz) ** 2)
            if distance_from_ball <= threshold_distance and vy < 0:  # If the ball is close to the paddle then the agent is actived
                state_np = initial_state[9:23]
                actions = player.update(initial_state)
                state_tensor = torch.Tensor(np.array([state_np])).to(device)
                actions_tensor = agent.calc_action(state_tensor, ou_noise)
                actions[9] = state_np[9] + actions_tensor[0][0].item()
                actions[10] = state_np[10] + actions_tensor[0][1].item()
                env.send_joints(actions)
                initial_next_state = env.get_state()

                if initial_state[31] == 0 and initial_next_state[31]:  # If the ball was hit
                    reward = get_reward(initial_state, initial_next_state, env, hit=True)
                    done = True
                else:
                    reward = get_reward(initial_state, initial_next_state, env)

                if reward != 0:
                    print("Reward -> ", reward)
                    epoch_return += reward
                    next_state_np = initial_next_state[9:23]
                    next_state_tensor = torch.Tensor(np.array([next_state_np])).to(device)
                    mask = torch.Tensor([done]).to(device)
                    reward = torch.Tensor([reward]).to(device)
                    memory.push(state_tensor, actions_tensor, mask, next_state_tensor, reward)

            else:
                actions = player.update(initial_state)
                env.send_joints(actions)
                new_state = env.get_state()
                initial_next_state = env.get_state()

            initial_state = initial_next_state
            epoch_value_loss = 0
            epoch_policy_loss = 0

            if len(memory) > args.batch_size:
                transitions = memory.sample(args.batch_size)

                batch = Transition(*zip(*transitions))

                # Update actor and critic according to the batch
                value_loss, policy_loss = agent.update_params(batch)
                epoch_value_loss += value_loss
                epoch_policy_loss += policy_loss

            if done:
                timestep += 1
                break

        rewards.append(epoch_return)
        value_losses.append(epoch_value_loss)
        policy_losses.append(epoch_policy_loss)
        writer.add_scalar('epoch/return', epoch_return, epoch)

        # Test every 10th episode (== 1e4) steps for a number of test_epochs epochs
        if timestep >= 100 * t:
            t += 1
            mean_train_rewards.append(np.mean(rewards[-100:]))
            for name, param in agent.actor.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
            for name, param in agent.critic.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

            writer.add_scalar('train/mean_train_return', mean_train_rewards[-1], epoch)
            logger.info("Epoch: {}, current timestep: {}, last reward: {}, "
                        "mean reward: {}, mean train reward {}".format(epoch,
                                                                       timestep,
                                                                       rewards[-1],
                                                                       np.mean(rewards[-100:]),
                                                                       mean_train_rewards))

            # Save if the mean of the last three averaged rewards while testing is greater than the specified reward threshold
            if mean_train_rewards[-1] >= max_reward:
                agent.save_checkpoint(timestep, memory)
                time_last_checkpoint = time.time()
                logger.info('Saved model at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))
                max_reward = max(mean_train_rewards)
        epoch += 1

    agent.save_checkpoint(timestep, memory)
    logger.info('Saved final model at endtime {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))
    logger.info('Stopping training at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))
    env.close()
