"""
Course: Machine Learning 2023/2024
Students :
Alberti Andrea	0622702370	a.alberti2@studenti.unisa.it
Attianese Carmine 0622702355 c.attianese13@studenti.unisa.it
Capaldo Vincenzo 0622702347 v.capaldo7@studenti.unisa.it
Esposito Paolo 0622702292 p.esposito57@studenti.unisa.it
 """
import gc
import logging
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils.nets import Actor, Critic
from utils.custom_action_space import CustomActionSpace

# Set up the logger for debugging and information purposes
logger = logging.getLogger('ddpg')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler()) # Add a stream handler to the logger to output logs to the console

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):  # Iterate over the parameters of both networks
        # Update the target parameter as a weighted average of the target and source parameters
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):  # Iterate over the parameters of both networks
        # Directly copy the parameters from the source to the target network
        target_param.data.copy_(param.data)


class Agent(object):
    # Initialize the Agent with actor and critic networks along with optimizers
    def __init__(self, gamma, tau, hidden_size, input_dim, output_dim, batch_size, checkpoint_dir=None, lr_actor=1e-4,
                 lr_critic=1e-3):
        """
        Arguments:
            gamma:          Discount factor
            tau:            Update factor for the actor and the critic
            hidden_size:    Number of units in the hidden layers of the actor and critic. Must be of length 2.
            input_dim:     Size of the input states
            output_dim:   Size of the output states
            checkpoint_dir: Path as String to the directory to save the networks. 
                            If None then "./saved_models/" will be used
        """

        # Setting variables
        self.gamma = gamma # Set the discount factor
        self.tau = tau # Set the update factor
        self.output_dim = output_dim # Set the size of the action space
        self.batch_size = batch_size # Set the batch size for training
        self.lr_actor = lr_actor # Set the learning rate for actor network
        self.lr_critic = lr_critic # Set the learning rate for the critic network
        self.action_space = CustomActionSpace() # Initialize the custom action space

        # Set the directory to save the models
        if checkpoint_dir is None: # if no checkpoint directory is provided
            self.checkpoint_dir = "./saved_models/" # use the default directory
        else:
            self.checkpoint_dir = checkpoint_dir  # Use the provided directory
        os.makedirs(self.checkpoint_dir, exist_ok=True) # Create the directory if it does not exist
        logger.info('Saving all checkpoints to {}'.format(self.checkpoint_dir)) # Log the directory used for saving checkpoints

        # Define the actor network and its target network
        self.actor = Actor(input_dim, hidden_size, self.output_dim).to(device) # Initialize the actor network
        self.actor_target = Actor(input_dim, hidden_size, self.output_dim).to(device) # Initialize the target actor network

        # Define the critic network and its target network
        self.critic = Critic(input_dim, hidden_size, self.output_dim).to(device) # Initialize the critic network
        self.critic_target = Critic(input_dim, hidden_size, self.output_dim).to(device) # Initialize the target critic network

        # Define the optimizers for both networks
        self.actor_optimizer = Adam(self.actor.parameters(), self.lr_actor)  # optimizer for the actor network
        self.critic_optimizer = Adam(self.critic.parameters(), self.lr_critic,
                                     weight_decay=1e-2)  # optimizer for the critic network

        # Make sure both target networks start with the same weights as their respective source networks
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def calc_action(self, state, action_noise=None):
        """
        Evaluates the action to perform in a given state

        Arguments:
            state:          State to perform the action on in the env. 
                            Used to evaluate the action.
            action_noise:   If not None, the noise to apply on the evaluated action
        """
        x = state.to(device) # Move the state to the appropriate device

        # Get the continuous action value to perform in the environment
        self.actor.eval()  # Set the actor in evaluation mode
        out = self.actor(x) # Compute the action using the actor network
        self.actor.train()  # Sets the actor in training mode
        out = out.data  # Get the data from the action tensor

        # During training, we add noise for exploration
        if action_noise is not None:
            noise = torch.Tensor(action_noise.noise()).to(device)
            out += noise  # add the noise to the action
        # Clip the output according to the action space of the env
        out = self.action_space.bound_actions(out)
        return out # return the computed action

    def update_params(self, batch):
        """
        Updates the parameters/networks of the agent according to the given batch.
        This means we ...
            1. Compute the targets
            2. Update the Q-function/critic by one step of gradient descent
            3. Update the policy/actor by one step of gradient ascent
            4. Update the target networks through a soft update

        Arguments:
            batch:  Batch to perform the training of the parameters
        """
        # Get tensors from the batch
        state_batch = torch.cat(batch.state).to(device)  # Concatenate and move the state batch to the appropriate device
        action_batch = torch.cat(batch.action).to(device)  # Concatenate and move the action batch to the appropriate device
        reward_batch = torch.cat(batch.reward).to(device)  # Concatenate and move the reward batch to the appropriate device
        done_batch = torch.cat(batch.done).to(device)  # Concatenate and move the done batch to the appropriate device
        next_state_batch = torch.cat(batch.next_state).to(device)  # Concatenate and move the next state batch to the appropriate device

        # Get the actions and the state values to compute the targets
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch.detach())

        # Compute the target
        reward_batch = reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqueeze(1)
        expected_values = reward_batch + (1.0 - done_batch) * self.gamma * next_state_action_values  # Compute the expected values using the Bellman equation

        # Update the critic network
        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)
        value_loss = F.mse_loss(state_action_batch, expected_values.detach())
        value_loss.backward()
        self.critic_optimizer.step()

        # Update the actor network
        self.actor_optimizer.zero_grad()  # Zero the gradients of the actor optimizer
        policy_loss = -self.critic(state_batch,self.actor(state_batch))  # Compute the policy loss as the negative Q value
        policy_loss = policy_loss.mean()  # Take the mean of the policy loss
        policy_loss.backward()  # Perform backpropagation to compute the gradients
        self.actor_optimizer.step()  # Update the actor network parameters

        # Update the target networks
        soft_update(self.actor_target, self.actor, self.tau)  # Soft update the target actor network
        soft_update(self.critic_target, self.critic, self.tau)  # Soft update the target critic network

        return value_loss, policy_loss

    def save_checkpoint(self, last_timestep, replay_buffer):
        """
        Saving the networks and all parameters to a file in 'checkpoint_dir'

        Arguments:
            last_timestep:  Last timestep in training before saving
            replay_buffer:  Current replay buffer
        """
        checkpoint_name = self.checkpoint_dir + '/ep_{}.pth.tar'.format(last_timestep)
        logger.info('Saving checkpoint...')
        checkpoint = {
            'last_timestep': last_timestep,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'replay_buffer': replay_buffer,
        }
        logger.info('Saving model at timestep {}...'.format(last_timestep))
        torch.save(checkpoint, checkpoint_name)
        gc.collect()
        logger.info('Saved model at timestep {} to {}'.format(last_timestep, self.checkpoint_dir))

    def get_path_of_latest_file(self):
        """
        Returns the latest created file in 'checkpoint_dir'
        """
        files = [file for file in os.listdir(self.checkpoint_dir) if (file.endswith(".pt") or file.endswith(".tar"))]
        filepaths = [os.path.join(self.checkpoint_dir, file) for file in files]
        last_file = max(filepaths, key=os.path.getctime)
        return os.path.abspath(last_file)

    def load_checkpoint(self, checkpoint_path=None):
        """
        Saving the networks and all parameters from a given path. If the given path is None
        then the latest saved file in 'checkpoint_dir' will be used.

        Arguments:
            checkpoint_path:    File to load the model from

        """

        if checkpoint_path is None:
            checkpoint_path = self.get_path_of_latest_file()

        if os.path.isfile(checkpoint_path):
            logger.info("Loading checkpoint...({})".format(checkpoint_path))
            key = 'cuda' if torch.cuda.is_available() else 'cpu'

            checkpoint = torch.load(checkpoint_path, map_location=key)
            start_timestep = checkpoint['last_timestep'] + 1
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.actor_target.load_state_dict(checkpoint['actor_target'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            replay_buffer = checkpoint['replay_buffer']

            gc.collect()
            logger.info('Loaded model at timestep {} from {}'.format(start_timestep, checkpoint_path))
            return start_timestep, replay_buffer
        else:
            raise OSError('Checkpoint not found')

    def set_eval(self):
        """
        Sets the model in evaluation mode

        """
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()

    def set_train(self):
        """
        Sets the model in training mode

        """
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()

    def get_network(self, name):
        if name == 'Actor':
            return self.actor
        elif name == 'Critic':
            return self.critic
        else:
            raise NameError('name \'{}\' is not defined as a network'.format(name))
