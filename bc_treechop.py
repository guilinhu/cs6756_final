from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import gym
import minerl
from minerl.data import BufferedBatchIter



# minerl.data.download('/Users/guilinhu/Documents/cs6756/')


"""
Your task: Implement behavioural cloning for MineRLTreechop-v0.

Behavioural cloning is perhaps the simplest way of using a dataset of demonstrations to train an agent:
learn to predict what actions they would take, and take those actions.
In other machine learning terms, this is almost like building a classifier to classify observations to
different actions, and taking those actions.

For simplicity, we build a limited set of actions ("agent actions"), map dataset actions to these actions
and train on the agent actions. During evaluation, we transform these agent actions (integerse) back into
MineRL actions (dictionaries).

To do this task, fill in the "TODO"s and remove `raise NotImplementedError`s.

Note: For this task you need to download the "MineRLTreechop-v0" dataset. See here:
https://minerl.readthedocs.io/en/latest/tutorials/data_sampling.html#downloading-the-minerl-dataset-with-minerl-data-download
"""


class ConvNet(nn.Module):
    """
    :param input_shape: A three-item tuple telling image dimensions in (C, H, W)
    :param output_dim: Dimensionality of the output vector
    """

    def __init__(self, input_shape, output_dim):
        super().__init__()
        # TODO Create a torch neural network here to turn images (of shape `input_shape`) into
        #      a vector of shape `output_dim`. This output_dim matches number of available actions.
        #      See examples of doing CNN networks here https://pytorch.org/tutorials/beginner/nn_tutorial.html#switch-to-cnn
        # raise NotImplementedError("TODO implement a simple convolutional neural network here")
        self.know_output_size = self.conv_output_size(input_shape)
        
        self.convs = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.know_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def conv_output_size(self, input_shape):
        zero_tensor = torch.zeros((1, input_shape))
        example_output = self.convs(zero_tensor)
        output_dim = int(np.prod(example_output.size()))
        return output_dim


    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # TODO with the layers you created in __init__, transform the `observations` (a tensor of shape (B, C, H, W)) to
        #      a tensor of shape (B, D), where D is the `output_dim`
        # raise NotImplementedError("TODO implement forward function of the neural network")
        x = self.convs(observations)
        x = self.fc(x)
        return x


def agent_action_to_environment(noop_action, agent_action):
    """
    Turn an agent action (an integer) into an environment action.
    This should match `environment_action_batch_to_agent_actions`,
    e.g. if attack=1 action was mapped to agent_action=0, then agent_action=0
    should be mapped back to attack=1.

    noop_action is a MineRL action that does nothing. You may want to
    use this as a template for the action you return.
    """
    # raise NotImplementedError("TODO implement agent_action_to_environment (see docstring)")
    action = noop_action.copy()

    if agent_action == 0:
        action['forward'] = 1
    elif agent_action == 1:
        action['jump'] = 1
    elif agent_action == 2:
        action['camera'] = [0, -5]
    elif agent_action == 3:
        action['camera'] = [0, 5]
    elif agent_action == 4:
        action['camera'] = [-5, 0]
    elif agent_action == 5:
        action['camera'] = [5, 0]
    elif agent_action == 6:
        action['attack'] = 1

    return action


def environment_action_batch_to_agent_actions(dataset_actions):
    """
    Turn a batch of actions from environment (from BufferedBatchIterator) to a numpy
    array of agent actions.

    Agent actions _have to_ start from 0 and go up from there!

    For MineRLTreechop, you want to have actions for the following at the very least:
    - Forward movement
    - Jumping
    - Turning camera left, right, up and down
    - Attack

    For example, you could have seven agent actions that mean following:
    0 = forward
    1 = jump
    2 = turn camera left
    3 = turn camera right
    4 = turn camera up
    5 = turn camera down
    6 = attack

    This should match `agent_action_to_environment`, by converting dictionary
    actions into individual integeres.

    If dataset action (dict) does not have a mapping to agent action (int),
    then set it "-1"
    """
    # There are dummy dimensions of shape one
    batch_size = len(dataset_actions["camera"])
    actions = np.zeros((batch_size,), dtype=np.int)

    for i in range(batch_size):
        # TODO
        if dataset_actions['attack'][i] == 1:
            actions[i] = 6
        elif dataset_actions['forward'][i] == 1:
            actions[i] = 0
        elif dataset_actions['jump'][i] == 1:
            actions[i] = 1
        elif dataset_actions['camera'][i][1] < 0:
            actions[i] = 2  # turn left
        elif dataset_actions['camera'][i][1] > 0:
            actions[i] = 3  # turn right
        elif dataset_actions['camera'][i][0] < 0:
            actions[i] = 4  # look up
        elif dataset_actions['camera'][i][0] > 0:
            actions[i] = 5  # look down
        else:
            actions[i] = -1

    return actions


def train():
    # Path to where MineRL dataset resides (should contain "MineRLTreechop-v0" directory)
    DATA_DIR = "."
    # How many times we train over dataset and how large batches we use.
    # Larger batch size takes more memory but generally provides stabler learning.
    EPOCHS = 1
    BATCH_SIZE = 32

    # TODO create data iterators for going over MineRL data using BufferedBatchIterator
    #      https://minerl.readthedocs.io/en/latest/tutorials/data_sampling.html#sampling-the-dataset-with-buffered-batch-iter
    #      NOTE: You have to download the Treechop dataset first for this to work, see:
    #           https://minerl.readthedocs.io/en/latest/tutorials/data_sampling.html#downloading-the-minerl-dataset-with-minerl-data-download
    data = minerl.data.make('MineRLTreechop-v0', data_dir=DATA_DIR)
    iterator = BufferedBatchIter(data)

    number_of_actions = 7
    
    model = ConvNet((3, 64, 64), number_of_actions)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    iter_count = 0
    losses = []
    for dataset_obs, dataset_actions, _, _, _ in tqdm(iterator.buffered_batch_iter(num_epochs=EPOCHS, batch_size=BATCH_SIZE)):
        # We only use camera observations here
        obs = dataset_obs["pov"].astype(np.float32)
        # Transpose observations to be channel-first (BCHW instead of BHWC)
        obs = obs.transpose(0, 3, 1, 2)
        # Normalize observations, otherwise the neural network will get spooked
        obs /= 255.0

        # Turn dataset actions into agent actions
        actions = environment_action_batch_to_agent_actions(dataset_actions)
        assert actions.shape == (obs.shape[0],), "Array from environment_action_batch_to_agent_actions should be of shape {}".format((obs.shape[0],))

        # Remove samples that had no corresponding action
        mask = actions != -1
        obs = obs[mask]
        actions = actions[mask]

    
        obs_tensor = torch.tensor(obs).float()
        actions_tensor = torch.tensor(actions).long()

        optimizer.zero_grad()
        action_predictions = model(obs_tensor)
        loss = loss_function(action_predictions, actions_tensor)
        loss.backward()
        optimizer.step()


        # Keep track of how training is going by printing out the loss
        iter_count += 1
        losses.append(loss.item())
        if (iter_count % 1000) == 0:
            mean_loss = sum(losses) / len(losses)
            tqdm.write("Iteration {}. Loss {:<10.3f}".format(iter_count, mean_loss))
            losses.clear()

    # Store the network
    torch.save(model, "behavioural_cloning.pth")


def enjoy():
    # Load up the trained network
    model = torch.load("behavioural_cloning.pth")

    env = gym.make('MineRLTreechop-v0')

    # Play 10 games with the model
    for game_i in range(10):
        obs = env.reset()
        done = False
        reward_sum = 0
        while not done:
            pov_obs = obs['pov'].astype(np.float32)
            pov_obs = pov_obs.transpose(2, 0, 1)
            pov_obs = np.expand_dims(pov_obs, axis=0) / 255.0

            pov_tensor = torch.tensor(pov_obs).float()
            logits = model(pov_tensor)

            probabilities = torch.softmax(logits, dim=1)[0]

            probabilities = probabilities.detach().cpu().numpy()
            
            agent_action = np.random.choice(len(probabilities), p=probabilities)

            noop_action = env.action_space.noop()
            environment_action = agent_action_to_environment(noop_action, agent_action)

            obs, reward, done, info = env.step(environment_action)
            reward_sum += reward
        print("Game {}, total reward {}".format(game_i, reward_sum))

    env.close()


if __name__ == "__main__":
    train()
    enjoy()