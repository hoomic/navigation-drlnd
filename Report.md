#Learning Algorithm

This repository contains 3 main files that allow the agent to learn to collect bananas in the environment

### model.py
This file contains the class definitions for the Q-Networks. The classes derive from torch.nn.Module and they define the neural network in pytorch. 

The class QNetwork is a fully connected neural network that is defined by its hidden layer sizes. QNetwork can take a list of arbitrary length that contains the sizes for each hidden layer. Each layer in this network is followed by a ReLU activation function except for the output layer. 

The class DuelingQNetwork is a fully connected neural network followed by two dueling streams corresponding to the value and advantage function that sum to make the Q function. Similar to the QNetwork, this class is defined by its hidden layer sizes and its dueling hidden layer sizes. The network is fully connected through the hidden layers, then it is split into an advantage and a value stream which are each fully connected, but separated from each other. The output of this network is the sum of the value stream and the advantage stream minus the mean of the advantage stream. 

### dqn_agent.py
This file contains the hyperparameters for the agent as well as the class definitions for the agent and the experience replay buffer.

The class Agent contains two Q-Networks that can either be the class QNetwork or the class DuelingQNetwork. One of these networks is the local, and the other is the target. The local network is updated on every step once there are enough samples in the replay buffer to make a batch. The target network is updated every UPDATE_EVERY steps so that it is sightly closer to the local version by the following equation: 

```target_parameters = TAU * local_parameters + (1 - TAU) * target_parameters```

The target network can be used in two ways depending on the hyperparameter DOUBLE_DQN. When training the local network, the target Q-value is based on the reward received plus the discounted Q-value of the next state. When DOUBLE_DQN is True, the Q-value of the next state is calculated based on the parameters of the target network and the action chosen by the *local* network. When DOUBLE_DQN is False, the Q-value of the next state is calculated based on the target network parameters and the action chosen by the *target* network.


The class ReplayBuffer keeps a memory of the last BUFFER_SIZE experiences and allows the agent to randomly sample from these experiences. When PRIORITIZED_EXPERIENCE is True, whenever the agent takes an action, it also records the absolute difference between its predicted Q-value and its target Q-value, and these priority values are used to put a probability distribution over the collected experiences so that observations where the model's prediction was close to the target have a lesser probability of being sampled than observations where the model's prediction was far from the target. If PRIORITIZED_EXPERIENCE is False, then the experiences are sampled uniformly.

### navigation.py
This file contains the training loop for the agent. This is where the interaction between the agent and the environment actually takes place. 
