import torch
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# This function generates tasks for the model to learn
# The task we will model is a simple working memory task. 
# The model is presented with a single input, followed by a delay period, followed by a response period.
# The input to the model is the stimulus and the output is the response.
def task(T_stim, T_delay, T_resp, batch_size, stim_dim):
    # Find the total length of the trial
    T_tot = T_stim + T_delay + T_resp
    # Create the stimulus input to the model
    X = torch.zeros(batch_size, T_tot, stim_dim + 1)
    # Create the target output of the model
    Y_tar = torch.zeros(batch_size, T_tot, stim_dim)

    # First fix the go cue for each trial:
    # The go cue is the time at which the model should start to produce a response, and occurs at T_stim + T_delay
    X[:, T_stim + T_delay:, stim_dim] = 1

    # Now generate the stimulus for each trial
    # The stimulus is a von Mises bump with concentration 2 centered at a random orientation
    sample_orientations = torch.linspace(0, 2 * np.pi, stim_dim + 1)[:-1]
    theta = torch.rand(batch_size) * 2 * np.pi
    bumps = stats.vonmises.pdf(sample_orientations[None,:], 1, theta[:,None])
    X[:, :T_stim, :stim_dim] = torch.from_numpy(bumps).float()[:,None,:]

    Y_tar[:, T_stim + T_delay:, :] = torch.from_numpy(bumps).float()[:,None,:]

    return X, Y_tar

# This function plots an example trial from the task
def plot_trial(T_stim, T_delay, T_resp, stim_dim):
    X, Y_tar = task(T_stim, T_delay, T_resp, 1, stim_dim)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(X[0,:,:stim_dim].T, aspect='auto')
    plt.title('Stimulus')
    plt.xlabel('Time')
    plt.ylabel('Neuron')
    plt.subplot(1, 2, 2)
    plt.imshow(Y_tar[0].T, aspect='auto')
    plt.title('Target response')
    plt.xlabel('Time')
    plt.ylabel('Neuron')
    plt.show()