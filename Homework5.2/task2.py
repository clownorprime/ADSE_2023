"""
Goal of Task 2:
    Derive speed feature from tracked data.
"""

import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from train import main_train
from bin.eval_velocity_influence import eval_velocity_influence
from bin.task2_prep import get_next_sample


class VelocityCalculator:
    # Defining __call__ method
    def __call__(self, sample=None, train_flag=False):
        """
        Function is executed on a call of the respective object e.g. example_object().

        inputs:
            sample (optional)
            train_flag (type: bool)

        given:
        previous_positions (type: np.ndarray, shape: (j, k, n)): with
            j: Length of tracked positions, e.g. 30 (= 30 timesteps observation)
            k: Batch size, e.g. 32 (= Number of trajectories for faster computing during training)
            n: Spatial dimension = 2, x-, y-Positions (relative coordinates)

        output:
            velocity array (type: np.ndarray, shape: (j, k)): with
                j: Length of tracking, so for every tracking step the associated velocity should be added
                k: Batch size, e.g. 32 (= Number of trajectories for faster computing during training)
        """

        if train_flag:
            previous_positions = sample
            t_hist = [np.arange(0, 3, 0.1)] * previous_positions.shape[1]
        elif sample is None:
            t_hist, previous_positions = get_next_sample()
        else:  # you can test your function over here (see below in __name__ == "__main__")
            t_hist, previous_positions = sample

        hist_len, n_batch, _ = previous_positions.shape

        velocity_array = np.zeros([hist_len, n_batch])
        n_interpolations = 3
        n_polyfit = 1

        # Task: Calculate the velocity from previous positions.
        ########################
        #  Start of your code  #
        ########################
        ########################
        #   End of your code   #
        ########################


def compare_performance(no_vel, with_vel):
    _, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

    n_steps = np.arange(len(no_vel["nll"]))
    ax1.plot(n_steps, no_vel["nll"], label="without velocity")
    ax1.plot(n_steps, with_vel["nll"], label="with velocity")
    ax1.set_ylabel("NLL")
    ax1.grid(True)
    ax1.legend()

    n_steps = np.arange(len(no_vel["rmse"]))
    ax2.plot(n_steps, no_vel["rmse"], label="without velocity")
    ax2.plot(n_steps, with_vel["rmse"], label="with velocity")
    ax2.set_xlabel("timesteps (dt=0.1s)")
    ax2.set_ylabel("RMSE")
    ax2.grid(True)
    ax2.legend()

    plt.show()


def plot_results(velocity_array, ax):
    ax.cla()
    hist, batch = velocity_array.shape
    for b in range(batch):
        ax.plot(np.arange(hist), velocity_array[:, b], label="your solution")
        ax.legend()
        ax.set_xlabel("timesteps (dt=0.1s)", size=14)
        ax.set_ylabel("v in m/s", size=14)
        ax.grid(True)
        ax.axis("equal")

        plt.pause(1.0)


if __name__ == "__main__":
    vel = VelocityCalculator()
    tensor_output = vel()
    n_plots = 20

    for _ in range(n_plots):
        ax = plt.gca()
        velocity_array = vel()
        plot_results(velocity_array, ax)

    ########################
    #    Optional Task     #
    ########################
    # Check out the influence of the new feature on the prediction performance
    # Uncomment the following lines and let the training run
    # Check out the results regarding the RMSE and the NLL
    # NLL describes 'how sure the net is with the prediction'
    # RMSE is the root mean squared error
    # What do you think about the results?

    # # Uncomment and run the script
    # # Net Trainig without velocity feature
    # net_weights = main_train(full_train=True)
    # no_velocity_metrics = eval_velocity_influence(net_weights)

    # # Net Trainig with (your) velocity feature
    # vel = VelocityCalculator()
    # net_weights = main_train(vel, full_train=True)
    # with_velocity_metrics = eval_velocity_influence(net_weights, vel)

    # # Comparison
    # compare_performance(no_velocity_metrics, with_velocity_metrics)
