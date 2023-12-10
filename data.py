# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as signal
from scipy.fft import fft, fftfreq

# Configure Matplotlib for LaTeX text rendering and set plot style parameters
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.5,
    "lines.markersize": 3,
    "figure.figsize": (3.39, 2.54),
})

# Define line styles and markers for plotting
line_styles = ["-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted"]
line_markers = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "s", "p", "*", "h", "H", "+", "x", "D", "d"]

# Define a class for analyzing child heartbeat data
class ChildHeartBeat():
    def __init__(self) -> None:
        """
        Initialize the ChildHeartBeat class.

        Sets up file and data frame names, loads data from files into Pandas data frames,
        and sets parameters for analysis.
        """
        self.folder = str("ECGdata_child\\")
        self.files = ["abdomen1.txt", "abdomen2.txt", "abdomen3.txt", "thorax1.txt", "thorax2.txt"]
        self.dfs = ["abd1", "abd2", "abd3", "thr1", "thr2"]
        self.data = {}

        for i in range(0, len(self.files)):
            self.data[self.dfs[i]] = pd.read_csv(self.folder + self.files[i], sep=" ", names=[self.dfs[i]])
        print(f"loaded {self.data.keys()}")

        self.Tend = 20  # seconds
        self.freqHz = int(len(self.data["abd1"]) / self.Tend)
        self.freqRad = self.freqHz * 2 * np.pi
        self.N = self.freqHz * self.Tend

    def show_fourier(self, data_key):
        """
        Display the Fourier transform of the given data.

        Parameters:
        - data_key (str): Key to access the data frame.

        Returns:
        - tuple: Frequencies and corresponding Fourier transform values.
        """
        data = np.array(self.data[data_key])
        yf = fft(data)
        xf = fftfreq(self.N, 1 / self.freqHz)[:self.N // 2]
        return xf, 2.0 / self.N * np.abs(yf[0:self.N // 2])

    def highpass(self, data_key, cutoff, order):
        """
        Apply a high-pass Butterworth filter to the given data.

        Parameters:
        - data_key (str): Key to access the data frame.
        - cutoff (float): Cutoff frequency for the high-pass filter.
        - order (int): Order of the Butterworth filter.

        Returns:
        - numpy.ndarray: High-pass filtered data.
        """
        data = self.data[data_key]
        nyq = 0.5 * self.freqHz
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
        y = signal.filtfilt(b, a, data, method='gust')
        return y

    def plot_signal(self, y, x=None):
        """
        Plot a signal.

        Parameters:
        - y (array-like): Signal values.
        - x (array-like, optional): X-axis values. Defaults to None.

        Returns:
        - matplotlib.figure.Figure: The created figure.
        """
        fig = plt.figure()
        return fig

# Entry point of the script
if __name__ == "__main__":
    # Create an instance of the ChildHeartBeat class
    C = ChildHeartBeat()

    # Create a subplot with two rows and one column, sharing the x-axis
    fig, axs = plt.subplots(2, layout = 'constrained', figsize=(10, 4), sharex=True)

    # Plot raw and high-pass filtered signals for the "thr1" data file
    for file in C.dfs:
        axs[0].plot(C.data[file])
        axs[1].plot(C.highpass(file, 50, 1))

    # Set titles for subplots
    axs[0].set_title("raw signal")
    axs[1].set_title("highpass output")
    axs[0].legend(C.dfs)
    axs[1].legend(C.dfs)

    # Display the plots
    plt.show()
