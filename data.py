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
    def __init__(self,resample_fac) -> None:
        """
        Initialize the ChildHeartBeat class.

        Sets up file and data frame names, loads data from files into Pandas data frames,
        and sets parameters for analysis.
        """
        self.folder = str("ECGdata_child\\")
        self.files = ["abdomen1.txt", "abdomen2.txt", "abdomen3.txt", "thorax1.txt", "thorax2.txt"]
        self.dfs = ["abd1", "abd2", "abd3", "thr1", "thr2"]
        self.data = pd.DataFrame()

        for i in range(0, len(self.files)):
            self.data[self.dfs[i]] = pd.read_csv(self.folder + self.files[i], sep=" ", names=[self.dfs[i]])[::resample_fac]


        self.data = self.data.reset_index(drop = True)

  
        self.Tend = 20  # seconds
        self.freqHz = int(len(self.data["abd1"]) / self.Tend)
        self.freqRad = self.freqHz * 2 * np.pi
        self.N = self.freqHz * self.Tend

        #plt.figure()
        self.primary = np.array(self.data['abd3'])
        #plt.plot(self.primary)
        self.primary = self.highpass('abd3', 2,1)
        
        #plt.plot(self.primary)
        #plt.show()
        self.primary = (self.primary - self.primary.min())/ (self.primary.max() - self.primary.min())
        
        #plt.figure()

        self.reference = np.array(self.data['thr1'])
        #plt.plot(self.reference)
        self.reference = self.highpass('thr1',2, 1)
        #plt.plot(self.reference)
        #plt.show()
        self.reference = (self.reference - self.reference.min())/ (self.reference.max() - self.reference.min())
        
 

    def train_weights(self,k_l = 5,plot= False):
        #k_l = weights size
        y = self.primary
        N = len(y)
        y_hat = np.zeros(N)
        x = self.reference
        X = np.zeros(shape = (N - k_l,k_l+1))
        for i in range(k_l,N):
            X[i-k_l,1:] = x[i-k_l:i]
            X[i-k_l,0] = 1
        B = np.dot( np.dot( np.linalg.inv(np.dot(np.transpose(X),X)), np.transpose(X)), y[k_l:N])
        
        for i in range(k_l,len(y)):
            y_hat[i] = np.dot(x[i-k_l:i],B[1:]) + B[0]

        error = y-y_hat
        if plot:
            plt.figure()
            plt.plot(y)
            plt.plot(y_hat)
            plt.plot(self.reference)
            plt.legend(['abd3','abd3_hat','input'])
            plt.show()

        return B,sum(abs(error))


    def noise_filter(self,weights = [1,1,1,1,1]):
        k_l = len(weights)
        N = len(self.primary)
        y = np.zeros(N)

        self.thr2 = np.array(self.data['thr2'])
        self.thr2 =  (self.thr2 - self.thr2.min())/ (self.thr2.max() -self.thr2.min())
        W = weights
        z = np.zeros(N)
        
        for i in range(k_l,len(y)):
            y[i] = np.dot(self.reference[i-k_l+1:i],W[1:]) + W[0]
            z[i] = y[i]
        

   
        plt.figure()
        plt.plot(z)
        plt.plot(self.primary)
        e = z - self.primary
        plt.plot(e)
        cutoff = 8
        order = 1
        nyq = 0.5 * self.freqHz
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
        z_filt = signal.filtfilt(b, a, e)
        plt.plot(z_filt)
        plt.legend(['filter','abd3','e','e smooth'])
        plt.show()
        return






    def show_fourier(self, data_key):
        """
        Display the Fourier transform of the given data.

        Parameters:
        - data_key (str): Key to access the data frame.

        Returns:
        - tuple: Frequencies and corresponding Fourier transform values.
        """
        data = np.array(self.data[data_key])
        print(len(data))
        yf = fft(data)
        xf = fftfreq(self.N, 1 / self.freqHz)[:self.N // 2]
        return xf, (2.0 / self.N * np.abs(yf[0:self.N // 2]))

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
        y = signal.filtfilt(b, a, data)
        return y


# Entry point of the script
if __name__ == "__main__":
    # Create an instance of the ChildHeartBeat class
    C = ChildHeartBeat(1)

    B,e = C.train_weights(500,True)
   #print(B)
    C.noise_filter(B)
    
    if False:
        # Create a subplot with two rows and one column, sharing the x-axis
        fig, axs = plt.subplots(3, layout = 'constrained', figsize=(10, 4))

        # Plot raw and high-pass filtered signals for the "thr1" data file
        for file in C.dfs:
            axs[0].plot(C.data[file])
            axs[1].plot(C.highpass(file, 1, 1))

            x,y = C.show_fourier(file)
        
            axs[2].plot(x,y)

        # Set titles for subplots
        axs[0].set_title("raw signal")
        axs[1].set_title("highpass output")
        axs[0].legend(C.dfs)
        axs[1].legend(C.dfs)
        axs[2].legend(C.dfs)
        axs[0].axhline(0)
        axs[1].axhline(0)
        axs[2].set_xlabel("freq [Hz]")
        # Display the plots
        plt.show()
