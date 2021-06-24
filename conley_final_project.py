import numpy as np
from numpy import sin, cos, pi, arange
from numpy.random import randint
import matplotlib.pyplot as plt
import scipy.signal as sig
import pandas as pd
import control as con
import scipy.fftpack as fft


###Define sampling frequenices and total periods
fs = 1e6
Ts = 1/fs
t_end = 50e-3

t = arange(0,t_end-Ts,Ts)

###Define frequencies for the incomming data
f1 = 1.8e3
f2 = 1.9e3
f3 = 2e3
f4 = 1.85e3
f5 = 1.87e3
f6 = 1.94e3
f7 = 1.92e3

info_signal = (2.5*cos(2*pi*f1*t) + 1.75*cos(2*pi*f2*t) + 2*cos(2*pi*f3*t) + 
               2*cos(2*pi*f4*t) + 1*cos(2*pi*f5*t) + 1*cos(2*pi*f6*t) +
               1.5*cos(2*pi*f7*t))

N = 25
my_sum = 0

##Create noise
for i in range(N+1):
    noise_amp     = 0.075*randint(-10,10,size=(1,1))
    noise_freq    = randint(-1e6,1e6,size=(1,1))
    noise_signal  = my_sum + noise_amp * cos(2*pi*noise_freq*t)
    my_sum = noise_signal

f6 = 50e3    #50kHz
f7 = 49.9e3
f8 = 51e3

#modify signal for power supply noise subtraction
pwr_supply_noise = 1.5*sin(2*pi*f6*t) + 1.25*sin(2*pi*f7*t) + 1*sin(2*pi*f8*t)

f9 = 60

low_freq_noise = 1.5*sin(2*pi*f9*t)

total_signal = info_signal + noise_signal + pwr_supply_noise + low_freq_noise
total_signal = total_signal.reshape(total_signal.size)


### Start to read in values from csv file
df = pd.DataFrame({'0':t,
                   '1':total_signal})

df.to_csv('NoisySignal.csv')

df=pd.read_csv('NoisySignal.csv')

t = df['0'].values
sensor_sig = df['1'].values

###Plot the signal from the csb
plt.figure (figsize = (10,7))
plt.plot (t,sensor_sig)
plt.grid ()
plt.title ('Noisy Input Signal')
plt.xlabel ('Time [s]')
plt.ylabel ('Amplitude [V]')
plt.show ()

#define stem graphs
def make_stem(ax,x,y, color='k',style='solid', label='', linewidths =2.5,**kwargs):
      ax.axhline(x[0] ,x[-1],0,color ='r')
      ax.vlines(x,0,y,color=color,linestyles=style, label=label, linewidths=linewidths)
      ax.set_ylim([1.05*y.min(),1.05*y.max()])
     

    #define Fast Fourier Transform
def NewFFT(x,fs):
    
    N = len( x ) 
    X_fft = fft.fft ( x )
    X_fft_shifted = fft.fftshift (X_fft) 
    freq = np.arange ( - N /2 , N /2) * fs / N 

    x_mag = np.abs( X_fft_shifted ) / N
    x_phi = np.angle ( X_fft_shifted ) 
    

    for k in range (len(x_mag)):
        if x_mag[k] < 1e-10:
            x_phi[k] = 0
            
    return x_mag, x_phi, freq


#### TASK 1
fs = 1e6
steps = 1/fs

### Signal with FFT applied and stem plot
### for signal analysis of magnitudes and 
#### phase angles
fig, ax = plt.subplots(figsize = (10,7))
x_mag1, x_phi, freq = NewFFT(sensor_sig, fs)
make_stem(ax, freq ,x_mag1)
plt.xlim([10,1e6])
plt.xscale('log')
plt.xlabel("F(hz)")
plt.ylabel("Amplitude")
plt.title("FFT of sensor signal")
plt.grid()
plt.show()

### Look at low frequencies
fig, ax = plt.subplots(figsize = (10,7))
x_mag, x_phi, freq = NewFFT(sensor_sig, fs)
make_stem(ax, freq ,x_mag)
plt.xlim([10, 1000])
plt.xscale('log')
plt.title('FFT - Low Frequency Check')
plt.ylim([0,0.8])
plt.xlabel("F(hz)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

###Look at bandpass area
fig, ax = plt.subplots(figsize = (10,7))
x_mag, x_phi, freq = NewFFT(sensor_sig, fs)
make_stem(ax, freq ,x_mag)
plt.xlim([1400, 2500])
plt.xscale('log')
plt.title('FFT - Sensor Frequency Check')
plt.xlabel("F(hz)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

###Look at high frequencies
fig, ax = plt.subplots(figsize = (10,7))
x_mag, x_phi, freq = NewFFT(sensor_sig, fs)
make_stem(ax, freq ,x_mag)
plt.ylim([0,0.8])
plt.xlim([5e3, 4e5])
plt.xscale('log')
plt.title('FFT - High Frequency Check')
plt.xlabel("F(hz)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()






####Task 3

###Develop an anolog RLC bandpass filter using
## by formulating a transfer function
steps= 1e1
f = np.arange(steps, 1e5+steps, steps)
w = 2*np.pi*f

#Values for bandpass filter
R = 125
L = 4.7e-3
C = 1.4929e-6


### Display a bode plot of the filtered signal
num = np.array([1/(R*C), 0])
den = np.array([1, 1/(R*C), 1/(L*C)])
sys = con.TransferFunction(num,den)
plt.figure(figsize = (10,7))
_ = con.bode(sys,w, dB = True, Hz = True, deg = True)
plt.title('Full Bode Plot')
plt.axhline(y=-0.3, color = 'r', linestyle = '-')
plt.axvline(x=1.8e3, color = 'b', linestyle = '-')
plt.axvline(x=2e3, color = 'b', linestyle = '-')
plt.tight_layout()

plt.show()

####Demonstration

#cut off frequencies with bode plot

plt.figure(figsize=(10,7))
low = np.arange(1.75e3, 2.05e3,1)*2*np.pi ### just on either side of the filter 
reX = np.arange(1.75e3, 2.05e3, 50)
_ = con.bode(sys,low, dB = True, Hz = True, deg = True)
plt.title('Cutt-off frequency zoom')
plt.axhline(y=-30, color = 'r', linestyle = '-')
plt.axvline(x=1.8e3, color = 'b', linestyle = '-')
plt.xticks(reX)
plt.tight_layout()

plt.show()


#within passing frequencies

plt.figure(figsize = (10,7))
within = np.arange(80, 110, 1)*2*np.pi
reX = np.arange(80, 110, 5)
_ = con.bode(sys,within, dB = True, Hz = True, deg = True)
plt.xticks(reX)
plt.tight_layout()
plt.title('Sensor frequency zoom')
plt.show()


#High frequencies (outside passing)

plt.figure(figsize = (10,7))
highF = np.arange(30e3, 50e3, 1e3)*2*np.pi
reX = np.arange(30e3, 50e3, 1e3)
_ = con.bode(sys,highF, dB = True, Hz = True, deg = True)
plt.title('High frequencies outside passing')
plt.axhline(y=-20, color = 'r', linestyle = '-')
plt.axvline(x=4e4, color = 'b', linestyle = '-')
plt.xticks(reX)
plt.tight_layout()

plt.show()


###Show that the filter is working with the sig.bilinear function


R = 125
L = 4.7e-3
C = 1.4929e-6


num = np.array([1/(R*C), 0])
den = np.array([1, 1/(R*C), 1/(L*C)])
fs = 1e6
steps = 1/fs

znum, zden = sig.bilinear(num,den,fs)

y = sig.lfilter(znum,zden,sensor_sig)


####filtered data
fs = 1e6
plt.figure (figsize = (10,7))
plt.plot (t,y)
plt.grid ()
plt.title ('Filtered Input Signal')
plt.xlabel ('Time [s]')
plt.ylabel ('Amplitude [V]')
plt.show ()

plt.show()

#### filtered data through FFT... Similar style 
#### to the above plots with high, low and 
#### bandpass frequencies

fig, ax = plt.subplots(figsize = (10,7))
x_mag, x_phi, freq = NewFFT(y, fs)
make_stem(ax, freq ,x_mag)
plt.xlim([10,1e6])
plt.title ('Filtered Input Signal with FFT')
plt.xscale('log')
plt.xlabel("F(hz)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize = (10,7))
x_mag, x_phi, freq = NewFFT(y, fs)
make_stem(ax, freq ,x_mag)
plt.xlim([10, 1000])
plt.title ('Filtered Input Signal with FFT (Low Frequency)')
plt.xscale('log')
plt.xlabel("F(hz)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()


fig, ax = plt.subplots(figsize = (10,7))
x_mag, x_phi, freq = NewFFT(y, fs)
make_stem(ax, freq ,x_mag)
plt.xlim([1400, 2500])
plt.title ('Filtered Input Signal with FFT (Sensor Frequency)')
plt.xscale('log')
plt.xlabel("F(hz)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize = (10,7))
x_mag, x_phi, freq = NewFFT(y, fs)
make_stem(ax, freq ,x_mag)
plt.xlim([5e3, 4e5])
plt.title ('Filtered Input Signal with FFT (High frequency)')
plt.xscale('log')
plt.xlabel("F(hz)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()


fig, ax = plt.subplots(figsize = (10,7))
x_mag, x_phi, freq = NewFFT(y, fs)
make_stem(ax, freq ,x_mag)
plt.xlim([100e3, 100e5])
plt.title ('Attenuation of frequency greater than 100kHz')
plt.xscale('log')
plt.xlabel("F(hz)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize = (10,7))
x_mag, x_phi, freq = NewFFT(y, fs)
make_stem(ax, freq ,x_mag, color = 'black')
make_stem(ax, freq, x_mag1, color = 'b')
plt.xlim([10,1e6])
plt.title ('Superimposed FFT. Filtered vs nonFilterd. (Black = Filtered)')
plt.xscale('log')
plt.xlabel("F(hz)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

