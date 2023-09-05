import pandas as pd
import matplotlib.pyplot as plt

data0 = pd.read_csv('/home/lautthom/Desktop/PartABiosignals/biosignals_raw/071309_w_21/071309_w_21-BL1-081_bio.csv', sep='\t')
data1 = pd.read_csv('/home/lautthom/Desktop/PartABiosignals/biosignals_raw/071309_w_21/071309_w_21-BL1-082_bio.csv', sep='\t')
data2 = pd.read_csv('/home/lautthom/Desktop/PartABiosignals/biosignals_raw/071309_w_21/071309_w_21-BL1-083_bio.csv', sep='\t')
data3 = pd.read_csv('/home/lautthom/Desktop/PartABiosignals/biosignals_raw/071309_w_21/071309_w_21-BL1-084_bio.csv', sep='\t')
data4 = pd.read_csv('/home/lautthom/Desktop/PartABiosignals/biosignals_raw/071309_w_21/071309_w_21-BL1-085_bio.csv', sep='\t')
data5 = pd.read_csv('/home/lautthom/Desktop/PartABiosignals/biosignals_raw/071309_w_21/071309_w_21-BL1-086_bio.csv', sep='\t')
data6 = pd.read_csv('/home/lautthom/Desktop/PartABiosignals/biosignals_raw/071309_w_21/071309_w_21-BL1-087_bio.csv', sep='\t')
data7 = pd.read_csv('/home/lautthom/Desktop/PartABiosignals/biosignals_raw/071309_w_21/071309_w_21-BL1-088_bio.csv', sep='\t')
data8 = pd.read_csv('/home/lautthom/Desktop/PartABiosignals/biosignals_raw/071309_w_21/071309_w_21-BL1-089_bio.csv', sep='\t')
data9 = pd.read_csv('/home/lautthom/Desktop/PartABiosignals/biosignals_raw/071309_w_21/071309_w_21-BL1-090_bio.csv', sep='\t')
data10 = pd.read_csv('/home/lautthom/Desktop/PartABiosignals/biosignals_raw/071309_w_21/071309_w_21-BL1-091_bio.csv', sep='\t')
data11 = pd.read_csv('/home/lautthom/Desktop/PartABiosignals/biosignals_raw/071309_w_21/071309_w_21-BL1-092_bio.csv', sep='\t')
data12 = pd.read_csv('/home/lautthom/Desktop/PartABiosignals/biosignals_raw/071309_w_21/071309_w_21-BL1-093_bio.csv', sep='\t')
data13 = pd.read_csv('/home/lautthom/Desktop/PartABiosignals/biosignals_raw/071309_w_21/071309_w_21-BL1-094_bio.csv', sep='\t')
data14 = pd.read_csv('/home/lautthom/Desktop/PartABiosignals/biosignals_raw/071309_w_21/071309_w_21-BL1-095_bio.csv', sep='\t')
data15 = pd.read_csv('/home/lautthom/Desktop/PartABiosignals/biosignals_raw/071309_w_21/071309_w_21-BL1-096_bio.csv', sep='\t')
data16 = pd.read_csv('/home/lautthom/Desktop/PartABiosignals/biosignals_raw/071309_w_21/071309_w_21-BL1-097_bio.csv', sep='\t')
data17 = pd.read_csv('/home/lautthom/Desktop/PartABiosignals/biosignals_raw/071309_w_21/071309_w_21-BL1-098_bio.csv', sep='\t')
data18 = pd.read_csv('/home/lautthom/Desktop/PartABiosignals/biosignals_raw/071309_w_21/071309_w_21-BL1-099_bio.csv', sep='\t')
data19 = pd.read_csv('/home/lautthom/Desktop/PartABiosignals/biosignals_raw/071309_w_21/071309_w_21-BL1-100_bio.csv', sep='\t')

fig, axs = plt.subplots(4, 5)
axs[0, 0].plot(range(2816), data0.gsr)
axs[0, 1].plot(range(2816), data1.gsr)
axs[0, 2].plot(range(2816), data2.gsr)
axs[0, 3].plot(range(2816), data3.gsr)
axs[0, 4].plot(range(2816), data4.gsr)
axs[1, 0].plot(range(2816), data5.gsr)
axs[1, 1].plot(range(2816), data6.gsr)
axs[1, 2].plot(range(2816), data7.gsr)
axs[1, 3].plot(range(2816), data8.gsr)
axs[1, 4].plot(range(2816), data9.gsr)
axs[2, 0].plot(range(2816), data10.gsr)
axs[2, 1].plot(range(2816), data11.gsr)
axs[2, 2].plot(range(2816), data12.gsr)
axs[2, 3].plot(range(2816), data13.gsr)
axs[2, 4].plot(range(2816), data14.gsr)
axs[3, 0].plot(range(2816), data15.gsr)
axs[3, 1].plot(range(2816), data16.gsr)
axs[3, 2].plot(range(2816), data17.gsr)
axs[3, 3].plot(range(2816), data18.gsr)
axs[3, 4].plot(range(2816), data19.gsr)
plt.show()

data_uncut = pd.read_csv('/home/lautthom/Desktop/PartC-Biosignals/biosignals_raw/072414_m_23.csv', sep='\t')
stimulus = pd.read_csv('/home/lautthom/Desktop/PartC-Biosignals/stimulus/072414_m_23.csv', sep='\t')
temperature = pd.read_csv('/home/lautthom/Desktop/PartC-Biosignals/temperature/072414_m_23.csv', sep='\t')
print(data_uncut)
print(stimulus)
print(temperature)

fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
ax0.plot(data_uncut.time, data_uncut.gsr)
ax1.plot(stimulus.time, stimulus.label)
ax2.plot(temperature.time, temperature.temperature)
plt.show()