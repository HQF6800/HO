import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams['font.sans-serif']=['Microsoft YaHei']#Used to display Chinese labels normally
plt.rcParams['axes.unicode_minus']=False#Used to display the negative sign normally

'''path(SPEED=10,HOM=6)
start_handover_position=[225.0, 425.0, 619.0, 819.0, 1025.0, 1219.0]
end_handover_position=[230.0, 430.0, 624.0, 824.0, 1030.0, 1224.0]'''
'''path(SPEED=10,HOM=10)
start_handover_position=[231.0, 431.0, 631.0, 831.0, 1031.0, 1231.0]
end_handover_position=[236.0, 436.0, 636.0, 836.0, 1036.0, 1236.0]'''
'''path(SPEED=20,HOM=6)
start_handover_position=[220.0, 420.0, 620.0, 820.0, 1020.0, 1220.0]
end_handover_position=[230.0, 430.0, 630.0, 830.0, 1030.0, 1230.0]'''
'''path(SPEED=20,HOM=10)
start_handover_position=[232.0, 432.0, 632.0, 832.0, 1032.0, 1232.0]
end_handover_position=[242.0, 442.0, 642.0, 842.0, 1042.0, 1242.0]'''
BS_position_X = [100,300,500,700,900,1100,1300]
BS_position_Y = [0,0,0,0,0,0,0]
x_start_points = [232.0, 432.0, 632.0, 832.0, 1032.0, 1232.0]
y_start_points = [0,0,0,0,0,0]
x_achieve_points = [0,242.0, 442.0, 642.0, 842.0, 1042.0, 1242.0]
y_achieve_points = [0,0,0,0,0,0,0]

fig = plt.figure()
ax = fig.add_subplot()
circle1 = plt.Circle((100, 0), 150, color='blue',fill=False)
circle2 = plt.Circle((300, 0), 150, color='blue',fill=False)
circle3 = plt.Circle((500, 0), 150, color='blue',fill=False)
circle4 = plt.Circle((700, 0), 150, color='blue',fill=False)
circle5 = plt.Circle((900, 0), 150, color='blue',fill=False)
circle6 = plt.Circle((1100, 0), 150, color='blue',fill=False)
circle7 = plt.Circle((1300, 0), 150, color='blue',fill=False)
BS_position = plt.scatter(BS_position_X, BS_position_Y,c='red',marker='2',s = 50)
# plt.plot(x_start_points,BS_position_Y)
# plt.axis('scaled')
plt.axis('equal')
ax.add_artist(circle1)
ax.add_artist(circle2)
ax.add_artist(circle3)
ax.add_artist(circle4)
ax.add_artist(circle5)
ax.add_artist(circle6)
ax.add_artist(circle7)
ax.add_artist(BS_position)
# ax.add_artist(path)
plt.xticks(range(-100,1600,100))
plt.plot(x_start_points,y_start_points,">",c= 'c',ms = 3,label='initiate handover')
plt.plot(x_achieve_points,y_achieve_points,"^",c= 'c',ms = 3,label='complete handover')
plt.plot([0,1400],[0,0],c= 'c',label='movement trajectory')
BS = [21,22,23,24,25,26,27]
BS_ = ['[21]','[22]','[23]','[24]','[25]','[26]','[27]']
for a, b,c in zip(BS_position_X, BS_position_Y,BS_):
    plt.text(a, b, c, ha='left', va='bottom', fontsize=7,color = 'red',)
for a, b,c in zip(x_achieve_points, y_achieve_points,BS):
    plt.text(a, b, c, ha='center', va='top', fontsize=7)
BS1 = [21,22,23,24,25,26]
for a, b,c in zip(x_start_points, y_start_points,BS1):
    plt.text(a, b, c, ha='center', va='bottom', fontsize=7)
ax.set_ylabel('Y position (m)', color='b',fontsize=10)
ax.set_xlabel('X position (m)', color='b',fontsize=10)
ax2 = ax.twinx()
data = pd.read_csv('.\\path\\hebing\\path(SPEED=20,HOM=10).csv')
ydata = data.loc[1:,'BS_RSRP']
# xdata = np.arange(0,1400,1400/7006)
xdata = np.arange(0,1400,1400/706)
ax2.plot(xdata,ydata,label='RSRP')
y1data = data.loc[1:,'BS_SINR']
ax2.plot(xdata,y1data,label='SINR')
ax2.axhline(-12, c = 'r',linestyle='--',label ='SINR threshold')
ax2.set_ylabel('SINR (dB) and RSRP (dBm)', color='g',fontsize=10)
fig.legend(loc=3,fontsize=9,bbox_to_anchor=(0,0), bbox_transform=ax.transAxes)
# plt.savefig('./path/figuers/speed=20 HOM=10.svg', format='svg',dpi = 300)
# plt.savefig('./path/figuers/speed=20 HOM=10.png', format='png',dpi = 300)
plt.show()

