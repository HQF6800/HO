import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif']=['Microsoft YaHei']#Used to display Chinese labels normally
plt.rcParams['axes.unicode_minus']=False#Used to display the negative sign normally
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
})
'''fig8(a)'''
# x = np.array(["0", "2", "4",'6',"8",'10'])
# y = np.array([25.7884,22.0182,18.9836,16.927,14.9442,13.4954])
# plt.ylabel('Average number of  HOs')
# plt.xlabel('HOM (dB)')
# plt.bar(x,y,color = 'c',width = 0.5)
# plt.savefig('./figures/Fig8a.svg', format='svg', dpi=300)
# plt.savefig('./figures/Fig8a.png', format='png', dpi=300)
# plt.show()
'''fig8(b)'''
# x = np.array(["0", "2", "4",'6',"8",'10'])
# y1= np.array([0.608038531805348,0.6526916583244932,0.6814947236878645,0.7130179797823906,0.48723625442784535,0.048349149705515466])
# y2=np.array([0.077331506730589,0.04659119491869,0.044562831771204,0.04850918390325,0.360965874100735,0.935812100047257])
# y3 = np.array([0.3609801591864372,0.3117632117617176,0.2768328242154957,0.24052781850451424,0.1548398275065455,0.020495491939176706])
# plt.ylabel('HO rate')
# plt.xlabel('HOM (dB)')
# plt.xlim((0,5))
# plt.ylim((0,1))
# plt.plot(x, y1, color='r', linewidth=1.2, label=r'$\mathit{R}_{\mathrm{SHO}}$', marker="*")
# plt.plot(x, y2, color='b', linewidth=1.2, label=r'$\mathit{R}_{\mathrm{HOF}}$', marker="*")
# plt.plot(x, y3, color='k', linewidth=1.2, label=r'$\mathit{R}_{\mathrm{PPHO}}$', marker="*")
# plt.legend(loc='upper left')
# plt.savefig('./figures/Fig8b.svg', format='svg', dpi=300)
# plt.savefig('./figures/Fig8b.png', format='png', dpi=300)
# plt.show()

# '''Fig9(a)'''
# x = np.array(["100", "300", "500",'700',"900"])
# y = np.array([28.3057,25.6311,22.0182,19.6841,17.2402])
# plt.ylabel('Average number of  HOs')
# plt.xlabel('TTT (ms)')
# plt.bar(x,y,color = 'c',width = 0.5)
# plt.savefig('./figures/Fig9a.svg', format='svg', dpi=300)
# plt.savefig('./figures/Fig9a.png', format='png', dpi=300)
# plt.show()
# '''fig9(b)'''
# x = np.array(["100", "300", "500",'700',"900"])
# y1 = np.array([0.6601286585417, 0.6595841231883,0.6526916583244932,0.6206823158037,0.61028791813]) # SHO
# y2 = np.array([0.0161864463988,0.019767084460650064 ,0.03554512991378922,0.08119731930202395,0.11414970084529996]) # HOF
# y3 = np.array([0.3236848950594,0.32064879235105 ,0.3117632117617176,0.29812036489427,0.2755623810247]) # PPHO
# plt.ylabel('HO rate')
# plt.xlabel('TTT (ms)')
# plt.xlim((0,4))
# plt.ylim((0,0.8))
# plt.plot(x,y1,color = 'r',linewidth=1.2,label = r'$\mathit{R}_{\mathrm{SHO}}$',marker = "*")  # 'sin$_x$'
# plt.plot(x,y2,color = 'b',linewidth=1.2,label = r'$\mathit{R}_{\mathrm{HOF}}$',marker = "*")  # r'$\mathrm{R}_{\mathit{PPHO}}$'
# plt.plot(x,y3,color = 'k',linewidth=1.2,label = r'$\mathit{R}_{\mathrm{PPHO}}$',marker = "*")
# plt.legend()
# plt.savefig('./figures/Fig9b.svg', format='svg', dpi=300)
# plt.savefig('./figures/Fig9b.png', format='png', dpi=300)
# plt.show()
# '''fig10(a)'''
# x = np.array(["10","20","40","60", "80", "100",'120'])
# y = np.array([7.8671,11.9206,15.3748,19.7172,23.965,27.781,32.853])
# plt.ylabel('Average number of  HOs')
# plt.xlabel("v (km/h)")
# plt.bar(x,y,color = 'c',width = 0.5)
# plt.savefig('./figures/Fig10a.svg', format='svg', dpi=300)
# plt.savefig('./figures/Fig10a.png', format='png', dpi=300)
# plt.show()
#
# '''fig10(b)'''
# x = np.array(["10","20","40","60", "80", "100",'120'])
# y1 = np.array([0.68094781267838,0.6760289461328,0.6843568161289,0.6769419593045666 ,0.6960484039223868, 0.6933011770634606, 0.6778193772258241]) # SHO
# y2 = np.array([0.0302957589311,0.0431003675742,0.050383761956738,0.07120686507211978 ,0.09237638222407678, 0.1242503869551132 ,0.16235960186284357]) # HOF
# y3 = np.array([0.28875642839052,0.28087068629300005,0.265259421914362,0.25185117562331366, 0.21157521385353642, 0.18244843598142615, 0.1598210209113323]) # PPHO
# plt.ylabel('HO rate') #,fontproperties='Times New Roman'
# plt.xlabel("v (km/h)")
# plt.xlim((0,6))
# plt.ylim((0,0.75))
# plt.plot(x,y1,color = 'r',linewidth=1.2,label = r'$\mathit{R}_{\mathrm{SHO}}$',marker = "*")  # 'sin$_x$'
# plt.plot(x,y2,color = 'b',linewidth=1.2,label = r'$\mathit{R}_{\mathrm{HOF}}$',marker = "*")  # r'$\mathrm{R}_{\mathit{PPHO}}$'
# plt.plot(x,y3,color = 'k',linewidth=1.2,label = r'$\mathit{R}_{\mathrm{PPHO}}$',marker = "*")
# plt.legend(loc='center right')
# plt.savefig('./figures/Fig10b.svg', format='svg', dpi=300)
# plt.savefig('./figures/Fig10b.png', format='png', dpi=300)
# plt.show()

