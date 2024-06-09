import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif']=['Microsoft YaHei']#Used to display Chinese labels normally
plt.rcParams['axes.unicode_minus']=False#Used to display the negative sign normally
'''Fig13c'''
x = np.array(["THA", 'Non-ML',"SLEHA", "QHA",'Proposed'])
'''Low density, low speed'''
y1 = np.array([0.315050197722374024,0.30845386533665836, 0.253846065004900532, 0.1943903796745646587, 0.183654982190154352])
'''Low density, high speed'''
y2 = np.array([0.231738531805348,0.2272161972001844,0.207947236878645,0.2305916583244932,0.22871797978239060])
'''High density, low speed'''
y3 = np.array([0.4035178155311945, 0.38534325542395975,0.35274147391379212, 0.2874987545198875, 0.265041956035125983])
'''High density, high speed'''
y4 = np.array([0.2179447617281661,0.20382209641662805,  0.17833224285449392,  0.1980769304124394,  0.185746065004900532])
plt.yticks([0,0.1,0.2,0.3,0.4])
plt.ylabel('$R_{PPHO}$')
index = np.arange(len(x))
bar_width=0.2
plt.bar(index,y1,color ='#DF7A5E',width = bar_width,label='low density and low speed')
plt.bar(index+bar_width,y2,color ='#3C405B',width =bar_width,label='low density and high speed')
plt.bar(index+2*bar_width,y3,color ='#82B29A',width = bar_width,label='high density and low speed')
plt.bar(index+3*bar_width,y4,color ='#F2CC8E',width = bar_width,label='high density and high speed')
plt.xticks(index + 0.2 / 2, x)
plt.legend(prop={'size': 9})
plt.savefig('./figures/fig13c.pdf', format='pdf',dpi = 300)
plt.savefig('./figures/fig13c.png', format='png',dpi = 300)
plt.show()
'''Fig13b'''
x = np.array(["THA", 'Non-ML',"SLEHA", "QHA",'Proposed'])
'''Low density, high speed'''
y1 = np.array([0.081850197722374024,0.0756877016764927,0.076846065004900532, 0.064903796745646587, 0.058054982190154352])
'''High density, high speed'''
y2 = np.array([0.0581313160148444,0.047860497739256773, 0.04982657721952613, 0.0417517270910648, 0.03430903796745646587])
plt.yticks(np.linspace(0,0.08,9,endpoint=True))
plt.ylabel('$R_{TLHO}$')
index = np.arange(len(x))
bar_width=0.3
plt.bar(index,y1,color ='#3C405B',width = bar_width,label='low density and high speed')
plt.bar(index+bar_width,y2,color ='#F2CC8E',width =bar_width,label='high density and high speed')
plt.xticks(index + 0.2 / 2, x)
plt.legend(prop={'size': 9})
plt.savefig('./figures/fig13b.pdf', format='pdf',dpi = 300)
plt.savefig('./figures/fig13b.png', format='png',dpi = 300)
plt.show()

'''Fig13a'''
x = np.array(["THA", 'Non-ML',"SLEHA", "QHA",'Proposed'])
'''Low density, low speed'''
y1 = np.array([0.0386181309008214837,0.03292743285537521,0.03214512991378924,0.021577245209663982,0.0203154201713095146])
'''High density, low speed'''
y2 = np.array([0.04634565511889,0.04281795511221945,0.039574399915331,0.0311026176532844,0.029871691949481])
plt.yticks(np.linspace(0,0.06,7,endpoint=True))
plt.ylabel('$R_{TEHO}$')
plt.ylim(0, 0.05)
index = np.arange(len(x))
bar_width=0.3
plt.bar(index,y1,color ='#DF7A5E',width = bar_width,label='low density and low speed')
plt.bar(index+bar_width,y2,color ='#82B29A',width =bar_width,label='high density and low speed')
plt.xticks(index + 0.2 / 2, x)
plt.legend(prop={'size': 9})
plt.savefig('./figures/fig13a.pdf', format='pdf',dpi = 300)
plt.savefig('./figures/fig13a.png', format='png',dpi = 300)
plt.show()


