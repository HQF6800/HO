import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read reward.CSV
file_path ='D:\pythonProject\HO\data/reward.csv'
data = pd.read_csv(file_path)
second_column_data1 = data.iloc[:, 0]
second_column_data2 = data.iloc[:, 1]
# plt.plot(np.arange(len(second_column_data1)), second_column_data1,label='with prediction')
plt.plot(np.arange(len(second_column_data2)), second_column_data2,label='without prediction')
plt.ylabel('Reward')
plt.xlabel('Train episodes')
plt.ylim(-0.8, None)
plt.yticks(np.append(plt.yticks()[0], -0.27))
plt.legend(loc='upper right', frameon=True, prop={'size': 8})
# plt.savefig('./figures/DQN_reward.pdf',format='pdf',dpi = 300)
plt.savefig('./figures/DQN+withoutprediction_reward.pdf',format='pdf',dpi = 300)
plt.show()
# read Cost.CSV
# file_path ='D:\pythonProject\HO\data/cost.csv'
# data = pd.read_csv(file_path)
# second_column_data1 = data.iloc[:, 0]
# second_column_data2 = data.iloc[:, 1]
# plt.plot(np.arange(len(second_column_data1)), second_column_data1,label='with prediction')
# plt.plot(np.arange(len(second_column_data2)), second_column_data2,label='without prediction')
# plt.ylabel('Cost')
# plt.xlabel('Train episodes')
# plt.legend(loc='upper right', frameon=True, prop={'size': 8})
# plt.savefig('./figures/DQN_Cost.pdf',format='pdf',dpi = 300)
# plt.show()


