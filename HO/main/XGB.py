# XGBClassifier
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy import stats
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, confusion_matrix    
from scipy.interpolate import interp1d
plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

# load data
data = pd.read_csv('../data/THA.csv')
# data = pd.read_csv('../data/THA_V.csv')

# data processing
label = data.pop('INDEX') # Label Extraction
scaler = StandardScaler() # standardization
scaled_data = scaler.fit_transform(data)
scaled_data_df = pd.DataFrame(scaled_data, columns=data.columns)
# second_column_data = scaled_data_df.iloc[:, 0:6]
X_train,X_test,Y_train,y_test = train_test_split(scaled_data,label,test_size=0.3) # 3. 7 is divided into test set and training set
# X_train,X_test,Y_train,y_test = train_test_split(second_column_data,label,test_size=0.3) # 3. 7 is divided into test set and training set
# print(X_train.shape,Y_train.shape)
smo = SMOTE(random_state=42) # Data balancing process
smo_X_train,smo_y_train = smo.fit_resample(X_train,Y_train)
'''training model'''
#XGB
clf_XGB = XGBClassifier(n_estimators=500,learning_rate=0.01,max_depth=25,min_child_weight=1,gamma=0.3)
clf_XGB.fit(smo_X_train,smo_y_train)
clf_XGB.save_model("../model/model.xgb")
prediction_XGB = clf_XGB.predict(X_test)

print('XGB accuracy',accuracy_score(prediction_XGB,y_test))
#MLP
# MLP =MLPClassifier(hidden_layer_sizes=(512, 512),max_iter=1000,learning_rate='adaptive',learning_rate_init=0.01,alpha=0.001,activation='relu',solver='adam',batch_size=64,early_stopping=True,validation_fraction=0.1)
# MLP.fit(smo_X_train,smo_y_train)
# prediction_MLP = MLP.predict(X_test)
# pickle.dump(MLP, open('MLP.pkl', "wb"))
# print('MLP accuracy',accuracy_score(prediction_MLP,y_test))
'''feature importance'''
from xgboost import plot_importance
# fig,ax = plt.subplots(figsize=(10,15))
# plot_importance(clf_XGB,height=0.5,max_num_features=64,ax=ax)
# plt.show()

# binarization
# prediction = label_binarize(prediction_XGB, classes=[0, 1, 2, 3])
# # prediction = label_binarize(prediction_MLP, classes=[0, 1, 2, 3])
# test_data = label_binarize(smo_y_test, classes=[0, 1, 2, 3])
# n_classes = prediction.shape[1]

'''ROC_AUC curved line'''
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i],tpr[i],_ = roc_curve(test_data[:,i],prediction[:,i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
# lw = 2
# classes = [0,1,2,3]
# all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# mean_tpr = np.zeros_like(all_fpr)
# for i in range(n_classes):
#     # mean_tpr +=interp1d(all_fpr, fpr[i], tpr[i])
#     f = interp1d(fpr[i], tpr[i])
#     mean_tpr += f(all_fpr)
# mean_tpr /= n_classes
# for i in range(n_classes):
#     plt.plot(fpr[i], tpr[i], lw=lw,
#              label='ROC curve of class {0} (auc = {1:0.2f})'
#              ''.format(classes[i], roc_auc[i]))
# # for i in range(n_classes):
# #     plt.plot(fpr[i], tpr[i], lw=lw,
# #              label='ROC curve of class {0} '
# #              ''.format(classes[i]))
# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.legend(loc="lower right")
# plt.savefig('./figures/fig11b.svg', format='svg',dpi = 300)
# plt.savefig('./figures/fig11b.png', format='png',dpi = 300)
# plt.show()

'''AUC curved line'''
# l = list(roc_auc.values())
# plt.plot(classes,list(roc_auc.values()))
# plt.title('AUC curved line')
# plt.xlabel('classes')
# plt.ylabel('Accuracy')
#
# plt.bar(classes,l)
# for i in range(4):
#     plt.text(classes[i], l[i]+0.01, '%.2f' %l[i], ha='center', va= 'bottom',fontsize=11)
# plt.title('AUC curved line')
# plt.xlabel('classes')
# plt.ylabel('Accuracy')
'''confusion matrix'''
# def plot_confusion_matrix(cm, labels_name, title):
#     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    #  normalize
#     plt.imshow(cm, interpolation='nearest')    # Displaying images on a specific window
#     plt.title(title)    # Image Title
#     plt.colorbar()
#     num_local = np.array(range(len(labels_name)))
#     plt.xticks(num_local, labels_name, rotation=90)    # Print labels on x-axis coordinates
#     plt.yticks(num_local, labels_name)    # Print labels on y-axis coordinates
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')

# cm = confusion_matrix(smo_y_test, prediction_XGB)
# plot_confusion_matrix(cm, classes, "Confusion Matrix")
# plt.show()










