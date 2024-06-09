import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import pickle
#Load data
data = pd.read_csv('../data/THA.csv')
#Draw heat map, values are correlation coefficients between two variables
# p = sns.heatmap(diabetes_data.corr(),annot=True) #.corr() indicates the correlation between the two variables in the data, the value range is [-1,1], the value close to -1, that is, the inverse correlation, similar to the inverse proportional function, the value close to 1, the table is positively correlated. annot=True indicates that the numbers are displayed on their own.
# plt.show()
# plt.savefig('cheatmap.pdf', format='pdf',dpi = 300,bbox_inches = 'tight')

label = data.pop('INDEX') # Label Extraction
scaler = StandardScaler() # Standardized
scaled_data = scaler.fit_transform(data)
X_train,X_test,Y_train,Y_test = train_test_split(scaled_data,label,test_size=0.3,random_state=10)  # 3 and 7 are divided into test set and training set
# print(X_test.shape,Y_test.shape)

smo = SMOTE(random_state=42) # Data balancing process
smo_x_train,smo_y_train = smo.fit_resample(X_train,Y_train)
# print(smo_X_train.shape,smo_y_train.shape)

classifiers = [
              KNeighborsClassifier(n_neighbors=20,weights='uniform',algorithm='kd_tree',leaf_size=2),
              LogisticRegression(max_iter=1000,class_weight='balanced',solver='newton-cg',multi_class='auto'),
              RandomForestClassifier(n_estimators=5,max_depth=17),
              MLPClassifier(hidden_layer_sizes=(512, 512),max_iter=500,learning_rate='adaptive',learning_rate_init=0.01,alpha=0.001,activation='relu',solver='adam',batch_size=64,early_stopping=True,validation_fraction=0.1),
              DecisionTreeClassifier(max_depth=18, min_samples_split=4),
              AdaBoostClassifier(n_estimators=5,learning_rate=0.01),
              XGBClassifier(n_estimators=500,learning_rate=0.01,max_depth=25,min_child_weight=1,gamma=0.3,objective='multi:softmax')
              ]

log = []

for clf in classifiers:
    clf.fit(smo_x_train,smo_y_train)
    name = clf.__class__.__name__
    print("="*30)
    print(name)

    print('*****RESULTS*****')
    test_predictions = clf.predict(X_test)
    print("Train Accuracy: {:.4%}".format(accuracy_score(smo_y_train, clf.predict(smo_x_train))))
    acc = accuracy_score(Y_test, test_predictions)
    print("Test Accuracy: {:.4%}".format(acc))

    # pickle.dump(clf, open('%s.pt' % name, "wb"))
    pickle.dump(clf, open('../model/%s.pkl' % name, "wb"))
    log.append([name, acc*100])

print("="*30)

log = pd.DataFrame(log)
log.rename(columns={0: 'Classifier', 1: 'Accuracy'}, inplace=True) #inplace=True  direct modification of the data
print(log)

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
sns.barplot(x='Accuracy', y='Classifier', data=log, color='seagreen')
plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')#The classification accuracy score is the percentage of all classifications that are correct

# plt.savefig('./figures/classify.pdf', format='pdf',dpi = 300,bbox_inches = 'tight')
plt.savefig('../figures/classify.png', format='png',dpi = 300,bbox_inches = 'tight')
# plt.show()