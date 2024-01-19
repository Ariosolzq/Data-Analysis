#1，调用库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#2，加载数据
data = pd.read_csv(r"C:\Users\76044\Desktop\data\Social_Network_Ads.csv")#根据年龄、工资及购买行为进行二分类并预测

#3，检查数据导入情况
data.head()
data.describe()
data.isnull().sum()

#4，选取数据及处理
Sex = pd.get_dummies(data['Gender'] , drop_first = True)#一种新方法，哑变量。因为性别是文字型变量，进行分类型变量编码处理
data['Sex'] = Sex
#data['Sex'] = data['Gender'].replace({'Male': 1, 'Female': 0})
data = data.drop('Gender' , axis =1) #删除原始的Gender列数据，axis=0表示跨行，axis=1表示跨列，删完第一列删第二列

#5，调用相关库，进行Z-score标准化处理，相关数据处理
from sklearn.preprocessing import StandardScaler #处理不同量度的数
ss = StandardScaler()
ss.fit(data.drop('Purchased', axis =1 ))#先删除购买与否的数据再进行拟合数据和标准化处理
ss_new = ss.transform( data.drop('Purchased', axis =1 ))#在fit的基础上，再次进行标准化、归一化等操作
#可以使用fit_transform是fit和transform的组合，既包括了训练又包含了转换，标准与单独的transform不同

scale = pd.DataFrame(ss_new , columns = data.columns[:-1])#之前在最后1列生成Sex列，数据内容为0，1
scale['Sex'] = scale['Purchased']
scale = scale.drop('Purchased'  ,axis =1)
scale#得到一个标准化之后的数据框架

x = scale
y = data['Purchased']#数据为0，1型，表示购买与否，如果原始数据为yes no 用replace可以替换

#6，划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train ,X_test , y_train , y_test = train_test_split( x , y , test_size = 0.3 , random_state = 42)

#7，调用逻辑回归的库
from sklearn.linear_model import LogisticRegression#调用逻辑回归的库
lj_model = LogisticRegression()
lj_model.fit(X_train,y_train)
predictions = lj_model.predict(X_test)

#8，得出结果
from sklearn.metrics import classification_report , confusion_matrix #引用混淆矩阵进行评估，二分类
print(confusion_matrix (y_test , predictions)) #混淆矩阵
print(classification_report(y_test , predictions)) #不用print结果为一行，有\n划分间隔,查准率=0.83，查全率=0.97 ，准确率=0.86

print(lj_model.score(X_train, y_train)) #训练集准确率 0.8214
print(lj_model.score(X_test, y_test)) #测试集准确率   0.8582