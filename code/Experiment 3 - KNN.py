#1，调用库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#2，加载数据
data = pd.read_csv(r"C:\Users\76044\Desktop\data\3-knn-classification.csv")#内容为对各类因素与成功与否的训练集、测试集分类

#3，检查数据导入情况
data.head()
data.describe()

#4，选取数据及处理
data['gender'] = data['gender'].replace({'Male': 0, 'Female': 1})#因为性别由文字分类，replace替换
data.corr() # 筛选数据进行KNN算法识别与成功相关较大的因素，求相关性，选择年龄和兴趣
data.isnull().sum()

import seaborn as sns#观察
sns.set_theme(style="whitegrid")
sns.boxplot(x="age", data=data, palette="Set3")
plt.title("Age Distribution") #人群的年龄分布
#data.age.min(), data.age.max()

sns.set_theme(style="whitegrid")
sns.boxplot(x="interest", data=data, palette="Set1")
plt.title("Interests Distribution")#生活兴趣指数分布
#data.interest.min(), data.interest.max()

data.success.value_counts().plot(kind='bar')
plt.xlabel("Success And Not Success")
plt.ylabel("Count")
plt.title("Success Acount")
sns.scatterplot(data=data, x="age", y="interest", hue="success") #hue将散点图按照hue指定的特征或类别进行区分

#5，调用KNN相关的库，划分数据
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

X = data[["age","interest"]]
Y = data["success"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)#注意大小写 X x Y y

#6，寻找K
error_rates = [] #后续采集错误率
for i in np.arange(1, 101): #寻找错误率较小的K的方法。np.arange返回步长为1的排列

    new_model = KNeighborsClassifier(n_neighbors = i)

    new_model.fit(X, Y)

    new_predictions = new_model.predict(X_test)

    error_rates.append(np.mean(new_predictions != y_test))

plt.plot(error_rates)#找到较合适的K值为5

#7，得出结果
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)
y_predict = neigh.predict(X_test)#预测y_pred

print(accuracy_score(y_test,y_predict))