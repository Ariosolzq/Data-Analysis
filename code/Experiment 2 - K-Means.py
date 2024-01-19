#1，调用库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#2，加载数据
data = pd.read_csv(r'C:\Users\76044\Desktop\data\Mall_Customers.csv')#内容为超市消费者的年收入和消费积分之间的聚类
#data = pd.read_csv('C:\Users\76044\Desktop\data\Mall_Customers.csv')缺少r
#data = pd.read_csv(r 'C:\Users\76044\Desktop\data\Mall_Customers.csv')避免空格
#含有中文 encoding=’gbk’  data = pd.read_csv(r'C:\Users\76044\Desktop\data\Mall_Customers.csv', encoding=’gbk’)  UnicodeDecodeError: 'utf-8' codec can't decode byte 0xd6 in position 2: invalid continuation byte

#3，检查数据导入情况
data.head() #前5行
data.shape#数据内容 200个 5列
data.isnull().sum()#查询空值，可以剔除
data.info()#数据的列名，空值个数，数据类型
#交互式仅会执行最后的代码，所有行代码执行时有需要加print

#4，选取数据及处理
X = data[["Annual Income (k$)" , "Spending Score (1-100)"]]#中括号选取
X.head()
sns.scatterplot(X["Annual Income (k$)"] , X['Spending Score (1-100)'])#散点图，会Warning :FutureWarning: Pass the following variables as keyword args: x, y.
#sns.scatterplot(x=X["Annual Income (k$)"] , y=X['Spending Score (1-100)'])#散点图,x y 关键字传参
#plt.scatter(X["Annual Income (k$)"] , X['Spending Score (1-100)']) 效果一样，sns函数封装，像是点套餐，需要符合要求；plt可以自由定制
X = data.iloc[:,[3,4]].values #索引切分数据，整理仅含Annual Income 和 Spending Score

#5，调用K-means的库
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)#初始化，类别K=5
kmeans.fit(X)
Y = kmeans.fit_predict(X) #预测结果赋值
kmeans.labels_ #分配集群

#6，画图得出结果
plt.figure(figsize=(7,7))#图片大小
plt.scatter(X[Y==0,0], X[Y==0,1], s=50, c='red', label='Cluster 1')
plt.scatter(X[Y==1,0], X[Y==1,1], s=50, c='yellow', label='Cluster 2')
plt.scatter(X[Y==2,0], X[Y==2,1], s=50, c='green', label='Cluster 3')
plt.scatter(X[Y==3,0], X[Y==3,1], s=50, c='blue', label='Cluster 4')
plt.scatter(X[Y==4,0], X[Y==4,1], s=50, c='purple', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='black', label='Centroids')# 画出最后的每类中心点
plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()