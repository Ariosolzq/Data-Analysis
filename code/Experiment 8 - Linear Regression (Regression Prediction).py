#1，调用库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#2，加载数据
data = pd.read_csv(r"C:\Users\76044\Desktop\data\Life Expectancy Data.csv")#根据不同国家的调查数据中因素对支出百分比进行回归分析并预测

#3，检查数据导入情况
data.head()
data.isnull().sum()#有空值
data = data.dropna()#原数据中有缺失项，进行剔除，被pycharm读取的数据改变，而在data中的原始数据集不会改变
data.corr()#可以分析各个因素之间的相关性

#4，调用线性回归的库，建模
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
x = data.GDP.values.reshape(-1,1)#转化为任意行，1列的矩阵
y = data['percentage expenditure'].values.reshape(-1,1)#转化为任意行，1列的矩阵，因为原始数据的标签名含有空格
lr_model.fit(x,y)
a = lr_model.coef_ #线性回归的系数为0.14705833
print("a: ", a)

#5，运用原始数据创建新的测试集
x_data = np.arange(min(data.GDP),max(data.GDP)).reshape(-1,1) #将数据的最小值到最大值排列并按照0.01递进
y_project = lr_model.predict(x_data) #预测

#6，得到结果并画图
plt.scatter(x,y,color="red")
plt.plot(x_data,y_project,color="blue")
plt.show()

#7，评估回归方程
from sklearn.metrics import  r2_score
print(r2_score(y, lr_model.predict(x)))#线性回归决定系数为0.92