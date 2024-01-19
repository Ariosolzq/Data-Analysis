#1，调用库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#2，加载数据
data = pd.read_csv(r"C:\Users\76044\Desktop\data\Life Expectancy Data.csv")#根据不同国家的调查数据中因素对支出百分比进行回归分析并预测

#3，数据处理
data.head()
data.isnull().sum()#有空值
data = data.dropna()#原数据中有缺失项，进行剔除，被pycharm读取的数据改变，而在data中的原始数据集不会改变
x = data.GDP.values.reshape(-1,1) #转化为任意行，1列的矩阵，不然无法读取
y = data['percentage expenditure'].values.reshape(-1,1)#任意行，1列,因为原始数据的标签名含有空格

#4，调用相关的库，建模
from sklearn.tree import DecisionTreeRegressor #决策树英文名：DecisionTree,决策树回归
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state= 42)

dt_model = DecisionTreeRegressor(max_depth = 3) # 建立模型 深度默认为3，调节之后score改变
dt_model.fit(x,y)
dt_model.score(x,y)#准确率 0.9354206797638399

dt_model.fit(x_train,y_train)
dt_model.score(x_train,y_train)#准确率 0.9546885289014841         深度为4时，0.9666387554915615  过拟合
dt_model.score(x_test,y_test) #  0.8655932140475638        0.8569448732874969


#5，运用原始数据创建新的测试集
x_data = np.arange(min(x),max(x),0.01).reshape(-1,1)   #将数据的最小值到最大值排列并按照0.01递进
y_predict = dt_model.predict(x_data) #预测

#6，得到结果并画图
plt.scatter(x,y, color="red") #以点表示实际的支出与GDP的关系
plt.plot(x_data,y_predict,color="blue")#以线预测支出与GDP的关系
plt.xlabel("GDP")
plt.ylabel("percentage expenditure")
plt.show()
