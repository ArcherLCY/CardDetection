import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
data = pd.read_csv("creditcard.csv")
data.head(6)
# 输出读取的数据
# print(data)

# 统计Class这一列中有多少不同的值，并排序出来
count_classes = pd.value_counts(data['Class'], sort=True).sort_index()
# 输出结果是：
# 0    284315
# 1       492
# Name: Class, dtype: int64
# print(count_classes)

count_classes.plot(kind='bar')
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")
# 绘图
# plt.show()


# 调用预处理模块,数据归一化
from sklearn.preprocessing import StandardScaler

# 标准化，并产生新的normAmount
# data['normAmount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1, 1))
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
# 删除无用的所在列
data = data.drop(['Time', 'Amount'], axis=1)
data.head()
# print(data)

# 下采集数据
# 取出所有属性。不包含Class这一列
X = data.ix[:, data.columns != 'Class']
# 另外y=class这一列
y = data.ix[:, data.columns == "Class"]
# 计算出Class这一列中有几个为1的元素
number_record_fraud = len(data[data.Class == 1])
# 取出Class这一列所有等于1的行索引
fraud_indices = np.array(data[data.Class == 1].index)
# 取出Class这一列所有等于0的行索引
normal_indices = np.array(data[data.Class == 0].index)

# 随机选着和1这个属性样本相同个数的0样本
random_normal_indices = np.random.choice(normal_indices, number_record_fraud, replace=False)
# 转换为numpy的格式
random_normal_indices = np.array(random_normal_indices)

# 将正负样本拼接在一起
under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])

# 下采集数据
under_sample_data = data.iloc[under_sample_indices, :]

# 下采集数据集的数据（除Class这列外）
X_undersample = under_sample_data.ix[:, under_sample_data.columns != "Class"]
# 下采集数据集的label（只取Class这列）
y_undersample = under_sample_data.ix[:, under_sample_data.columns == "Class"]

# 输出
# 打印正样本数目
print("Percentage of normal transactions:",
      len(under_sample_data[under_sample_data.Class == 0]) / len(under_sample_data))
# 打印负样本数目
print("Percentage of fraud transactions:",
      len(under_sample_data[under_sample_data.Class == 1]) / len(under_sample_data))
# 打印总数
print("Total number of transaction in resampled data:", len(under_sample_data))

# 交叉验证模块引入
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

# 训练集和数据切分
# 对整个训练集进行切分，testsize表示训练集大小，state=0在切分时进行数据重新洗牌的标识位
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("Number transactions train dataset:", len(X_train))
print("Number transactions test dataset:", len(X_test))
print("Total number of transactions:", len(X_train) + len(X_test))

# 对下采样数据切分
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample,
                                                                                                    y_undersample,
                                                                                                    test_size=0.3,
                                                                                                    random_state=0)
print("切分:")
print("Number transactions train dataset:", len(X_train_undersample))
print("Number transactions test dataset:", len(X_test_undersample))
print("Total number of transactions:", len(X_train_undersample) + len(X_test_undersample))

# 调用逻辑回归模型
from sklearn.linear_model import LogisticRegression
# 调用K折交叉验证
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, recall_score, classification_report


# 引用混淆矩阵，召回率
def printing_Kfold_scores(x_train_data, y_train_data):
    # 第一个参数：训练集长度，第二个参数：输入为几折交叉验证
    # from sklearn.model_selection import KFold这个版本的库无需传入总数
    fold = KFold(5, shuffle=False)
    # from sklearn.cross_validation import KFold版本需要传入总数
    # fold = KFold(len(y_train_data), 5, shuffle=False)
    # 传入选择正则化的参数
    c_param_range = [0.01, 0.1, 1, 10, 100]

    results_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_paramter', "Mean recall score"])
    results_table['C_parameter'] = c_param_range

    j = 0
    for c_param in c_param_range:
        print("---------------------------------------")
        print("C parameter:", c_param)
        print("---------------------------------------")
        # 第一个for循环用来了打印每个正则化参数下的输出

        recall_accs = []
        # from sklearn.cross_validation import KFold写法
        # for iteration, indices in enumerate(fold, start=1):
        # from sklearn.model_selection import KFold版本写法；
        for iteration, indices in enumerate(fold.split(x_train_data)):
            # 传入正则化参数下的输出
            # 用一个确定的c参数调用逻辑回归模型，把c_param_range代入到逻辑回归模型中，并使用了l1正则化
            lr = LogisticRegression(C=c_param, penalty='l1', solver='liblinear')
            # 使用训练数据拟合模型，在这个例子中，我们使用这交叉部分训练模型
            # 套路：使训练模型fit模型,使用indices[0]的数据进行拟合曲线，使用indices[1]的数据进行误差测试
            # lr.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0], :].values.ravel())
            lr.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0], :].values.ravel())
            # 在训练集数据中，使用测试指标来预测值
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1], :].values)

            # 评估the recall score
            recall_acc = recall_score(y_train_data.iloc[indices[1], :].values, y_pred_undersample)
            recall_accs.append(recall_acc)
            print("Iteration", iteration, ':recall score =', recall_acc)

        # 这些recall scores的平均值，就是我们想要的指标
        results_table.loc[j, 'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print("")
        print("Mean recall score", np.mean(recall_accs))

    best_c = results_table.loc[results_table['Mean recall score'].values.argmax()]['C_parameter']
    # 最后，验证那个c参数是最好的选择
    print("Best model to choose from cross validation is with parameter=", best_c)
    return best_c


best_c = printing_Kfold_scores(X_train_undersample, y_train_undersample)

# 混淆矩阵
import itertools


# 这个方法输出和画出混淆矩阵
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    # cm为数据，interpolation=‘nearest'使用最近邻插值，cmap颜色图谱（colormap），默认绘制为RGB（A）颜色空间
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    # xticks为刻度下标
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    # text()命令可以在任意位置添加文字
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else 'black')
    # 自动紧凑布局
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


lr = LogisticRegression(C=best_c, penalty='l1', solver='liblinear')
lr.fit(X_train_undersample, y_train_undersample.values.ravel())
y_pred_undersample = lr.predict(X_test_undersample.values)
# 计算混淆矩阵
cnf_matrix = confusion_matrix(y_test_undersample, y_pred_undersample)
# 输出精度为小数点后两位
np.set_printoptions(precision=2)
print("Recall metric in the testing dataset:", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))
# 画出非标准化的混淆矩阵
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()

# 下采样数据进行训练，使用原始数据进行测试
lr = LogisticRegression(C=best_c, penalty='l1', solver='liblinear')
lr.fit(X_train_undersample, y_train_undersample.values.ravel())
y_pred = lr.predict(X_test.values)
# 计算混淆矩阵
cnf_matrix = confusion_matrix(y_test, y_pred)
# 输出精度为小数点后两位
np.set_printoptions(precision=2)
print("Recall metric in the testing dataset :", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))
# 画出非标准化的混淆矩阵
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()

print('************************************************')
# 原始数据进行K折交叉验证
best_c = printing_Kfold_scores(X_train, y_train)

print('==============================================')
# 使用原始数据进行训练与测试
lr = LogisticRegression(C=best_c, penalty='l1', solver='liblinear')
lr.fit(X_train, y_train.values.ravel())
y_pred = lr.predict(X_test.values)
# 计算混淆矩阵
cnf_matrix = confusion_matrix(y_test, y_pred)
# 输出精度为小数点后两位
np.set_printoptions(precision=2)
print("Recall metric in the testing dataset :", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))
# 画出非标准化的混淆矩阵
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()

print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
# 使用下采样数据训练与测试（不同的阈值对结果的影响）
lr = LogisticRegression(C=best_c, penalty='l1', solver='liblinear')
lr.fit(X_train_undersample, y_train_undersample.values.ravel())
y_pred_undersample_proba = lr.predict_proba(X_test_undersample.values)
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
plt.figure(figsize=(10, 10))
j = 1
for i in thresholds:
    y_test_predictions_high_recall = y_pred_undersample_proba[:, 1] > i
    plt.subplot(3, 3, j)
    j += 1
    # 计算混淆矩阵
    cnf_matrix = confusion_matrix(y_test_undersample, y_test_predictions_high_recall)
    # 输出精度为小数点后两位
    np.set_printoptions(precision=2)
    print("Recall metric in the testing dataset:", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))
    # 画出非标准化的混淆矩阵
    class_names = [0, 1]
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Threshold>=%s' % i)
plt.show()

import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# 读取数据
credit_cards = pd.read_csv('creditcard.csv')
columns = credit_cards.columns
# 为了获得特征列，移除最后一列标签列
features_columns = columns.delete(len(columns) - 1)
features = credit_cards[features_columns]
labels = credit_cards['Class']
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2,
                                                                            random_state=0)
oversampler = SMOTE(random_state=0)
os_features, os_labels = oversampler.fit_sample(features_train, labels_train)
print('采样过后，1的样本的个数为：', len(os_labels[os_labels == 1]))

print('##############################################')
# K折交叉验证得到最好的C parameter
os_features = pd.DataFrame(os_features)
os_labels = pd.DataFrame(os_labels)
best_c = printing_Kfold_scores(os_features, os_labels)

print('*********************************************')
lr = LogisticRegression(C=best_c, penalty='l1', solver='liblinear')
lr.fit(os_features, os_labels.values.ravel())
y_pred = lr.predict(features_test.values)
# 计算混淆矩阵
cnf_matrix = confusion_matrix(labels_test, y_pred)
np.set_printoptions(precision=2)
print("Recall metric in the testing dataset:", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))
# 画出非规范化的混淆矩阵
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()
