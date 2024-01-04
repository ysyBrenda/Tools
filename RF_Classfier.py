from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

#================ (1) 数据 ================
all_file = './data/TrainData.csv'
train_df = pd.read_csv(all_file)
X = train_df.values[:, 3:7]   #读第3，4，5，6列  （从0开始计数）
y = (train_df.values[:, 7] == 1).astype(np.float64)  #读第7列  （从0开始计数）
# 划分训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=27)

# 读入预测数据 （可选）
pred_file = './data/PredictData.csv' #预测文件
pred_df=pd.read_csv(pred_file)
pred_x=pred_df.values[:,3:7]
predxyz=pred_df.values[:, 0:3]  #X,Y,Z

#================ (2) 模型构建与训练 ================
# 建立模型
rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42) #更多参数，如max_depth，min_samples_leaf
#训练模型
rnd_clf.fit(train_x, train_y)
# test
y_pred_rf = rnd_clf.predict_proba(test_x)

#================ (3) 模型评价 ================
predict_prob_train = rnd_clf.predict_proba(train_x)
train_auc = metrics.roc_auc_score(train_y, predict_prob_train[:, 1])
print("train_auc=",train_auc)

test_auc = metrics.roc_auc_score(test_y, y_pred_rf[:, 1])
print("test_auc=",test_auc)


#================ (4) 模型预测 ================ (可选)
#输出文件路径
output_file = open("./data/output/prob_RandomForests_.csv", 'w')
predictions = []
#预测
known_preds = rnd_clf.predict_proba(pred_x)
for i, unknown_pred in enumerate(known_preds):
    pred_prob = known_preds[:, 1]
    # pred_label = unknown_pred.argmax(axis=0)
    predictions.append(pred_prob)
    #输出文件：index，坐标x，坐标y，坐标z，预测的概率
    output_file.write(
        str(i) + ', ' + str(predxyz[i, 0]) + ', ' + str(predxyz[i, 1]) + ', ' + str(predxyz[i, 2]) + ', ' + str(
            pred_prob[i]) + '\n')
output_file.close()

#================ (5) 其他结果绘制 ================ (可选)
# 绘制随机森林特征重要性 柱状图
from matplotlib import pyplot as plt
feature_name = np.array(['dF', 'gF', 'cF', 'wF'])
print(rnd_clf.feature_importances_)
sorted_idx = rnd_clf.feature_importances_.argsort()
bar = plt.barh(feature_name[sorted_idx], rnd_clf.feature_importances_[sorted_idx])
text=plt.title('Feature importances')
savefigname = './feature_importances.jpg'
plt.savefig(savefigname, dpi=800)
plt.show()


