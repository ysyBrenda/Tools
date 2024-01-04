import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_fscore_support


# 从CSV文件读取预测结果，假设CSV文件包含两列：真实标签和预测概率.把csv两列表头改成 label和pred。不限次序
csv_file_path = './output/label_predscore.csv'
df = pd.read_csv(csv_file_path,header=0)

# 真实标签和预测概率的列名称，根据CSV文件中的列名称进行替换
true_labels_column = 'label'
predicted_probabilities_column = 'pred'

# 获取真实标签和预测概率的值
true_labels = df[true_labels_column]
predicted_probabilities = df[predicted_probabilities_column]

# 计算AUC值
auc_value = roc_auc_score(true_labels, predicted_probabilities)

fpr, tpr, thresholds = roc_curve(true_labels, predicted_probabilities)

# youden_index计算
youden_index = tpr - fpr
optimal_threshold_index = np.argmax(youden_index)
max_youden_index = youden_index[optimal_threshold_index]
optimal_threshold = thresholds[optimal_threshold_index]
print("最大Youden指数:", max_youden_index)
print("最优阈值:", optimal_threshold)

# 根据最优阈值将预测概率分为两类
binary_predictions = (predicted_probabilities > optimal_threshold).astype(int)

precision, recall, f1_score, _ = precision_recall_fscore_support\
    (true_labels, binary_predictions, average='binary')
# 输出评价指标
print("AUC:", auc_value)
print("precision准确率:", precision)
# print("recall召回率:", recall)
print("f1_score:", f1_score)

# 计算特异度和敏感度
TN = ((true_labels == 0) & (binary_predictions == 0)).sum()  # 真阴性
FP = ((true_labels == 0) & (binary_predictions == 1)).sum()  # 假阳性
FN = ((true_labels == 1) & (binary_predictions == 0)).sum()  # 假阴性
TP = ((true_labels == 1) & (binary_predictions == 1)).sum()  # 真阳性

specificity = TN / (TN + FP)  # 特异度  specificity  =1-FPR
sensitivity = TP / (TP + FN)  # 敏感度 sensitivity = recall召回率 =TPR真阳性率
# 计算准确率
accuracy = (TP + TN) / (TP + TN + FP + FN)  # 准确率  accuracy
print("specificity特异度:", specificity)
print("sensitivity敏感度=recall召回率=TPR真阳性率:", sensitivity)
print("accuracy:", accuracy)

# 画ROC曲线
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_value)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')  #'1-特异性'
plt.ylabel('True Positive Rate')   #'敏感度'
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
# plt.savefig(savefigname, dpi=600)