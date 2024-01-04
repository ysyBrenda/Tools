from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#================ (1) 数据 ================
all_file = './data/TrainData.csv'
train_df = pd.read_csv(all_file)
X = train_df.values[:, 3:7]   #读第3，4，5，6列  （从0开始计数）
y = train_df.values[:, 7]   #读第7列  （从0开始计数）
# 划分训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=27)

# 读入预测数据 （可选）
pred_file = './data/PredictData.csv' #预测文件
pred_df=pd.read_csv(pred_file)
pred_x=pred_df.values[:,3:7]
predxyz=pred_df.values[:, 0:3]  #X,Y,Z


if __name__ == '__main__':
    # ================ (2) 模型构建与训练 ================
    # 初始化随机森林回归模型
    rnd_clf = RandomForestRegressor(n_estimators = 300,random_state=42)
    # 训练模型
    rnd_clf.fit(train_x, train_y)
    # 使用训练好的模型进行预测
    y_pred_rf = rnd_clf.predict(test_x)

    #================ (3) 模型评价 ================
    mse = mean_squared_error(test_y, y_pred_rf)
    print("均方误差（MSE）：", mse)
    y_pred_rftrain = rnd_clf.predict(train_x)
    trainmse = mean_squared_error(train_y, y_pred_rftrain)
    print("均方误差（trainMSE）：", trainmse)

    r2 = r2_score(test_y, y_pred_rf)
    print("决定系数（R^2 score）：", r2)

    rmse = mean_squared_error(test_y, y_pred_rf, squared=False)
    print("均方根误差（RMSE）：", rmse)

    mae = mean_absolute_error(test_y, y_pred_rf)
    print("平均绝对误差（MAE）：", mae)

    # ================ (4) 模型预测 ================ (可选)
    #预测并输出文件
    output_file = open("./output/prob_RandomForests_300.csv", 'w')
    predictions = []
    known_preds = rnd_clf.predict(pred_x)
    for i, unknown_pred in enumerate(known_preds):
        pred_prob = known_preds
        # pred_label = unknown_pred.argmax(axis=0)
        predictions.append(pred_prob)
        output_file.write(
            str(i) + ', ' + str(predxyz[i, 0]) + ', ' + str(predxyz[i, 1]) + ', ' + str(predxyz[i, 2]) + ', ' + str(
                pred_prob[i]) + '\n')
    output_file.close()

    # ================ (5) 其他结果绘制 ================ (可选)
    # 绘制随机森林特征重要性 柱状图
    from matplotlib import pyplot as plt

    feature_name = np.array(['dF', 'gF', 'cF', 'wF'])
    print(rnd_clf.feature_importances_)
    sorted_idx = rnd_clf.feature_importances_.argsort()
    bar = plt.barh(feature_name[sorted_idx], rnd_clf.feature_importances_[sorted_idx])
    text = plt.title('Feature importances')
    savefigname = './feature_importances.jpg'
    plt.savefig(savefigname, dpi=800)
    plt.show()
