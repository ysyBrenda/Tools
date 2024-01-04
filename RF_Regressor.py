from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#read data
all_file = './20240104/Fault_TrainData.csv'    # SSDTrainData(1).csv'
train_df = pd.read_csv(all_file)
X = train_df.values[:, 3:7]   #3:7
y =train_df.values[:, 7]
# y = (train_df.values[:, 7] == 1).astype(np.float64)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=27)

filepred = './20240104/weizhi0308.csv'   #预测文件    SSDPredictData(1).csv'
pred_df=pd.read_csv(filepred)
pred_x=pred_df.values[:,3:7];
predxyz=pred_df.values[:, 0:3];  #X,Y,Z

if 1: #Rf 0.9229  &0.81
    # rnd_clf = RandomForestClassifier(n_estimators=5, random_state=42,max_depth=5,min_samples_leaf=10)
    rnd_clf = RandomForestRegressor(n_estimators=100, random_state=42)    #,min_samples_leaf=1
    rnd_clf.fit(train_x, train_y)
    y_pred_rf = rnd_clf.predict(test_x)
    print(y_pred_rf)
    print(test_x)

    # 评估模型性能
    mse = mean_squared_error(test_y, y_pred_rf)
    print("均方误差（MSE）：", mse)
    y_pred_rftrain = rnd_clf.predict(train_x)
    trainmse = mean_squared_error(train_y, y_pred_rftrain)
    print("均方误差（trainMSE）：", trainmse)

    from sklearn.metrics import r2_score
    r2 = r2_score(test_y, y_pred_rf)
    print("决定系数（R^2 score）：", r2)

    from sklearn.metrics import mean_squared_error
    rmse = mean_squared_error(test_y, y_pred_rf, squared=False)
    print("均方根误差（RMSE）：", rmse)

    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(test_y, y_pred_rf)
    print("平均绝对误差（MAE）：", mae)
    # #----------输出所有train的预测-用来画预测图
    # output_file = open("./data/output/label_prob_RandomForests_JJ.csv", 'w')
    # predictions = []
    # known_preds = rnd_clf.predict_proba(X)  #将所有的train数据计算，用来画预测曲线
    # for i, unknown_pred in enumerate(known_preds):
    #     pred_prob = known_preds[:, 1]
    #     # pred_label = unknown_pred.argmax(axis=0)
    #     predictions.append(pred_prob)
    #     output_file.write(
    #         str(i) + ', ' + str(y[i]) + ', ' + str(pred_prob[i]) + '\n')
    # output_file.close()
    # #--------------
    #
    #预测并输出文件
    output_file = open("./20240104/output/prob_RandomForests_100.csv", 'w')
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

    # 绘制随机森林特征重要性 柱状图
    from matplotlib import pyplot as plt
    feature_name = np.array(['dF', 'gF', 'cF', 'wF'])
    print(rnd_clf.feature_importances_)
    sorted_idx = rnd_clf.feature_importances_.argsort()
    bar = plt.barh(feature_name[sorted_idx], rnd_clf.feature_importances_[sorted_idx])
    text=plt.title('ZP')
    savefigname = './20240104/output/feature_importances.jpg'
    plt.savefig(savefigname, dpi=800)
    plt.show()


elif 1:#svm 0.80/0.75  &0.96/0.80
    rbf_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=1, C=1e+1, probability=True))])
    #gamma=1., C=1e+1, 0.80/0.75 RBF高斯核
    #gamma = 1., C = 1e+1, 0.896/0.845   0.5, C=1e+1  0.84
    rbf_kernel_svm_clf.fit(train_x, train_y)
    predict_prob_y = rbf_kernel_svm_clf.predict_proba(test_x)
    #print(predict_prob_y)
    #print(test_y)
    # end svm ,start metrics
    predict_prob_train = rbf_kernel_svm_clf.predict_proba(train_x)
    train_auc = metrics.roc_auc_score(train_y, predict_prob_train[:, 1])
    print(train_auc)
    test_auc = metrics.roc_auc_score(test_y, predict_prob_y[:, 1])
    print(test_auc)

    output_file = open("./data/output/prob_SVM_SSD.csv", 'w')
    predictions = []
    known_preds = rbf_kernel_svm_clf.predict_proba(pred_x)
    for i, unknown_pred in enumerate(known_preds[:,1]):
        pred_prob = known_preds
        predictions.append(pred_prob)
        output_file.write(
            str(i) + ', ' + str(predxyz[i, 0]) + ', ' + str(predxyz[i, 1]) + ', ' + str(predxyz[i, 2]) + ', ' + str(
                pred_prob[i,1]) + '\n')
    output_file.close()
elif 1:   #MLP  #0.86/0.83  &0.99/0.82
    mlp_clf = Pipeline([("scaler", StandardScaler()),
                    ('mlp_clf', MLPClassifier(solver='lbfgs',random_state=42, hidden_layer_sizes=200))])#lbfgs 200.200.200
    # solver = 'adam'\lbfgs\sgd, max_iter = 500, learning_rate_init = 0.001, shuffle = True, alpha = 0.001, random_state = 42, hidden_layer_sizes = [
        # 200, 200, 200]0.89 learning_rate='invscaling'\adaptive
    mlp_clf.fit(train_x,train_y)
    predict_prob_y = mlp_clf.predict_proba(test_x)
    print(predict_prob_y)
    print(test_y)
    # end,start metrics
    predict_prob_train=mlp_clf.predict_proba(train_x)
    train_auc = metrics.roc_auc_score(train_y, predict_prob_train[:,1])
    print(train_auc)
    test_auc = metrics.roc_auc_score(test_y, predict_prob_y[:,1])
    print(test_auc)

    # ----------输出所有train的预测-用来画预测图
    output_file = open("./data/output/label_prob_MLP_SSD.csv", 'w')
    predictions = []
    known_preds = mlp_clf.predict_proba(X)  # 将所有的train数据计算，用来画预测曲线
    for i, unknown_pred in enumerate(known_preds):
        pred_prob = known_preds[:, 1]
        # pred_label = unknown_pred.argmax(axis=0)
        predictions.append(pred_prob)
        output_file.write(
            str(i) + ', ' + str(y[i]) + ', ' + str(pred_prob[i]) + '\n')
    output_file.close()
    # --------------

    output_file = open("./data/output/prob_MLP_SSD_test.csv", 'w')
    # output_file = open("./XIADIANOUTPUT2/prob_4_MLP_200_3_0123.csv", 'w')
    predictions = []
    known_preds = mlp_clf.predict_proba(pred_x)
    for i, unknown_pred in enumerate(known_preds[:, 1]):
        pred_prob = known_preds
        predictions.append(pred_prob)
        output_file.write(
            str(i) + ', ' + str(predxyz[i, 0]) + ', ' + str(predxyz[i, 1]) + ', ' + str(predxyz[i, 2]) + ', ' + str(
                pred_prob[i,1]) + '\n')
    output_file.close()

else :

    # LogisticasRegression
    log_clf = Pipeline(
        [('scaler', StandardScaler()), ('log_clf', LogisticRegression(solver="lbfgs", random_state=42))])  # todo
    log_clf.fit(train_x, train_y)
    y_pred = log_clf.predict_proba(test_x)
    print(log_clf.__class__.__name__, metrics.roc_auc_score(test_y, y_pred[:, 1]))  # 训练auc
    y_trian = log_clf.predict_proba(train_x)
    print(log_clf.__class__.__name__, metrics.roc_auc_score(train_y, y_trian[:, 1]))  # 验证auc
    # 预测并输出成文件
    output_file = open("./SSD/OUTPUT/label_prob_SSD_LogisticRegression.csv", 'w')  # todo
    predictions = []
    # output_file.write("Prediction , " + "Actual , " + "Accuracy" + '\n')
    known_preds = log_clf.predict_proba(pred_x)
    for i, unknown_pred in enumerate(known_preds):
        pred_prob = known_preds[:, 1]
        # pred_label = unknown_pred.argmax(axis=0)
        predictions.append(pred_prob)
        output_file.write(
            str(i) + ', ' + str(predxyz[i, 0]) + ', ' + str(predxyz[i, 1]) + ', ' + str(predxyz[i, 2]) + ', ' + str(
            pred_prob[i, 1]) + '\n')
    output_file.close()
