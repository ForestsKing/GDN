import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, f1_score
from sklearn.preprocessing import MinMaxScaler


def adjust_predicts(label, pred):
    anomaly_state = False

    for i in range(len(label)):
        if label[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if label[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(label)):
                if label[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1

        elif label[i] == 0:
            anomaly_state = False

        if anomaly_state:
            pred[i] = 1

    return pred


def bestf1_threshold(test_scores, test_label, adjust=True, start=0, end=1, search_step=1000):
    best_f1 = 0.0
    best_threshold = 0.0

    for i in range(search_step):
        threshold = start + i * ((end - start) / search_step)
        test_pred = (test_scores > threshold).astype(int)
        if adjust:
            test_pred = adjust_predicts(test_label, test_pred)
        f1 = f1_score(test_label, test_pred)

        if f1 > best_f1:
            best_threshold = threshold
            best_f1 = f1

    return best_threshold


def myevaluate(thre_score, thre_label, test_score, test_label):
    thre_score = np.mean(thre_score, axis=-1)
    test_score = np.mean(test_score, axis=-1)

    scaler = MinMaxScaler()
    thre_score = scaler.fit_transform(thre_score.reshape(-1, 1)).reshape(-1)
    test_score = scaler.transform(test_score.reshape(-1, 1)).reshape(-1)

    thresh = bestf1_threshold(thre_score, thre_label)
    test_pred = (test_score > thresh).astype(int)
    test_pred = adjust_predicts(test_label, test_pred)
    precision_adjust, recall_adjust, f_score_adjust, _ = precision_recall_fscore_support(test_label, test_pred,
                                                                                         average='binary')

    thresh = bestf1_threshold(thre_score, thre_label, adjust=False)
    test_pred = (test_score > thresh).astype(int)
    precision, recall, f_score, _ = precision_recall_fscore_support(test_label, test_pred, average='binary')

    auc = roc_auc_score(test_label, test_score)
    return precision, precision_adjust, recall, recall_adjust, f_score, f_score_adjust, auc


def evaluate(valid_score, test_score, test_label, before_num=3):
    median = np.percentile(valid_score, 50, axis=0)
    iqr = np.percentile(valid_score, 75, axis=0) - np.percentile(valid_score, 25, axis=0)

    valid_score = (valid_score - median) / (iqr + 1e-6)
    test_score = (test_score - median) / (iqr + 1e-6)

    valid_score = np.max(valid_score, axis=-1)
    test_score = np.max(test_score, axis=-1)

    smoothed_valid_score = valid_score
    for i in range(before_num, len(valid_score)):
        smoothed_valid_score[i] = np.mean(valid_score[i - before_num:i + 1])

    smoothed_test_score = test_score
    before_num = 3
    for i in range(before_num, len(test_score)):
        smoothed_test_score[i] = np.mean(test_score[i - before_num:i + 1])

    thresold = np.max(smoothed_valid_score)
    test_pred = (smoothed_test_score > thresold).astype(int)

    precision, recall, f_score, _ = precision_recall_fscore_support(test_label, test_pred, average='binary')

    test_pred = adjust_predicts(test_label, test_pred)
    precision_adjust, recall_adjust, f_score_adjust, _ = precision_recall_fscore_support(test_label, test_pred,
                                                                                         average='binary')
    auc = roc_auc_score(test_label, test_score)

    return precision, precision_adjust, recall, recall_adjust, f_score, f_score_adjust, auc
