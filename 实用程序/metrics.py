# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np


class RunningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        cls_iu = dict(zip(range(self.n_classes), iu))


        # F1 score
        F1_score = 2.0 * np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=0) + self.confusion_matrix.sum(axis=1)) # 
        mean_F1_score = np.nanmean(F1_score)

        cls_f1 =  dict(zip(range(self.n_classes),F1_score))

        print(" Class_IoU:" + str(cls_iu) +  " miou:" + str(np.nanmean(iu)))
        print(" Class_F1:" + str(cls_f1) +  " mean_F1:" + str(mean_F1_score))

        return  cls_iu, iu, cls_f1,mean_F1_score

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
