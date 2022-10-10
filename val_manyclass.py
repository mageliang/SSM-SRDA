from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.visualizer import save_segment_result
from util.metrics import RunningScore
from util import util
import time
import os
import numpy as np
import torch.nn as nn

best_result = 0
best_F1score = 0
label_values = ['class 1', 'class 2', 'class 3', 'class 4']

# # 混淆矩阵可视化
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
import itertools
#画混淆矩阵
# def plot_confusion_matrix(cm,
#                           target_names,
#                           title='Confusion matrix',
#                         #   cmap=plt.cm.Reds, # 设置混淆矩阵的颜色主题
#                           cmap=plt.cm.PuBu, # 设置混淆矩阵的颜色主题

#                           normalize=True):
   
 
#     accuracy = np.trace(cm) / float(np.sum(cm))
#     misclass = 1 - accuracy

#     if cmap is None:
#         cmap = plt.get_cmap('Blues')

#     plt.figure(figsize=(15, 12))
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()

#     if target_names is not None:
#         tick_marks = np.arange(len(target_names))
#         plt.xticks(tick_marks, target_names, rotation=45)
#         plt.yticks(tick_marks, target_names)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


#     thresh = cm.max() / 1.5 if normalize else cm.max() / 2
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         if normalize:
#             plt.text(j, i, "{:0.4f}".format(cm[i, j]),
#                         horizontalalignment="center",
#                         color="white" if cm[i, j] > thresh else "black")
#         else:
#             plt.text(j, i, "{:,}".format(cm[i, j]),
#                         horizontalalignment="center",
#                         color="white" if cm[i, j] > thresh else "black")


#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    
#     plt.show()



if __name__ == '__main__':
    # 验证设置
    opt_val = TestOptions().parse()

    # 设置显示验证结果存储的设置
    web_dir = os.path.join(opt_val.checkpoints_dir, opt_val.name, 'val')
    image_dir = os.path.join(web_dir, 'images')
    util.mkdirs([web_dir, image_dir])

    # 设置验证数据集
    dataset_val = create_dataset(opt_val)
    dataset_val_size = len(dataset_val)
    print('The number of valling images = %d' % dataset_val_size)

    # 创建验证模型
    model_val = create_model(opt_val)
    model_val.eval()

    # 训练设置
    opt_train = TrainOptions().parse()

    # 设置显示训练结果的类
    visualizer = Visualizer(opt_train)
    # for epoch in range(1, 61):
    


    for epoch in range(1,201):
        epoch_iters = 0
        epoch_start_time = time.time()

        # 验证结果
        metrics = RunningScore(opt_val.num_classes)

        model_val.opt.epoch = epoch
        model_val.setup(model_val.opt)

        # for i, data in enumerate(dataset_val):
        for i, data in enumerate(dataset_val):

            model_val.set_input(data)
            model_val.forward()
            gt = np.squeeze(data["label"].numpy(), axis=1)  # [N, W, H]
            pre = model_val.pre
            pre = pre.data.max(1)[1].cpu().numpy()  # [N, W, H]
            # print("gt:",gt)
            # print("gt.shape:",gt.shape) # gt.shape: (2, 256, 256)
            metrics.update(gt, pre)
            confusion_matrix = metrics.update(gt, pre)
            # 保存结果
            if i % opt_train.display_freq == 0:  # 逻辑有点问题
                save_segment_result(model_val.get_current_visuals(), epoch, opt_train.display_winsize, image_dir,
                                    web_dir, opt_train.name)
        # confusion_matrix_end = confusion_matrix
        # plot_confusion_matrix(confusion_matrix_end, normalize=False,target_names=label_values,title='Confusion Matrix')
        val_class_iou, iu, val_class_f1, mean_f1 = metrics.get_scores()
        # with open(web_dir + "_result.txt","a") as f:
        #     f.write("epoch" + str(epoch) + ":" + str(val_class_iou) + " mean_iou:" + str(iu)+"\n")

        # if best_result < np.mean(iu):
        #     best_result = np.mean(iu)
        #     with open(web_dir + "best_result.txt", mode="w+") as f:
        #         f.write("epoch" + str(epoch) + ":" + str(val_class_iou) + " best_mean_iou:" + str(best_result))


        with open(web_dir + "1_result-iou-score.txt","a") as f:
            f.write("epoch" + str(epoch) + ":" + str(val_class_iou) + " mean_iou:" + str(np.mean(iu))+"\n")
            f.write("epoch" + str(epoch) + ":" + str(val_class_f1) + " mean_f1:" + str(mean_f1)+"\n")

        if best_result < np.mean(iu):
            best_result = np.mean(iu)
            with open(web_dir + "1_best_result-2-iou.txt", mode="w+") as f:
                f.write("epoch" + str(epoch) + ":" + str(val_class_iou) + " best_mean_iou:" + str(best_result))

        if best_F1score < mean_f1:
            best_F1score = mean_f1
            with open(web_dir + "1_best_result-2-F1.txt", mode="w+") as f:
                f.write("epoch" + str(epoch) + ":" + str(val_class_f1) + " best_mean_f1:" + str(best_F1score))


        # 一个epoch 改变一次学习率
        print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt_train.niter + opt_train.niter_decay, time.time() - epoch_start_time))

        





