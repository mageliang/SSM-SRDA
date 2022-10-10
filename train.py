from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import time
import os
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    # 训练设置
    opt_train = TrainOptions().parse()

    # 加载训练数据集
    dataset_train = create_dataset(opt_train)
    dataset_train_size = len(dataset_train)
    print('The number of training images = %d' % dataset_train_size)

    # 创建训练模型
    model_train = create_model(opt_train)
    model_train.train()
    opt_train.continue_train = False 
    # opt_train.continue_train = True

    model_train.setup(opt_train)
    
    if not os.path.exists(opt_train.tensorboard_dir):
        os.makedirs(opt_train.tensorboard_dir)
    writer = SummaryWriter(opt_train.tensorboard_dir)

    # 设置显示训练结果的类
    visualizer = Visualizer(opt_train)
    num = 0
    for epoch in range(opt_train.epoch_count, opt_train.niter + opt_train.niter_decay + 1):
        epoch_iters = 0
        epoch_start_time = time.time()

        for i, data in enumerate(dataset_train):
            # label, image.shape == [N, C, W, H]
            iter_start_time = time.time()
            
            epoch_iters += 1
            num += 1
            # 训练一次
            model_train.set_input(data)
            model_train.optimize_parameters()

            # 保存训练出来的图像
            if epoch_iters % opt_train.display_freq == 0:
                visualizer.display_current_results_segment(model_train.get_current_visuals(), epoch)

            # 控制台打印loss的值，存储log信息到磁盘
            if epoch_iters % opt_train.print_freq == 0:
                losses = model_train.get_current_losses()
                t_comp = time.time() - iter_start_time
                visualizer.print_current_losses(epoch, epoch_iters, losses, t_comp)
                # writer.add_scalar("cross_entropy", losses['cross_entropy'], num)  #BASELINE_tensorboard
                writer.add_scalar("G-loss", losses['G'], num)
                writer.add_scalar("D-loss", losses['D'], num)

        if epoch % opt_train.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d' % epoch)
            epoch_s = epoch 
            print('saving the model at the end of epoch %d' % epoch_s)
            # model_train.save_networks(epoch)
            model_train.save_networks(epoch_s)
            # print('saving the model at the end of epoch %d' % epoch)
            # # model_train.save_networks(epoch)
            # model_train.save_networks(epoch)

        # 一个epoch 改变一次学习率
        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt_train.niter + opt_train.niter_decay, time.time() - epoch_start_time))
        model_train.update_learning_rate()

    writer.close()
    





