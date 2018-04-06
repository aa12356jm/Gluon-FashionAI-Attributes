import mxnet as mx
import numpy as np
import os, time, logging, math, argparse

from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models

#参数设置
def parse_args():
    parser = argparse.ArgumentParser(description='Gluon for FashionAI Competition',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #任务类别，必须指定
    parser.add_argument('--task', required=True, type=str,
                        help='name of the classification task')
    #预训练模型，必须指定
    parser.add_argument('--model', required=True, type=str,
                        help='name of the pretrained model from model zoo.')
    #cpu核心数
    parser.add_argument('-j', '--workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    #GPU数量
    parser.add_argument('--num-gpus', default=0, type=int,
                        help='number of gpus to use, 0 indicates cpu only')
    #迭代的epoch数量
    parser.add_argument('--epochs', default=40, type=int,
                        help='number of training epochs')
	#batch-size大小    
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        help='mini-batch size')
    #学习率设置
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        help='initial learning rate')
    #冲量设置
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    
    #权重衰减(通常使用L2范数)，对权重的惩罚，使权重不至于过大，保持在较小的范围内
    parser.add_argument('--weight-decay', '--wd', dest='wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    
    #学习率衰减比例，就是每次衰减为原来的0.75
    parser.add_argument('--lr-factor', default=0.75, type=float,
                        help='learning rate decay ratio')
    
    #学习率衰减频率，就是每次隔多少个epoch按照比例进行衰减
    parser.add_argument('--lr-steps', default='10,20,30', type=str,
                        help='list of learning rate decay epochs as in str')
    args = parser.parse_args()
    return args

#计算AP精度
def calculate_ap(labels, outputs):
    cnt = 0
    ap = 0.
    for label, output in zip(labels, outputs):
        for lb, op in zip(label.asnumpy().astype(np.int),
                          output.asnumpy()):
            op_argsort = np.argsort(op)[::-1]
            lb_int = int(lb)
            ap += 1.0 / (1+list(op_argsort).index(lb_int))
            cnt += 1
    return ((ap, cnt))

#对训练数据进行crop增强
def ten_crop(img, size):
    H, W = size
    iH, iW = img.shape[1:3]

    if iH < H or iW < W:
        raise ValueError('image size is smaller than crop size')

    img_flip = img[:, :, ::-1]
    crops = nd.stack(
        img[:, (iH - H) // 2:(iH + H) // 2, (iW - W) // 2:(iW + W) // 2],
        img[:, 0:H, 0:W],
        img[:, iH - H:iH, 0:W],
        img[:, 0:H, iW - W:iW],
        img[:, iH - H:iH, iW - W:iW],

        img_flip[:, (iH - H) // 2:(iH + H) // 2, (iW - W) // 2:(iW + W) // 2],
        img_flip[:, 0:H, 0:W],
        img_flip[:, iH - H:iH, 0:W],
        img_flip[:, 0:H, iW - W:iW],
        img_flip[:, iH - H:iH, iW - W:iW],
    )
    return (crops)

#训练数据增广函数
def transform_train(data, label):
    im = data.astype('float32') / 255
    #进行数据增强
    auglist = image.CreateAugmenter(data_shape=(3, 224, 224), resize=256,
                                    rand_crop=True, rand_mirror=True,
                                    mean = np.array([0.485, 0.456, 0.406]),
                                    std = np.array([0.229, 0.224, 0.225]))
    for aug in auglist:
        im = aug(im) #对每张图像进行数据增强
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar())

#训练数据增广函数
def transform_val(data, label):
    im = data.astype('float32') / 255
    im = image.resize_short(im, 256) #对数据按照短边进行crop为256*256
    im, _ = image.center_crop(im, (224, 224)) #对数据进行中心裁剪为224*224
    im = nd.transpose(im, (2,0,1))  #
    im = mx.nd.image.normalize(im, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))#归一化操作
    return (im, nd.array([label]).asscalar()) #返回图像和标签

def transform_predict(im):
    im = im.astype('float32') / 255
    im = image.resize_short(im, 256)
    im = nd.transpose(im, (2,0,1))
    im = mx.nd.image.normalize(im, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    im = ten_crop(im, (224, 224))
    return (im)

#训练过程的进度条
def progressbar(i, n, bar_len=40):
    percents = math.ceil(100.0 * i / float(n))
    filled_len = int(round(bar_len * i / float(n)))
    prog_bar = '=' * filled_len + '-' * (bar_len - filled_len)
    print('[%s] %s%s' % (prog_bar, percents, '%'), end = '\r')

#使用验证集对网络进行调参
def validate(net, val_data, ctx):
    metric = mx.metric.Accuracy()
    L = gluon.loss.SoftmaxCrossEntropyLoss() #交叉熵损失
    AP = 0.
    AP_cnt = 0
    val_loss = 0
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(X) for X in data] #每一张图像经过网络前向计算得到的输出值
        metric.update(label, outputs) 
        loss = [L(yhat, y) for yhat, y in zip(outputs, label)]#根据真实标签和计算结果来计算loss
        val_loss += sum([l.mean().asscalar() for l in loss]) / len(loss) #总验证loss
        ap, cnt = calculate_ap(label, outputs) #计算AP精度
        AP += ap
        AP_cnt += cnt
    _, val_acc = metric.get()
    return ((val_acc, AP / AP_cnt, val_loss / len(val_data)))


#训练函数
def train():
    logging.info('Start Training for Task: %s\n' % (task))

    # Initialize the net with pretrained model，使用预训练好的模型参数
    pretrained_net = gluon.model_zoo.vision.get_model(model_name, pretrained=True)
	
	#使用此网络结构
    finetune_net = gluon.model_zoo.vision.get_model(model_name, classes=task_num_class) 
    finetune_net.features = pretrained_net.features  #拷贝预训练模型的参数
    finetune_net.output.initialize(init.Xavier(), ctx = ctx) #对网络进行初始化参数
    finetune_net.collect_params().reset_ctx(ctx) #参数放在gpu上
    finetune_net.hybridize()

    # Define DataLoader定义数据加载器
    train_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(
            os.path.join('data/train_valid', task, 'train'),
            transform=transform_train),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, last_batch='discard')

    val_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(
            os.path.join('data/train_valid', task, 'val'),
            transform=transform_val),
        batch_size=batch_size, shuffle=False, num_workers = num_workers)

    # Define Trainer 训练器
    trainer = gluon.Trainer(finetune_net.collect_params(), 'sgd', {
        'learning_rate': lr, 'momentum': momentum, 'wd': wd})
    metric = mx.metric.Accuracy()
    L = gluon.loss.SoftmaxCrossEntropyLoss()#损失函数
    lr_counter = 0
    num_batch = len(train_data)#训练数据有多少个batch-size

    # Start Training
    for epoch in range(epochs): #每次训练一个epoch
        if epoch == lr_steps[lr_counter]: 
        	#学习率衰减为原来的lr_factor
            trainer.set_learning_rate(trainer.learning_rate*lr_factor)
            lr_counter += 1

        tic = time.time()
        train_loss = 0
        metric.reset()
        AP = 0.
        AP_cnt = 0

        #每次从训练数据中拿出batch-size个数据进行训练
        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            with ag.record():
                outputs = [finetune_net(X) for X in data] #每个图像经过网络计算得到结果
                loss = [L(yhat, y) for yhat, y in zip(outputs, label)]#计算loss
            for l in loss:
                l.backward() #计算梯度

            trainer.step(batch_size) #每次迭代batch-size个数据
            train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)

            metric.update(label, outputs)
            ap, cnt = calculate_ap(label, outputs)
            AP += ap
            AP_cnt += cnt

            progressbar(i, num_batch-1) #训练进度条

        train_map = AP / AP_cnt
        _, train_acc = metric.get()
        train_loss /= num_batch

        val_acc, val_map, val_loss = validate(finetune_net, val_data, ctx) #计算验证精度

        logging.info('[Epoch %d] Train-acc: %.3f, mAP: %.3f, loss: %.3f | Val-acc: %.3f, mAP: %.3f, loss: %.3f | time: %.1f' %
                 (epoch, train_acc, train_map, train_loss, val_acc, val_map, val_loss, time.time() - tic))

    logging.info('\n')
    return (finetune_net)

#使用test图像进行测试
def predict(task):
    logging.info('Training Finished. Starting Prediction.\n')
    f_out = open('submission/%s.csv'%(task), 'w')  #将测试结果写入到此文件

    #加载测试集中的图像，将网络检测结果写入到文件中
    with open('data/rank/Tests/question.csv', 'r') as f_in:
        lines = f_in.readlines()
    tokens = [l.rstrip().split(',') for l in lines]
    task_tokens = [t for t in tokens if t[1] == task]
    n = len(task_tokens)
    cnt = 0
    for path, task, _ in task_tokens:
        img_path = os.path.join('data/rank', path)
        with open(img_path, 'rb') as f:
            img = image.imdecode(f.read())
        data = transform_predict(img)
        out = net(data.as_in_context(mx.gpu(0)))
        out = nd.SoftmaxActivation(out).mean(axis=0)

        pred_out = ';'.join(["%.8f"%(o) for o in out.asnumpy().tolist()])
        line_out = ','.join([path, task, pred_out])
        f_out.write(line_out + '\n')
        cnt += 1
        progressbar(cnt, n)
    f_out.close()

# Preparation
args = parse_args() #解析命令行参数

task_list = {
    'collar_design_labels': 5,
    'skirt_length_labels': 6,
    'lapel_design_labels': 5,
    'neckline_design_labels': 10,
    'coat_length_labels': 8,
    'neck_design_labels': 5,
    'pant_length_labels': 6,
    'sleeve_length_labels': 9
}
task = args.task  #参数中指定的任务类别
task_num_class = task_list[task] #此任务中的有多少个类别

model_name = args.model #参数中指定的预训练模型类型

epochs = args.epochs  #参数中指定的epoch数量
lr = args.lr    #参数中指定的学习率
batch_size = args.batch_size  #参数中指定的学习率
momentum = args.momentum  #参数中指定的momentum的大小
wd = args.wd  #参数中指定的weight decay大小

lr_factor = args.lr_factor #参数中指定的每次学习率衰减参数
lr_steps = [int(s) for s in args.lr_steps.split(',')] + [np.inf]  #参数中指定的学习率衰减频率

num_gpus = args.num_gpus  #参数中指定的GPU数量
num_workers = args.num_workers   #参数中指定的cpu核心数
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
batch_size = batch_size * max(num_gpus, 1)  #batchsize根据GPU个数来确定

#log信息
logging.basicConfig(level=logging.INFO,
                    handlers = [
                        logging.StreamHandler(),
                        logging.FileHandler('training.log')
                    ])

if __name__ == "__main__":
    net = train() #训练
    predict(task) #测试

