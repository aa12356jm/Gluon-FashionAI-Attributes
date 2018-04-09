from mxnet import init
from mxnet.gluon.model_zoo import vision
from mxnet.gluon import nn
from mxnet import image
import numpy as np
from mxnet import nd
import mxnet as mx

#对输入的2个网络的特征层进行融合，返回融合后的一个网络特征
class  ConcatNet(nn.HybridBlock):
    def __init__(self,net1,net2,**kwargs):
        super(ConcatNet,self).__init__(**kwargs)
        self.net1 = nn.HybridSequential()
        self.net1.add(net1)
        self.net1.add(nn.GlobalAvgPool2D())
        self.net2 = nn.HybridSequential()
        self.net2.add(net2)
        self.net2.add(nn.GlobalAvgPool2D())
    def hybrid_forward(self,F,x1,x2):
        return F.concat(*[self.net1(x1),self.net2(x2)])

#将特征层和输出层合并为一个网络
class  OneNet(nn.HybridBlock):
    def __init__(self,features,output,**kwargs):
        super(OneNet,self).__init__(**kwargs)
        self.features = features
        self.output = output
    def hybrid_forward(self,F,x1,x2):
        return self.output(self.features(x1,x2))


#通过使用多个预训练模型来合并为一个网络
class Net():
    def __init__(self,ctx,num_class,nameparams=None):
        inception = vision.inception_v3(pretrained=True,ctx=ctx).features  #加载预训练模型
        resnet = vision.resnet152_v1(pretrained=True,ctx=ctx).features     #加载预训练模型
        self.features = ConcatNet(resnet,inception)		#融合2个网络并得到网络特征
        self.output = self.__get_output(ctx,num_class,nameparams) #得到网络输出层
        self.net = OneNet(self.features,self.output)    #将特征层和输出层合并为一个网络
    #构造网络输出层
    def __get_output(self,ctx,num_class,ParamsName=None):
        net = nn.HybridSequential("output")
        with net.name_scope():
            net.add(nn.Dense(256,activation='relu'))
            net.add(nn.Dropout(.5))
            net.add(nn.Dense(num_class))  #改为由外部传入
        if ParamsName is not None:
            net.collect_params().load(ParamsName,ctx)
        else:
            net.initialize(init = init.Xavier(),ctx=ctx)
        return net