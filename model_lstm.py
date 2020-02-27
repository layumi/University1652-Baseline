import argparse
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
        
def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        init.orthogonal_(m.weight)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

def fix_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x


# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        if pool =='avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft
            #self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool=='avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            self.model = model_ft
            #self.classifier = ClassBlock(2048, class_num, droprate)
        elif pool=='max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft

        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
            x = x.view(x.size(0), x.size(1))
        #x = self.classifier(x)
        return x
    
class two_view_net_seq(nn.Module):
    def __init__(self, class_num, droprate, stride = 2, pool = 'avg', share_weight = False):
        super(two_view_net_seq, self).__init__()
        self.model_1 =  ft_net(class_num, stride=stride, pool = pool)
        if share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 =  ft_net(class_num, stride = stride, pool = pool)
        self.classifier = ClassBlock(2048, class_num, droprate)
        if pool =='avg+max':
            self.classifier = ClassBlock(4096, class_num, droprate)

    def forward(self, x1, x2):
        # model_1 satellite
        # model_2 drone
        if x1 is None:
            y1 = None
        else:
            x1 = self.model_1(x1)
            y1 = self.classifier(x1)

        if x2 is None:
            y2 = None
        else:
            n = x2.size(0)
            cnn_embed_seq = torch.FloatTensor(n,2048).zero_().cuda()
            for t in range(x2.size(1)):
                # ResNet CNN
                x = self.model_2(x2[:, t, :, :, :])  # ResNet
                x = x.view(x.size(0), -1)             # flatten output of conv
        
                cnn_embed_seq += x
            cnn_embed_seq.div(x2.size(1))
#             print(cnn_embed_seq)
            
            x2 = cnn_embed_seq
#             x2 = self.model_2(x2[:, 0, :, :, :])
            y2 = self.classifier(x2)
        return y1, y2

class two_view_net_lstm(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride = 2, pool = 'avg', share_weight = False, fc_hidden1=512, fc_hidden2=512, drop_p=0.5, CNN_embed_dim=512):
        super(two_view_net_lstm, self).__init__()
        self.model_1 =  ft_net(class_num, stride=stride, pool = pool)
        if share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 =  ft_net(class_num, stride = stride, pool = pool)
        self.classifier = ClassBlock(2048, class_num, droprate)
        if pool =='avg+max':
            self.classifier = ClassBlock(4096, class_num, droprate)
        lstm = nn.LSTM(
            input_size = 512,
            hidden_size = 2048,        
            num_layers = 3,       
            batch_first = True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        
#         for name, param in lstm.named_parameters():
#             if 'bias' in name:
#                 init.constant_(param, 0.0)
#             elif 'weight' in name:
#                 init.xavier_normal_(param)
                
        lstm.apply(weights_init_kaiming)
        self.LSTM = lstm
        
        
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        
        lstm_cb = []
        lstm_cb += [nn.Linear(2048, fc_hidden1)]
        lstm_cb += [nn.BatchNorm1d(fc_hidden1, momentum=0.01)]
        lstm_cb += [nn.LeakyReLU(0.1)]
        lstm_cb += [nn.Linear(fc_hidden1, fc_hidden2)]
        lstm_cb += [nn.BatchNorm1d(fc_hidden2, momentum=0.01)]
        lstm_cb += [nn.LeakyReLU(0.1)]
        lstm_cb += [nn.Dropout(p=drop_p)]
        lstm_cb += [nn.Linear(fc_hidden2, CNN_embed_dim)]
        lstm_cb = nn.Sequential(*lstm_cb)
#         lstm_cb.apply(weights_init_orthogonal)
        lstm_cb.apply(weights_init_kaiming)
        
        self.lstm_cb = lstm_cb

    def forward(self, x1, x2):
        # model_1 satellite
        # model_2 drone
        if x1 is None:
            y1 = None
        else:
            x1 = self.model_1(x1)
            y1 = self.classifier(x1)

        if x2 is None:
            y2 = None
        else:
            cnn_embed_seq = []
            for t in range(x2.size(1)):
                # ResNet CNN
#                 with torch.no_grad():
                x = self.model_2(x2[:, t, :, :, :])  # ResNet
                x = x.view(x.size(0), -1)             # flatten output of conv
                
                # FC layers
#                 x = self.bn1(self.fc1(x))
#                 x = F.relu(x)
#                 x = self.bn2(self.fc2(x))
#                 x = F.relu(x)
#                 x = F.dropout(x, p=self.drop_p, training=self.training)
#                 x = self.fc3(x)
                x = self.lstm_cb(x)
        
                cnn_embed_seq.append(x)

            # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
            cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
            # cnn_embed_seq: shape=(batch, time_step, input_size)
        
#             x2 = self.model_2(x2)
            self.LSTM.flatten_parameters()
#             x2 = x2.unsqueeze(0)
            RNN_out, (h_n, h_c) = self.LSTM(cnn_embed_seq, None)  
            x2 = RNN_out[:, -1, :]
            y2 = self.classifier(x2)
        return y1, y2
    
class two_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride = 2, pool = 'avg', share_weight = False):
        super(two_view_net, self).__init__()
        self.model_1 =  ft_net(class_num, stride=stride, pool = pool)
        if share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 =  ft_net(class_num, stride = stride, pool = pool)
        self.classifier = ClassBlock(2048, class_num, droprate)
        if pool =='avg+max':
            self.classifier = ClassBlock(4096, class_num, droprate)

    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
        else:
            x1 = self.model_1(x1)
            y1 = self.classifier(x1)

        if x2 is None:
            y2 = None
        else:
            x2 = self.model_2(x2)
            y2 = self.classifier(x2)
        return y1, y2


class three_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride = 2, pool = 'avg', share_weight = False):
        super(three_view_net, self).__init__()
        self.model_1 =  ft_net(class_num, stride = stride, pool = pool)
        self.model_2 =  ft_net(class_num, stride = stride, pool = pool)
        if share_weight:
            self.model_3 = self.model_1
        else:
            self.model_3 =  ft_net(class_num, stride = stride, pool = pool)
        self.classifier = ClassBlock(2048, class_num, droprate)
        if pool =='avg+max':
            self.classifier = ClassBlock(4096, class_num, droprate)

    def forward(self, x1, x2, x3, x4 = None): # x4 is extra data
        if x1 is None:
            y1 = None
        else:
            x1 = self.model_1(x1)
            y1 = self.classifier(x1)

        if x2 is None:
            y2 = None
        else:
            x2 = self.model_2(x2)
            y2 = self.classifier(x2)

        if x3 is None:
            y3 = None
        else:
            x3 = self.model_3(x3)
            y3 = self.classifier(x3)

        if x4 is None:
            return y1, y2, y3
        else:
            x4 = self.model_2(x4)
            y4 = self.classifier(x4)
            return y1, y2, y3, y4


if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    net = two_view_net_lstm(751, stride=1)
    net.classifier.classifier = nn.Sequential()
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 256, 256))
    input2 = Variable(torch.FloatTensor(8, 5, 3, 256, 256))
    output, output2 = net(input, input2)
    print('net output size:')
    print(output.shape)
    print(output2.shape)
