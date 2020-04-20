import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator as tg
import os
import math
import argparse
import scipy as sp
import scipy.stats

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 5)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 1)#每一类支撑集数量
parser.add_argument("-b","--batch_num_per_class",type = int, default = 15)#每一类测试集数量
parser.add_argument("-e","--episode",type = int, default= 500000)
parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
args = parser.parse_args()


# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = 1
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit
a = 0.5
m = 0.5

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0),-1)
        return out # 64
    def myloss(self, sample_features, batch_features, batch_labels):
        z = torch.cat([sample_features, batch_features], 0) #25x64*5*5
        z_dim = z.size(-1)
        z_proto = z[:CLASS_NUM*SAMPLE_NUM_PER_CLASS].view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, z_dim).mean(1) #way to proto
        zq = z[CLASS_NUM*SAMPLE_NUM_PER_CLASS:]

        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1).view(CLASS_NUM, BATCH_NUM_PER_CLASS, -1)

        target_inds = torch.arange(0, CLASS_NUM).view(CLASS_NUM, 1, 1).expand(CLASS_NUM, BATCH_NUM_PER_CLASS, 1).long() #goal class
        target_inds = Variable(target_inds, requires_grad=False)

        batch_true = torch.from_numpy(np.array(batch_labels))
        target = batch_true.view(CLASS_NUM, BATCH_NUM_PER_CLASS, -1)


        loss_val = -log_p_y.gather(2, target).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target.squeeze(2)).float().mean()

        # Aug-margin Loss
        l2_dists = torch.norm(dists, dim = 1)
        # print(dists)
        l2_norm = l2_dists.unsqueeze(1).expand(CLASS_NUM*BATCH_NUM_PER_CLASS, 5)
        # print(l2_norm)
        # print(l2_norm.size())
        l2_dists = dists/l2_norm
        # print("l2", l2_dists)
        # print("batch_la", batch_labels.size())
        label = batch_labels.unsqueeze(1)
        # print(label)
        label_one_hot = torch.zeros(CLASS_NUM*BATCH_NUM_PER_CLASS, CLASS_NUM).scatter_(1,label,1)
        # print("onehot", label_one_hot)
        one = torch.ones(CLASS_NUM*BATCH_NUM_PER_CLASS , CLASS_NUM)
        torch_m = m * torch.ones(CLASS_NUM*BATCH_NUM_PER_CLASS , CLASS_NUM)
        # print("m",torch_m)
        # print(label_one_hot)
        Loss_pull = a * label_one_hot * l2_dists.pow(2)
        com = torch_m - l2_dists 
        zero = torch.zeros_like(com)
        max = torch.where(com < 0, zero, com)
        Loss_push = (1-a) * (one - label_one_hot) * max.pow(2)

        Loss_margin = (torch.sum(Loss_pull) + torch.sum(Loss_push))/CLASS_NUM/CLASS_NUM*BATCH_NUM_PER_CLASS
        
        loss_val = loss_val + Loss_margin

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(128,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size*3*3,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        #out = F.sigmoid(self.fc2(out))
        out = self.fc2(out).sigmoid()
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())



def main():
    # Step 1: init data folders
    print("init data folders!")
    # init character folders for dataset construction
    metatrain_folders,metatest_folders = tg.mini_imagenet_folders()

    # Step 2: init neural networks
    print("init neural networks!")

    feature_encoder = CNNEncoder()
    

    feature_encoder.apply(weights_init)
    

    #feature_encoder.cuda(GPU)
   

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=100000,gamma=0.5)
   

    # if os.path.exists(str("./models/newminiimagenet_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
    #     feature_encoder.load_state_dict(torch.load(str("./models/newminiimagenet_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"), map_location='cpu'))
    #     print("load feature encoder success, feature encoder:", str("./models/newminiimagenet_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
    

    # Step 3: build graph
    print("Training...")

    last_accuracy = 0.0

    for episode in range(EPISODE):

        

        # init dataset
        # sample_dataloader is to obtain previous samples for compare
        # batch_dataloader is to batch samples for training
        task = tg.MiniImagenetTask(metatrain_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
        sample_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
        batch_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True)

        # sample datas
        samples,sample_labels = sample_dataloader.__iter__().next()
        batches,batch_labels = batch_dataloader.__iter__().next()

        # calculate features
        sample_features = feature_encoder(Variable(samples))      #.cuda(GPU)) # 5x64*5*5
        batch_features = feature_encoder(Variable(batches))       #.cuda(GPU)) # 20x64*5*5

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a 100x128 matrix for relation network
        # sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        # batch_features_ext = batch_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        # batch_features_ext = torch.transpose(batch_features_ext,0,1)
        # relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,FEATURE_DIM*2,19,19)
        # relations = relation_network(relation_pairs).view(-1,CLASS_NUM*SAMPLE_NUM_PER_CLASS)

        # mse = nn.MSELoss()   #.cuda(GPU)
        # one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1), 1))   #.cuda(GPU)
        # loss = mse(relations,one_hot_labels)


        # training

        feature_encoder.zero_grad()
       

        myloss = feature_encoder.myloss(sample_features, batch_features, batch_labels) 
        loss = myloss[0]

        loss.backward()

        torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(),0.5)
        
        feature_encoder_optim.step()
        
        feature_encoder_scheduler.step(episode)


        if (episode+1)%100 == 0:
                print("episode:",episode+1,"loss",loss.item())

        if (episode+1)%5000 == 0:

            # test
            print("Testing...")
            accuracies = []
            for i in range(TEST_EPISODE):
                total_acc = 0
                counter = 0
                task = tg.MiniImagenetTask(metatest_folders,CLASS_NUM,1,15)
                sample_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=1,split="train",shuffle=False)

                num_per_class = 3
                test_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=num_per_class,split="test",shuffle=True)
                sample_images,sample_labels = sample_dataloader.__iter__().next()
                for test_images,test_labels in test_dataloader:
                    batch_size = test_labels.shape[0]
                    # calculate features
                    sample_features = feature_encoder(Variable(sample_images))    #.cuda(GPU)) # 5x64
                    test_features = feature_encoder(Variable(test_images))       #.cuda(GPU)) # 20x64

                    # calculate relations
                    # each batch sample link to every samples to calculate relations
                    # to form a 100x128 matrix for relation network
                    # sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size,1,1,1,1)
                    # test_features_ext = test_features.unsqueeze(0).repeat(1*CLASS_NUM,1,1,1,1)
                    # test_features_ext = torch.transpose(test_features_ext,0,1)
                    # relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,19,19)
                    # relations = relation_network(relation_pairs).view(-1,CLASS_NUM)

                    # _,predict_labels = torch.max(relations.data,1)

                    # rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(batch_size)]

                    # total_rewards += np.sum(rewards)
                    counter += batch_size
                    test = torch.cat([sample_features, test_features], 0)
                    test_dim = test.size(-1)
                    test_proto = test[:CLASS_NUM].view(CLASS_NUM, 1, test_dim).mean(1) 
                    testq = test[CLASS_NUM:]

                    dists = euclidean_dist(testq, test_proto)

                    log_p_y = F.log_softmax(-dists, dim=1).view(CLASS_NUM, num_per_class, -1)

                    # target_inds = torch.arange(0, CLASS_NUM).view(CLASS_NUM, 1, 1).expand(CLASS_NUM, SAMPLE_NUM_PER_CLASS, 1).long() #
                    # target_inds = Variable(target_inds, requires_grad=False)

                    _, testy_hat = log_p_y.max(2)

                    test_target = test_labels.view(CLASS_NUM, -1).unsqueeze(0)
                
                    ac = torch.eq(testy_hat, test_target.squeeze(0)).float()
                

                    acc = torch.sum(ac).item() 
                    #print(acc)
                    total_acc += acc

                    accuracy = total_acc/1.0/counter
                    accuracies.append(accuracy)

            test_accuracy,h = mean_confidence_interval(accuracies)

            print("test accuracy:",test_accuracy,"h:",h)

            if test_accuracy > last_accuracy:

                # save networks
                torch.save(feature_encoder.state_dict(),str("./models/newminiimagenet_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                

                print("save networks for episode:",episode + 1)

                last_accuracy = test_accuracy





if __name__ == '__main__':
    main()
