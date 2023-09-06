"""
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
"""

import argparse
import sys
import time
import torch
import warnings
import glob
import os
import random
from torch import nn
from torchvision import models
from PIL import Image
from torchvision import transforms

parser = argparse.ArgumentParser(description="ECAPA_trainer")

parser.add_argument('--max_epoch', type=int, default=80, help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--n_cpu', type=int, default=8, help='Number of loader threads')
parser.add_argument('--test_step', type=int, default=1, help='Test and save every [test_step] epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument("--lr_decay", type=float, default=0.97, help='Learning rate decay every [test_step] epochs')
parser.add_argument('--save_path', type=str, default="../output", help='Path to save the score.txt and models')
parser.add_argument('--data_path', type=str, default="../fruitveg81", help='Path to save the score.txt and models')
parser.add_argument('--initial_model', type=str, default="", help='Path of the initial_model')
parser.add_argument('--m', type=float, default=0.2, help='Loss margin in AAM softmax')
parser.add_argument('--s', type=float, default=30, help='Loss scale in AAM softmax')

# Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
os.makedirs(args.save_path, exist_ok=True)
score_save_path = os.path.join(args.save_path, 'score.txt')
model_save_path = os.path.join(args.save_path)
score_file = open(score_save_path, "a+")
model_files = glob.glob('%s/model_0*.model' % model_save_path)
model_files.sort()


class data_loader(object):
    def __init__(self, data_path, train=True):
        self.data_path = data_path
        self.data_list = []
        self.data_label = []
        labels = ['apples', 'apricots', 'avocados', 'bananas', 'beans', 'cabbage', 'carambolas', 'carrots', 'cauliflower', 'celeries', 'chanterelles', 'cherries', 'clementines', 'coconuts', 'cucumbers', 'damsons', 'eggplants', 'fennels', 'garlics', 'gingers', 'grapefruits', 'grapes', 'herbs', 'kiwanos', 'kiwis', 'kumquats', 'leeks', 'lemons', 'limes', 'mangos', 'mangosteens', 'melons', 'mushrooms', 'nectarines', 'onions', 'oranges', 'passion_fruits', 'peaches', 'pears', 'peppers', 'pineapples', 'pitayas', 'plums', 'pomegranates', 'potatoes', 'pumpkins', 'radishes', 'red_beet', 'salads', 'tamarillos', 'tomatos', 'turnips', 'zucchinis']
        dictkeys = {key: i for i, key in enumerate(labels)}
        for label in labels:
            file_label = dictkeys[label]
            file_paths = glob.glob(os.path.join("../fruitveg81", label, "*/*/*/*.jpg"))
            random.shuffle(file_paths)
            if train:
                files_len = len(file_paths) * 5 // 6
            else:
                files_len = len(file_paths) * 1 // 6
            print(label,files_len)
            for i in range(files_len):
                self.data_label.append(file_label)
                self.data_list.append(file_paths[i])
        print(len(self.data_list),len(self.data_label))

    def __getitem__(self, index):
        image = Image.open(self.data_list[index])
        image = image.resize((224, 224))
        transform = transforms.ToTensor()
        image_tensor = transform(image)
        return image_tensor, self.data_label[index]

    def __len__(self):
        return len(self.data_list)


# Define the data loader
trainloader = data_loader(data_path=args.data_path, train=True)
trainLoader = torch.utils.data.DataLoader(trainloader, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu,drop_last=True)
evalloader = data_loader(data_path=args.data_path, train=False)
evalLoader = torch.utils.data.DataLoader(evalloader, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu,drop_last=False)




class CrossEntropy(nn.Module):
    def __init__(self, **kwargs):
        super(CrossEntropy, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        
    def accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def forward(self, outputs, label=None):
        outputs = outputs.squeeze(1)
        loss = self.ce(outputs, label)
        prec1 = self.accuracy(outputs.detach(), label.detach())[0]
        return loss, prec1


model = models.resnet152(pretrained=True).cuda()
model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 53), nn.LogSoftmax(dim=1)).cuda()
speaker_loss = CrossEntropy()
optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=2e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.test_step, gamma=args.lr_decay)

if args.initial_model != "":
    model = torch.load(args.intial_model)
    epoch = 1
elif len(model_files) >= 1:
    model = torch.load(model_files[-1])
    epoch = int(os.path.splitext(os.path.basename(model_files[-1]))[0][6:]) + 1
else:
    epoch = 1

    
    
    
if __name__ == '__main__':
    while 1:
        model.train()
        scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = optim.param_groups[0]['lr']
        for num, (data, labels) in enumerate(trainLoader, start=1):
            model.zero_grad()
            data, labels = data.cuda(), labels.cuda()
            embedding = model.forward(data)
            nloss, prec = speaker_loss(embedding, labels)
            nloss.backward()
            optim.step()
            index += len(labels)
            top1 += prec
            loss += nloss.detach().cpu().numpy()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") +
                             " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100 * (num / trainLoader.__len__())) +
                             " Loss: %.5f, ACC: %2.2f%% \r" % (loss / num, top1 / index * len(labels)))
            sys.stderr.flush()
        sys.stdout.write("\n")
        loss, lr, train_acc = loss / num, lr, top1 / index * len(labels)

        if epoch % args.test_step == 0:
            torch.save(model, os.path.join(model_save_path, "model_%03d.model" % epoch))
            model.eval()  # 这句话也是必须的
            with torch.no_grad():
                total_correct = 0
                total_num = 0
                for data, labels in evalLoader:
                    data, labels = data.cuda(), labels.cuda()
                    embedding = model.forward(data)
                    nloss, prec = speaker_loss(embedding, labels)
                    total_correct += data.size(0) * prec
                    total_num += data.size(0)
                eval_acc = total_correct / total_num

            print(time.strftime("%Y-%m-%d %H:%M:%S"),
                  "%d epoch, train_acc %2.2f%%, eval_acc %2.2f%%" % (epoch, train_acc, eval_acc))
            score_file.write("%d epoch, lr %f, loss %f, train_acc %2.2f%%, eval_acc %2.2f%%\n" % (
                epoch, lr, loss, train_acc, eval_acc))
            score_file.flush()

        if epoch >= args.max_epoch:
            quit()

        epoch += 1
