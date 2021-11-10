import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, weight, gamma=2):
        """
            alpha_list: ratio of samples for num_class
            gamma: gamma value for pow.
        """
        super(FocalLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.alpha = weight
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
            inputs (torch.Tensor): prediction matrix (before softmax) with
                shape (batch_size, num_classes).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
                Each position contains the label index.
        """
        log_probs = self.logsoftmax(inputs)
        tmp = torch.pow(1 - self.softmax(inputs), self.gamma)

        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = targets.cuda()

        alpha_weight = torch.ones(log_probs.size()).cuda()
        for i in range(log_probs.size(1)):
            alpha_weight[:, i] = alpha_weight[:, i] / self.alpha[i]
        #alpha_weight = 1 / np.array(self.alpha)

        return (- tmp * targets * log_probs * alpha_weight).mean(0).sum()


class FocalLoss_Ori(nn.Module):
    """
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """
    def __init__(self, alpha=0.5, gamma=2, balance_index=-1, num_class = 6, size_average=True):
        super(FocalLoss_Ori, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.eps = 1e-6

        if isinstance(self.alpha, (list, tuple)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.tensor(list(self.alpha))
        elif isinstance(self.alpha, (float, int)):
            assert 0 < self.alpha <= 1.0, 'alpha should be in `(0,1)`)'
            assert balance_index > -1
            alpha = torch.ones(self.num_class)
            # alpha *= 1-self.alpha
            # alpha[balance_index] = self.alpha
            self.alpha = alpha
        elif isinstance(self.alpha, torch.Tensor):
            self.alpha = self.alpha
        else:
            raise TypeError('Not support alpha type, expect `int|float|list|tuple|torch.Tensor`')

    def forward(self, logit, target):

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.transpose(1, 2).contiguous() # [N,C,d1*d2..] -> [N,d1*d2..,C]
            logit = logit.view(-1, logit.size(-1)) # [N,d1*d2..,C]-> [N*d1*d2..,C]
        target = target.view(-1, 1) # [N,d1,d2,...]->[N*d1*d2*...,1]

        # -----------legacy way------------
        #  idx = target.cpu().long()
        # one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        # one_hot_key = one_hot_key.scatter_(1, idx, 1)
        # if one_hot_key.device != logit.device:
        #     one_hot_key = one_hot_key.to(logit.device)
        # pt = (one_hot_key * logit).sum(1) + epsilon

        # ----------memory saving way--------
        pt = logit.gather(1, target).view(-1) + self.eps # avoid apply
        logpt = pt.log()

        if self.alpha.device != logpt.device:
            alpha = self.alpha.to(logpt.device)
            alpha_class = alpha.gather(0,target.view(-1))
            logpt = alpha_class*logpt
        loss = -1 * torch.pow(torch.sub(1.0, pt), self.gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


if __name__ == "__main__":
    num_class = 5
    # alpha = np.random.randn(num_class)
    # input = torch.randn(10, num_class).cuda()
    # target = torch.LongTensor(10).random_(num_class).cuda()
    # loss0 = FL(input, target)
    # print(loss0)
    nodes = 100
    N = 100
    # model1d = torch.nn.Linear(nodes, num_class).cuda()
    model2d = nn.Linear(16, num_class).cuda()
    FL = FocalLoss_Ori(num_class=num_class, alpha=0.3,  gamma=2, balance_index=0)
    for i in range(10):
        # input = torch.rand(N, nodes) * torch.randint(1, 100, (N, nodes)).float()
        # input = input.cuda()
        # target = torch.LongTensor(N).random_(num_class).cuda()
        # loss0 = FL(model1d(input), target)
        # print(loss0)
        # loss0.backward()

        input = torch.rand(3, 16).cuda()
        target = torch.rand(3).random_(num_class).cuda()
        target = target.long().cuda()
        output = model2d(input)
        output = F.softmax(output, dim=1)
        loss = FL(output, target)
        print(loss.item())
