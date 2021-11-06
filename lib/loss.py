import torch
import torch.nn as nn
from torch.autograd import Variable


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = get_sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        cost_s_max = cost_s.max(1)[0]
        cost_im_max = cost_im.max(0)[0]
        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum(), cost_s_max.detach(), cost_im_max.detach()

    def forward2(self, img, cap, img_neg_ind, cap_neg_ind):
        img_neg = img[img_neg_ind]
        cap_neg = cap[cap_neg_ind]
        pos = (img * cap).sum(1)
        i2t_neg = (img * cap_neg).sum(1)
        t2i_neg = (cap * img_neg).sum(1)
        cost_s = (self.margin + i2t_neg - pos).clamp(min=0).detach()
        cost_im = (self.margin + t2i_neg - pos).clamp(min=0).detach()
        return cost_s, cost_im


def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities

