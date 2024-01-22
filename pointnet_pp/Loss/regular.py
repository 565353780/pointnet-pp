import torch
import torch.utils.data
import torch.nn.parallel


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    i = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        i = i.cuda()
    loss = torch.mean(
        torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - i, dim=(1, 2))
    )
    return loss
