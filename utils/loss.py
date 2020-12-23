from torch import nn


class CrossEntropyLossGoogLeNet(nn.CrossEntropyLoss):
    """
    Weighted average of cross entropy on logits and auxilary logits
    """
    def forward(self, input, target) -> Tensor:
        if type(input).__name__ == 'GoogLeNetOutputs':
            logits, aux_logits2, aux_logits1 = input  # takes GoogLeNetOutputs with aux_logits
            loss1 = super(CrossEntropyLossGoogLeNet, self).forward(logits, target)
            loss2 = super(CrossEntropyLossGoogLeNet, self).forward(aux_logits2, target)
            loss3 = super(CrossEntropyLossGoogLeNet, self).forward(aux_logits1, target)
            return loss1 + 0.3 * (loss2 + loss3)
        else:
            return super(CrossEntropyLossGoogLeNet, self).forward(input, target)
