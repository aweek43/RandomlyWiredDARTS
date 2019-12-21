import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
from utils import drop_path


class Cell(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    print(C_prev_prev, C_prev, C)

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
    
    if reduction:
      op_names, indices = zip(*genotype.reduce)
      concat = genotype.reduce_concat
    else:
      op_names, indices = zip(*genotype.normal)
      concat = genotype.normal_concat
    self._compile(C, op_names, indices, concat, reduction)

  def _compile(self, C, op_names, indices, concat, reduction):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 2 if reduction and index < 2 else 1
      op = OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, s0, s1, drop_prob):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


class AuxiliaryHeadImageNet(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 14x14"""
    super(AuxiliaryHeadImageNet, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
      # Commenting it out for consistency with the experiments in the paper.
      # nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x

'''
class NetworkCIFAR(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype):
    super(NetworkCIFAR, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary

    stem_multiplier = 3
    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
    
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
      if i == 2*layers//3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2*self._layers//3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits, logits_aux
'''


class NetworkCIFAR(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype):
    super(NetworkCIFAR, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary

    stem_multiplier = 3
    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )

    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
      if i == 2*layers//3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self.SC_0_9 = nn.Sequential(
            nn.Conv2d(144, 288, 1, 1, padding=0),
            nn.BatchNorm2d(288),
            nn.MaxPool2d(2)
            )
    self.SC_0_17 = nn.Sequential(
            nn.Conv2d(144, 576, 1, 1, padding=0),
            nn.BatchNorm2d(576),
            nn.MaxPool2d(4)
            )
    self.SC_1_11 = nn.Sequential(
            nn.Conv2d(144, 288, 1, 1, padding=0),
            nn.BatchNorm2d(288),
            nn.MaxPool2d(2)
            )
    self.SC_1_16 = nn.Sequential(
            nn.Conv2d(144, 576, 1, 1, padding=0),
            nn.BatchNorm2d(576),
            nn.MaxPool2d(4)
            )
    self.SC_1_18 = nn.Sequential(
            nn.Conv2d(144, 576, 1, 1, padding=0),
            nn.BatchNorm2d(576),
            nn.MaxPool2d(4)
            )
    self.SC_1_19 = nn.Sequential(
            nn.Conv2d(144, 576, 1, 1, padding=0),
            nn.BatchNorm2d(576),
            nn.MaxPool2d(4)
            )
    self.SC_2_5 = nn.Sequential(
            nn.Conv2d(144, 144, 1, 1, padding=0),
            nn.BatchNorm2d(144)
            )
    self.SC_2_6 = nn.Sequential(
            nn.Conv2d(144, 144, 1, 1, padding=0),
            nn.BatchNorm2d(144)
            )
    self.SC_2_14 = nn.Sequential(
            nn.Conv2d(144, 576, 1, 1, padding=0),
            nn.BatchNorm2d(576),
            nn.MaxPool2d(4)
            )
    self.SC_2_18 = nn.Sequential(
            nn.Conv2d(144, 576, 1, 1, padding=0),
            nn.BatchNorm2d(576),
            nn.MaxPool2d(4)
            )
    self.SC_3_11 = nn.Sequential(
            nn.Conv2d(144, 288, 1, 1, padding=0),
            nn.BatchNorm2d(288),
            nn.MaxPool2d(2)
            )
    self.SC_4_7 = nn.Sequential(
            nn.Conv2d(144, 288, 1, 1, padding=0),
            nn.BatchNorm2d(288),
            nn.MaxPool2d(2)
            )
    self.SC_4_9 = nn.Sequential(
            nn.Conv2d(144, 288, 1, 1, padding=0),
            nn.BatchNorm2d(288),
            nn.MaxPool2d(2)
            )
    self.SC_4_12 = nn.Sequential(
            nn.Conv2d(144, 288, 1, 1, padding=0),
            nn.BatchNorm2d(288),
            nn.MaxPool2d(2)
            )
    self.SC_5_10 = nn.Sequential(
            nn.Conv2d(144, 288, 1, 1, padding=0),
            nn.BatchNorm2d(288),
            nn.MaxPool2d(2)
            )
    self.SC_5_11 = nn.Sequential(
            nn.Conv2d(144, 288, 1, 1, padding=0),
            nn.BatchNorm2d(288),
            nn.MaxPool2d(2)
            )
    self.SC_5_13 = nn.Sequential(
            nn.Conv2d(144, 288, 1, 1, padding=0),
            nn.BatchNorm2d(288),
            nn.MaxPool2d(2)
            )
    self.SC_5_17 = nn.Sequential(
            nn.Conv2d(144, 576, 1, 1, padding=0),
            nn.BatchNorm2d(576),
            nn.MaxPool2d(4)
            )
    self.SC_6_10 = nn.Sequential(
            nn.Conv2d(288, 288, 1, 1, padding=0),
            nn.BatchNorm2d(288)
            )
    self.SC_7_19 = nn.Sequential(
            nn.Conv2d(288, 576, 1, 1, padding=0),
            nn.BatchNorm2d(576),
            nn.MaxPool2d(2)
            )
    self.SC_8_13 = nn.Sequential(
            nn.Conv2d(288, 288, 1, 1, padding=0),
            nn.BatchNorm2d(288)
            )
    self.SC_8_17 = nn.Sequential(
            nn.Conv2d(288, 576, 1, 1, padding=0),
            nn.BatchNorm2d(576),
            nn.MaxPool2d(2)
            )
    self.SC_9_12 = nn.Sequential(
            nn.Conv2d(288, 288, 1, 1, padding=0),
            nn.BatchNorm2d(288)
            )
    self.SC_9_19 = nn.Sequential(
            nn.Conv2d(288, 576, 1, 1, padding=0),
            nn.BatchNorm2d(576),
            nn.MaxPool2d(2)
            )
    self.SC_10_16 = nn.Sequential(
            nn.Conv2d(288, 576, 1, 1, padding=0),
            nn.BatchNorm2d(576),
            nn.MaxPool2d(2)
            )
    self.SC_10_19 = nn.Sequential(
            nn.Conv2d(288, 576, 1, 1, padding=0),
            nn.BatchNorm2d(576),
            nn.MaxPool2d(2)
            )
    self.SC_11_18 = nn.Sequential(
            nn.Conv2d(288, 576, 1, 1, padding=0),
            nn.BatchNorm2d(576),
            nn.MaxPool2d(2)
            )
    self.SC_15_19 = nn.Sequential(
            nn.Conv2d(576, 576, 1, 1, padding=0),
            nn.BatchNorm2d(576)
            )
    

  def forward(self, input):
    logits_aux = None
    s0 = s1 = self.stem(input)
    # print("s0:",s0.shape)

    # for i, cell in enumerate(self.cells):
    #  s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
    #   if i == 2*self._layers//3:
    #    if self._auxiliary and self.training:
    #       logits_aux = self.auxiliary_head(s1)


    c0 = self.cells[0](s0, s1, self.drop_path_prob)
    if 0 == 2*self._layers//3:
      if self._auxiliary and self.training:
        logits_aux = self.auxiliary_head(c0)

    c1 = self.cells[1](s1, c0, self.drop_path_prob)
    if 1 == 2*self._layers//3:
      if self._auxiliary and self.training:
        logits_aux = self.auxiliary_head(c1)

    c2 = self.cells[2](c0, c1, self.drop_path_prob)
    if 2 == 2*self._layers//3:
      if self._auxiliary and self.training:
        logits_aux = self.auxiliary_head(c2)

    c3 = self.cells[3](c1, c2, self.drop_path_prob)
    if 3 == 2*self._layers//3:
      if self._auxiliary and self.training:
        logits_aux = self.auxiliary_head(c3)

    c4 = self.cells[4](c2, c3, self.drop_path_prob)
    if 4 == 2*self._layers//3:
      if self._auxiliary and self.training:
        logits_aux = self.auxiliary_head(c4)

    c4 += self.SC_2_5(c2)
    c5 = self.cells[5](c3, c4, self.drop_path_prob)
    if 5 == 2*self._layers//3:
      if self._auxiliary and self.training:
        logits_aux = self.auxiliary_head(c5)

    c5 += self.SC_2_6(c2)
    c6 = self.cells[6](c4, c5, self.drop_path_prob)
    if 6 == 2*self._layers//3:
      if self._auxiliary and self.training:
        logits_aux = self.auxiliary_head(c6)

    c6 += self.SC_4_7(c4)
    c7 = self.cells[7](c5, c6, self.drop_path_prob)
    if 7 == 2*self._layers//3:
      if self._auxiliary and self.training:
        logits_aux = self.auxiliary_head(c7)

    c8 = self.cells[8](c6, c7, self.drop_path_prob)
    if 8 == 2*self._layers//3:
      if self._auxiliary and self.training:
        logits_aux = self.auxiliary_head(c8)

    c8 += self.SC_0_9(c0) + self.SC_4_9(c4)
    c9 = self.cells[9](c7, c8, self.drop_path_prob)
    if 9 == 2*self._layers//3:
      if self._auxiliary and self.training:
        logits_aux = self.auxiliary_head(c9)

    c9 += self.SC_5_10(c5) + self.SC_6_10(c6)
    c10 = self.cells[10](c8, c9, self.drop_path_prob)
    if 10 == 2*self._layers//3:
      if self._auxiliary and self.training:
        logits_aux = self.auxiliary_head(c10)

    c10 += self.SC_1_11(c1) + self.SC_3_11(c3) + self.SC_5_11(c5)
    c11 = self.cells[11](c9, c10, self.drop_path_prob)
    if 11 == 2*self._layers//3:
      if self._auxiliary and self.training:
        logits_aux = self.auxiliary_head(c11)

    c11 += self.SC_4_12(c4) + self.SC_9_12(c9)
    c12 = self.cells[12](c10, c11, self.drop_path_prob)
    if 12 == 2*self._layers//3:
      if self._auxiliary and self.training:
        logits_aux = self.auxiliary_head(c12)

    c12 += self.SC_5_13(c5) + self.SC_8_13(c8)
    c13 = self.cells[13](c11, c12, self.drop_path_prob)
    if 13 == 2*self._layers//3:
      if self._auxiliary and self.training:
        logits_aux = self.auxiliary_head(c13)

    c13 += self.SC_2_14(c2)
    c14 = self.cells[14](c12, c13, self.drop_path_prob)
    if 14 == 2*self._layers//3:
      if self._auxiliary and self.training:
        logits_aux = self.auxiliary_head(c14)

    c15 = self.cells[15](c13, c14, self.drop_path_prob)
    if 15 == 2*self._layers//3:
      if self._auxiliary and self.training:
        logits_aux = self.auxiliary_head(c15)

    c15 += self.SC_1_16(c1) + self.SC_10_16(c10)
    c16 = self.cells[16](c14, c15, self.drop_path_prob)
    if 16 == 2*self._layers//3:
      if self._auxiliary and self.training:
        logits_aux = self.auxiliary_head(c16)

    c16 += self.SC_0_17(c0) + self.SC_5_17(c5) + self.SC_8_17(c8)
    c17 = self.cells[17](c15, c16, self.drop_path_prob)
    if 17 == 2*self._layers//3:
      if self._auxiliary and self.training:
        logits_aux = self.auxiliary_head(c17)

    c17 += self.SC_1_18(c1) + self.SC_2_18(c2) + self.SC_11_18(c11)
    c18 = self.cells[18](c16, c17, self.drop_path_prob)
    if 18 == 2*self._layers//3:
      if self._auxiliary and self.training:
        logits_aux = self.auxiliary_head(c18)

    c18 += self.SC_1_19(c1) + self.SC_7_19(c7) + self.SC_9_19(c9) + self.SC_10_19(c10) + self.SC_15_19(c15)
    c19 = self.cells[19](c17, c18, self.drop_path_prob)
    if 19 == 2*self._layers//3:
      if self._auxiliary and self.training:
        logits_aux = self.auxiliary_head(c19)



    out = self.global_pooling(c19)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits, logits_aux


class NetworkImageNet(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype):
    super(NetworkImageNet, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary

    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    C_prev_prev, C_prev, C_curr = C, C, C

    self.cells = nn.ModuleList()
    reduction_prev = True
    for i in range(layers):
      if i in [layers // 3, 2 * layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
      if i == 2 * layers // 3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AvgPool2d(7)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2 * self._layers // 3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits, logits_aux

