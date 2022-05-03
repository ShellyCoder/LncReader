import torch.nn as nn
import torch
import math


class Attention(nn.Module):

    def __init__(self, dk, drop_p):
        super(Attention, self).__init__()
        self.dk = math.sqrt(dk)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, Q, K, V):
        """
        :param Q: A 3d tensor with shape of [N, T_q, d_model].
        :param K: A 3d tensor with shape of [N, T_k, d_model].
        :param V: A 3d tensor with shape of [N, T_k, d_model].
        :return:
        """
        QKt = torch.matmul(Q, torch.transpose(K, dim0=-2, dim1=-1)) / self.dk  # (N, T_q, T_k)
        output = torch.softmax(QKt, dim=-1)
        output = self.dropout(output)
        output = torch.matmul(output, V)  # (N, T_q, d_v)
        return output

class Heads(nn.Module):

    def __init__(self, h, d_model, dk, drop_p):
        super(Heads, self).__init__()
        self.h = h
        self.dk = dk
        self.d_model = d_model
        self.QLinear = nn.Linear(d_model, d_model)
        self.KLinear = nn.Linear(d_model, d_model)
        self.VLinear = nn.Linear(d_model, d_model)
        self.attention = Attention(dk, drop_p)

    def forward(self, Q, K, V):
        bs = Q.size(0)
        qL = self.QLinear(Q).view([bs, -1, self.h, self.dk]).transpose(1, 2)
        kL = self.KLinear(K).view([bs, -1, self.h, self.dk]).transpose(1, 2)
        vL = self.VLinear(V).view([bs, -1, self.h, self.dk]).transpose(1, 2)
        scores = self.attention(qL, kL, vL)
        catTensor = scores.transpose(1, 2).contiguous().view([bs, -1, self.d_model])
        return catTensor

class MultiHeadAttention(nn.Module):

    def __init__(self, h=8, d_model=64 * 8, drop_p=0.1):
        super(MultiHeadAttention,self).__init__()
        assert d_model % h == 0, ValueError("d_model can not be divisible by heads number. ")
        dk = d_model // h
        self.outLinear = nn.Linear(d_model, d_model)
        self.heads = Heads(h, d_model, dk, drop_p)

    def forward(self, x):
        q = x
        k = torch.clone(x)
        v = torch.clone(x)
        head = self.heads(q, k, v)
        return self.outLinear(head)

class FeedForward(nn.Module):

    def __init__(self, d_model, drop_p=0.1):
        super(FeedForward, self).__init__()
        dff = d_model * 4
        self.linear1 = nn.Linear(d_model, dff)
        self.linear2 = nn.Linear(dff, d_model)
        self.activation = nn.GELU()
        self.ln = nn.LayerNorm(dff)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.ln(x1)
        x1 = self.activation(x1)
        x2 = self.linear2(x1)
        return self.dropout(x2)

class TransformerBlock(nn.Module):

    def __init__(self, h=8, d_model=512, drop_p=0.1):
        super(TransformerBlock, self).__init__()
        self.multiAttention = MultiHeadAttention(h, d_model)
        self.ln1 = nn.LayerNorm(d_model, eps=1e-3)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-3)
        self.feedForward = FeedForward(d_model, drop_p)

    def forward(self, x):
        ### main
        xOri = x.clone()
        attention = self.multiAttention(x)
        ln1 = self.ln1(xOri + attention)  # add and laynomralized
        ln1Ori = ln1.clone()
        fft = self.feedForward(ln1)
        ln2 = self.ln2(fft + ln1Ori)  # add and laynomralized
        return ln2

def add_n(tensorsList):
    return torch.stack(tensorsList,dim=-1).sum(dim=-1,keepdim=False)

def mean_n(tensorsList):
    return torch.stack(tensorsList,dim=-1).mean(dim=-1,keepdim=False)

class ALBERT(nn.Module):

    def __init__(self, in_features, d_model, heads, sequence_len, num_labels,
                 cross_layers=8, parallel_Transformers=8,
                 total_reuse_times=8, drop_p=0.15):
        super(ALBERT, self).__init__()
        self.cross_layers = cross_layers
        self.parallel_transformers = parallel_Transformers
        self.total_reuse_times = total_reuse_times
        self.num_labels = num_labels
        '''
            first linear layer of the Model
        '''
        self.firstLinear = nn.Sequential(*[
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model, eps=1e-3)  # layernomalized
        ])
        '''
            N = cross_layers, the number of transformer layer
        '''
        encoderLayers = {}
        for l in range(cross_layers):
            currentParallelBlocks = {}
            for p in range(parallel_Transformers):
                currentParallelBlocks["Parallel" + str(l) + str(p)] = TransformerBlock(h=heads, d_model=d_model, drop_p=drop_p)
            encoderLayers["Layer" + str(l)] = nn.ModuleDict(currentParallelBlocks)

        self.encoders = nn.ModuleDict(encoderLayers)
        self.linear = nn.Linear(sequence_len * d_model, num_labels)


    def forward(self, x):
        """
        :param x: x shape is [N, Length, embeddingDim]
        :return:
        """
        inputTensor = self.firstLinear(x)
        for _ in range(self.total_reuse_times):
            for l in range(self.cross_layers):
                tempTensors = []
                thisParallelBlocks = self.encoders["Layer" + str(l)]
                ### parallel encoders
                for p in range(self.parallel_transformers):
                    parallel_module = thisParallelBlocks["Parallel" + str(l) + str(p)]
                    outT = parallel_module(inputTensor)
                    tempTensors.append(outT)
                inputTensor = mean_n(tempTensors)
        encodedTensor = inputTensor.clone()
        _, s, h = encodedTensor.shape
        flatten = encodedTensor.view([-1, s * h])
        outputTensor = self.linear(flatten)
        return outputTensor



if __name__ == "__main__":

    # from torch.utils.tensorboard import SummaryWriter
    testInput = torch.randn(size=[5, 48, 1])
    testModel = ALBERT(in_features=1, d_model=256, sequence_len=48, heads=8, num_labels=1, total_reuse_times=1,
                       cross_layers=2, parallel_Transformers=2,
                       )
    print(testModel)

    # writer = SummaryWriter(log_dir="./run/")
    # writer.add_graph(testModel, testInput)
    # writer.close()

