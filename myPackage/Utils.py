import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F

################### CODE FOR THE BETA MODEL  ########################

def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)


def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) /x_bar
    return alpha, beta


class BetaMixture1D(object):
    def __init__(self, max_iters=10,
                 alphas_init=(1, 2),
                 betas_init=(2, 1),
                 weights_init=(0.5, 0.5)):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 10000
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r =  np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unsable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):

            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()
        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t

    def look_lookup(self, x):
        #x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]


class DNoiseLoss(nn.Module):

    def __init__(self, class_weight=None, reduction = "mean"):
        """
        :param class_weight: It is a list.
        :param reduction: "mean" or "sum"
        """
        super().__init__()
        self.crossEn = nn.CrossEntropyLoss(reduction="none")
        self.re = reduction
        self.class_weight = list(class_weight)

    @staticmethod
    def fit_bmmModel(lossesEpoch):
        minLoss = np.min(lossesEpoch)
        maxLoss = np.max(lossesEpoch)
        epoch_norm_losses = (lossesEpoch - minLoss) / (maxLoss - minLoss + 1e-4)
        epoch_norm_losses[epoch_norm_losses >= 1] = 1 - 10e-3
        epoch_norm_losses[epoch_norm_losses <= 0] = 10e-3
        bmmModel = BetaMixture1D(max_iters=100)
        bmmModel.fit(epoch_norm_losses)
        bmmModel.create_lookup(1)
        return bmmModel

    @staticmethod
    def compute_probabilities_batch(lossesBatch, bmmModel):
        minLoss = np.min(lossesBatch)
        maxLoss = np.max(lossesBatch)
        batch_norm_losses = (lossesBatch - minLoss) / (maxLoss - minLoss + 1e-4)
        batch_norm_losses[batch_norm_losses >= 1] = 1 - 10e-7
        batch_norm_losses[batch_norm_losses <= 0] = 10e-7
        # print("Loss {}".format(batch_norm_losses))
        weights = bmmModel.look_lookup(batch_norm_losses)
        weights[weights <= 0.1] = 0.1
        weights[weights >= 0.9] = 0.9
        # print("Weig {}".format(weights))
        return weights

    def forward(self,net_output, targets, bmmModel, epoch):
        device = net_output.device
        tensorType = net_output.dtype
        n, c = net_output.shape[0], net_output.shape[1]
        with torch.no_grad():
            if self.class_weight is None:
                class_weight = torch.ones(size=[1, c]).type(tensorType).to(device)
            else:
                class_weight = torch.tensor(self.class_weight).type(tensorType).to(device).view([1, c])
            if epoch == 1:
                batch_weights = 0.5 * torch.ones(size=[n]).to(device=device).view([-1, 1]).type(tensorType)
            else:
                entropyLoss = self.crossEn(net_output, targets).detach_().cpu().numpy()
                batch_weights = self.compute_probabilities_batch(entropyLoss, bmmModel=bmmModel)
                batch_weights = torch.from_numpy(batch_weights).to(device=device).view([-1, 1]).type(tensorType)
            # batch_weights = torch.zeros(size=[n]).view([-1,1]).to(device) # this is used to test if this loss is correct or not.
            maxIndices = torch.argmax(net_output, dim=-1).view(size=[-1, 1])
            temp = torch.zeros(size=[n, c]).type(tensorType).to(device)
            z = torch.scatter(temp, dim=1, index=maxIndices, value=1.)
            targets = targets.view(size=[-1, 1])
            y = torch.scatter(temp, dim=1, index=targets, value=1.)
            weightTarget = (1. - batch_weights) * y + batch_weights * z
        batchLoss = torch.mul(weightTarget, torch.log_softmax(net_output, dim=-1)) * class_weight
        batchLoss = torch.neg(torch.sum(batchLoss,dim=-1))
        if self.re == "mean":
            sumLoss = torch.sum(batchLoss)
            sumWeight = torch.sum(y * class_weight)
            return torch.div(sumLoss, sumWeight)
        else:
            return torch.sum(batchLoss)



if __name__ == "__main__":
    epoch_losses = np.random.rand(70000)
    testLoss = DNoiseLoss(class_weight=[0.1, 0.5, 0.4],reduction="sum")
    bmmModel = testLoss.fit_bmmModel(epoch_losses)

    testOutput = torch.rand(size=[5,3]).float().to("cuda")
    testTarget = torch.tensor([0, 1, 2, 2, 1]).long().to("cuda")
    c_weight = torch.tensor([0.1, 0.5, 0.4]).float().to("cuda")
    entropyLoss = F.cross_entropy(testOutput, testTarget, weight= c_weight, reduction="sum")
    print(entropyLoss)
    outLoss = testLoss(testOutput, testTarget, bmmModel, 2)
    print(outLoss)

