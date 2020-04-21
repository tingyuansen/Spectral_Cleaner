# In [0]:
# import packages
import numpy as np

import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
from torch.autograd import Variable

from astropy.io import fits
from scipy import interpolate


#========================================================================================================
# restore data
#temp = np.load("../mock_all_spectra_no_noise_resample_prior_large.npz")
#y_tr = temp["spectra"]

# temp = np.load("../apogee_spectra_0.npz")
# y_tr = temp["spectra"]
# y_tr[y_tr < 0.3] = 1.
# y_tr[y_tr > 1.5] = 1.

#-------------------------------------------------------------------------------------------------------
# read H3 spectra
temp = np.load("../H3_spectra.npz")
h3_flag = temp["h3_flag"]
flux_spectra = temp["flux_spectra"]
err_array = temp["err_array"]
rest_wave = temp["rest_wave"]

print(h3_flag.shape)
print(flux_spectra.shape)
print(err_array.shape)

#-------------------------------------------------------------------------------------------------------
## SNR cut
SNR_cut = 20

# define a uniform wavelength grid
uniform_wave = np.linspace(5150,5290,flux_spectra.shape[1])

# interpolate to the rest frame
for i in range(flux_spectra.shape[0]):
    if np.median(flux_spectra[i]) != 0:
        f_flux_spec = interpolate.interp1d(rest_wave[i,:], flux_spectra[i,:],\
                                          bounds_error=False,\
                                          fill_value=(flux_spectra[i,0], flux_spectra[i,-1]))
        flux_spectra[i,:] = f_flux_spec(uniform_wave)

# cull empty spectra
ind = np.where((np.median(flux_spectra, axis=1) != 0)*(err_array > SNR_cut)*(h3_flag == 0) == 1)[0]
flux_spectra = flux_spectra[ind,:]
print(flux_spectra.shape)

# save the restriction array
np.save("ind_cut_real_nvp_SNR=" + str(SNR_cut) + ".npz", ind)

#-------------------------------------------------------------------------------------------------------
# normalize spectra
y_tr = (flux_spectra.T/np.median(flux_spectra, axis=1)).T

# exclude bad pixels
y_tr[np.isnan(y_tr)] = 1.
y_tr[y_tr < 0.] = 0.
y_tr[y_tr > 2] = 2.
print(y_tr.shape)

# convert into torch
x_tr = torch.from_numpy(y_tr[:,200::]).type(torch.cuda.FloatTensor)
y_tr = torch.from_numpy(y_tr[:,:200]).type(torch.cuda.FloatTensor)


#=======================================================================================================
# In [2]:
# define normalizing flow
class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()

        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(masks))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(masks))])
        self.s2 = torch.nn.ModuleList([nets2() for _ in range(len(masks))])
        self.t2 = torch.nn.ModuleList([nett2() for _ in range(len(masks))])

    def g(self, z, y):
        x = z
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            s1 = self.s[i](x_)
            s2 = self.s2[i](y)
            s = s1*s2*(1 - self.mask[i])
            t1 = self.t[i](x_)
            t2 = self.t2[i](y)
            t = t1*t2*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x, y):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s1 = self.s[i](z_)
            s2 = self.s2[i](y)
            s = s1*s2*(1 - self.mask[i])
            t1 = self.t[i](z_)
            t2 = self.t2[i](y)
            t = t1*t2*(1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self,x,y):
        z, logp = self.f(x,y)
        return self.prior.log_prob(z) + logp

    def sample(self, z,y):
        x = self.g(z,y)
        return x



#=======================================================================================================
# In [3]:
# define network
device = torch.device("cuda")
num_neurons = 300

# input dimension
dim_in = y_tr.shape[-1]
nets = lambda: nn.Sequential(nn.Linear(dim_in, num_neurons), nn.LeakyReLU(),\
                             nn.Linear(num_neurons, num_neurons), nn.LeakyReLU(),\
                             nn.Linear(num_neurons, dim_in), nn.Tanh()).cuda()
nett = lambda: nn.Sequential(nn.Linear(dim_in, num_neurons), nn.LeakyReLU(),\
                             nn.Linear(num_neurons, num_neurons), nn.LeakyReLU(),\
                             nn.Linear(num_neurons, dim_in)).cuda()

# conditional layer
dim_cond = x_tr.shape[-1]
nets2 = lambda: nn.Sequential(nn.Linear(dim_cond, num_neurons), nn.LeakyReLU(),\
                              nn.Linear(num_neurons, num_neurons), nn.LeakyReLU(),\
                              nn.Linear(num_neurons, dim_in)).cuda()
nett2 = lambda: nn.Sequential(nn.Linear(dim_cond, num_neurons), nn.LeakyReLU(),\
                              nn.Linear(num_neurons, num_neurons), nn.LeakyReLU(),\
                              nn.Linear(num_neurons, dim_in)).cuda()


#=======================================================================================================
# define mask
num_layers = 10
masks = []
for i in range(num_layers):
    mask_layer = np.random.randint(2,size=(dim_in))
    masks.append(mask_layer)
    masks.append(1-mask_layer)
masks = torch.from_numpy(np.array(masks).astype(np.float32))
masks.to(device)

# set prior
prior = distributions.MultivariateNormal(torch.zeros(dim_in, device='cuda'),\
                                         torch.eye(dim_in, device='cuda'))

# intiate flow
flow = RealNVP(nets, nett, masks, prior)
flow.cuda()



#=======================================================================================================
# In [4]
# number of epoch and batch size
num_epochs = int(1e5)
batch_size = 2048

# break into batches
nsamples = y_tr.shape[0]
nbatches = nsamples // batch_size

# optimizing flow models
optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad==True], lr=1e-4)

# record loss function
loss_array = []

#-------------------------------------------------------------------------------------------------------
# train the network
for e in range(num_epochs):

    # randomly permute the data
    perm = torch.randperm(nsamples)
    perm = perm.cuda()

    # For each batch, calculate the gradient with respect to the loss and take
    # one step.
    for i in range(nbatches):
        idx = perm[i * batch_size : (i+1) * batch_size]
        loss = -flow.log_prob(y_tr[idx], x_tr[idx]).mean()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

#-------------------------------------------------------------------------------------------------------
    # the average loss.
    if e % 10 == 0:
        print('iter %s:' % e, 'loss = %.3f' % loss)
        loss_array.append(loss.item())

    # save models periodically
    if e % 1e3 == 0:
        torch.save(flow, 'flow_final_conditional_lr=-4_SNR=' + str(SNR_cut) + '.pt')
        np.savez("../loss_results_conditional_lr=-4_SNR=" + str(SNR_cut) + ".npz",\
                 loss_array = loss_array)

#========================================================================================================
# save models
torch.save(flow, 'flow_final_conditional_lr=-4_SNR=' + str(SNR_cut) + '.pt')
np.savez("../loss_results_conditional_lr=-4_SNR=" + str(SNR_cut) + ".npz",\
         loss_array = loss_array)
