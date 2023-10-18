import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tifffile as tiff

# Load the IMC file
tiff_file = 'raw/TMA1_45B.tiff'
img = tiff.imread(tiff_file)
x = np.array(img)

# Sanity preprocessing to remove crazy high pixels
for i in range(x.shape[0]):
    x[i] = x[i].clip(0, np.percentile(x[i], 98))

#### Create a combinatorial mask
import itertools

# Cartesian product of color assignments, dropping the null all-zero one
masks = np.array([np.array(p) for p in itertools.product((0, 1), repeat=int(np.ceil(np.log2(x.shape[0]))))][
                 1:x.shape[0] + 1]).astype(bool)

# Compress the image
x_comp = np.array([x[m].sum(axis=0) for m in masks.T])

# Partially decompress the binary image because we know any pixel that is
# zero must be zero for all mask members
# x_binary = x > 0
# x_partial = np.zeros((masks.shape[0],) + x.shape[1:])
# for c in range(x_partial.shape[0]):
#     x_partial[c] = x_binary[masks[c]].min(axis=0)

# Show the compressed image assuming it's 6 channels
fig, axarr = plt.subplots(2, 3, figsize=(15, 10))
for i in range(6):
    axarr[i // 3, i % 3].imshow(x_comp[i])
plt.show()


#### Decompress the image

# def decompress_image(data, masks, lam_min=1e-1, lam_max=50, n_lam=30):
#     import torch
#     from autograd_minimize import minimize
#     from tqdm import tqdm
#
#     # Estimates of the latent rates
#     cur_Rates = np.ones((masks.shape[0],) + data.shape[1:]) * 2
#     Rates = np.zeros((n_lam,) + cur_Rates.shape)
#
#     # Build the torch data tensors
#     t_Data = torch.Tensor(data)
#     t_Masks = torch.Tensor(masks)
#
#     lams = np.exp(np.linspace(np.log(lam_min), np.log(lam_max), n_lam))[::-1]
#     for lam_idx, lam in tqdm(enumerate(lams)):
#         print(f'{lam_idx + 1}/{n_lam} Lam={lam}')
#
#         # Estimate the rates as Poisson counts with a fused lasso penalty
#         def loss(t_rates):
#             '''
#             t_rates = torch.nn.functional.softplus(t_rates)
#
#             # Calculate the reconstruction loss
#             recon = (t_rates[:,None] * t_Masks[:,:,None,None]).sum(axis=0)
#             l = ((t_Data - recon)**2).mean()
#             '''
#
#             # Calculate the cross-entropy loss on the binarized signal
#             l = torch.nn.functional.binary_cross_entropy(t_rates, t_Data)
#
#             # Add the fused lasso penalty
#             if lam > 0:
#                 rows = torch.abs(t_rates[:, 1:] - t_rates[:, :-1]).reshape(-1)
#                 cols = torch.abs(t_rates[:, :, 1:] - t_rates[:, :, :-1]).reshape(-1)
#                 diag1 = torch.abs(t_rates[:, 1:, 1:] - t_rates[:, :-1, :-1]).reshape(-1)
#                 diag2 = torch.abs(t_rates[:, 1:, :-1] - t_rates[:, :-1, 1:]).reshape(-1)
#                 pen = lam * (rows.sum() + cols.sum() + diag1.sum() + diag2.sum()) / (
#                             rows.shape[0] + cols.shape[0] + diag1.shape[0] + diag2.shape[0])
#                 lasso = torch.abs(t_rates).mean() * lam
#                 print(f'recon: {l:.4f} penalty: {pen:.4f} lasso: {lasso:.4f}')
#                 l += pen + lasso
#
#             print(l)
#
#             return l
#
#         # Optimize using a 2nd order method with autograd for gradient calculation.
#         res = minimize(loss, cur_Rates, method='L-BFGS-B', backend='torch',
#                        # bounds=(1e-4,None),
#                        tol=1e-6)
#         cur_Rates = res.x
#
#         # Save the results
#         Rates[lam_idx] = cur_Rates
#
#     return Rates[::-1], lams[::-1]
#
#
# def decompress_nmf(data, masks, n_components=10, lam=1):
#     import torch
#     from autograd_minimize import minimize
#     from tqdm import tqdm
#
#     imshape = data.shape[1:]
#     flat_data = data.reshape((data.shape[0], -1))
#     n_pix = flat_data.shape[1]
#
#     W_cur = np.abs(np.random.normal(size=(masks.shape[0], n_components)))
#     V_cur = np.abs(np.random.normal(size=(n_pix, n_components)))
#
#     # Build the torch data tensors
#     t_Data = torch.Tensor(flat_data)
#     t_Masks = torch.Tensor(masks)
#
#     # Loss function
#     def loss(t_W, t_V):
#         t_Signal_hat = (t_W[:, None] * t_V[None]).sum(axis=2)
#         print(t_Signal_hat.shape, t_Masks.shape)
#         t_Data_hat = (t_Signal_hat[:, None] * t_Masks[:, :, None]).sum(axis=0)
#         l = ((t_Data - t_Data_hat) ** 2).mean()
#
#         # Add a lasso penalty to the latent factors
#         if lam > 0:
#             l += lam * t_W.sum() / (t_W.shape[0] * t_W.shape[1])
#             l += lam * t_V.sum() / (t_V.shape[0] * t_V.shape[1])
#
#         print(l)
#
#         return l
#
#     # Optimize using a 2nd order method with autograd for gradient calculation.
#     res = minimize(loss, [W_cur, V_cur], method='L-BFGS-B', backend='torch',
#                    bounds=(0, None),
#                    tol=1e-6)
#
#     W, V = res.x
#     V = V.T
#     V = V.reshape((V.shape[0],) + imshape)
#
#     return res.x
#
#
# def decompress_nmf(data, masks, n_components=10, lam=0, max_outer_steps=100, max_inner_steps=1000, tol=1e-6):
#     import torch
#     from autograd_minimize import minimize
#     from tqdm import tqdm
#
#     imshape = data.shape[1:]
#     flat_data = data.reshape((data.shape[0], -1))
#     n_pix = flat_data.shape[1]
#
#     W = np.abs(np.random.normal(size=(masks.shape[0], n_components)))
#     V = np.abs(np.random.normal(size=(n_pix, n_components)))
#
#     # Build the torch data tensors
#     t_Data = torch.Tensor(flat_data)
#     t_Masks = torch.Tensor(masks)
#
#     for outer_step in range(max_outer_steps):
#         print(f'Step {outer_step + 1}/{max_outer_steps}')
#         print('Fitting W')
#
#         t_V = torch.Tensor(V)
#
#         # Loss function
#         def loss_W(t_W):
#             t_Signal_hat = (t_W[:, None] * t_V[None]).sum(axis=2)
#             t_Data_hat = (t_Signal_hat[:, None] * t_Masks[:, :, None]).sum(axis=0)
#             l = ((t_Data - t_Data_hat) ** 2).mean()
#
#             # Add a lasso penalty to the latent factors
#             if lam > 0:
#                 l += lam * t_W.sum() / (t_W.shape[0] * t_W.shape[1])
#
#             return l
#
#         # Optimize using a 2nd order method with autograd for gradient calculation.
#         res = minimize(loss_W, W, method='L-BFGS-B', backend='torch',
#                        bounds=(0, None),
#                        tol=tol,
#                        options={'maxiter': max_outer_steps})
#         W = res.x
#
#         print('Fitting V')
#         t_W = torch.Tensor(W)
#
#         # Loss function
#         def loss_V(t_V):
#             t_Signal_hat = (t_W[:, None] * t_V[None]).sum(axis=2)
#             t_Data_hat = (t_Signal_hat[:, None] * t_Masks[:, :, None]).sum(axis=0)
#             l = ((t_Data - t_Data_hat) ** 2).mean()
#
#             # Add a lasso penalty to the latent factors
#             if lam > 0:
#                 l += lam * t_V.sum() / (t_V.shape[0] * t_V.shape[1])
#
#             return l
#
#         # Optimize using a 2nd order method with autograd for gradient calculation.
#         res = minimize(loss_V, V, method='L-BFGS-B', backend='torch',
#                        bounds=(0, None),
#                        tol=tol,
#                        options={'maxiter': max_outer_steps})
#
#         V = res.x
#
#     V = V.T
#     V = V.reshape((V.shape[0],) + imshape)
#
#     return W, V


def decompress_nmf(data, masks, truth, n_components=10, lam=0, max_outer_steps=100, max_inner_steps=1000, tol=1e-6):
    import torch
    from autograd_minimize import minimize
    from tqdm import tqdm
    from sklearn.decomposition import NMF

    imshape = data.shape[1:]
    flat_data = data.reshape((data.shape[0], -1))
    n_pix = flat_data.shape[1]

    # Cheat. Learn the true W on the real signal first.
    print('Fitting W')
    truth_shape = truth.shape[1:]
    flat_truth = truth.reshape((truth.shape[0], -1))
    n_flat_pix = flat_truth.shape[1]
    nmf = NMF(n_components=n_components)
    W = nmf.fit_transform(flat_truth)
    t_W = torch.Tensor(W)
    print(f'W shape: {W.shape}')

    V = np.abs(np.random.normal(size=(n_pix, n_components)))

    # Build the torch data tensors
    t_Data = torch.Tensor(flat_data)
    t_Masks = torch.Tensor(masks)

    print('Fitting V')

    # Loss function
    def loss_V(t_V):
        t_Signal_hat = (t_W[:, None] * t_V[None]).sum(axis=2)
        t_Data_hat = (t_Signal_hat[:, None] * t_Masks[:, :, None]).sum(axis=0)
        l = ((t_Data - t_Data_hat) ** 2).mean()

        # Add a lasso penalty to the latent factors
        if lam > 0:
            l += lam * t_V.sum() / (t_V.shape[0] * t_V.shape[1])

        return l

    # Optimize using a 2nd order method with autograd for gradient calculation.
    res = minimize(loss_V, V, method='L-BFGS-B', backend='torch',
                   bounds=(0, None),
                   tol=tol,
                   options={'maxiter': max_outer_steps})

    V = res.x
    V = V.T
    V = V.reshape((V.shape[0],) + imshape)

    return W, V


def softplus_np(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


start = 300
width = 200

# rates, lams = decompress_image(x_partial[:,start:start+width,start:start+width], masks, lam_max=500)
# rates = softplus_np(rates)

W, V = decompress_nmf(x_comp[:, start:start + width, start:start + width],
                      masks,
                      x[:, start:start + width, start:start + width], n_components=30, tol=1e-3, max_outer_steps=20,
                      lam=0.1)
rates = (W[:, :, None, None] * V[None]).sum(axis=1).reshape(x[:, start:start + width, start:start + width].shape)

# Show the compressed image assuming it's 6 channels
Xs = [x[:, start:start + width, start:start + width],
      x_comp[:, start:start + width, start:start + width],
      rates]
# Names = ['True', 'Compressed', f'Lam={lams[5]:.2f}',f'Lam={lams[10]:.2f}',f'Lam={lams[15]:.2f}',f'Lam={lams[20]:.2f}']
Names = ['True', 'Compressed', 'Reconstruction']
fig, axarr = plt.subplots(len(Xs), 6, figsize=(30, 5 * len(Xs)))
for i, x_plot in enumerate(Xs):
    x_plot = Xs[i]
    for j in range(6):
        axarr[i, j].imshow(x_plot[j])#, cmap='rocket')
        axarr[i, j].set_title(f'{Names[i]} Channel {j + 1}')
plt.tight_layout()
plt.savefig('plots/fused-compressed-sensing.pdf', bbox_inches='tight')
plt.close()

if __name__ == "__main__":
    pass
