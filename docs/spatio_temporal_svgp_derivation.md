# Spatio-Temporal SVGP with Separable Kernel and Kronecker Inducing Structure

## 1. Standard Sparse Variational GP (SVGP)

We follow the variational framework of Titsias (2009) and the scalable formulation of Hensman et al. (2013, 2015).

Consider a GP model with latent function $f \sim \mathcal{GP}(0, k)$ and observations $y_n \mid f(\mathbf{x}_n) \sim p(y_n \mid f(\mathbf{x}_n))$ for $n = 1, \ldots, N$.

We introduce $M$ inducing variables $\mathbf{u} = f(\mathbf{Z})$ at inducing locations $\mathbf{Z} = \{\mathbf{z}_1, \ldots, \mathbf{z}_M\}$.

The joint prior over function values and inducing variables is:

$$
\begin{pmatrix} \mathbf{f} \\ \mathbf{u} \end{pmatrix}
\sim \mathcal{N}\!\left(
\mathbf{0},
\begin{pmatrix} \mathbf{K}_{ff} & \mathbf{K}_{fu} \\ \mathbf{K}_{uf} & \mathbf{K}_{uu} \end{pmatrix}
\right)
$$

where $[\mathbf{K}_{ff}]_{nn'} = k(\mathbf{x}_n, \mathbf{x}_{n'})$, $[\mathbf{K}_{fu}]_{nm} = k(\mathbf{x}_n, \mathbf{z}_m)$, and $[\mathbf{K}_{uu}]_{mm'} = k(\mathbf{z}_m, \mathbf{z}_{m'})$.

The conditional prior is:

$$
p(\mathbf{f} \mid \mathbf{u}) = \mathcal{N}\!\left(
\mathbf{K}_{fu} \mathbf{K}_{uu}^{-1} \mathbf{u},\;
\mathbf{K}_{ff} - \mathbf{K}_{fu} \mathbf{K}_{uu}^{-1} \mathbf{K}_{uf}
\right)
$$

We posit a variational distribution $q(\mathbf{u}) = \mathcal{N}(\mathbf{m}, \mathbf{S})$ and form the approximate posterior $q(\mathbf{f}) = \int p(\mathbf{f} \mid \mathbf{u})\, q(\mathbf{u})\, d\mathbf{u}$, whose marginals are:

$$
q(f_n) = \mathcal{N}(\mu_n, \sigma_n^2)
$$

$$
\mu_n = \mathbf{k}_n^\top \mathbf{K}_{uu}^{-1} \mathbf{m}, \qquad
\sigma_n^2 = k_{nn} - \mathbf{k}_n^\top \mathbf{K}_{uu}^{-1} \mathbf{k}_n + \mathbf{k}_n^\top \mathbf{K}_{uu}^{-1} \mathbf{S}\, \mathbf{K}_{uu}^{-1} \mathbf{k}_n
$$

where $\mathbf{k}_n = \mathbf{K}_{fu}[n, :]^\top \in \mathbb{R}^M$ and $k_{nn} = k(\mathbf{x}_n, \mathbf{x}_n)$.

The evidence lower bound (ELBO) is (Hensman et al., 2013, Eq. 5):

$$
\mathcal{L} = \sum_{n=1}^{N} \mathbb{E}_{q(f_n)}\!\left[\log p(y_n \mid f_n)\right] - \mathrm{KL}\!\left(q(\mathbf{u}) \,\|\, p(\mathbf{u})\right)
\tag{1}
$$

where the KL divergence between two Gaussians is:

$$
\mathrm{KL}\!\left(q(\mathbf{u}) \,\|\, p(\mathbf{u})\right) = \frac{1}{2}\left[
\mathrm{tr}\!\left(\mathbf{K}_{uu}^{-1} \mathbf{S}\right)
+ \mathbf{m}^\top \mathbf{K}_{uu}^{-1} \mathbf{m}
- M
+ \log\frac{|\mathbf{K}_{uu}|}{|\mathbf{S}|}
\right]
\tag{2}
$$


## 2. Separable Kernel Assumption

We consider inputs $(\mathbf{x}_n, t_n) \in \mathbb{R}^D \times \mathbb{R}$, where $\mathbf{x}_n$ is a spatial coordinate and $t_n$ is a temporal coordinate. The kernel factorises as:

$$
k\!\left((\mathbf{x}, t),\, (\mathbf{x}', t')\right) = k_s(\mathbf{x}, \mathbf{x}') \cdot k_t(t, t')
\tag{3}
$$

where $k_s$ is a spatial kernel and $k_t$ is a temporal kernel.


## 3. Kronecker Inducing Structure

We place inducing variables on a **Cartesian product grid**:

- **Spatial inducing points**: $\mathbf{Z}_s = \{\mathbf{z}_s^1, \ldots, \mathbf{z}_s^{M_s}\}$
- **Temporal inducing features**: $\mathbf{Z}_t = \{z_t^1, \ldots, z_t^{M_t}\}$

The total number of inducing variables is $M = M_s \cdot M_t$. Each inducing variable is indexed by a pair $(i, j)$ with $i \in \{1, \ldots, M_s\}$ and $j \in \{1, \ldots, M_t\}$.

By the separability of the kernel (Eq. 3), the inducing-inducing covariance matrix factorises as a **Kronecker product**:

$$
\mathbf{K}_{uu} = \mathbf{K}_{uu}^{(s)} \otimes \mathbf{K}_{uu}^{(t)} \in \mathbb{R}^{M \times M}
\tag{4}
$$

where $[\mathbf{K}_{uu}^{(s)}]_{ii'} = k_s(\mathbf{z}_s^i, \mathbf{z}_s^{i'}) \in \mathbb{R}^{M_s \times M_s}$ and $[\mathbf{K}_{uu}^{(t)}]_{jj'} \in \mathbb{R}^{M_t \times M_t}$ is the temporal inducing covariance.

For each data point $(\mathbf{x}_n, t_n)$, the cross-covariance row factorises as:

$$
\mathbf{K}_{fu}[n, :] = \mathbf{k}_s(\mathbf{x}_n)^\top \otimes \mathbf{k}_t(t_n)^\top \in \mathbb{R}^{1 \times M}
\tag{5}
$$

where $\mathbf{k}_s(\mathbf{x}_n) \in \mathbb{R}^{M_s}$ with $[\mathbf{k}_s(\mathbf{x}_n)]_i = k_s(\mathbf{x}_n, \mathbf{z}_s^i)$ and $\mathbf{k}_t(t_n) \in \mathbb{R}^{M_t}$ with $[\mathbf{k}_t(t_n)]_j = [\mathbf{K}_{fu}^{(t)}]_{nj}$.

Stacking across all $N$ data points, the full cross-covariance is a **Khatri-Rao product** (row-wise Kronecker):

$$
\mathbf{K}_{fu} = \mathbf{K}_{fu}^{(s)} \odot \mathbf{K}_{fu}^{(t)} \in \mathbb{R}^{N \times M}
\tag{6}
$$

where $\mathbf{K}_{fu}^{(s)} \in \mathbb{R}^{N \times M_s}$ and $\mathbf{K}_{fu}^{(t)} \in \mathbb{R}^{N \times M_t}$, and row $n$ of the result is the Kronecker product of the corresponding rows.

The diagonal of the data-data covariance is:

$$
k_{nn} = k_s(\mathbf{x}_n, \mathbf{x}_n) \cdot k_t(t_n, t_n)
\tag{7}
$$


## 4. ELBO with Kronecker Structure

Substituting the Kronecker structure into the ELBO (Eq. 1), we exploit the following standard identities:

$$
\mathbf{K}_{uu}^{-1} = \left(\mathbf{K}_{uu}^{(s)}\right)^{-1} \otimes \left(\mathbf{K}_{uu}^{(t)}\right)^{-1}
\tag{8}
$$

$$
|\mathbf{K}_{uu}| = |\mathbf{K}_{uu}^{(s)}|^{M_t} \cdot |\mathbf{K}_{uu}^{(t)}|^{M_s}
\tag{9}
$$

$$
\mathrm{chol}(\mathbf{K}_{uu}) = \mathrm{chol}(\mathbf{K}_{uu}^{(s)}) \otimes \mathrm{chol}(\mathbf{K}_{uu}^{(t)})
\tag{10}
$$

### 4.1 Predictive Mean

Using Eqs. (5) and (8), the predictive mean at data point $n$ is:

$$
\mu_n = \left(\mathbf{k}_s(\mathbf{x}_n)^\top \otimes \mathbf{k}_t(t_n)^\top\right)
\left(\left(\mathbf{K}_{uu}^{(s)}\right)^{-1} \otimes \left(\mathbf{K}_{uu}^{(t)}\right)^{-1}\right) \mathbf{m}
$$

$$
= \left(\mathbf{k}_s(\mathbf{x}_n)^\top \left(\mathbf{K}_{uu}^{(s)}\right)^{-1}\right) \otimes \left(\mathbf{k}_t(t_n)^\top \left(\mathbf{K}_{uu}^{(t)}\right)^{-1}\right) \mathbf{m}
\tag{11}
$$

Define $\boldsymbol{\alpha}_s(\mathbf{x}_n) = \left(\mathbf{K}_{uu}^{(s)}\right)^{-1} \mathbf{k}_s(\mathbf{x}_n) \in \mathbb{R}^{M_s}$ and $\boldsymbol{\alpha}_t(t_n) = \left(\mathbf{K}_{uu}^{(t)}\right)^{-1} \mathbf{k}_t(t_n) \in \mathbb{R}^{M_t}$.

If we reshape $\mathbf{m} \in \mathbb{R}^{M}$ into $\mathbf{M} \in \mathbb{R}^{M_s \times M_t}$, then:

$$
\mu_n = \boldsymbol{\alpha}_s(\mathbf{x}_n)^\top \mathbf{M}\, \boldsymbol{\alpha}_t(t_n)
\tag{12}
$$

### 4.2 Predictive Variance

The predictive variance (Eq. in Section 1) has three terms. Using the Kronecker structure:

$$
\tilde{k}_n = k_{nn} - \mathbf{k}_n^\top \mathbf{K}_{uu}^{-1} \mathbf{k}_n
= k_s(\mathbf{x}_n, \mathbf{x}_n) k_t(t_n, t_n) - \left(\boldsymbol{\alpha}_s(\mathbf{x}_n)^\top \mathbf{k}_s(\mathbf{x}_n)\right)\left(\boldsymbol{\alpha}_t(t_n)^\top \mathbf{k}_t(t_n)\right)
\tag{13}
$$

$$
= \tilde{k}_s(\mathbf{x}_n) \cdot k_t(t_n, t_n) + \left(\boldsymbol{\alpha}_s(\mathbf{x}_n)^\top \mathbf{k}_s(\mathbf{x}_n)\right) \cdot \tilde{k}_t(t_n)
$$

where $\tilde{k}_s(\mathbf{x}_n) = k_s(\mathbf{x}_n, \mathbf{x}_n) - \mathbf{k}_s(\mathbf{x}_n)^\top (\mathbf{K}_{uu}^{(s)})^{-1} \mathbf{k}_s(\mathbf{x}_n)$ is the spatial prior variance reduction, and similarly $\tilde{k}_t(t_n)$ for temporal.

The full predictive variance is:

$$
\sigma_n^2 = \tilde{k}_n + \left(\boldsymbol{\alpha}_s(\mathbf{x}_n) \otimes \boldsymbol{\alpha}_t(t_n)\right)^\top \mathbf{S} \left(\boldsymbol{\alpha}_s(\mathbf{x}_n) \otimes \boldsymbol{\alpha}_t(t_n)\right)
\tag{14}
$$

### 4.3 KL Divergence

The KL divergence (Eq. 2) uses the Kronecker determinant identity (Eq. 9):

$$
\mathrm{KL} = \frac{1}{2}\!\left[
\mathrm{tr}\!\left(\left((\mathbf{K}_{uu}^{(s)})^{-1} \otimes (\mathbf{K}_{uu}^{(t)})^{-1}\right) \mathbf{S}\right)
+ \mathbf{m}^\top \left((\mathbf{K}_{uu}^{(s)})^{-1} \otimes (\mathbf{K}_{uu}^{(t)})^{-1}\right) \mathbf{m}
- M
+ M_t \log|\mathbf{K}_{uu}^{(s)}| + M_s \log|\mathbf{K}_{uu}^{(t)}|
- \log|\mathbf{S}|
\right]
\tag{15}
$$


## 5. Connection to HiPPO Inducing Features

In HiPPO-SVGP (Zhu et al., 2023), the temporal inducing variables are not standard function evaluations at temporal locations, but rather **inter-domain features** (Lázaro-Gredilla & Figueiras-Vidal, 2009): projections of the function onto HiPPO (Legendre polynomial) basis functions applied to random Fourier features (RFF).

Crucially, the SVGP ELBO (Eq. 1) depends only on the kernel matrices $\mathbf{K}_{uu}$, $\mathbf{K}_{fu}$, and $k_{nn}$ — not on how these matrices are computed. The inducing strategy is an implementation detail that produces these matrices.

For the **Spatio-Temporal HiPPO-SVGP**, we compute each factor independently:

| Component | Spatial | Temporal |
|-----------|---------|----------|
| Inducing type | Standard inducing points $\mathbf{Z}_s$ | HiPPO features (Analytic or ODE) |
| $\mathbf{K}_{uu}$ | $k_s(\mathbf{Z}_s, \mathbf{Z}_s) \in \mathbb{R}^{M_s \times M_s}$ | HiPPO-computed $\in \mathbb{R}^{M_t \times M_t}$ |
| $\mathbf{K}_{fu}$ | $k_s(\mathbf{X}, \mathbf{Z}_s) \in \mathbb{R}^{N \times M_s}$ | HiPPO-computed $\in \mathbb{R}^{N \times M_t}$ |
| $k_{nn}$ | $k_s(\mathbf{x}_n, \mathbf{x}_n)$ | $k_t(t_n, t_n)$ |

The combined matrices are then:

$$
\mathbf{K}_{uu} = \mathbf{K}_{uu}^{(s)} \otimes \mathbf{K}_{uu}^{(t)}, \qquad
\mathbf{K}_{fu} = \mathbf{K}_{fu}^{(s)} \odot \mathbf{K}_{fu}^{(t)}, \qquad
k_{nn} = k_s(\mathbf{x}_n, \mathbf{x}_n) \cdot k_t(t_n, t_n)
\tag{16}
$$

These are substituted directly into the ELBO (Eq. 1) and the predictive equations (Eqs. 11--14).


## 6. Computational Complexity

Without Kronecker structure, standard SVGP costs $\mathcal{O}(NM^2) = \mathcal{O}(N M_s^2 M_t^2)$.

With Kronecker structure, the Cholesky factorisation of $\mathbf{K}_{uu}$ (Eq. 10) costs $\mathcal{O}(M_s^3 + M_t^3)$ instead of $\mathcal{O}(M_s^3 M_t^3)$. Solves with $\mathbf{K}_{uu}^{-1}$ factorise similarly. The dominant cost becomes $\mathcal{O}(N(M_s + M_t))$ for the Khatri-Rao structure in $\mathbf{K}_{fu}$, plus $\mathcal{O}(M_s^3 + M_t^3)$ for the inducing covariance factorisation.

In the **online setting**, the temporal matrices $\mathbf{K}_{uu}^{(t)}$ and $\mathbf{K}_{fu}^{(t)}$ are updated via HiPPO as new temporal data arrives, while the spatial matrices remain fixed (or are updated independently).


## References

- Titsias, M. K. (2009). Variational Learning of Inducing Variables in Sparse Gaussian Processes. *AISTATS*.
- Hensman, J., Fusi, N., & Lawrence, N. D. (2013). Gaussian Processes for Big Data. *UAI*.
- Hensman, J., Matthews, A. G., & Ghahramani, Z. (2015). Scalable Variational Gaussian Process Classification. *AISTATS*.
- Lázaro-Gredilla, M. & Figueiras-Vidal, A. (2009). Inter-domain Gaussian Processes for Sparse Inference using Inducing Features. *NeurIPS*.
- Zhu, H., Maddox, W. J., & Bonilla, E. V. (2023). HiPPO-SVGP: Online Sparse Variational Gaussian Processes with HiPPO Inducing Features.
- Hamelijnck, O., Wilkinson, W., Loppi, N., Solin, A., & Damoulas, T. (2021). Spatio-Temporal Variational Gaussian Processes. *NeurIPS*.
