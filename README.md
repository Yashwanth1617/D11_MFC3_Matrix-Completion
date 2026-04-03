<div align="center">

# 🧮 ADMM-Factorization Algorithm for Low Rank Matrix Completion

<img src="https://img.shields.io/badge/Course-22MAT220%20Mathematics%20for%20Computing%203-blue?style=for-the-badge" />
<img src="https://img.shields.io/badge/Institution-Amrita%20Vishwa%20Vidyapeetham-red?style=for-the-badge" />
<img src="https://img.shields.io/badge/Language-MATLAB-orange?style=for-the-badge&logo=matlab" />
<img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge" />

<br/>

> *Recovering the invisible — reconstructing low-rank matrices from partial observations using ADMM with matrix factorization.*

<br/>

**Team D11 · Amrita School of Artificial Intelligence, Coimbatore**

</div>

---

## 👥 Team Members

| Name | Roll Number |
|------|------------|
| Hemanth S N | CB.SC.U4AIE24321 |
| Jaswanth Saravanan | CB.SC.U4AIE24324 |
| Mahakisore M | CB.SC.U4AIE24333 |
| Yashwanth B | CB.SC.U4AIE24360 |

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Mathematical Foundation](#-mathematical-foundation)
- [Algorithm: ADMM-Factorization](#-algorithm-admm-factorization)
- [Update Equations](#-update-equations)
- [Convergence Analysis](#-convergence-analysis)
- [Datasets](#-datasets)
- [Experiments & Results](#-experiments--results)
- [Repository Structure](#-repository-structure)
- [How to Run](#-how-to-run)
- [Key Achievements](#-key-achievements)
- [References](#-references)

---

## 🔭 Overview

Real-world data is inherently incomplete. Rating matrices in recommender systems, sensor readings in IoT networks, traffic flow matrices in urban monitoring — all of these are **partially observed** datasets where recovering the missing entries is critical to downstream decision-making.

This project implements and validates the **ADMM-Factorization algorithm** for **Low Rank Matrix Completion**, based on the paper by *Taleghani and Salahi (2019)*. The core insight is that many real-world data matrices are **low-rank** — governed by a small number of latent factors — and this structure can be exploited to accurately recover missing entries.

We replicate and validate results on:
- 🎬 **MovieLens 100K** — a real-world collaborative filtering benchmark (943 users, 1,682 movies, 100,000 ratings)
- 🧑 **ORL Face Database** — 400 grayscale face images across 40 subjects (image inpainting)
- 🔲 **Synthetic matrices** — noiseless and noisy random low-rank matrices of various dimensions

---

## 📐 Problem Statement

Given a partially observed matrix $M \in \mathbb{R}^{m \times n}$, recover a **low-rank** matrix $X \in \mathbb{R}^{m \times n}$ such that:

$$\min_{X \in \mathbb{R}^{m \times n}} \text{rank}(X) \quad \text{subject to} \quad X_{ij} = M_{ij}, \quad \forall (i,j) \in \Omega$$

where $\Omega \subseteq \{(i,j) : 1 \leq i \leq m, 1 \leq j \leq n\}$ is the index set of **known (observed) entries**.

### Why is this hard?

The rank function is:
- **NP-hard** to minimize directly — no polynomial-time algorithm is known for the general case
- **Non-convex** — a convex combination of two rank-1 matrices can have rank 2, violating the convexity condition:

$$\text{rank}(\theta X_1 + (1-\theta) X_2) \leq \theta \cdot \text{rank}(X_1) + (1-\theta) \cdot \text{rank}(X_2) \quad \text{(violated!)}$$

---

## 🧠 Mathematical Foundation

### Step 1: Convex Relaxation via Nuclear Norm

To overcome non-convexity, we relax the rank to the **nuclear norm** $\|X\|_* = \sum_i \sigma_i(X)$ (sum of singular values):

$$\min_{X} \|X\|_* \quad \text{subject to} \quad X_{ij} = M_{ij}, \quad \forall (i,j) \in \Omega$$

This is convex but requires an **SVD at every iteration** — prohibitively expensive for large matrices.

### Step 2: Matrix Factorization

Instead of working directly with $X$, we express $X = UV$ where $U \in \mathbb{R}^{m \times r}$, $V \in \mathbb{R}^{r \times n}$, and $r \ll \min(m,n)$ is the **rank estimate**. Introducing auxiliary variable $Z \in \mathbb{R}^{m \times n}$, the problem becomes:

$$\min_{U, V, Z} \frac{1}{2} \|UV - Z\|_F^2 \quad \text{subject to} \quad P_\Omega(Z) = P_\Omega(M)$$

where $P_\Omega$ is the **projection operator** that retains only the observed entries.

### Step 3: Augmented Lagrangian

To enforce the constraint, we build the **Augmented Lagrangian**:

$$\mathcal{L}_\rho(U, V, Z, \Lambda) = \frac{1}{2}\|UV - Z\|_F^2 + \Lambda \cdot (P_\Omega(Z) - P_\Omega(M)) + \frac{\rho}{2} \|P_\Omega(Z) - P_\Omega(M)\|_F^2$$

where $\Lambda \in \mathbb{R}^{m \times n}$ is the **matrix of Lagrange multipliers** and $\rho > 0$ is the **penalty parameter**.

---

## ⚙️ Algorithm: ADMM-Factorization

The **Alternating Direction Method of Multipliers (ADMM)** decomposes the problem into three sub-problems, each solved in closed form:

```
ADMM-Factorization Algorithm
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:  Y₀, Z₀, Λ₀, tolerance ε, penalty ρ, maxiter
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For k = 1, 2, ..., maxiter:

  1. Update X:   Xₖ₊₁ = Zₖ Yₖᵀ (Yₖ Yₖᵀ)⁻¹
  2. Update Y:   Yₖ₊₁ = (Xₖ₊₁ᵀ Xₖ₊₁)⁻¹ Xₖ₊₁ᵀ Zₖ
  3. Update Z:   Zₖ₊₁ = P_Ωᶜ(A) + 1/(1+ρ) [P_Ω(A) + ρ P_Ω(M) - P_Ω(Λₖ)]
                 where A = Xₖ₊₁ Yₖ₊₁
  4. Update Λ:   P_Ω(Λₖ₊₁) = P_Ω(Λₖ) + γρ [P_Ω(Zₖ₊₁) - P_Ω(M)]

  Check stopping criterion:
    ‖M - XY‖_F / max(1, ‖M‖_F) ≤ ε  →  exit

Output: (Xₖ₊₁, Yₖ₊₁, Zₖ₊₁, Λₖ₊₁)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Computational Cost per Iteration:**

| Step | Cost |
|------|------|
| Update X (Eq. 8) | $2mnr + 2nr^2 + O(r^3)$ |
| Update Y (Eq. 9) | $2mnr + 2mr^2 + O(r^3)$ |
| Update Z (Eq. 10) | $2 |\Omega^c|r + 2|\Omega|(2r+1)$ |
| Update Λ (Eq. 11) | $2|\Omega|r + |\Omega|$ |
| **Total** | $6mnr + 2(m+n)r^2 + |\Omega|(4r+3)$ |

> ✅ **No SVD required** — all steps are closed-form least squares, making this algorithm far more scalable than nuclear norm methods.

---

## 📐 Update Equations

### X-Update (Eq. 8)
Minimizing $\frac{1}{2}\|XY_k - Z_k\|_F^2$ w.r.t. $X$:
$$X_{k+1} = Z_k Y_k^T (Y_k Y_k^T)^{-1}$$

### Y-Update (Eq. 9)
Minimizing $\frac{1}{2}\|X_{k+1}Y - Z_k\|_F^2$ w.r.t. $Y$:
$$Y_{k+1} = (X_{k+1}^T X_{k+1})^{-1} X_{k+1}^T Z_k$$

### Z-Update (Eq. 10)
Split by observed/unobserved entries:
- **Unobserved** $(i,j) \notin \Omega$: $\quad Z_{ij} = A_{ij}$ (set to the factorization)
- **Observed** $(i,j) \in \Omega$: $\quad Z_{ij} = \frac{1}{1+\rho}(A_{ij} + \rho M_{ij} - \Lambda_{ij})$

Combined:
$$Z_{k+1} = P_{\Omega^c}(A) + \frac{1}{1+\rho}(P_\Omega(A) + \rho P_\Omega(M) - P_\Omega(\Lambda_k))$$

### Λ-Update (Eq. 11)
Dual variable gradient ascent step:
$$P_\Omega(\Lambda_{k+1}) = P_\Omega(\Lambda_k) + \gamma\rho(P_\Omega(Z_{k+1}) - P_\Omega(M))$$
where $\gamma \in (0, 1.618)$ is the step length.

---

## 📊 Convergence Analysis

**Lemma:** If $\{\Lambda_k\}$ is bounded and $\sum_{k=0}^{\infty} \|\Lambda_{k+1} - \Lambda_k\|_F^2 < \infty$, then $\|W_{k+1} - W_k\|_F^2 \to 0$ as $k \to \infty$, where $W_k = (X_k, Y_k, Z_k)$.

**Theorem:** Any accumulation point $(X^*, Y^*, Z^*, \Lambda^*)$ of the iterates satisfies the **KKT conditions**:

$$
(X^*Y^* - Z^*)(Y^*)^T = 0, \quad (X^*)^T(X^*Y^* - Z^*) = 0
$$
$$
P_{\Omega^c}(Z^*) = P_{\Omega^c}(X^*Y^*), \quad P_\Omega(Z^*) = P_\Omega(M)
$$

This guarantees convergence to a **stationary point** of the non-convex problem.

---

## 📦 Datasets

### 1. MovieLens 100K

| Property | Value |
|----------|-------|
| Total Ratings | 100,000 |
| Users | 943 (each with ≥ 20 ratings) |
| Movies | 1,682 |
| Rating Scale | 1 – 5 |
| Sampling Ratio | ~0.06 |
| Collection Period | Sep 1997 – Apr 1998 |
| Source | GroupLens Research, University of Minnesota |

Used to benchmark the algorithm on **real-world collaborative filtering** with an extremely sparse observation matrix.

### 2. ORL (AT&T) Face Database

| Property | Value |
|----------|-------|
| Total Images | 400 |
| Subjects | 40 |
| Images per Subject | 10 |
| Resolution | 92 × 112 pixels |
| Bit Depth | 8-bit grayscale |
| Format | PGM |
| Variations | Expressions, glasses, head pose |
| Source | Olivetti Research Laboratory, Cambridge UK |

Used for **image inpainting** — randomly masking >50% of pixels and recovering the original face images using low-rank structure.

---

## 🔬 Experiments & Results

### Experiment 1: Basic ADMM on Synthetic Matrices

Implementation of the core ADMM-Factorization algorithm verified against the paper's baseline. The algorithm consistently achieves relative errors in the range of **~9×10⁻¹¹** matching the paper.

### Experiment 2: Noiseless Synthetic Matrices (First Class)

**Setup:** Random low-rank matrices $M = M_L M_R$ with $M_L \in \mathbb{R}^{m \times r}$, $M_R \in \mathbb{R}^{r \times n}$, sampling ratio SR = 0.56.

| Matrix Size [m,n,r] | ADMM Time (s) | ADMM Rel.Err | ScaledASD Time (s) | LMaFit Time (s) |
|---------------------|--------------|-------------|-------------------|----------------|
| [300, 2000, 3] | **0.64** | 9.24×10⁻¹¹ | 0.65 | 1.50 |
| [500, 2000, 3] | **0.88** | 9.61×10⁻¹¹ | 0.94 | 1.91 |
| [700, 2000, 3] | **1.10** | 6.67×10⁻¹¹ | 1.05 | 2.72 |
| [300, 4000, 3] | **1.17** | 6.86×10⁻¹¹ | 1.19 | 2.83 |
| [500, 4000, 3] | **1.64** | 7.03×10⁻¹¹ | 1.76 | 3.60 |
| [700, 4000, 3] | **2.04** | 6.30×10⁻¹¹ | 1.99 | 5.16 |

✅ ADMM-Factorization achieves **comparable or better CPU times** than ScaledASD, while being **significantly faster** than FR (14–120s) and OptSpace (28–128s).

### Experiment 3: Noisy Matrices (Second Class)

**Setup:** Noisy observations $M_\epsilon = M + \epsilon \frac{\|M\|_F}{\|P_\Omega(N)\|_F} P_\Omega(N)$ with $\epsilon = 10^{-7}$, SR = 0.25.

| Matrix Size [m,n,r] | ADMM Time (s) | ADMM Rel.Err | ScaledASD Time (s) | FR Time (s) |
|---------------------|--------------|-------------|-------------------|------------|
| [1000, 1000, 2] | **4.18** | 1.57×10⁻⁷ | 11.40 | 6.18 |
| [2000, 2000, 2] | **16.13** | 1.54×10⁻⁷ | 43.22 | 34.71 |
| [4000, 4000, 2] | **60.72** | 1.57×10⁻⁷ | 168.29 | 286.87 |
| [5000, 5000, 3] | **94.53** | 1.57×10⁻⁷ | 256.25 | **Timeout** |

✅ ADMM is **2–3× faster** than ScaledASD and **3–5× faster** than FR on large noisy problems. OptSpace fails entirely (timeout) for $5000 \times 5000$ matrices.

### Experiment 4: Face Image Reconstruction (ORL Database)

**Setup:** AT&T face images (112×92), >50% pixels randomly removed, estimated rank = 10.

| Algorithm | Time (s) | Relative Error |
|-----------|----------|---------------|
| **ADMM-Factorization** | **0.15** | 9.93×10⁻² |
| LMaFit | 0.25 | 9.93×10⁻² |
| ASD | 0.19 | 9.92×10⁻² |
| ScaledASD | 0.29 | 9.99×10⁻² |

✅ ADMM achieves the **lowest runtime** with reconstruction quality matching all competing algorithms. Face identity and structure are faithfully recovered despite >50% missing pixels.

### Experiment 5: MovieLens 100K Recommendation

**Setup:** 943 users × 1682 movies rating matrix, sampling ratio ≈ 0.06, extremely sparse.

Stopping criterion:
$$\text{Residual Error} = \frac{\|P_\Omega(M) - P_\Omega(XY)\|_F}{\|P_\Omega(M)\|_F} \leq \text{tol}$$

ADMM-Factorization achieved **consistently lower CPU times** than ASD and ScaledASD across all rank settings (50–130), demonstrating real-world applicability for collaborative filtering.

> *Note: LMaFit was excluded from MovieLens comparison as its relative errors exceeded 10⁻¹ for all rank values tested.*

### Experiment 6: One-Square Black Image Completion

A custom experiment on a structured binary image — a black square on white background — to verify the algorithm's behavior on structured/piecewise-constant data. The algorithm successfully recovered the original structure with low error, demonstrating flexibility beyond conventional photographic images.

---

## 📁 Repository Structure

```
D11_MFC3_Matrix-Completion/
├── 📁 Base Paper
│   └── 📕 An ADMM-Factorization Algorithm for Low Rank Matrix.pdf
├── 📁 Dataset & Codes
│   ├── 📄 AT&T Faces.mlx
│   ├── 📄 BasicLMFusingADMM.mlx
│   ├── 📦 Dataset.zip
│   ├── 📄 Movielens.mlx
│   └── 📄 Onesquareblack.mlx
├── 📘 Analog_Report.docx
├── 📕 MFC_3.pdf
├── 📝 README.md
└── 📄 mfc3_report.mlx
```

---

## 🚀 How to Run

### Prerequisites

- **MATLAB R2018a or later** (tested on R2018a as per the original paper)
- No additional toolboxes required

### Steps

1. **Clone this repository**
   ```bash
   git clone https://github.com/your-username/admm-matrix-completion.git
   cd admm-matrix-completion
   ```

2. **Extract the datasets**
   ```
   Unzip Dataset.zip into the project root directory
   ```

3. **Run the core algorithm**
   ```matlab
   % Open MATLAB and navigate to the project folder
   open BasicLMFusingADMM.mlx
   % Click Run All
   ```

4. **Run individual experiments**
   ```matlab
   open Movielens.mlx          % MovieLens recommendation experiment
   open AT_T_Faces.mlx         % Face image reconstruction
   open Onesquareblack.mlx     % Binary image completion
   open mfc3_report.mlx        % Full report with all experiments
   ```

### Configuration Parameters

```matlab
rho    = 1e8;       % Penalty parameter (ρ)
gamma  = 1.0;       % Dual step length (γ ∈ (0, 1.618))
tol    = 1e-10;     % Stopping tolerance
maxiter = 500;      % Maximum iterations
r      = 10;        % Rank estimate (tune per problem)

% Initialization
Y0 = eye(r, n);     % Identity matrix
Z0 = P_Omega(M);    % Observed entries
Lambda0 = ones(m,n); % Ones matrix
```

---

## 🏆 Key Achievements

| Achievement | Detail |
|-------------|--------|
| ✅ **Paper Replication** | Successfully reproduced all 5 classes of numerical experiments from Taleghani & Salahi (2019) |
| ✅ **Competitive Speed** | Matched or beat ScaledASD on all problem classes; significantly outperformed LMaFit, ASD, FR, and OptSpace |
| ✅ **Real-World Validation** | Demonstrated on MovieLens 100K (collaborative filtering) and ORL Face Database (image inpainting) |
| ✅ **Scalability** | Handles matrices up to 5000×5000 where competing methods (OptSpace, FR) timeout or fail |
| ✅ **SVD-Free** | Algorithm avoids expensive SVD computations, relying entirely on closed-form least squares updates |
| ✅ **Convergence Proven** | Theoretical convergence to KKT stationary points proven and validated empirically |
| ✅ **Novel Extension** | Applied the algorithm to a custom structured image (black square) beyond the paper's test cases |

---

## 📈 Algorithm Comparison Summary

```
Performance Ranking (CPU Time, lower is better):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Class 1 (Noiseless):   ADMM ≈ ScaledASD  ≪  LMaFit < ASD < ScGrassMC ≪ FR ≪ OptSpace
Class 2 (Noisy):       ADMM  ≪  ScaledASD ≪ ASD ≈ LMaFit < ScGrassMC < FR < OptSpace
Class 3 (PSD):         ADMM ≈ ScaledASD  ≪  ASD ≈ LMaFit
Class 4 (Images):      ADMM  <  LMaFit ≈ ASD  <  ScaledASD
Class 5 (MovieLens):   ADMM  <  ASD  <  ScaledASD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Relative errors are comparable across all algorithms.
```

---

## 📚 References

1. **Taleghani, R. & Salahi, M.** (2019). An ADMM-Factorization Algorithm for Low Rank Matrix Completion. *Applications and Applied Mathematics: An International Journal (AAM)*, Vol. 14, Issue 2, Article 34. [Link](https://digitalcommons.pvamu.edu/aam/vol14/iss2/34)

2. **Wen, Z., Yin, W. & Zhang, Y.** (2012). Solving a low-rank factorization model for matrix completion by a nonlinear successive over-relaxation algorithm. *Mathematical Programming Computation*, 4(4), 333–361.

3. **Tanner, J. & Wei, K.** (2016). Low rank matrix completion by alternating steepest descent methods. *Applied and Computational Harmonic Analysis*, 40(2), 417–429.

4. **Candès, E.J. & Recht, B.** (2009). Exact matrix completion via convex optimization. *Foundations of Computational Mathematics*, 9(6), 717–772.

5. **Keshavan, R.H., Montanari, A. & Oh, S.** (2010). Matrix completion from a few entries. *IEEE Transactions on Information Theory*, 56(6), 2980–2998.

6. **Harper, F.M. & Konstan, J.A.** (2016). The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems*, 5(4), 19.

7. **Huang, S. & Wolkowicz, H.** (2018). Low-rank matrix completion using nuclear norm minimization and facial reduction. *Journal of Global Optimization*, 72(1), 5–26.

---

<div align="center">

**Amrita School of Artificial Intelligence · Amrita Vishwa Vidyapeetham, Coimbatore**

*22MAT220 Mathematics for Computing 3 · August 2025*

</div>
