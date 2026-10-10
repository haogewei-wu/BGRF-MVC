# BGRF-MVC v2.0.1 Experimental Prototype — Frozen Method Specification

Version 2.0.1 preserves the complete v2.0 algorithm framework. It only tightens probability normalization, degenerate-radius behavior, radial-standard-deviation convention, deterministic candidate splitting, radius interpretation, and parent/child size guards. It introduces no new algorithmic module.

## 4.1 Input

The multi-view input is

$$X=\{X^{(1)},\ldots,X^{(V)}\}, \qquad X^{(v)}\in\mathbb{R}^{n\times d_v}.$$

Ground-truth labels $y$ are not algorithm inputs. They do not participate in preprocessing, fitting, transform, granular-ball construction, relation construction, reliability computation, view weighting, fusion, KMeans, hyperparameter selection, dataset selection, or determination of $K$.

## 4.2 Independent per-view standardization

Apply `StandardScaler` independently to each view. Concatenating before standardization, using $y$, supervised feature selection, and default PCA are forbidden. The default is `use_pca = false`.

## 4.3 Independent recursive granular-ball construction per view

For a current ball $B$, define its center and mean radius as

$$c(B)=\frac{1}{|B|}\sum_{x\in B}x,$$

$$r(B)=\frac{1}{|B|}\sum_{x\in B}\|x-c(B)\|_2.$$

Here $r(B)$ is a **characteristic radius**, also called an **effective granular radius**. It describes the typical scale of the local granular structure; it is not an enclosing radius required to contain every member. Consequently, “inside”, “boundary”, and “outside” below mean position relative to this characteristic/effective granular boundary, not strict set containment by a physical ball.

Each view starts with one ball containing all samples and is recursively processed independently of every other view and of $y$.

## 4.4 Candidate binary split

Use unsupervised 2-means only as a **candidate splitter**,

$$B\rightarrow B_1+B_2.$$

It must not be claimed that 2-means directly minimizes the mean-radius objective defined here. For candidate children, calculate

$$r_{\mathrm{split}}=\frac{|B_1|r(B_1)+|B_2|r(B_2)}{|B|},$$

$$\operatorname{Gain}(B)=\frac{r(B)-r_{\mathrm{split}}}{r(B)+\varepsilon}.$$

A split is allowed only if every condition holds:

1. $|B|\geq\mathtt{min\_parent\_size}$;
2. $|B|\geq2\,\mathtt{min\_child\_size}$;
3. $\operatorname{Gain}(B)>\eta$;
4. $|B_1|\geq\mathtt{min\_child\_size}$;
5. $|B_2|\geq\mathtt{min\_child\_size}$;
6. $\mathtt{depth}<\mathtt{max\_depth}$.

The implementation should combine the first two guards as

$$|B|\geq\max(\mathtt{min\_parent\_size},2\,\mathtt{min\_child\_size}).$$

Defaults are `eta = 0.05`, `min_parent_size = 10`, `min_child_size = 5`, `max_depth = 20`, and `eps = 1e-12`. Ground-truth labels must never determine splitting.

## 4.5 Deterministic internal 2-means

The granular-ball candidate splitter is fixed as:

```python
KMeans(n_clusters=2, n_init=10, max_iter=100, random_state=0)
```

Its `random_state` remains 0 and does not follow the final experiment seed. This freezes granularization structure so repeated-seed experiments mainly expose final-KMeans initialization variation.

## 4.6 Boundary-aware granular position

For sample $x_i$ and ball $GB_j$, define

$$d_{ij}=\|x_i-c_j\|_2,$$

$$\delta_{ij}=\frac{d_{ij}-r_j}{r_j+\varepsilon}.$$

Thus $\delta<0$ denotes a sample inside the characteristic granular radius, $\delta=0$ the characteristic granular boundary, and $\delta>0$ outside the characteristic granular radius. It is a normalized radial position relative to the typical scale of local granular structure.

## 4.7 Boundary relation

For a normal radius $r_j>\mathtt{radius\_eps}$,

$$R_{ij}=\frac{1}{1+\exp(\delta_{ij}/\tau)}=\operatorname{sigmoid}(-\delta_{ij}/\tau).$$

The default is `tau = 0.5`. It follows that $\delta<0\Rightarrow R>0.5$, $\delta=0\Rightarrow R=0.5$, and $\delta>0\Rightarrow R<0.5$. Implementations must use a numerically stable sigmoid and avoid exponential overflow.

## 4.8 Zero-radius and degenerate granular balls

If $r_j\leq\mathtt{radius\_eps}$, all or nearly all ball members coincide and ordinary $\delta$ semantics are unreliable. Define

$$R_{ij}=\begin{cases}
1.0,&d_{ij}\leq\mathtt{radius\_eps},\\
\mathtt{relation\_floor},&d_{ij}>\mathtt{radius\_eps}.
\end{cases}$$

Defaults are `radius_eps = 1e-12` and `relation_floor = 1e-12`. A coincident member must not be handled as $\delta=0\Rightarrow R=0.5$. Its radial standard deviation is normally also near zero, so reliability $q_j$ may approach 1.

## 4.9 Granular-ball reliability

Do not use `mean(distance) / radius` as compactness because the radius is the mean distance and the ratio is identically one. For member radial distances $d_{kj}=\|x_k-c_j\|_2$, define the **population radial standard deviation**

$$\sigma_j=\sqrt{\frac{1}{n_j}\sum_{x_k\in GB_j}(d_{kj}-r_j)^2}.$$

This strictly means `ddof = 0`, or `np.std(distances, ddof=0)`, never the Pandas default `ddof=1`. Then

$$CV_j=\frac{\sigma_j}{r_j+\varepsilon}, \qquad q_j=\exp(-CV_j),$$

where theoretically $0<q_j\leq1$. If $r_j\leq\mathtt{radius\_eps}$ and $\sigma_j\leq\mathtt{radius\_eps}$, set $CV_j=0$ and $q_j=1$. The corrected relation is

$$\widetilde R_{ij}=q_jR_{ij}.$$

## 4.10 Strict probability normalization

The v2.0 expression $\widetilde R_{ij}/(\sum_j\widetilde R_{ij}+\varepsilon)$ is retired as a formal definition because its sum is strictly below one. Version 2.0.1 defines

$$p_{ij}=\frac{\widetilde R_{ij}+\varepsilon}{\sum_{l=1}^{m_v}\widetilde R_{il}+m_v\varepsilon}.$$

It must satisfy $\sum_jp_{ij}=1$, allowing only floating-point error $|\sum_jp_{ij}-1|<10^{-10}$. Since $p_{ij}>0$, entropy uses $\log(p_{ij})$ directly without `log(p + eps)`.

## 4.11 Granular-relation entropy

For $m_v>1$, define

$$H_i^{(v)}=-\frac{\sum_{j=1}^{m_v}p_{ij}^{(v)}\log p_{ij}^{(v)}}{\log(m_v)}.$$

For $m_v=1$, set $H_i^{(v)}=0$. The theoretical range is $0\leq H_i^{(v)}\leq1$. Implementations may clip small floating-point drift into $[0,1]$, but a material excursion is a bug and must not be hidden by clipping.

$H$ is not ground-truth correctness. It is granular-relation uncertainty, or a structural-certainty surrogate: lower entropy means concentrated relations and greater structural certainty; higher entropy means uniform relations and greater ambiguity.

## 4.12 Sample-level view weights

Define

$$\operatorname{score}_i^{(v)}=1-H_i^{(v)}+\varepsilon,$$

$$\alpha_i^{(v)}=\frac{\operatorname{score}_i^{(v)}}{\sum_{l=1}^{V}\operatorname{score}_i^{(l)}}.$$

The weights must satisfy $\alpha_i^{(v)}\geq0$ and $\sum_v\alpha_i^{(v)}=1$. If every view has $H_i^{(v)}=1$, all scores equal $\varepsilon$ and the weights automatically become $\alpha_i^{(v)}=1/V$.

## 4.13 Relation-block normalization

For every sample/view block,

$$\overline R_i^{(v)}=\frac{\widetilde R_i^{(v)}}{\|\widetilde R_i^{(v)}\|_2+\varepsilon}.$$

If a block is theoretically all zero, the implementation must use a stable fallback and must not produce NaN.

## 4.14 Multi-view fusion

Define

$$F_i=\left[\sqrt{\alpha_i^{(1)}}\overline R_i^{(1)},\ldots,\sqrt{\alpha_i^{(V)}}\overline R_i^{(V)}\right].$$

Use $\sqrt{\alpha}$ rather than $\alpha$: once blocks are L2-normalized, this makes each view's contribution to squared Euclidean distance approximately linear in $\alpha$. After concatenation, L2-normalize each $F_i$ once more as a numerical safeguard.

## 4.15 Final KMeans

Use

```python
KMeans(
    n_clusters=K,
    n_init=20,
    max_iter=300,
    random_state=seed,
)
```

Final KMeans uses the experiment `seed`; internal split 2-means always uses `random_state=0`. `K` must come from `configs/datasets.yaml` or another pre-frozen registry. `K = len(unique(y))` and every other form of label-derived $K$ are strictly forbidden.

## Excluded methods

Version 2.0.1 does not add PCA, spectral clustering, graph learning, contrastive learning, multi-granularity, hyperspheres, attention, tensors, pseudo-labels, or additional deep models.
