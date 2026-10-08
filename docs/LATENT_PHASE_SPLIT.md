# Latent-space phase split: how-to

This splits a single-arm RLBench demonstration into phases using only camera images. Each frame is embedded with a
frozen image encoder, and the method looks for where the sequence of embeddings changes character. No labels,
object detection or robot state are used to find the boundaries. Robot state is used only afterwards, as a
reference.

| Piece | File |
|---|---|
| Encoder + change-point maths | `src/entry_vla/perception/latent_split.py` |
| Batch runner (embed, split, compare, phase videos) | `scripts/latent_phase_split.py` |
| t-SNE plots | `scripts/plot_latent_tsne.py` |
| Unit tests (synthetic trajectories) | `tests/test_latent_split.py` |

Environments:
- generation runs in `peract2` (CoppeliaSim);
- everything else runs in `entryvla`, which needs torch, transformers, scikit-learn, matplotlib, opencv and an
  `ffmpeg` binary.

## Step 0: Generate demonstrations (peract2 env)

```bash
source scripts/RLBench_env.sh
python scripts/rlbench_single_generator.py --task push_button --variation 4 --episodes 20 \
  --seed 42 --image_size 256 256 --save_path data/raw/rlbench_single_seed42
```

Output:
- `data/raw/rlbench_single_seed42/push_button/variation4/episodes/episode<N>/`;
- in each episode: `<cam>_rgb/`, `<cam>_depth/` and `<cam>_mask/` for the cameras `front`, `over_shoulder_left`,
  `over_shoulder_right`, `overhead` and `wrist`, plus `low_dim_obs.pkl`.

## Step 1 (optional): State-based references

The split is unsupervised. To judge it, compute reference frames from the recorded robot state:

```bash
conda activate entryvla
for ep in data/raw/rlbench_single_seed42/push_button/variation4/episodes/episode*; do
  python scripts/detect_transition_rlbench.py --episode $ep \
    --out data/entry_process/push_button_seed42/variation4/$(basename $ep) --no-video
done
```

This writes `transition.json` for each episode:
- `entry_frame`: the arm stopped above the target, a PerAct-style keyframe;
- `contact_frame`: the object first moves or a touch force appears.

`perception/comparison.json` is used too when it exists; it comes from `scripts/perception_phase_split.py`.

These are heuristic references, **not human phase labels**. RLBench stores no phase labels.

## Step 2: Embed every frame

`FrameEncoder` (default `facebook/dinov2-small`) maps each RGB image to the CLS token of the last layer,
L2-normalised: one 384-d vector per frame per camera. This is done for all 5 cameras and cached as
`episode<N>/latent/embeddings_<cam>.npy`, so later runs skip it. `--force` recomputes.

The encoder is frozen. `--encoder` takes any `transformers` vision model with a CLS token (e.g.
`facebook/dinov2-base`).

## Step 3: Build the latent trajectory per camera set

| View set | Cameras |
|---|---|
| `exo` | front + over_shoulder_left |
| `wrist` | wrist |
| `all` | all 5 |

The per-camera embeddings are concatenated, then each dimension is z-scored **within the episode**
(`standardise`). That removes the episode's static appearance and leaves how the view changes over time. The
episode becomes a trajectory `X` with shape `(T, D)`.

## Step 4: Find change points

Three detectors run on `X`:

- **`k2`: one boundary.** Exact least-squares segmentation into 2 contiguous segments. It minimises the summed
  squared distance of each frame to its segment mean (`optimal_segmentation`: dynamic programming over a
  cumulative-sum cost table, exact, O(K·T²)).
- **`k3`: two boundaries.** The same, with 3 segments.
- **`novelty`.** For each frame t, the cosine distance between the mean embedding of frames `[t-w, t)` and of
  `[t, t+w)` (default `w=5`). The strongest peak is the boundary, and the whole curve is saved.

`--k3-min-len` sets the minimum segment length (5 frames). Boundaries are the first frame of the new segment.

## Algorithm in detail

Notation:
- an episode has $T$ frames $t = 0,\dots,T-1$;
- $I_t^c$ is the RGB image of camera $c$ at frame $t$;
- a view set is an ordered list of cameras $\mathcal{C} = (c_1,\dots,c_M)$.

### 1. Frame embedding

The frozen encoder $f$ (DINOv2-small, a ViT-S/14) maps an image to a sequence of tokens; we keep the CLS token
$h^{\text{CLS}}\in\mathbb{R}^{384}$ and L2-normalise it:

$$
z_t^{c} = \frac{h^{\text{CLS}}\!\left(f(I_t^{c})\right)}{\left\lVert h^{\text{CLS}}\!\left(f(I_t^{c})\right)\right\rVert_2}
\in \mathbb{R}^{384}, \qquad \lVert z_t^c\rVert_2 = 1 .
$$

The view-set feature concatenates the cameras: $x_t = \left[z_t^{c_1};\dots;z_t^{c_M}\right] \in \mathbb{R}^{D}$,
$D = 384M$.

### 2. Per-episode standardisation

Each dimension $d$ is z-scored over the frames of *this* episode only (no labels, nothing shared between
episodes):

$$
\mu_d = \frac{1}{T}\sum_{t=0}^{T-1} x_{t,d}, \qquad
\sigma_d = \sqrt{\frac{1}{T}\sum_{t=0}^{T-1}\left(x_{t,d}-\mu_d\right)^2}, \qquad
\tilde{x}_{t,d} = \frac{x_{t,d}-\mu_d}{\max(\sigma_d,\,10^{-8})} .
$$

This removes the episode's constant appearance (table texture, lighting) and puts every feature on the same scale,
so the segmentation responds to how the view *changes*. Dimensions that barely move are amplified as well, which
can add noise.

### 3. Segment cost

For a candidate segment of frames $[i, j)$ ($0 \le i < j \le T$), with mean
$m_{ij} = \frac{1}{j-i}\sum_{t=i}^{j-1}\tilde{x}_t$, the cost is its within-segment sum of squared errors:

$$
C(i,j) = \sum_{t=i}^{j-1}\left\lVert \tilde{x}_t - m_{ij}\right\rVert_2^2 .
$$

With the prefix sums $S_1(j)=\sum_{t<j}\tilde{x}_t \in\mathbb{R}^D$ and $S_2(j)=\sum_{t<j}\lVert\tilde{x}_t\rVert_2^2$
(with $S_1(0)=0$, $S_2(0)=0$), every cost takes $O(D)$ time:

$$
C(i,j) = \big(S_2(j)-S_2(i)\big) - \frac{\left\lVert S_1(j)-S_1(i)\right\rVert_2^2}{j-i} .
$$

`segment_cost_table` fills the whole $(T+1)\times(T+1)$ table in $O(T^2 D)$.

### 4. Optimal K-segment split (`k2`, `k3`)

We look for boundaries $0=b_0<b_1<\dots<b_{K-1}<b_K=T$, each segment at least $L$ frames long
($b_k-b_{k-1}\ge L$), that minimise the total within-segment error:

$$
\left(\hat b_1,\dots,\hat b_{K-1}\right) = \arg\min_{b_1<\dots<b_{K-1}} \sum_{k=1}^{K} C(b_{k-1}, b_k) .
$$

This is a piecewise-constant approximation of the latent trajectory: each phase is summarised by one mean
embedding, and the boundaries are where switching the mean reduces the error most. It is solved **exactly** by
dynamic programming over $D_s(j)$, the best cost of splitting frames $[0,j)$ into $s$ segments:

$$
D_0(0) = 0, \qquad
D_s(j) = \min_{\,(s-1)L \,\le\, i \,\le\, j-L} \Big[ D_{s-1}(i) + C(i,j) \Big], \qquad s = 1,\dots,K ,
$$

storing the minimising $i$ as a back-pointer. The boundaries are then read off by backtracking from $D_K(T)$.

- **Cost.** $O(K\,T^2)$ table lookups. For $T\approx 100$ and $K=3$ this is negligible next to the encoder.
- **Output.** The code reports $\hat b_1,\dots,\hat b_{K-1}$, each the first frame of a new segment.
- **Minimum length.** $L=3$ for `k2`; $L$ = `--k3-min-len` (default 5) for `k3`.

**What K = 2 maximises.** For a single boundary $b$, the reduction in error equals the between-segment term

$$
C(0,T) - \big[C(0,b) + C(b,T)\big] = \frac{b\,(T-b)}{T}\,\left\lVert m_{0b} - m_{bT}\right\rVert_2^2 ,
$$

so `k2` picks the cut where the means of "before" and "after" differ most, weighted by $b(T-b)/T$. That weight
favours cuts near the middle, which is one reason `exo k2` lands mid-travel.

**Why the wrist `k3` boundary falls at the stop.** Once the arm stops above the button, $\tilde{x}_t$ barely
changes until the press ends. That segment has a tiny $C(\cdot,\cdot)$, so the optimal 3-segment solution puts a
boundary exactly where the near-static segment begins.

### 5. Novelty curve (`novelty`)

With window $w$ (default 5), compare the mean embedding of the $w$ frames before and after each frame $t$:

$$
a_t = \frac{1}{w}\sum_{\tau=t-w}^{t-1}\tilde{x}_\tau, \qquad
c_t = \frac{1}{w}\sum_{\tau=t}^{t+w-1}\tilde{x}_\tau, \qquad
n_t = 1 - \frac{a_t^\top c_t}{\lVert a_t\rVert_2\,\lVert c_t\rVert_2}
\quad (w \le t \le T-w), \qquad n_t = 0 \text{ otherwise} .
$$

$n_t\in[0,2]$ is large when the view right after $t$ points in a different latent direction from the view right
before it. `novelty_peaks` takes the largest values of $n_t$, at least `min_gap` = 5 frames apart, and the
`novelty` boundary is the strongest one: $\hat b = \arg\max_t n_t$.

Unlike the DP, this is local. It finds the sharpest change within $\pm w$ frames, but ignores how long each phase
lasts, so a brief camera jolt can beat a real phase change.

### 6. Evaluation against references

For a detector boundary $\hat b$ and a reference frame $b^{\text{ref}}$ (stop, contact or perception split), the
signed error is $e = \hat b - b^{\text{ref}}$ (negative = early). Over the $N$ episodes where both exist,
`latent_summary.json` reports:

$$
\bar e = \tfrac{1}{N}\textstyle\sum e_n, \quad
\overline{|e|} = \tfrac{1}{N}\textstyle\sum |e_n|, \quad
\operatorname{median}|e|, \quad
\Pr(|e|\le 5) = \tfrac{1}{N}\textstyle\sum \mathbb{1}[|e_n|\le 5] .
$$

### 7. t-SNE visualisation

t-SNE embeds $\{\tilde{x}_t\}$ in 2-D only for display; it plays no part in the split.

1. **Neighbour probabilities in latent space.** Each frame gets Gaussian neighbour probabilities
   $p_{j|i} \propto \exp\!\left(-\lVert\tilde{x}_i-\tilde{x}_j\rVert^2 / 2\sigma_i^2\right)$, with each $\sigma_i$
   set so the perplexity is $\text{Perp} = \min\!\big(30,\ \max(5, \lfloor (T-1)/3\rfloor)\big)$. They are
   symmetrised as $p_{ij} = (p_{j|i}+p_{i|j})/2T$.
2. **Neighbour probabilities in 2-D.** Points $y_t\in\mathbb{R}^2$ use a heavy-tailed kernel
   $q_{ij} \propto \left(1+\lVert y_i-y_j\rVert^2\right)^{-1}$.
3. **Fit.** The $y_t$ minimise $\mathrm{KL}(P\Vert Q) = \sum_{i\ne j} p_{ij}\log\frac{p_{ij}}{q_{ij}}$, starting
   from PCA with a fixed random seed.

Because the KL term preserves neighbourhoods, frames that look alike stay together. Cluster sizes and the gaps
between clusters are *not* metric. A phase that forms its own cluster with a long jump into it (wrist phase 3) is
a qualitative sign that the boundary separates genuinely different views.

## Step 5: Run it (background)

```bash
nohup python scripts/latent_phase_split.py \
  --episodes-root data/raw/rlbench_single_seed42/push_button/variation4/episodes \
  --out data/entry_process/push_button_seed42/variation4 \
  > data/entry_process/push_button_seed42/variation4/latent_run.log 2>&1 &
tail -f data/entry_process/push_button_seed42/variation4/latent_run.log
```

Speed on the RTX 4090: about 7 s to load the encoder, then about 1 s per episode.

Outputs:
- `episode<N>/latent/latent_split.json`: the boundaries for every view set and detector, the novelty curves and the
  references.
- `episode<N>/latent/phase<i>_<start>-<end>.mp4`: one video per phase, all 5 cameras with RGB above depth, cut at
  `--video-split` (default `wrist:k3`). Use `exo:k2`, `all:novelty`, … to change it, or `none` to skip videos.
- `latent_summary.csv`: one row per episode.
- `latent_summary.json`: error of each detector against each reference (mean, |mean|, median, share within 5
  frames).

## Step 6: Visualise clusters with t-SNE (background)

```bash
nohup python scripts/plot_latent_tsne.py --out data/entry_process/push_button_seed42/variation4 \
  --view wrist --method k3 > data/entry_process/push_button_seed42/variation4/tsne_run.log 2>&1 &
```

Outputs:
- `episode<N>/latent/tsne_wrist_k3.png`: each frame is a point coloured by phase. Clusters are named
  "phase i: frame a → frame b", a grey line shows time order, and a ring marks the first frame.
- `latent_tsne_overview_wrist_k3.png`: all episodes as small multiples.

t-SNE preserves neighbourhoods, not distances. Read it as "do the phases form separate groups", not "how far apart
are they". `--seed` fixes the layout.

## Step 7: Read the result

On push_button variation 4 (20 episodes, seed 42), compared with the stop frame:

| Detector | Result |
|---|---|
| `wrist k3`, 2nd boundary | **exact in 17/20**, otherwise 11–19 frames early |
| `wrist novelty` | exact in 12/20, misses up to 61 frames |
| `wrist k2` | mean 15 frames early |
| `exo` / `all` `k2` | 27–30 frames early: a mid-travel cut when the arm enters the view |

The wrist k3 segments correspond to:
1. far travel;
2. the final approach, as the button grows in the wrist view;
3. stopped above the button and pressing, an almost static wrist view.

In t-SNE, phase 3 forms a separate cluster with a jump to it in nearly every episode. Phases 1 and 2 form one
continuous path.

Per-run details are in `data/entry_process/push_button_seed42/variation4/LATENT_SPLIT.md`.

## Limits and choices to revisit

- **K is fixed** (2 or 3). Tasks with several skills need a larger K or a penalty-based choice of K, which is not
  implemented.
- **The wrist result depends on a static final segment.** Skills that keep moving after the approach (carry, pour,
  wipe) will not show one.
- **Encoder choice.** DINOv2 CLS is a general visual feature. Robot-specific encoders (R3M, VC-1) or the VLA's own
  vision tower may separate phases differently.
- **Validation.** Boundaries are checked only against heuristic state references from one task. Agreement on more
  tasks, with manual labels, is needed before using them as training labels.
