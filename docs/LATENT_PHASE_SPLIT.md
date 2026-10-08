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
