# SO-101 recording guide: minimal smoke-test dataset

This is a checklist for recording a small SO-101 dataset good enough to prove the
`lerobot-policy-turbovla-so101` training/inference pipeline actually works end-to-end. It is
deliberately scoped as a **smoke test, not a benchmark** — the goal is "does the loop run and does
loss go down," not "what's the real success rate." Do this first, before spending a full recording
session on a large dataset.

It exists because an earlier SmolVLA dataset recorded on the same rig trained badly, and inspecting
the actual footage (not just the training config) turned up real, fixable problems. See
[Diagnosing a bad dataset](#diagnosing-a-bad-dataset-worked-example) below for the full worked
example.

## Camera setup

- **2 cameras**: a front/top overview + a wrist close-up. The wrist camera in the original dataset
  was already good (sharp, well-framed, no contamination) — keep that framing.
- **Front/top camera — the one that actually broke last time**:
  - Mount it **right-side up**. Check the live preview, not just "it's plugged in."
  - Position it so it **physically cannot see your hand or the leader arm**, even when you lean in
    to reset the scene between episodes. This was the single worst problem in the old dataset: a
    hand was visible at the frame edge through entire episodes, not just at boundaries — a signal
    present during every recording but categorically absent during autonomous rollout, which is
    exactly the kind of spurious correlation vision-based imitation learning latches onto.
  - Don't worry about matching `image_size=224` at capture time — the policy resizes internally.
    Whatever native resolution your camera gives (640x480 is fine) works.
- **FPS**: 30. Matches the default `chunk_size=12` (→ 0.4s chunks).

## Object and task

- Same object every episode for the smoke test — no need for color/object diversity yet.
- **One full imperative task string**, identical across all episodes — e.g. `"pick up the blue
  cup"`. Not `"Pick colored block"`: a telegraphic, ungrammatical instruction is out-of-distribution
  for the pretrained language backbone relative to what it saw in pretraining. Full sentences,
  every time.
- Keep the background static and uncluttered if easy. Not fatal for a single fixed setup, but it's
  one less variable.

## Episode count and length

- **20-30 episodes.** Enough to prove the pipeline works for a single task; not enough to claim a
  real success rate. Do a second, larger recording pass (80-150+ episodes) only after this one
  trains cleanly.
- **Trim tight**: start recording at the reach, stop right after grasp+lift. Target **60-120 frames
  (2-4s)** per episode. The earlier dataset averaged ~190 frames/episode with a lot of idle padding.
- Watch teleop speed during transit moves specifically — the earlier front camera showed real motion
  blur during fast segments in some episodes (not all; it was intermittent, localized to the fastest
  moves), while slow/careful segments were sharp throughout.

## Recording command

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.cameras='{"front_top": {...}, "wrist": {...}}' \
  --dataset.repo_id=${HF_USER}/turbovla_smoketest \
  --dataset.num_episodes=25 \
  --dataset.single_task="pick up the blue cup" \
  --dataset.fps=30
```

Adjust the camera config keys to your actual camera identifiers/indices.

## Before you train: check the footage, not just the config

Costs a few minutes, saves a wasted training run. Pull a handful of frames spread across several
episodes and actually look at them — don't assume "it recorded" means "it recorded correctly."

```bash
# Requires ffmpeg/ffprobe. Point REPO_ID at your new dataset.
REPO_ID="your-username/turbovla_smoketest"
CAMERA="observation.images.front_top"   # check every camera, not just one
OUT=/tmp/dataset_check
mkdir -p "$OUT"

curl -sL "https://huggingface.co/datasets/${REPO_ID}/resolve/main/meta/episodes/chunk-000/file-000.parquet" -o "$OUT/episodes.parquet"
curl -sL "https://huggingface.co/datasets/${REPO_ID}/resolve/main/videos/${CAMERA}/chunk-000/file-000.mp4" -o "$OUT/video.mp4"

# List episode lengths — look for suspiciously short ones (aborted takes) or wild outliers.
python3 -c "
import pyarrow.parquet as pq
t = pq.read_table('$OUT/episodes.parquet')
for r in t.to_pylist():
    print(r['episode_index'], 'len', r['length'])
"

# Grab 3 frames (start / mid / end) from a few episodes spread across the file and actually view them.
for t in 2 5 8; do
  ffmpeg -y -ss $t -i "$OUT/video.mp4" -frames:v 1 -q:v 2 "$OUT/frame_${t}.jpg" -loglevel error
done
```

Check every frame you pull for: correct orientation, no hand/leader-arm in frame, no motion blur
during the segment you sampled, object clearly visible. If anything's off, fix the rig and
re-record — it's much cheaper than diagnosing it after a failed training run.

## Diagnosing a bad dataset (worked example)

An earlier SmolVLA dataset (`pick_block_20260727_165601`, 375 episodes) trained badly. Downloading
the actual parquet metadata and video files (not just reading the training config) found, in order
of severity:

1. **A human hand persistently visible** at the front camera's frame edge, confirmed across an
   entire episode's timestamp span (not just at reset boundaries) — the most likely dominant cause,
   since it's a signal present in 100% of training data and 0% of inference.
2. **Front camera mounted upside-down** — every frame, confirmed via readable-but-inverted text in
   frame. Not fatal alone, but a sign the mount was unstable.
3. **Intermittent motion blur** during fast transit moves in some episodes, confirmed by pulling
   frames every ~1.5s across an episode's span — present in roughly 1-3 second bursts during the
   fastest segments, absent in slower episodes and the careful/precise parts of the same episode.
   Initially mis-reported as "always blurry" from a couple of arbitrarily-timestamped frames in the
   raw concatenated video file — sampling systematically across multiple full episodes corrected
   that to "real, but localized to fast segments."
4. **5 episodes at exactly 11 frames** (0.37s) — aborted/false-start recordings mixed into the
   training set.
5. **Telegraphic task string** (`"Pick colored block"` — no verb form, no article) — real but likely
   secondary for a single-task setup with nothing to disambiguate.

The action/state statistics themselves were clean (real joint ranges, no degenerate/constant
dimensions) — the problem was in the footage, which stats alone don't reveal. This is the reason the
checklist above exists: check the actual images before training, every time.
