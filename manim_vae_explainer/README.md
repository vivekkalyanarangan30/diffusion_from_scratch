# Manim Variational Autoencoder Explainer

This directory contains scenes and narration assets for a 20-minute, voiceover
-driven explainer video on variational autoencoders.  The animation is designed
for a learner already comfortable with classical autoencoders and mirrors the
visual pacing of a professional "3Blue1Brown" production.

## Contents

- `scenes.py` – Manim scenes implemented with [manim-voiceover](https://github.com/Mathigon/manim-voiceover).
- `voiceover_script.md` – High-level narration outline and timing guidance.

## Rendering

1. Install dependencies:
   ```bash
   pip install manim manim-voiceover[all]
   ```
2. Render a specific scene:
   ```bash
   manim -pqh manim_vae_explainer/scenes.py AutoencoderProblemScene
   ```
3. Render the entire sequence by invoking each scene, or write a shell script to
   iterate through `all_scenes()`.

The scenes are configured to use the Google Text-to-Speech service provided by
`manim-voiceover`.  Ensure you have network access during rendering or swap in a
local recorder implementation.
