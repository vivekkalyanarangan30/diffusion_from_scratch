# Variational Autoencoder Explainer Voiceover

This transcript is synchronized with the `manim_vae_explainer/scenes.py` scenes
and totals roughly twenty minutes of narration when rendered with GTTS at the
configured pace.  The content assumes the audience already knows how standard
autoencoders work.

## Scene 1 – Autoencoder Limitations (approx. 4 minutes)
- Introduce the classical encoder–decoder pipeline.
- Emphasize deterministic latent codes and the issue of decoding unseen regions.
- Illustrate failure cases and motivate probabilistic latents.

## Scene 2 – KL Divergence Intuition (approx. 3 minutes)
- Compare two Gaussian curves.
- Explain "directional distance" intuition and overlap penalty.

## Scene 3 – Deriving the ELBO (approx. 4 minutes)
- Start from log-likelihood integral.
- Insert approximate posterior and apply Jensen's inequality.
- Highlight reconstruction term vs. KL regularizer.

## Scene 4 – Latent Space Geometry (approx. 3 minutes)
- Show clusters blending into a smooth manifold.
- Demonstrate interpolation semantics.

## Scene 5 – Reparameterization Trick (approx. 3 minutes)
- Discuss gradient issues with sampling.
- Introduce epsilon trick and differentiable path.

## Scene 6 – Training Loop (approx. 2 minutes)
- Walk through iterative ELBO optimization.
- Mention β-VAE trade-off.

## Scene 7 – Applications and Outlook (approx. 1 minute)
- Highlight downstream uses and future directions.
- End with an inspirational call to explore.

Each bullet corresponds to the narration strings inside the scene definitions.
During production you can refine timing by adjusting `speech_rate` or adding
pauses within each scene's `voiceover` context manager.
