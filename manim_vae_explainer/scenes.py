"""Manim scenes for a 20-minute VAE explainer.

This module uses manim-voiceover to create narrated scenes that guide the viewer
from the shortcomings of deterministic autoencoders to the full variational
autoencoder framework.  Each scene is designed to flow like a professional
"3Blue1Brown" production, with layered animations, camera movement, and a rich
visual experience.
"""
from __future__ import annotations

from typing import Iterable, List

from manim import (
    BLUE,
    BLUE_E,
    GREEN,
    GREEN_E,
    GREY_BROWN,
    GREY_E,
    Lighten,
    ORANGE,
    RED,
    TEAL,
    WHITE,
    YELLOW,
    Circle,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    Group,
    LaggedStart,
    Line,
    MathTex,
    Mobject,
    Polygon,
    Rectangle,
    ReplacementTransform,
    Scene,
    Square,
    Tex,
    Transform,
    VGroup,
    Write,
)
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class RichVoiceoverScene(VoiceoverScene):
    """Base class that configures GTTS narration and helper utilities."""

    speech_rate: float = 0.98

    def construct(self) -> None:  # pragma: no cover - Manim scene entry point
        self.set_speech_service(
            GTTSService(lang="en", tld="com", speed=self.speech_rate)
        )
        self.setup_scene()

    def setup_scene(self) -> None:
        raise NotImplementedError

    # Helper methods -----------------------------------------------------
    def add_title(self, text: str, color=WHITE) -> Tex:
        title = Tex(text, color=color).to_edge(UP)
        self.play(FadeIn(title, shift=0.5 * UP))
        return title

    def narration(self, text: str, animation) -> None:
        with self.voiceover(text=text):
            self.play(animation)


class AutoencoderProblemScene(RichVoiceoverScene):
    """Introduces the limitations of standard autoencoders."""

    speech_rate = 0.95

    def setup_scene(self) -> None:
        title = self.add_title("Why Autoencoders Struggle")

        encoder = Rectangle(width=3, height=2, color=BLUE_E).shift(3 * LEFT)
        decoder = Rectangle(width=3, height=2, color=GREEN_E).shift(3 * RIGHT)
        bottleneck = Square(side_length=1.3, color=YELLOW)

        encoder_label = Tex("Encoder").move_to(encoder)
        decoder_label = Tex("Decoder").move_to(decoder)
        latent_label = Tex("Latent Code", color=YELLOW).next_to(bottleneck, DOWN)

        data_cloud = VGroup(
            *[Dot(color=BLUE).shift(0.4 * RIGHT * i + 0.3 * UP * j)
              for i, j in [(-2, 1), (-1.5, -0.5), (-0.5, 0.5), (0, -1), (0.8, 1.2)]]
        ).shift(4 * LEFT)

        with self.voiceover(
            "Autoencoders compress data into a latent code and try to reconstruct "
            "the original input."
        ):
            self.play(LaggedStart(
                FadeIn(encoder), FadeIn(decoder), FadeIn(bottleneck),
                FadeIn(encoder_label), FadeIn(decoder_label), FadeIn(latent_label),
                FadeIn(data_cloud)
            ))

        arrow1 = Line(data_cloud.get_right(), encoder.get_left(), buff=0.3)
        arrow2 = Line(bottleneck.get_right(), decoder.get_left(), buff=0.3)
        arrow3 = Line(decoder.get_right(), data_cloud.get_right() + 5 * RIGHT)

        with self.voiceover(
            "But deterministic codes squeeze every input into a single point. "
            "If the data has regions we never saw, the decoder has no clue what to do."
        ):
            self.play(LaggedStart(FadeIn(arrow1), FadeIn(arrow2), FadeIn(arrow3)))

        missing_data = VGroup(
            *[Dot(color=RED).shift(4 * RIGHT + 0.5 * RIGHT * i + 0.4 * UP * j)
              for i, j in [(-1, 1), (0.3, -0.8), (1.2, 0.6)]]
        )
        cross = Polygon(
            missing_data.get_left() + 0.4 * UP,
            missing_data.get_right() + 0.4 * UP,
            missing_data.get_right() + 0.4 * DOWN,
            missing_data.get_left() + 0.4 * DOWN,
            color=RED,
        )

        with self.voiceover(
            "As a result, latent space arithmetic is brittle. Neighboring points "
            "may decode to unrelated samples."
        ):
            self.play(FadeIn(missing_data), Write(cross))

        blur = Rectangle(width=14, height=8, color=GREY_E, fill_opacity=0.3)
        with self.voiceover(
            "Variational autoencoders fix this by making the latent space probabilistic."
        ):
            self.play(FadeIn(blur), FadeOut(title))
        self.play(*map(FadeOut, [encoder, decoder, bottleneck, encoder_label,
                                 decoder_label, latent_label, data_cloud,
                                 arrow1, arrow2, arrow3, missing_data, cross, blur]))


class KLConceptScene(RichVoiceoverScene):
    """Explains KL divergence through visual intuition."""

    speech_rate = 0.97

    def gaussian(self, mean: float, color=BLUE) -> Mobject:
        curve = VGroup()
        for x in range(-5, 6):
            height = 2.5 * (1 / (1 + (x - mean) ** 2 / 4))
            dot = Dot(point=(x * 0.5, height - 2, 0), color=color)
            curve.add(dot)
        return curve

    def setup_scene(self) -> None:
        title = self.add_title("Measuring Distribution Mismatch")

        p = self.gaussian(0, color=BLUE)
        q = self.gaussian(2, color=GREEN)
        labels = VGroup(
            Tex(r"$p(z)$", color=BLUE).next_to(p, UP),
            Tex(r"$q(z)$", color=GREEN).next_to(q, UP),
        )

        with self.voiceover(
            "To make latent codes smooth, we compare distributions. The Kullback--"
            "Leibler divergence measures how surprised one distribution is when "
            "seeing samples from another."
        ):
            self.play(LaggedStart(FadeIn(p), FadeIn(q), FadeIn(labels)))

        area = Polygon(
            *[dot.get_center() for dot in p],
            *[dot.get_center() for dot in reversed(q)],
            color=ORANGE,
            fill_opacity=0.3,
        )
        with self.voiceover(
            "If q places mass where p does not, the overlap shrinks, and the KL "
            "penalty grows."
        ):
            self.play(FadeIn(area))

        arrow = Line(p.get_center(), q.get_center(), color=ORANGE)
        with self.voiceover(
            "We can think of KL divergence as a directional distance—zero only when "
            "the two distributions match everywhere."
        ):
            self.play(Write(arrow))

        self.play(*map(FadeOut, [title, p, q, labels, area, arrow]))


class ELBOScene(RichVoiceoverScene):
    """Derives the Evidence Lower Bound step by step."""

    speech_rate = 0.96

    def setup_scene(self) -> None:
        title = self.add_title("From Likelihood to the ELBO")

        equation1 = MathTex(r"\log p_\theta(x) = \log \int p_\theta(x, z) \\ \mathrm{d}z")
        equation2 = MathTex(
            r"\log p_\theta(x) = \log \mathbb{E}_{q_\phi(z \mid x)} \left["
            r"\frac{p_\theta(x, z)}{q_\phi(z \mid x)}\right]"
        )
        equation3 = MathTex(
            r"\log p_\theta(x) \geq \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]"
            r" - \text{KL}\left(q_\phi(z \mid x) \| p(z)\right)"
        )

        with self.voiceover(
            "Starting from the marginal likelihood, we insert an auxiliary distribution "
            "q and apply Jensen's inequality."
        ):
            self.play(Write(equation1))
        with self.voiceover(
            "This trick lets us express the log evidence as an expectation under q."
        ):
            self.play(ReplacementTransform(equation1, equation2))
        with self.voiceover(
            "Jensen tells us the log can move inside, producing a lower bound we can "
            "optimize."
        ):
            self.play(ReplacementTransform(equation2, equation3))

        highlight_recon = Rectangle(color=TEAL, width=8, height=1.1).surround(
            equation3[0][6:26]
        )
        highlight_kl = Rectangle(color=ORANGE, width=8, height=1.1).surround(
            equation3[0][27:]
        )

        with self.voiceover(
            "Two forces emerge: a reconstruction term pulling the decoder toward the "
            "data, and a KL regularizer pushing q toward the prior."
        ):
            self.play(FadeIn(highlight_recon), FadeIn(highlight_kl))

        self.play(*map(FadeOut, [title, equation3, highlight_recon, highlight_kl]))


class LatentSpaceScene(RichVoiceoverScene):
    """Visualizes latent space organization."""

    speech_rate = 0.94

    def setup_scene(self) -> None:
        title = self.add_title("Shaping Latent Space")

        plane = Rectangle(width=6, height=6, color=GREY_BROWN).set_fill(
            color=Lighten(GREY_BROWN, 0.7), opacity=0.3
        )
        clusters = VGroup()
        colors = [BLUE, GREEN, TEAL, ORANGE]
        centers = [LEFT + UP, RIGHT + UP, LEFT + DOWN, RIGHT + DOWN]

        for center, color in zip(centers, colors):
            cluster = VGroup(*[Dot(center + 0.4 * RIGHT * i + 0.35 * UP * j,
                                   color=color)
                               for i, j in [(-1, 0), (0, 1), (1, -1), (0.5, -0.5)]])
            clusters.add(cluster)

        with self.voiceover(
            "Instead of isolated points, VAEs sculpt a latent landscape where nearby "
            "codes decode to coherent variations."
        ):
            self.play(FadeIn(plane), FadeIn(clusters))

        manifold = Circle(radius=2.5, color=YELLOW).set_stroke(width=6)
        with self.voiceover(
            "Regularization compresses the space into a dense manifold, making "
            "interpolation smooth."
        ):
            self.play(Write(manifold))

        path = DashedLine(start=centers[0], end=centers[1], color=WHITE)
        decoded_samples = VGroup(
            Tex("cat").next_to(path.get_start(), DOWN),
            Tex("tiger").next_to(path.point_from_proportion(0.5), DOWN),
            Tex("lion").next_to(path.get_end(), DOWN),
        )
        with self.voiceover(
            "Walking through latent space creates gradual morphs between examples, "
            "even for classes never explicitly paired."
        ):
            self.play(Write(path), FadeIn(decoded_samples, shift=0.5 * DOWN))

        self.play(*map(FadeOut, [title, plane, clusters, manifold, path, decoded_samples]))


class ReparameterizationScene(RichVoiceoverScene):
    """Explains the reparameterization trick visually."""

    speech_rate = 0.95

    def setup_scene(self) -> None:
        title = self.add_title("The Reparameterization Trick")

        encoder_box = Rectangle(width=3, height=2, color=BLUE).shift(3 * LEFT)
        stats = VGroup(
            MathTex(r"\mu_\phi(x)").next_to(encoder_box, RIGHT, buff=1.2),
            MathTex(r"\sigma_\phi(x)").next_to(encoder_box, RIGHT, buff=1.2, DOWN)
        )
        epsilon_label = MathTex(r"\epsilon \sim \mathcal{N}(0, I)").shift(3 * RIGHT + UP)
        sample_formula = MathTex(r"z = \mu + \sigma \odot \epsilon").shift(3 * RIGHT + DOWN)

        with self.voiceover(
            "Gradients cannot flow through random sampling. We rewrite the draw as a "
            "deterministic function of noise."
        ):
            self.play(LaggedStart(FadeIn(encoder_box), FadeIn(stats)))

        epsilon_cloud = VGroup(
            *[Dot(3 * RIGHT + 0.4 * RIGHT * i + 0.4 * UP * j,
                 color=WHITE)
              for i, j in [(-1, 1), (0, 0), (1, -1), (1, 1), (-1, -1)]]
        )

        with self.voiceover(
            "We sample epsilon from a fixed standard Gaussian, detached from the data."
        ):
            self.play(FadeIn(epsilon_label), FadeIn(epsilon_cloud))

        arrow = Line(stats[0].get_right(), sample_formula.get_left(), color=TEAL)
        with self.voiceover(
            "Combining learned statistics with epsilon yields z, letting gradients flow."
        ):
            self.play(FadeIn(sample_formula), Write(arrow))

        backprop = Tex("Backprop OK!", color=GREEN).next_to(sample_formula, DOWN)
        with self.voiceover(
            "Now the entire computation graph is differentiable."
        ):
            self.play(FadeIn(backprop, shift=0.5 * UP))

        self.play(*map(FadeOut, [title, encoder_box, stats, epsilon_label, epsilon_cloud,
                                 sample_formula, arrow, backprop]))


class TrainingLoopScene(RichVoiceoverScene):
    """Shows the iterative optimization of the ELBO."""

    speech_rate = 0.95

    def setup_scene(self) -> None:
        title = self.add_title("Training the VAE")

        steps = VGroup(
            Tex("Sample batch x").to_edge(LEFT).shift(2 * UP),
            Tex("Encode to μ, σ").next_to(title, DOWN, buff=1.2),
            Tex("Sample z = μ + σ ⊙ ε").shift(2 * DOWN),
            Tex("Decode and compute loss").to_edge(RIGHT).shift(0.5 * DOWN),
        )

        arrows = VGroup(
            Line(steps[0].get_right(), steps[1].get_left(), color=TEAL),
            Line(steps[1].get_bottom(), steps[2].get_top(), color=TEAL),
            Line(steps[2].get_right(), steps[3].get_left(), color=TEAL),
            Line(steps[3].get_top(), steps[0].get_right() + 0.1 * UP, color=TEAL),
        )

        with self.voiceover(
            "Each iteration encodes a batch, samples latent variables, decodes, and "
            "optimizes the ELBO."
        ):
            self.play(LaggedStart(FadeIn(steps), FadeIn(arrows)))

        loss_eq = MathTex(
            r"\mathcal{L}(x) = -\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]"
            r" + \beta \cdot \text{KL}(q_\phi(z \mid x) \| p(z))"
        ).shift(2.5 * DOWN)

        with self.voiceover(
            "We can even tune a beta parameter to control the trade-off between "
            "fidelity and disentanglement."
        ):
            self.play(Write(loss_eq))

        self.play(*map(FadeOut, [title, steps, arrows, loss_eq]))


class ApplicationsScene(RichVoiceoverScene):
    """Highlights downstream applications and extensions."""

    speech_rate = 0.97

    def setup_scene(self) -> None:
        title = self.add_title("Beyond the Basics")

        bullets = VGroup(
            Tex("Disentangled representations").to_edge(LEFT).shift(UP),
            Tex("Semi-supervised learning").to_edge(LEFT),
            Tex("Generative pipelines and diffusion").to_edge(LEFT).shift(DOWN),
        )

        with self.voiceover(
            "VAEs underpin disentanglement research, semi-supervised tasks, and even "
            "modern diffusion models."
        ):
            self.play(LaggedStart(*[FadeIn(b, shift=0.3 * RIGHT) for b in bullets]))

        outro = Tex("Keep exploring latent worlds!", color=YELLOW).shift(2 * DOWN)
        with self.voiceover(
            "With the variational toolkit, we can craft smooth, controllable latent "
            "spaces."
        ):
            self.play(FadeIn(outro))

        self.play(*map(FadeOut, [title, bullets, outro]))


def all_scenes() -> List[Scene]:
    """Return the scenes in viewing order for convenience."""

    return [
        AutoencoderProblemScene(),
        KLConceptScene(),
        ELBOScene(),
        LatentSpaceScene(),
        ReparameterizationScene(),
        TrainingLoopScene(),
        ApplicationsScene(),
    ]


__all__ = [
    "RichVoiceoverScene",
    "AutoencoderProblemScene",
    "KLConceptScene",
    "ELBOScene",
    "LatentSpaceScene",
    "ReparameterizationScene",
    "TrainingLoopScene",
    "ApplicationsScene",
    "all_scenes",
]
