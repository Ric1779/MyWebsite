---
title: "Evolution of Diffusion Model"
date: 2025-04-30T23:17:00+09:00
slug: DiffusionModelEvolution
category: DiffusionModelEvolution
tags:
  - AI
  - GenAI
  - Image-Generation
  - Diffusion-Model
  - Score-Based-Model
  - Dall-E
  - Stable-Diffusion
  - LoRA
summary:
description:
cover:
  image: "covers/diffusion_model_evolution_16_9.png"
  alt:
  caption:
  relative: true
showtoc: true
draft: false
---

## Introduction

---

In the past few years, artificial intelligence (AI) has made huge progress in creating realistic images, videos, music, and even human-like voices. A big reason behind this progress is the invention of generative models. These models are designed to create new things - like pictures of people who don’t exist or completely new songs that sound like they were made by a professional artist.

At the core of generative models is the concept of probability distributions. Consider an image with a resolution of 1024x1024 pixels, where each pixel has an RGB intensity ranging from 0 to 255. The total number of possible images in this space is finite but astronomically large. However, only a tiny subset of these images make sense as natural images - coherent scenes, recognizable objects, and meaningful patterns. {{< mathjax/inline>}}<span style="color: #0084a5;">The true distribution of natural images exists within this vast space, and generative models aim to learn and approximate this distribution either explicitly or implicitly.</span>{{< /mathjax/inline>}}

Among the different types of generative models, you might have heard about [GANs](https://lilianweng.github.io/posts/2017-08-20-gan/), [VAEs](https://lilianweng.github.io/posts/2018-08-12-vae/), and [Flow-based models](https://lilianweng.github.io/posts/2018-10-13-flow-models/) ( Lilian Weng has these amazing blog posts, which explain these models in more detail ). All of these models try to capture the underlying patterns of the data they are trained on, allowing them to generate new samples that resemble real-world data. However, each approach comes with its own challenges, such as unstable training, limited diversity in generated outputs, or difficulties in scaling to high-resolution images.

Then came a new family of models that changed the game - **Diffusion Models**.

**What Are Diffusion Models?** Diffusion models work in a very clever way. Imagine starting with a clear image and slowly adding noise to it, like static on an old TV screen. Eventually, the image becomes pure noise and loses all its details. Now, what if we could train a model to reverse this process? Starting from random noise, it could "denoise" the image step by step until a realistic picture appears. That’s exactly what diffusion models do. They learn how to carefully remove noise and rebuild the image. The result? These models can create stunningly realistic images, videos, and more. {{< mathjax/inline>}}<span style="color: #ff8459;">I know this seems completely random, but bear with me and things will get more clear</span>{{< /mathjax/inline>}}

<!-- In probability terms, instead of trying to directly learn the true distribution of natural images - which is extremely complex and intractable - diffusion models take a different route: they try to model the true distribution conditioned on some random variable, in our case it's gaussian - {{< mathjax/inline>}}<span style="color: #ff8459;">I know this seems completely random, but bear with me and things will get more clear</span>{{< /mathjax/inline>}}. This makes the problem more manageable and allows the model to recover realistic images step by step. -->

**Why Are Diffusion Models Important?** Diffusion models are important because they solve many of the problems older models struggled with:

- **More Stable Training:** Unlike GANs, diffusion models don’t need to balance two competing networks, making training more predictable and easier.
- **High-Quality Outputs:** They create very detailed and diverse results, outperforming GANs.
- **Flexibility:** These models can be used not just for generating images but also for tasks like inpainting (filling missing parts), super-resolution (making blurry images sharp), and even generating audio or 3D models.

This blog post is an attempt to trace the evolution of diffusion models - both conceptually and historically. To do that, we’ll journey back to the very beginning, to a 2015 paper that framed learning as a nonequilibrium thermodynamic process, a view that has quietly shaped the models we now know as DDPMs (Denoising Diffusion Probabilistic Models).

Along the way, we'll explore a fascinating parallel evolution in score-based modeling, and highlight how the various formulations eventually converge into a unified understanding.

By the end of this post, you’ll not only see how the pieces fit together but also gain a sense of where this rapidly evolving field might be headed next.

## The Origin: Nonequilibrium Thermodynamics

---

To truly understand where diffusion models come from, we need to take a step outside computer science and peek into the world of physics - specifically, the behavior of systems that evolve over time, like smoke dispersing in the air or a cup of hot coffee cooling down. This area of study is called **nonequilibrium thermodynamics**, and in 2015, a groundbreaking paper by Jascha Sohl-Dickstein and colleagues brought these ideas into machine learning.

### The Forward Process: Turning Order into Noise

Imagine starting with a clear photograph and slowly corrupting it by adding tiny bits of noise at every step. Over time, the image becomes more and more scrambled until it eventually looks like static. This process is very gradual and happens in many small steps. {{< mathjax/inline>}}<span style="color: #ff8459;">The key idea is that the system is moving toward chaos - just like physical systems tend to move toward disorder or maximum entropy.</span>{{< /mathjax/inline>}}

This “destruction” process was carefully designed to be **reversible** in theory, but it's not feasible in real life and we use deep learning models to approximate this process. That is, if we could understand how the noise was added at each step, we could potentially undo it - and recover the original image.

### The Reverse Process: Learning to Rebuild

Now comes the clever part: what if we could teach a computer to reverse that destruction — to go from pure noise back to a realistic image? It’s like trying to put shattered glass back together, one fragment at a time. This reverse journey is much harder than the forward one because we’re trying to go from randomness to structure. But here’s the trick: the noise we add isn’t just random chaos — it’s **Gaussian noise**, which follows a well-understood statistical pattern. Because this noise has a known structure, it gives the model something to latch onto when learning how to reverse the process. In simple terms, it means: _“given this specific noisy input, generate the most likely original image.”_ And since Gaussian noise is easy to generate and manipulate, the model starts with a reliable foundation for learning the denoising process.

To make this possible, researchers trained a deep neural network to predict how to remove just a **small amount of noise at each step**. This network doesn’t rebuild the image all at once — instead, it denoises it gradually, in a step-by-step process. Crucially, it’s **the same model reused at every step**, taking in the noisy image at the current step and predicting the denoised version for the previous step. It learns to handle different noise levels using either an explicit time-step embedding or by conditioning on the noise schedule. Over many iterations, the model steadily transforms random noise into a coherent and realistic image.

This recursive process — using the **same model architecture over many noise levels** — is at the heart of what we now call a **diffusion model**. Through this repeated denoising, the model gradually reconstructs structure out of randomness, guided by the statistical regularity of Gaussian noise and the learned understanding of what real images look like.

### Why This Approach Matters

At the time, most generative models (like GANs and VAEs) were trying to directly generate images in one big leap, which often made them unstable or hard to train. Diffusion models, on the other hand, followed a slow and steady path - adding and then carefully removing noise.

What made this method special was that it was built on solid physical intuition: the idea that systems naturally move toward disorder, and we can learn to reverse that process. It also introduced a new way of training models by comparing how well they can match forward and reverse paths, rather than playing a game between two networks like GANs.

### Limitations and What Came Next

The original diffusion model was powerful but slow - it needed many small steps to work properly. While that made it more stable, it also meant it wasn’t practical for real-world use at the time. But it planted the seed for a new kind of generative modeling that was **simple, stable, and grounded in physics**.

## Parallel Histories: Diffusion vs. Score-Based

---

While diffusion models were developing along this machine-learning-meets-physics path, another stream of research was evolving in parallel - {{< mathjax/inline>}}<span style="color: #0084a5;">score-based generative models</span>{{< /mathjax/inline>}}. What’s fascinating is that both approaches were solving similar problems but starting from very different angles. Eventually, researchers realized they were _more connected than they first appeared_.

### Diffusion Models - Noise Prediction Perspective

Diffusion models, especially as simplified in {{< mathjax/inline>}}<span style="color: #0084a5;">Ho et al. (2020)</span>{{< /mathjax/inline>}}, view the problem like this:

- Add Gaussian noise step by step until your data turns into pure noise.
- Train a model to predict and remove that noise.
- Over time, the model learns to reverse the process, going from random noise back to realistic samples.

This makes the problem intuitive for neural networks - learning to predict noise at each step.

{{< rawhtml>}}

<p align="center">
  <img src="../images/DiffusionEvolution/1.png" alt="Image description" class="img-fluid" style="max-width: 100%; height: auto; border-radius: 10px; width: 100%"/>
</p>
<p align="center">
  <em>Figure 1: Forward Process of Diffusion Models</em>
</p>
{{< /rawhtml>}}

### Score-Based Models - Learning the Gradient of Distribution

Meanwhile, score-based models, inspired by {{< mathjax/inline>}}<span style="color: #0084a5;">score matching (Hyvärinen, 2005)</span>{{< /mathjax/inline>}}, focus on directly learning the "score function":

{{< mathjax/inline>}}\[
\nabla*x \log p*{\text{data}}(x)
\]{{< /mathjax/inline>}}

This means learning how the probability density of data changes - essentially figuring out which direction to nudge a noisy sample to make it more likely to be real.

Song & Ermon (2019) pushed this further by using stochastic differential equations (SDEs):

- Instead of discrete steps, model the continuous path of noise being added or removed.
- Sampling becomes solving an SDE guided by the learned score function.

{{< rawhtml>}}

<p align="center">
  <img src="../images/DiffusionEvolution/3.png" alt="Image description" class="img-fluid" style="max-width: 100%; height: auto; border-radius: 10px; width: 100%"/>
</p>
{{< /rawhtml>}}

### The Surprising Connection: They’re the Same at the Core

- Diffusion models learn to {{< mathjax/inline>}}<span style="color: #0084a5;">denoise</span>{{< /mathjax/inline>}} step by step.
- Score-based models learn the {{< mathjax/inline>}}<span style="color: #0084a5;">gradient of the log-probability</span>{{< /mathjax/inline>}} (the score).

**Mathematically, they are doing the same thing**:

- Predicting the noise in diffusion models is equivalent to predicting the score function up to a scaling factor. This equivalence was shown in the denoising score matching literature and formalized in Song et al.'s work. When you train a diffusion model to predict the noise ε, it's mathematically equivalent to predicting the score of the noisy distribution, scaled by the variance of the noise.
- Both can be described as solving a reverse-time SDE. Both frameworks can be unified using stochastic differential equations (SDEs). Song et al. (2021) showed that you can describe the forward (noising) and reverse (generation) processes using Itô SDEs, and that the reverse-time SDE governs how to denoise or sample by estimating the score function.

- Both generate data by starting from noise and iteratively refining it toward the data distribution. In both approaches, generation begins with a noise sample (often Gaussian), and through iterative refinement — either via denoising steps (DDPM) or solving an ODE/SDE guided by the learned score — they gradually arrive at a data sample.

This realization unified the two worlds - leading to models like {{< mathjax/inline>}}<span style="color: #0084a5;">"Score-Based Generative Models via SDEs" (Song et al., 2021)</span>{{< /mathjax/inline>}} that connect diffusion and score-based methods seamlessly.

Understanding this dual heritage - machine learning noise prediction and physics-based score modeling - helps explain why diffusion models suddenly exploded in popularity and became the backbone of models like Stable Diffusion, DALL·E 2, and Imagen.

## Diffusion in Practice

---

After years of theoretical development and steady improvement, diffusion models made a major leap: they started working shockingly well in practice. Suddenly, these once-slow, academic models began producing photorealistic images, outshining GANs in terms of fidelity, diversity, and stability. What had begun as a physics-inspired framework in 2015 had, by 2022, become the engine behind some of the most exciting advancements in generative AI.

In this section, we’ll explore how diffusion models are used in practice - how they evolved into modern architectures, how real-world systems implement them, and why they’ve been so successful in domains like image, video, audio, and even molecular generation.

### The Breakthrough Moment: DDPMs and Improved Sampling

The turning point came in late 2020 with the introduction of **Denoising Diffusion Probabilistic Models (DDPMs)** by **Jonathan Ho et al.**, which made a few key contributions:

> A simple and effective noise prediction objective, which trains the model to estimate the noise added to an image at a given timestep. A U-Net-based architecture with attention mechanisms, borrowed from segmentation and image-to-image tasks. A practical reverse process, implemented as a Markov chain with hundreds or thousands of timesteps.

Despite its slow sampling speed, DDPMs delivered image quality that rivaled and even surpassed GANs, particularly on metrics like FID (Fréchet Inception Distance).

This model was a demonstration that theoretical elegance could yield practical results - and it inspired a flurry of research focused on making diffusion faster, sharper, and more scalable.

### Scaling Up: Imagen, DALLE-2, and Stable Diffusion

After DDPMs proved the concept, several key systems brought diffusion to the mainstream:

#### Imagen (Google Research, 2022)

Imagen focused on **text-to-image generation** and pushed image quality to new heights.

> A frozen, pre-trained large language model (T5) to encode the text prompt. A diffusion model trained in a super-resolution cascade, first generating a low-res image, then progressively upscaling it. A focus on classifier-free guidance, improving alignment between text and image while boosting image quality.

Imagen's images were hyper-detailed and well-aligned with prompts, setting a new benchmark in text-to-image synthesis.

#### DALLE-2 (OpenAI, 2022)

DALLE-2 took a slightly different route:

> It used a CLIP model to embed the text prompt. A prior model (also a diffusion model) generated an image embedding from the text embedding. A decoder model (another diffusion model) converted that embedding into a full image.

This two-step architecture showed how diffusion could be paired with **contrastive learning** and **discrete representation spaces**, opening new possibilities for editing, inpainting, and variation control.

#### Stable Diffusion (CompVis, 2022)

Perhaps the most impactful in terms of accessibility and adoption, Stable Diffusion:

> Operated in latent space (using a pre-trained VAE to compress image data), massively reducing memory and compute requirements. Was open-sourced, enabling a creative explosion of use cases and community-driven tools. Introduced prompt engineering into the mainstream, turning text-to-image generation into an interactive art form.

Stable Diffusion made diffusion fast and accessible, and kickstarted the era of generative AI tools for the public.

<!-- ### Challenges in Practice

Despite the stunning results, deploying diffusion models at scale comes with some real challenges:

#### Slow Sampling
- The generation process requires **50 to 1000 sequential steps**, making it far slower than GANs (which generate images in one pass).
- Researchers have developed **sampling acceleration techniques** - like DDIM, fast ODE solvers, and fewer-step distillation methods (e.g. *Progressive Distillation* or *Consistency Models*).

#### Computational Cost
- Training diffusion models is computationally expensive due to the long forward/backward process and repeated noise injections.
- Latent diffusion (as used in Stable Diffusion) helps reduce this by training in a compressed space.

#### Controllability
- While diffusion models are inherently flexible, controlling outputs (e.g., generating objects in specific locations or with certain attributes) can still be tricky.
- Solutions include conditioning on layout maps, masks, embeddings, or using **guidance strategies**. -->

<!-- ### Beyond Images: Multimodal & Scientific Applications

Diffusion models aren’t limited to images. Their flexibility and stability make them powerful tools across a range of domains:

#### Audio
- Models like **DiffWave** and **WaveGrad** use diffusion to generate raw audio waveforms with high fidelity.
- Text-to-speech systems are starting to integrate diffusion for natural-sounding prosody and control.

#### Video
- Temporal coherence is a challenge, but progress is happening with models like **Video Diffusion Models** and **Generative Query Networks with diffusion**.
- Techniques involve training on 3D spatial-temporal data or extending 2D frame models to handle time.

#### 3D and Scene Generation
- Diffusion has been applied to **NeRF-like representations**, 3D object generation, and even robotics environments.
- Some models learn to diffuse over implicit neural representations rather than pixel grids.

#### Molecular Modeling
- Diffusion models are used for [**protein folding**](https://www.youtube.com/watch?v=P_fHJIYENdI&ab_channel=Veritasium), **drug discovery**, and **molecular graph generation**, offering a structured yet stochastic way to sample valid molecular configurations. -->

<!-- ### The Real-World Impact

What began as a niche research idea has become the **backbone of modern generative AI**. Diffusion models now power:
- Creative tools for artists and designers
- Content generation for games, film, and marketing
- Accessibility tools (e.g., generating visuals from descriptive text for visually impaired users)
- Scientific discovery and simulation

Thanks to open-source models like **Stable Diffusion**, the barrier to entry has dropped dramatically, enabling a wave of innovation at the grassroots level - everything from fine-tuned anime generators to highly controlled product rendering pipelines.
 -->

## What’s Next? The Future of Diffusion Models

---

Diffusion models have already transformed how we generate images, but the journey is far from over. One of the biggest areas of focus right now is **speed**. While current diffusion models can produce beautiful results, they require many steps to do so - sometimes thousands. This makes them slow and impractical for real-time applications like video games, interactive media, or mobile apps. To overcome this, researchers are developing faster sampling methods that can generate images in just a few steps - or even one. These breakthroughs are bringing us closer to real-time, high-quality generation.

Another exciting frontier is **control and customization**. Future diffusion models are becoming better at understanding prompts and user intent, allowing for more accurate and nuanced generation. They’re learning to respond to specific instructions about layout, pose, style, and even personal preferences. You might soon be able to fine-tune a model to match your unique artistic taste with just a few clicks or examples. These improvements make diffusion models not only more powerful but also more useful for everyday creators and professionals alike.

Diffusion is also moving **beyond images**. We’re now seeing models that can generate **videos**, **3D scenes**, and even **audio**, opening the door to entirely new forms of creative expression. Some models are designed to handle multiple types of data at once - like text, images, and video - blurring the lines between different kinds of media. This makes them useful for building AI systems that can understand and generate rich, multimodal experiences.

Interestingly, diffusion is also being used to generate **structured content**, like computer code, mathematical formulas, and scientific graphs. These aren’t just pretty pictures - they’re precise and logical outputs that require the model to reason step-by-step. By combining diffusion with tools from programming and symbolic logic, researchers are building systems that could someday help solve complex problems in science and engineering.

As these models grow in ability, there's also a growing need to make them more **interpretable and responsible**. Researchers are working to understand how diffusion models “think” - what they learn internally, how they represent concepts like “a smiling face” or “a city at night,” and how we can better control them. Alongside this, there’s active work in reducing bias, detecting AI-generated content, attributing authorship fairly, and cutting down on the energy costs of training large models.

In short, diffusion models are evolving fast. They’re getting quicker, smarter, more creative, and more aligned with how we want to use them. As this technology matures, it’s not just reshaping digital art - it’s redefining what machines can imagine.

### Closing Thoughts

The story of diffusion models is far from over. What began as a theoretical curiosity grounded in nonequilibrium thermodynamics has grown into a **generative powerhouse**, capable of creating images, videos, 3D worlds, and even chemical molecules.

With each innovation - whether it's faster sampling, deeper multimodal reasoning, or tighter control - we move closer to **generative systems that are not just tools, but collaborators** in science, art, design, and discovery.

The question is no longer whether diffusion models will shape the future, but **how far they’ll take us**.

## References

---

1. Luo, Calvin. _Understanding Diffusion Models: A Unified Perspective_. [https://www.calvinyluo.com/2022/08/26/diffusion-tutorial.html](https://www.calvinyluo.com/2022/08/26/diffusion-tutorial.html)
   – A clear and intuitive explanation of diffusion models with an extensive mathematical proof and explanation.

2. Weng, Lilian. _What are Diffusion Models?_ [https://lilianweng.github.io/posts/2021-07-11-diffusion-models/](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
   – A deep-dive blog post covering the theory, math, and evolution of diffusion models in detail.
