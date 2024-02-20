---
title: "Volumetric Rendering using NeRF 💡"
date: 2024-01-03T23:15:00+09:00
slug: NeRF
category: projects
summary:
description:
cover:
  image: "covers/NeRF.png"
  alt:
  caption:
  relative: true
showtoc: true
draft: false
---

## Introduction
---
The project delves into Neural Radiance Fields (NeRF), a cutting-edge methodology in computer graphics and computer vision. NeRF has had a significant impact in 3D scene synthesis by modeling the volumetric scene function, allowing for the creation of highly realistic and detailed virtual environments. Its ability to comprehend radiance at each point in a scene opens up new possibilities for rendering scenes with unparalleled accuracy. The motivation behind this undertaking lies in the aspiration to contribute to the forefront of 3D scene reconstruction and rendering. In contrast to traditional graphics methods, which often struggle with capturing intricate lighting effects and scene details, NeRF offers a data-driven approach that promises to overcome these limitations. The project seeks to explore and showcase the potential of NeRF in generating immersive and visually striking 3D scenes.

To bring this vision to life, the project strategically integrates two pivotal technologies—Blender's Python API and PyTorch. Blender's Python API was used for creating and manipulating 3D scenes, providing a versatile environment for dataset generation. This choice aligns seamlessly with the project's objective of producing high-quality training data for the NeRF model. Complementing this, PyTorch serves as the backbone for implementing and training the NeRF model. The synergistic use of Blender's Python API and PyTorch enables a fluid integration of synthetic 3D data creation and neural network training, forming the basis for this project.

## Dataset
---
In the process of crafting a tailored dataset essential for training the NeRF model, the decision to construct a custom dataset arose from the unique capabilities offered by Blender's Python API. Unlike real-world images, Blender provided a controlled environment where obtaining crucial camera intrinsic and extrinsic parameters was feasible. This was a pivotal aspect, as NeRF's training relies heavily on accurate knowledge of the camera's orientation and calibration parameters, which can be challenging to acquire in real-world scenarios.

By utilizing Blender's Python API, the project not only addressed the need for diverse and detailed scenes but also tackled a fundamental challenge in NeRF training – the availability of precise camera parameters. The API facilitated the extraction of intrinsic details, such as focal length, as well as extrinsic details like rotation and translation matrices. This level of control over the camera parameters allowed for a more realistic and nuanced training of the NeRF model. In navigating these intricacies, the project highlighted the important role of Blender's Python API not only in generating diverse scenes but also in providing the crucial camera information required for a robust and accurate NeRF model.

{{< rawhtml>}}
<style>
  .image-container {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
  }

  .image-container img {
    max-width: 45%;
    height: auto;
    border-radius: 5px;
    margin: 10px; /* Adjust this value to set the gap between images */
  }
</style>

<div class="image-container">
  <img src="../images/NeRF/0001.png" alt="Image 1" class="img-fluid" />
  <img src="../images/NeRF/0002.png" alt="Image 2" class="img-fluid" />
  <img src="../images/NeRF/0003.png" alt="Image 3" class="img-fluid" />
  <img src="../images/NeRF/0004.png" alt="Image 4" class="img-fluid" />
  <img src="../images/NeRF/0005.png" alt="Image 5" class="img-fluid" />
  <img src="../images/NeRF/0006.png" alt="Image 6" class="img-fluid" />
</div>
<p align="center">
  <em>Figure 1: Training Samples</em>
</p>
{{< /rawhtml>}}

## NeRF
---
### Introduction to NeRF Architecture

NeRF conceptualizes a static scene as a continuous 5D function, defining the radiance emitted in each direction {{< mathjax/inline >}}\((\theta, \phi)\){{< /mathjax/inline >}} at every point {{< mathjax/inline >}}\((x, y, z)\){{< /mathjax/inline >}} in space. Additionally, a density parameter at each point serves as a differential opacity, controlling the radiance accumulation along rays passing through the scene. This continuous scene representation is achieved through a deep fully-connected neural network,without any convolutional layers. The neural network efficiently maps a single 5D coordinate {{< mathjax/inline >}}\((x, y, z, \theta, \phi)\){{< /mathjax/inline >}} to a corresponding volume density and view-dependent RGB color.

The NeRF rendering process unfolds in a sequential manner: first, camera rays traverse the scene, generating a sampled set of 3D points. Subsequently, these points and their corresponding 2D viewing directions serve as inputs to the neural network, producing an output set of colors and densities. Finally, classical volume rendering techniques are employed to accumulate these colors and densities into a 2D image. This inherently differentiable process allows the utilization of gradient descent for model optimization, minimizing the error between observed images and their corresponding views generated from the neural representation.

However, the initial implementation faces challenges, struggling with convergence to sufficiently high-resolution representations and inefficiencies in the required number of samples per camera ray. NeRF addresses these issues through ingenious strategies. Firstly, input 5D coordinates undergo positional encoding, enabling the MLP to represent higher frequency functions. Secondly, a hierarchical sampling procedure is introduced, reducing the number of queries needed to calculate the color of the pixel.

The advantages of NeRF extend beyond representation quality; it inherits the benefits of volumetric representations, excelling in modeling complex real-world geometry and appearance. Crucially, it overcomes the storage costs associated with discretized voxel grids, particularly beneficial when modeling intricate scenes at high resolutions.

Importantly, NeRF showcases superior performance in both quantitative and qualitative evaluations when compared to state-of-the-art view synthesis methods. This includes outperforming works utilizing neural 3D representations and those training deep convolutional networks for volumetric predictions. As a milestone achievement, this paper introduces the first continuous neural scene representation capable of rendering high-resolution photorealistic novel views of real objects and scenes from RGB images captured in natural settings.

### NeRF Scene Representation

The core of NeRF architecture lies in its representation of a continuous scene as a 5D vector-valued function. This function takes a 3D location {{< mathjax/inline >}}\(x = (x, y, z)\){{< /mathjax/inline >}} and a 2D viewing direction {{< mathjax/inline >}}\((\theta, \phi)\){{< /mathjax/inline>}}, where {{< mathjax/inline >}}\(\theta\){{< /mathjax/inline >}} and {{< mathjax/inline >}}\(\phi\){{< /mathjax/inline >}} are angles representing the direction, and outputs an emitted color {{< mathjax/inline >}}\(c = (r, g, b)\){{< /mathjax/inline >}} and volume density {{< mathjax/inline >}}\(\sigma\){{< /mathjax/inline >}}. To facilitate practical implementation, the direction is expressed as a 3D Cartesian unit vector *d*. NeRF approximates this continuous 5D scene representation using an MLP network {{< mathjax/inline >}}\(F_\theta: (x, d) \rightarrow (c, \sigma)\){{< /mathjax/inline >}}, and optimizes its weights {{< mathjax/inline >}}\(\Theta\){{< /mathjax/inline >}} to map each input 5D coordinate to its corresponding volume density and directional emitted color.

In order to promote multiview consistency within the representation, the network is designed to predict the volume density {{< mathjax/inline >}}\(\sigma\){{< /mathjax/inline >}} as a function of only the location *x*. Simultaneously, the RGB color *c* is predicted as a function of both the location and the viewing direction. This architectural decision enhances the coherence of the representation across multiple views.

The MLP {{< mathjax/inline >}}\(F_\theta\){{< /mathjax/inline >}} undergoes a sequential processing of the input 3D coordinate *x*. Initially, it passes through 8 fully-connected layers, each equipped with ReLU activations and 256 channels, resulting in the output of {{< mathjax/inline >}}\(\sigma\){{< /mathjax/inline >}} and a 256-dimensional feature vector. Subsequently, this feature vector is concatenated with the camera ray's viewing direction and fed into an additional fully-connected layer. This layer, utilizing a ReLU activation and consisting of 128 channels, outputs the view-dependent RGB color.

This design choice showcases NeRF's flexibility in capturing nuanced scene details, ensuring that the model can adapt to various lighting conditions and material properties by incorporating both spatial and directional information into its continuous scene representation. The next section delves into the challenges faced by the basic implementation of NeRF and the innovative solutions introduced to overcome these hurdles.

### Volume Rendering

The volume density {{< mathjax/inline >}}\(\sigma(x)\){{< /mathjax/inline >}} within NeRF is construed as the differential probability of a ray terminating at an infinitesimal particle located at {{< mathjax/inline >}}\(x\){{< /mathjax/inline >}}. The expected color {{< mathjax/inline >}}\(C(r)\){{< /mathjax/inline >}} of a camera ray {{< mathjax/inline >}}\(r(t) = o + td\){{< /mathjax/inline >}} with near and far bounds {{< mathjax/inline >}}\(t_n\) and \(t_f\){{< /mathjax/inline >}} is expressed through the integral equation, as outlined in Kutulakos and Seitz's work. This integral requires estimating the accumulated transmittance along the ray, denoted as {{< mathjax/inline >}}\(T(t)\){{< /mathjax/inline >}}, from {{< mathjax/inline >}}\(t_n\) to \(t\){{< /mathjax/inline >}}. 

{{< mathjax/block >}}
\[
C(r) = \int_{t_n}^{t_f} T(t) \sigma(r(t)) c(r(t), d) \, dt ,\quad where\quad T(t) = \exp(-\int_t^{t_n}\sigma(r(s))ds)
\]
{{< /mathjax/block >}}

Rendering a view from NeRF requires numerically estimating this continuous integral {{< mathjax/inline >}}\(C(r)\){{< /mathjax/inline >}} for each camera ray traced through every pixel of the virtual camera. Traditional approaches, commonly applied in rendering discretized voxel grids, would impose limitations on our representation's resolution. To overcome this constraint, NeRF employs a stratified sampling approach, partitioning {{< mathjax/inline >}}\([t_n, t_f]\) into \(N\){{< /mathjax/inline >}} evenly-spaced bins and drawing one sample uniformly at random from within each bin.

The discrete set of samples obtained through stratified sampling facilitates the representation of a continuous scene. Throughout the optimization process, the MLP is evaluated at continuous positions, enhancing the overall accuracy of the rendered scenes. Estimating {{< mathjax/inline >}}\(C(r)\){{< /mathjax/inline >}} is achieved through a quadrature rule inspired by Max's volume rendering review [Max, N.: Optical models for direct volume rendering. IEEE Transactions on Visualization and Computer Graphics (1995)].

{{< mathjax/block >}}
\[
   \hat{C}(r) = \sum_{i=1}^{N}T_i \cdot (1 - \exp(-\sigma_i \delta_i))c_i, \quad where \quad T_i = \exp(-\sum_{j=1}^{i-1}\sigma_j \delta_j)
\]
{{< /mathjax/block >}}

The function {{< mathjax/inline >}}\(\hat{C}(r)\){{< /mathjax/inline >}} represents the accumulated transmittance for each sample, and {{< mathjax/inline >}}\(\delta_i\){{< /mathjax/inline >}} is the distance between adjacent samples. The differentiability of this function is pivotal, allowing for seamless integration into NeRF's optimization process.

In essence, this intricate process of volume rendering within NeRF transforms abstract continuous scene representations into visually realistic images, demonstrating the model's capability to synthesize high-fidelity views with nuanced lighting effects and material interactions.

### Positional Encoding

The authors introduce two pivotal improvements aimed at empowering the representation of high-resolution and intricate scenes. The first involves positional encoding of input coordinates, aiding the MLP in capturing high-frequency functions. The second entails a hierarchical sampling procedure designed to efficiently sample this high-frequency representation.

Despite the universal function approximation capabilities of neural networks, the empirical findings by the authors reveal that direct operation of the network {{< mathjax/inline >}}\(F_\Theta\){{< /mathjax/inline >}} on {{< mathjax/inline >}}\((x,y,z,\theta,\phi)\){{< /mathjax/inline >}} coordinates yields renderings that inadequately capture high-frequency variations in color and geometry. This observation aligns with recent work by Rahaman et al., demonstrating the inherent bias of deep networks towards learning lower frequency functions. Rahaman et al. further propose that mapping inputs to a higher-dimensional space using high-frequency functions before passing them to the network facilitates better fitting of data containing high-frequency variation.

Drawing inspiration from these insights, the authors reformulate {{< mathjax/inline >}}\(F_\Theta\){{< /mathjax/inline >}} as a composition of two functions: {{< mathjax/inline >}}\(F_\Theta = F_\Theta' \cdot \gamma\){{< /mathjax/inline >}}, where {{< mathjax/inline >}}\(F_\Theta'\){{< /mathjax/inline >}} is a learned function, and {{< mathjax/inline >}}\(\gamma\){{< /mathjax/inline >}} is a mapping from {{< mathjax/inline >}}\(\mathbb{R}\){{< /mathjax/inline >}} to a higher-dimensional space {{< mathjax/inline >}}\(\mathbb{R}^{2L}\){{< /mathjax/inline >}}. Notably, {{< mathjax/inline >}}\(F_\Theta'\){{< /mathjax/inline >}} remains a regular MLP. The encoding function {{< mathjax/inline >}}\(\gamma(p)\){{< /mathjax/inline >}} takes the form:

{{< mathjax/block >}}
\[
\gamma(p) = (\sin(2^0 \pi p), \cos(2^0 \pi p), \ldots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p))
\]
{{< /mathjax/block >}}

This encoding function {{< mathjax/inline >}}\(\gamma(\cdot)\){{< /mathjax/inline >}} is applied separately to each of the three coordinate values in {{< mathjax/inline >}}\(x\){{< /mathjax/inline >}} (normalized to lie in {{< mathjax/inline >}}\([-1, 1]\){{< /mathjax/inline >}}) and to the three components of the Cartesian viewing direction unit vector {{< mathjax/inline >}}\(d\){{< /mathjax/inline >}} (constructed to lie in {{< mathjax/inline >}}\([-1, 1]\){{< /mathjax/inline >}}). In our experiments, we set {{< mathjax/inline >}}\(L = 10\) for \(\gamma(x)\) and \(L = 4\) for \(\gamma(d)\){{< /mathjax/inline >}}. This approach, reminiscent of positional encoding in the Transformer architecture, empowers the MLP to more effectively approximate higher-frequency functions, contributing to the overall improvement in performance.

It's worth noting that a similar coordinate mapping technique is employed in the popular Transformer architecture, known as positional encoding. However, Transformers utilize this for a distinct purpose—providing discrete positions of tokens in a sequence as input to an architecture devoid of any inherent notion of order. In contrast, in NeRF these functions is geared towards mapping continuous input coordinates into a higher-dimensional space, enhancing the MLP's ability to approximate higher-frequency functions with greater ease.

### Hierarchical Volume Sampling

The authors of NeRF project have introduced a novel rendering strategy that efficiently evaluates the NeRF network along camera rays by addressing inherent inefficiencies. Drawing inspiration from early work in volume rendering [Levoy, M.: Efficient ray tracing of volume data. ACM Transactions on Graphics (1990)], they propose a hierarchical representation to enhance rendering efficiency by allocating samples proportionally to their expected impact on the final rendering.

In this innovative approach, the authors optimize two networks simultaneously: one referred to as "coarse" and another as "fine." Initially, a set of {{< mathjax/inline >}}\(N_c\){{< /mathjax/inline >}} locations is sampled using stratified sampling, and the "coarse" network is evaluated at these locations, following the equations described in the preceding sections. Subsequently, given the output of the "coarse" network, a more informed sampling of points along each ray is generated, biasing samples towards the relevant parts of the volume.

To achieve this, the authors rewrite the alpha-composited color from the coarse network {{< mathjax/inline >}}\(\hat{C}_c(r)\){{< /mathjax/inline >}} as a weighted sum of all sampled colors {{< mathjax/inline >}}\(c_i\){{< /mathjax/inline >}} along the ray:

{{< mathjax/block >}}
\[
\hat{C}_c(r) = \sum_{i=1}^{N_c} w_i c_i \quad \text{where} \quad w_i = T_i(1 - \exp(-\sigma_i \delta_i))
\]
{{< /mathjax/block >}}

Normalizing these weights as {{< mathjax/inline >}}\(\hat{w}_i = w_i/\sum_{j=1}^{N_c} w_j\){{< /mathjax/inline >}} produces a piecewise-constant Probability Density Function (PDF) along the ray. Subsequently, a second set of {{< mathjax/inline >}}\(N_f\){{< /mathjax/inline >}} locations is sampled from this distribution using inverse transform sampling. The "fine" network is then evaluated at the union of the first and second sets of samples, and the final rendered color of the ray {{< mathjax/inline >}}\(\hat{C}_f(r)\){{< /mathjax/inline >}} is computed using equation given in volume rendering section but utilizing all {{< mathjax/inline >}}\(N_c + N_f\){{< /mathjax/inline >}} samples.

This hierarchical sampling procedure ensures that more samples are allocated to regions expected to contain visible content. While addressing a similar goal as importance sampling, the authors' method employs the sampled values as a nonuniform discretization of the entire integration domain. This approach differs from treating each sample as an independent probabilistic estimate of the entire integral, offering an efficient solution to the challenges posed by traditional rendering strategies. The loss function used was a simple total squared error between the rendered and true pixel colors for both the coarse and fine renderings.

{{< rawhtml>}}
<p align="center">
  <img src="../images/NeRF/test_sample.gif" alt="Image description" class="img-fluid" style="max-width: 100%; height: auto; border-radius: 10px; width: 100%"/>
</p>
<p align="center">
  <em>Figure 2: Test Samples</em>
</p>
{{< /rawhtml>}}