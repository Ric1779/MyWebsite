---
title: "Overview of Graphics Pipeline 🎨"
date: 2024-04-15T23:15:00+09:00
slug: graphicsPipeline
category: graphicsPipeline
summary:
description:
cover:
  image: "covers/graphics-pipeline.png"
  alt:
  caption:
  relative: true
showtoc: true
draft: false
---

## Introduction
---
In computer graphics, there's a fascinating process called the graphics pipeline. It's like a factory assembly line, but instead of producing physical objects, it creates the stunning images we see on our screens. Let's break it down into simpler terms.

The graphics pipeline works by taking geometric shapes, like triangles or squares, and transforming them into pixels on our screen. Imagine starting with a bunch of dots in space, representing the corners of these shapes. The pipeline then does some math to figure out where each dot should go on the screen, making sure everything lines up perfectly.

Once the dots are in the right place, the pipeline moves on to rasterization. This step is like coloring in a coloring book: it figures out which dots are inside each shape and fills them in with color. It also blends colors together where shapes overlap.

After rasterization, the pipeline adds some finishing touches to each pixel, making sure they look just right. Finally, all the pixels come together to create the complete image we see on our screen.

The graphics pipeline is used in everything from video games to movie special effects, making it a crucial part of modern technology. In the next sections, we'll take a closer look at each step of the pipeline, breaking it down into easy-to-understand concepts and exploring how it brings virtual worlds to life.

## Rasterization
---
### Line Drawing : The Midpoint Algorithm

In computer graphics, drawing lines accurately is crucial. Whether sketching shapes or outlining objects, knowing the ins and outs of line drawing algorithms is essential. Let's dive into one of the most popular methods: the midpoint algorithm.

Drawing lines typically involves connecting two points on a screen. But when dealing with any two points, finding a way to represent the line between them becomes a challenge. This is where line equations come into play, offering methods like implicit and parametric approaches. Here, we'll focus on the implicit method and its use in the midpoint algorithm.

The midpoint algorithm is all about drawing the thinnest line possible between two points without leaving any gaps. To start, we need to find the implicit equation for the line, which looks like this:

{{< mathjax/inline>}}\[ f(x, y) ≡ (y_0 - y_1)x + (x_1 - x_0)y + x_0y_1 - x_1y_0 = 0 \]{{< /mathjax/inline>}}

Here, {{< mathjax/inline>}}\( (x_0, y_0) \){{< /mathjax/inline>}} and {{< mathjax/inline>}}\( (x_1, y_1) \){{< /mathjax/inline>}} represent the endpoints of the line. If {{< mathjax/inline>}}\( x_0 ≤ x_1 \){{< /mathjax/inline>}}, we're good to go. Otherwise, we just switch the points to make it true.

The slope {{< mathjax/inline>}}\( m \){{< /mathjax/inline>}} of the line is given by:

{{< mathjax/inline>}}\[ m = \frac{y_1 - y_0}{x_1 - x_0} \]{{< /mathjax/inline>}}

We will consider the case where {{< mathjax/inline>}}\( m ∈ (0, 1] \){{< /mathjax/inline>}} (which means the line is flatter than it is tall). Analogous discussions can be derived for {{< mathjax/inline>}}\( m ∈ (−∞,−1]\){{< /mathjax/inline>}}, {{< mathjax/inline>}}\( m ∈ (−1, 0]\){{< /mathjax/inline>}}, and {{< mathjax/inline>}}\( m ∈ (1,∞)\){{< /mathjax/inline>}}. The key assumption of the midpoint algorithm is that we draw the thinnest line possible that has no gaps. A diagonal connection between two pixels is not considered a gap.

{{< rawhtml>}}
<p align="center">
  <img src="../images/graphics_pipeline/diagonal.png" alt="Image description" class="img-fluid" style="max-width: 50%; height: auto; border-radius: 10px; width: 100%"/>
</p>
<p align="center">
  <em>Figure 1: Three “reasonable” lines that go seven pixels horizontally and three pixels vertically.</em>
</p>
{{< /rawhtml>}}

The core algorithm operates as follows:

```latex
y = y0
for x from x0 to x1 do
    draw_pixel(x, y)
    if (some condition) then
        y = y + 1
```

In simpler terms, this algorithm instructs to draw pixels sequentially from left to right, occasionally adjusting the vertical position upward based on a specified condition. The efficiency of this algorithm hinges on optimizing the decision-making process within the conditional statement.

This decision is based on whether {{< mathjax/inline>}}\( f(x + 1, y + 0.5) \){{< /mathjax/inline>}} (evaluated from the implicit equation) is less than zero. If it is, we move up, depicted in image 1 of Figure 2; if not, we keep going sideways, depicted in image 2 of Figure 2. This ensures our line is drawn smoothly without any gaps.

{{< rawhtml>}}
<p align="center">
  <img src="../images/graphics_pipeline/f.png" alt="Image description" class="img-fluid" style="max-width: 50%; height: auto; border-radius: 10px; width: 100%"/>
</p>
<p align="center">
  <em>Figure 2: Top: the line goes above the midpoint so the top pixel is drawn. Bottom: the line goes below the midpoint so the  bottom pixel is drawn.</em>
</p>
{{< /rawhtml>}}

In short, the midpoint algorithm is a handy tool for drawing lines precisely in computer graphics. By understanding its principles and tricks, we can create smooth and accurate lines in digital worlds effortlessly.

### Triangle Rasterization

When it comes to rendering 2D triangles, a process known as triangle rasterization is employed. This procedure involves converting geometrically defined triangles into pixel representations on a screen. While similar to line drawing, triangle rasterization poses its own set of challenges and nuances.

**Interpolation and Gouraud Shading**

In triangle rasterization, we often encounter the need to interpolate properties such as color across the triangle's surface. This is achievable through barycentric coordinates, which determine how much each vertex contributes to a point within the triangle. For instance, employing barycentric coordinates {{< mathjax/inline>}}\( (\alpha, \beta, \gamma) \){{< /mathjax/inline>}}, the color at any point within the triangle can be calculated as:

{{< mathjax/inline>}}\[ c = \alpha c_0 + \beta c_1 + \gamma c_2 \]{{< /mathjax/inline>}}

This interpolation technique, known as [{{< mathjax/inline>}}<span style="color: #ffa700;">Gouraud shading</span>{{< /mathjax/inline>}}]({{< ref "blogs/gouraud-shading" >}}) , after its creator Henri Gouraud, ensures smooth color transitions across the triangle's surface.

**Handling Adjacent Triangles**

Rasterizing adjacent triangles presents another challenge, particularly concerning shared vertices and edges. A common approach involves drawing the outline of each triangle using the midpoint algorithm and subsequently filling in the interior pixels. However, this method can lead to inconsistencies in the final image when adjacent triangles possess differing colors, as the image's outcome depends on the drawing order.

To address this issue and ensure seamless rendering without gaps, a convention is adopted where pixels are drawn only if their centers lie within the triangle. This convention dictates that the barycentric coordinates of the pixel center must fall within the interval {{< mathjax/inline>}}\((0, 1)\){{< /mathjax/inline>}}. However, handling pixels exactly on the triangle edges poses a dilemma, which various methods seek to resolve.

**Efficient Triangle Rasterization**

The brute-force approach to triangle rasterization involves iterating over all pixels within the triangle's bounding box and computing their barycentric coordinates. If the coordinates satisfy the condition that {{< mathjax/inline>}}\( \alpha, \beta, \gamma \){{< /mathjax/inline>}} are all within the range {{< mathjax/inline>}}\([0, 1]\){{< /mathjax/inline>}}, the pixel is drawn with the interpolated color. However, to enhance efficiency, optimizations limit the pixel iteration to a smaller set of candidates and streamline the barycentric coordinate computation process.

Triangle rasterization lies at the heart of many rendering algorithms, providing the foundation for rendering complex scenes with intricate geometry and shading. By understanding the principles and techniques behind triangle rasterization, graphics programmers can create visually stunning images with accuracy and efficiency.

## Clipping
---
#### The Need for Clipping

When transforming geometric primitives into screen space, the possibility arises of certain primitives extending beyond the bounds of the view volume. This scenario is particularly prevalent with primitives positioned behind the observer's eye. Such occurrences can lead to erroneous rendering if left unaddressed. Consider a triangle with two vertices within the view volume and one behind the eye. Without proper handling, the projection transformation could erroneously map the vertex behind the eye to an invalid location on the image plane with a z-coordinate in the z-buffer exceeding the far plane, resulting in incorrect rasterization.

#### Clipping Operation Overview

Clipping is a fundamental operation in graphics, essential for handling scenarios where one geometric entity intersects or extends beyond another. For instance, when clipping a triangle against a specific plane, portions of the triangle lying on the 'wrong' side of the plane are typically discarded. In the context of preparing primitives for rasterization, the 'wrong' side refers to areas outside the view volume. While clipping against all six faces of the view volume is a foolproof approach, many systems suffice with clipping solely against the near plane to optimize performance.

#### Implementation Approaches

There are two primary approaches to implementing clipping:

1. **Clipping in World Coordinates:** This method involves utilizing the six planes that bound the truncated viewing pyramid. Each triangle undergoes clipping against these planes to ensure visibility within the view volume.

2. **Clipping in Homogeneous Coordinates:** Surprisingly, the commonly implemented approach involves clipping in homogeneous coordinates before the homogeneous divide. This 4D representation of the view volume is bounded by simple hyperplanes, allowing for efficient clipping operations.

#### Clipping Procedure

No matter which method of implementation is chosen, the basic process of clipping against a plane remains unchanged. Using the implicit equation of a plane, we can determine whether points or line segments are situated on opposite sides of the plane. For example, if the result of {{< mathjax/inline>}}\( f(x, y) \){{< /mathjax/inline>}} is positive, it indicates that the point lies on the "correct" side of the plane. Here, {{< mathjax/inline>}}\( f \){{< /mathjax/inline>}} represents the implicit equation of the plane.

When a plane intersects a line segment, we can calculate the precise point of intersection and then adjust the segment accordingly. Similarly, when dealing with triangles, the process involves iteratively clipping against each plane that defines the boundaries of the viewing volume. This ensures that only the visible parts of the triangle contribute to the final image.

{{< rawhtml>}}
<p align="center">
  <img src="../images/graphics_pipeline/clipping.png" alt="Image description" class="img-fluid" style="max-width: 40%; height: auto; border-radius: 10px; width: 100%"/>
</p>
<p align="center">
  <em>Figure 3: A polygon is clipped against a clipping plane. The portion “inside” the plane is retained.</em>
</p>
{{< /rawhtml>}}

In essence, clipping serves as a crucial precursor to rasterization, ensuring that only visible portions of primitives contribute to the final rendered image. By discarding extraneous geometry lying outside the view volume, clipping lays the foundation for accurate and visually appealing graphics.

## Operations Before and After Rasterization
---
#### Preparing for Rasterization: The Vertex-Processing Stage

Before a primitive can undergo rasterization, it must be meticulously prepared. This crucial task falls upon the vertex-processing stage of the graphics pipeline. Here, incoming vertices undergo a series of transformations, including modeling, viewing, and projection transformations. These transformations map the vertices from their original coordinates into screen space, where positions are measured in pixels. Simultaneously, other essential information such as colors, surface normals, or texture coordinates are transformed as necessary, setting the stage for subsequent operations.

#### Post-Rasterization Processing: Fragment Operations and Blending

Once rasterization is complete, further processing ensues to compute crucial attributes for each fragment. This processing can range from simple tasks like passing through interpolated colors and utilizing rasterizer-computed depth to complex shading operations that breathe life into the scene. Following fragment processing, the blending phase comes into play. Here, fragments generated by overlapping primitives are combined to compute the final color. A common blending approach involves selecting the color of the fragment with the smallest depth, indicating proximity to the observer.

In its most elementary form, the pipeline requires minimal intervention in the vertex and fragment stages. Primitives are supplied directly in pixel coordinates, with the rasterizer handling the bulk of the workload. This setup forms the backbone of many simplistic APIs for drawing user interfaces, plots, graphs, and other 2D content. Whether drawing solid color shapes or smoothly varying colors through interpolation, this basic arrangement suffices for a myriad of applications.

Transitioning to 3D rendering requires a minor adjustment to the 2D drawing pipeline: a single matrix transformation. The vertex-processing stage applies this transformation to incoming vertices, resulting in screen-space triangles ready for rendering. However, the minimal 3D pipeline introduces challenges with occlusion relationships. To address this, primitives must be drawn in back-to-front order, akin to the painter's algorithm for hidden surface removal.

#### Challenges and Drawbacks:

While the painter's algorithm is a valid approach for hidden surface removal, it has inherent limitations. Triangles that intersect or form occlusion cycles pose significant challenges for back-to-front rendering. Additionally, sorting primitives by depth can be computationally intensive, especially for large scenes, disrupting the efficient flow of data.

In the next section, we'll delve deeper into strategies and advancements that address these challenges, paving the way for more efficient and visually stunning graphics rendering. Stay tuned for an exploration of cutting-edge techniques and optimizations in the realm of computer graphics.

#### Understanding the Z-Buffer Algorithm

The z-buffer algorithm revolutionizes hidden surface removal by employing a straightforward approach: at each pixel, it maintains the distance to the closest surface encountered thus far. This depth information is stored alongside the traditional red, green, and blue color values in what is known as the z-buffer. During the fragment blending phase, fragments' depths are compared with the current z-buffer values. Fragments closer to the observer overwrite the values in the z-buffer and contribute to the final image, while farther fragments are discarded.

Implementing the z-buffer algorithm is relatively straightforward. Fragments carry depth information interpolated from vertex attributes, ensuring accurate depth comparisons during blending. This simplicity, coupled with its effectiveness, has cemented the z-buffer as the dominant approach for hidden surface removal in both hardware and software graphics pipelines.

While the z-buffer algorithm offers unparalleled simplicity, it does encounter precision challenges, particularly concerning the storage of depth values. To mitigate these issues, z-values are typically stored as non-negative integers rather than true floating-point numbers, conserving valuable memory resources. However, this approach introduces precision limitations, especially when rendering scenes with significant depth variations.

To ensure optimal precision in z-buffer implementations, careful consideration of various factors is required. Adjusting the number of bits allocated for depth storage (represented by B) and the separation between the near and far clipping planes (n and f, respectively) are crucial steps. Additionally, understanding the relationship between post-perspective divide depth (z) and world depth (zw) is essential for accurately determining depth bin sizes.

Incorporating the z-buffer algorithm into graphics rendering pipelines offers a reliable and efficient solution for hidden surface removal. However, managing precision issues requires meticulous attention to detail, including optimizing depth value storage and carefully choosing clipping plane distances. By understanding the intricacies of z-buffer implementation, developers can unlock the full potential of this indispensable technique in creating visually stunning and immersive graphics experiences.

#### Per-Vertex Shading

Traditionally, the application is responsible for setting colors for triangles sent into the pipeline, with the rasterizer interpolating these colors across the surface. While this suffices for some applications, per-vertex shading offers a more sophisticated approach. In per-vertex shading, illumination equations are evaluated at each vertex, taking into account factors like light direction, eye direction, and surface normal. The resulting color is then passed to the rasterizer as the vertex color. This technique, sometimes referred to as Gouraud shading, enhances the visual appeal of rendered objects by simulating realistic lighting effects.

One crucial decision in per-vertex shading is the choice of coordinate system for performing shading computations. Options like world space or eye space are commonly employed. It's imperative to select an orthonormal coordinate system, particularly in world space, to ensure accurate shading calculations. Eye space offers the advantage of simplifying camera position tracking, as the camera is always located at the origin in eye space.

While per-vertex shading offers significant improvements in realism, it does come with limitations. Notably, it cannot capture shading details smaller than the primitives used to draw the surface. This is because shading is computed only once for each vertex and does not interpolate between vertices. As a result, large triangles or surfaces may exhibit interpolation artifacts, leading to inaccuracies in shading, especially in regions with significant curvature or complex lighting effects.

To address the limitations of per-vertex shading, developers often resort to techniques like subdividing large primitives into smaller ones or employing more sophisticated shading models. These approaches help ensure that shading details are accurately represented, even on curved surfaces or in regions with intricate lighting effects.

Per-vertex shading stands as a powerful tool for enhancing the visual fidelity of rendered scenes, providing a balance between computational efficiency and realism. While it may not capture every nuance of lighting and shading, its simplicity and effectiveness make it a valuable asset in the arsenal of graphics rendering techniques. In the quest for lifelike virtual environments, per-vertex shading serves as a foundational building block, paving the way for immersive and captivating visual experiences.

#### Per-Fragment Shading

Per-fragment shading, also known as Phong shading, presents a solution to the interpolation artifacts encountered in per-vertex shading by evaluating shading equations individually for each fragment in the graphics pipeline. Unlike per-vertex shading, which computes shading at vertices and interpolates the results, per-fragment shading ensures smooth and accurate lighting across surfaces by performing shading computations after interpolation. This technique requires coordination between the vertex and fragment stages to pass geometric information such as eye-space surface normals and vertex positions through the rasterizer as attributes. By eliminating interpolation artifacts and providing smoother shading transitions, per-fragment shading enhances realism in rendered images, making it a crucial tool for achieving lifelike rendering in computer graphics.

## Enhancing Visual Quality with Antialiasing Techniques
---
Just as in ray tracing, where jagged lines and edges can mar the visual appeal of rendered images, rasterization techniques can suffer from similar issues if an all-or-nothing determination is made for each pixel's coverage by a primitive. This technique, known as standard or aliased rasterization, produces pixel sets identical to those mapped by a ray tracer sending one ray through each pixel's center. To address these jagged edges and improve visual quality, antialiasing techniques come into play. One common approach is box filtering, which averages pixel values over a square area to blur edges. While more sophisticated filters exist, the simplicity of the box filter suffices for most applications. Implementing box-filter antialiasing often involves supersampling: generating high-resolution images, downsampling, and averaging pixel groups to approximate the final image. Although supersampling can be computationally expensive, it effectively reduces aliasing artifacts caused by sharp primitive edges. An optimization strategy involves sampling visibility at a higher rate than shading, focusing resources on computing color values for fewer points within each pixel. This approach, known as multisample antialiasing, is particularly prevalent in hardware pipelines, where per-fragment shading is employed. By combining efficient rendering techniques with antialiasing strategies, developers can significantly enhance the visual fidelity of rasterized images, ensuring smooth edges and realistic rendering in computer graphics applications.

## Strategies for Culling Primitives
---
While object-order rendering offers efficiency by traversing all geometry in a single pass, this strength can become a weakness in complex scenes where much of the geometry isn't visible. For example, in a city model, only a fraction of buildings might be visible at any given viewpoint, rendering processing of hidden geometry wasteful. To mitigate this inefficiency, culling techniques are employed to identify and discard invisible geometry, thus conserving computational resources. Three commonly implemented culling strategies work in tandem to optimize rendering performance:

1. **View Volume Culling**: This strategy involves removing geometry lying outside the view volume, which encompasses the region visible to the camera. By discarding primitives beyond the view frustum, computational resources are spared from processing irrelevant geometry.

2. **Occlusion Culling**: Occlusion culling targets geometry that, while potentially within the view volume, is obscured or occluded by other objects closer to the camera. By identifying and discarding these occluded primitives, redundant rendering calculations are avoided, leading to significant performance gains.

3. **Backface Culling**: Primitives facing away from the camera, known as backfaces, contribute nothing to the final image and can be safely discarded. Backface culling efficiently removes these invisible primitives from the rendering pipeline, further enhancing performance.

By implementing these culling strategies, rendering engines can significantly improve efficiency, particularly in scenes with complex geometry. These techniques not only optimize computational resources but also contribute to smoother and more responsive rendering in real-time applications, ultimately enhancing the user experience in interactive 3D environments.

In conclusion, the graphics pipeline serves as the backbone of modern rendering, orchestrating a series of intricate processes to transform geometric primitives into vibrant, lifelike images. From vertex processing to rasterization, shading, and beyond, each stage of the pipeline plays a critical role in shaping the final visual output. Through advancements in rendering algorithms, optimization techniques, and hardware acceleration, the graphics pipeline continues to evolve, unlocking new levels of realism and interactivity in virtual environments. By understanding the inner workings of the graphics pipeline and embracing innovative techniques, developers can harness the power of computer graphics to create immersive digital experiences that captivate and inspire audiences worldwide.