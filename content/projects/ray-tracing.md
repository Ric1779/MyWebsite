---
title: "Ray Tracing Essentials: From Concepts to Stunning Visuals"
date: 2022-04-04T23:15:00+07:00
slug: ray-tracing
category: projects
summary:
description:
cover:
  image: "covers/final_render.png"
  alt:
  caption:
  relative: true
showtoc: true
draft: false
---

## Overview

The journey into the realm of ray tracing begins with a promise of simplicity and yet a powerful outcome, as Peter Shirley, a seasoned educator in graphics, distills years of teaching experience into a practical how-to guide. For someone looking to start their journey into Computer Graphics, this is the best tutorial on the internet, with easy-to-follow code structure and amazing image renders. He sets the stage by emphasizing the underlying principles, aiming to steer beginners toward a fulfilling experience of creating impressive imagery. In demystifying the term "ray tracing" \( rendering technique that simulates the behavior of light to generate highly realistic images \) Shirley clarifies that the tutorial primarily focuses on a path tracer, a fundamental technique allowing enthusiasts to construct a general ray tracer. The emphasis here isn't on complexity but rather on grasping the core concepts through relatively simple code implementation, enabling computers to handle the bulk of the workload. It's important to note that in the broader context of computer graphics, applications like Blender with its Cycles and Eevee engines also utilize ray tracing techniques. While Shirley's tutorial doesn't encompass the breadth of functionalities seen in Blender's engines, it serves as an accessible starting point for understanding the fundamental principles of ray tracing. Cycles, for instance, is known for its path tracing capabilities, providing highly realistic results suitable for animations and still images. Eevee, on the other hand, employs rasterization techniques and real-time rendering, delivering quick results for interactive applications. The tutorial primarily employs C++ for its implementation, due to its efficiency, portability, and prevalence in professional-grade renderers used in movies and video games. Although he adheres to a relatively conventional subset of C++, Shirley leverages critical features such as inheritance and operator overloading, essential in crafting ray tracers. [Github Link](https://github.com/Ric1779/Ray-tracing) for the implementation.

## Output an Image

Commencing a renderer often necessitates a way to visualize images, and the tutorial introduces a straightforward method to write images, specifically, the portable pixmap (PPM) file format. PPM files are appreciated for their simplicity when compared to the complexity found in many other file formats. In a PPM file, image data is stored in plain text. The file starts with a header that defines the image format. For example, the "P3" header indicates a PPM file with ASCII characters. Following the header, the image dimensions (width and height) are specified. Then, the maximum color value (often 255 for an 8-bit color depth) is declared. After these parameters, the RGB pixel values for each individual pixel are listed. Each pixel's color information is presented in a sequence of three values (red, green, and blue) in the range of 0 to the maximum color value (e.g., 0 to 255) and the rows are separated by newline characters. The pixels are listed row by row, usually starting from the top-left corner of the image and moving left to right. The format allows for the creation of images without compression, enabling straightforward representation and easy comprehension. PPM files are versatile and widely supported, suitable for simple image storage and manipulation, making them an accessible choice for beginners in image rendering and processing.

```cpp
#include <iostream>

int main() {

    // Image

    int image_width = 256;
    int image_height = 256;

    // Render

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = 0; j < image_height; ++j) {
        for (int i = 0; i < image_width; ++i) {
            auto r = double(i) / (image_width-1);
            auto g = double(j) / (image_height-1);
            auto b = 0;

            int ir = static_cast<int>(255.999 * r);
            int ig = static_cast<int>(255.999 * g);
            int ib = static_cast<int>(255.999 * b);

            std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }
}
```

The C++ code snippet demonstrates the generation of the first PPM image. This code produces a gradient that traverses the image horizontally from black to bright red and vertically from black to green. This combination results in a yellow hue in the bottom-right corner, establishing the foundational steps for creating and understanding image output in the PPM format.

{{< rawhtml>}}
<p align="center">
  <img src="../images/first-ppm-image.png" alt="Image description" width="750" height="425" style="border-radius: 10px;"/>
</p>
<p align="center">
  <em>Figure 1: First PPM Image</em>
</p>
{{< /rawhtml>}}

## The 'vec3' class

In the project's evolution, a pivotal segment involved defining a fundamental class named vec3. This class is a versatile and multipurpose entity capable of encapsulating both geometric vectors and colors, consolidating their representation into a three-coordinate structure. Despite the conventional divergence in dealing with distinct vector types for positions and colors, the tutorial chose a unified approach with vec3, emphasizing simplicity and minimized code. This design choice, although allowing unconventional operations like subtracting a position from a color, aims at optimizing clarity and ease of comprehension in the code. Notably, the introduction of aliases—point3 and color—for vec3 serves as a guiding organizational element, enhancing readability without imposing strict barriers between different vector types. The project's adherence to this approach showcases a balance between efficiency and pragmatic code design.

## Rays, a Simple Camera, and Background

In the project's progression, a pivotal phase encompassed the establishment of a ray class that encapsulated the essential principles of rays as mathematical functions, enabling the computation of observed colors along these rays. Rays were characterized as originating from a specific point *(A)* and directed by a unit vector *(b)*, governed by a real number parameter *(t)* as shown in the equation below. 

{{< mathjax/block >}}
\[P(t) = A + tb\]
{{< /mathjax/block >}}

Implementation-wise, these foundational ray concepts were solidified within the codebase as a class, allowing the calculation of points along the ray's path through the use the parameter t. Subsequently, the project dived into the core functionalities of a ray tracer: calculating ray trajectories from the "eye" through each pixel, identifying intersecting objects, and determining the observed color at the closest intersection point. Additionally, the tutorial elucidated the creation of a simple camera system and the specifics of constructing a non-square image with a 16:9 aspect ratio, emphasizing the significance of defining image dimensions in alignment with the desired aspect ratio for consistency.

Furthermore, the tutorial delved into the concept of the viewport—a virtual 2D space housing the grid of image pixel locations—highlighting the relationships among pixel spacing, viewport boundaries, and the aspect ratio of the rendered image. It detailed the alignment and orientation of the camera center within the 3D space, emphasizing the transformation between the right-handed coordinates of 3D space and the inverted Y-axis in image coordinates to facilitate proper image scanning. Lastly, the section concluded by expounding on the process of establishing a gradient in color based on the normalized y-coordinate of the ray direction. This section illustrated a linear interpolation technique, blending white and blue colors to create a visually appealing gradient effect, achieved through the use of linear blending principles, known as linear interpolation, to smoothly transition between the two color values, influencing the resulting color at each point along the ray.

{{< rawhtml>}}
<p align="center">
  <img src="../images/blue-to-white.png" alt="Image description" width="750" height="425" style="border-radius: 10px;"/>
</p>
<p align="center">
  <em>Figure 2: A blue-to-white gradient depending on ray Y coordinate</em>
</p>
{{< /rawhtml>}}

## Adding Geometric Primitives
 Integrating spheres into the ray tracing process serves as an initial step in constructing a comprehensive rendering system. Defining a sphere necessitates the translation of fundamental mathematical equations to determine possible intersections between rays and these spherical shapes within a 3D space. The primary equation for a sphere centered at the origin, {{< mathjax/inline >}}\(x^2+y^2+z^2 = r^2\){{< /mathjax/inline>}}, lays the groundwork, which then evolves to accommodate spheres at any arbitrary position {{< mathjax/inline >}}\((Cx, Cy, Cz)\){{< /mathjax/inline>}}. Through the transformation of equations into vector form, utilizing dot products and employing vector algebra, the process culminates in the establishment of a quadratic equation. This equation allows for the exploration of potential solutions for the parameter *t*, connecting the abstract realm of mathematical formulas to tangible geometric representations. This detailed mathematical exploration enhances the ray tracing process, providing a bridge between algebraic calculations and their direct interpretation within the visual realm, resulting in more accurate and realistic renderings.

In the second part of the tutorial, expanding the ray tracing toolkit to include quadrilaterals—specifically, parallelograms—introduces a new set of geometric entities to the rendering repertoire. The definition of a quad in this context involves three primary components: *Q*, representing the lower-left corner; *u*, a vector characterizing one side that, when added to *Q*, defines an adjacent corner; and *v*, a vector representing the second side, determining the other adjacent corner when added to *Q*. The fourth corner of the quad, positioned opposite to *Q*, is identified as {{< mathjax/inline >}}\(Q+u+v\){{< /mathjax/inline>}} . Although a quad exists as a two-dimensional object, the values used to define it are three-dimensional. 
## Surface Normals and Multiple Objects
In the progression towards creating a comprehensive ray tracer, the concept of surface normals emerges as a pivotal factor in achieving realistic shading effects. These normals, serving as vectors perpendicular to the surfaces at points of intersection, play a fundamental role in simulating light interactions. The tutorial introduces a critical design decision in implementing these normal vectors—whether to maintain them at arbitrary lengths or to normalize them to a unit length. While the avoidance of the computationally intensive square root operation involved in normalizing vectors may seem appealing, practical considerations lead to the adoption of unit-length normals. The decision is motivated by the inevitable requirement of unit-length normals at various stages of the rendering process. Moreover, by strategizing the generation of these vectors within specific geometry classes or functions, such as the constructor or the hit() function, efficiency can be maximized. For instance, in the case of spheres, normal vectors can be made unit length by simply dividing by the sphere's radius, entirely bypassing the need for square root calculations.

In the context of spheres, determining the outward normal involves computing the direction from the hit point to the sphere's center. Analogously, envisioning this in terms of Earth, the outward normal represents the vector from the Earth's center to a given point—pointing directly outwards. The tutorial proceeds to showcase an initial shading visualization, primarily focusing on the depiction of these normals with a color map due to the absence of lighting components in the scene. To vividly illustrate these surface orientations, a simple yet effective technique is employed. By mapping each component of the normal vector (assumed to be of unit length - so each component is between -1 and 1) to the interval between 0 and 1 and then further mapping these components (x, y, z) to the RGB color channels, a color representation of the normals is generated.
```cpp
color = 0.5*color(N.x()+1, N.y()+1, N.z()+1);
```

 As the tutorial iterates through these steps, concentrating on visualizing the normals at the closest hit point for the single sphere within the scene, the significance of these preliminary renderings sets the stage for future refinements in simulating more intricate lighting effects and material interactions within the ray tracer.
{{< rawhtml>}}
<p align="center">
  <img src="../images/normals-sphere.png" alt="Image description" width="750" height="425" style="border-radius: 10px;"/>
</p>
<p align="center">
  <em>Figure 3: A sphere colored according to its normals</em>
</p>
{{< /rawhtml>}}

## Moving Camera Code Into Its Own Class

## Antialiasing

## Materials

### Diffuse Materials

### Metal 

### Dielectric

## Positionable Camera

## Defocus Blur

## Conclusion

## Full-text article
[Read article](https://peerj.com/articles/2322/)
