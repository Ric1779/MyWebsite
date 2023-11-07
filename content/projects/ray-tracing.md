---
title: "Ray Tracing Essentials: From Concepts to Stunning Visuals"
date: 2022-04-04T23:15:00+07:00
slug: ray-tracing
category: projects
summary:
description:
cover:
  image: "/covers/final_render.png"
  alt:
  caption:
  relative: true
showtoc: true
draft: false
---

## Overview

The journey into the realm of ray tracing begins with a promise of simplicity and yet a powerful outcome, as Peter Shirley, a seasoned educator in graphics, distills years of teaching experience into a practical how-to guide. For someone looking to start their journey into Computer Graphics, this is the best tutorial on the internet, with easy-to-follow code structure and amazing image renders. He sets the stage by emphasizing the underlying principles, aiming to steer beginners toward a fulfilling experience of creating impressive imagery. In demystifying the term "ray tracing" \( rendering technique that simulates the behavior of light to generate highly realistic images \) Shirley clarifies that the tutorial primarily focuses on a path tracer, a fundamental technique allowing enthusiasts to construct a general ray tracer. The emphasis here isn't on complexity but rather on grasping the core concepts through relatively simple code implementation, enabling computers to handle the bulk of the workload. It's important to note that in the broader context of computer graphics, applications like Blender with its Cycles and Eevee engines also utilize ray tracing techniques. While Shirley's tutorial doesn't encompass the breadth of functionalities seen in Blender's engines, it serves as an accessible starting point for understanding the fundamental principles of ray tracing. Cycles, for instance, is known for its path tracing capabilities, providing highly realistic results suitable for animations and still images. Eevee, on the other hand, employs rasterization techniques and real-time rendering, delivering quick results for interactive applications. The tutorial primarily employs C++ for its implementation, due to its efficiency, portability, and prevalence in professional-grade renderers used in movies and video games. Although he adheres to a relatively conventional subset of C++, Shirley leverages critical features such as inheritance and operator overloading, essential in crafting ray tracers.

## Output an Image

Commencing a renderer often necessitates a way to visualize images, and the tutorial introduces a straightforward method to write images, specifically, the portable pixmap (PPM) file format. PPM files are appreciated for their simplicity when compared to the complexity found in many other file formats. In a PPM file, image data is stored in plain text. The file starts with a header that defines the image format. For example, the "P3" header indicates a PPM file with ASCII characters. Following the header, the image dimensions (width and height) are specified. Then, the maximum color value (often 255 for an 8-bit color depth) is declared. After these parameters, the RGB pixel values for each individual pixel are listed. Each pixel's color information is presented in a sequence of three values (red, green, and blue) in the range of 0 to the maximum color value (e.g., 0 to 255). The pixels are listed row by row, usually starting from the top-left corner of the image and moving left to right. Each pixel's RGB values are listed in ASCII characters, separated by spaces, and the rows are separated by newline characters. The order of pixels representation typically goes from the top row to the bottom row. The format allows for the creation of images without compression, enabling straightforward representation and easy comprehension. PPM files are versatile and widely supported, suitable for simple image storage and manipulation, making them an accessible choice for beginners in image rendering and processing.

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
  <img src="/images/first-ppm-image.png" alt="Image description" width="750" height="425" style="border-radius: 10px;"/>
</p>
<p align="center">
  <em>Figure 1: First PPM Image</em>
</p>
{{< /rawhtml>}}

## The 'vec3' class

In the project's evolution, a pivotal segment involved defining a fundamental class named vec3. This class is a versatile and multipurpose entity capable of encapsulating both geometric vectors and colors, consolidating their representation into a three-coordinate structure. Despite the conventional divergence in dealing with distinct vector types for positions and colors, the tutorial chose a unified approach with vec3, emphasizing simplicity and minimized code. This design choice, although allowing unconventional operations like subtracting a position from a color, aims at optimizing clarity and ease of comprehension in the code. Notably, the introduction of aliases—point3 and color—for vec3 serves as a guiding organizational element, enhancing readability without imposing strict barriers between different vector types. The project's adherence to this approach showcases a balance between efficiency and pragmatic code design.

## Rays, a Simple Camera, and Background

In the project's progression, a pivotal phase encompassed the establishment of a ray class that encapsulated the essential principles of rays as mathematical functions, enabling the computation of observed colors along these rays. Rays were characterized as originating from a specific point (A) and directed by a vector (b), governed by a real number parameter (t). Implementation-wise, these foundational ray concepts were solidified within the codebase as a class, allowing the calculation of points along the ray's path through the use of the function ray::at(t). Subsequently, the project dived into the core functionalities of a ray tracer: calculating ray trajectories from the "eye" through each pixel, identifying intersecting objects, and determining the observed color at the closest intersection point. Additionally, the tutorial elucidated the creation of a simple camera system and the specifics of constructing a non-square image with a 16:9 aspect ratio, emphasizing the significance of defining image dimensions in alignment with the desired aspect ratio for consistency.

Furthermore, the tutorial delved into the concept of the viewport—a virtual space housing the grid of image pixel locations—highlighting the relationships among pixel spacing, viewport boundaries, and the aspect ratio of the rendered image. It detailed the alignment and orientation of the camera center within the 3D space, emphasizing the transformation between the right-handed coordinates of 3D space and the inverted Y-axis in image coordinates to facilitate proper image scanning. Lastly, the section concluded by expounding on the process of establishing a gradient in color based on the normalized y-coordinate of the ray direction. This section illustrated a linear interpolation technique, blending white and blue colors to create a visually appealing gradient effect, achieved through the use of linear blending principles, known as linear interpolation, to smoothly transition between the two color values, influencing the resulting color at each point along the ray.

## Adding Geometric Primitives

### Spheres

### Quads

## Surface Normals and Multiple Objects

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
