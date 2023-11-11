---
title: "Ray Tracing Essentials: From Concepts to Stunning Visuals🔦"
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

## Antialiasing
In the pursuit of enhancing the visual quality of rendered images, the tutorial introduces the concept of antialiasing to mitigate the jagged or "stair-stepping" effect visible along edges. Addressing this aliasing issue is essential in creating more realistic images, emulating how a true image of the world, unlike computer-generated images, maintains a continuous nature due to effectively infinite resolution. The strategy involves implementing a sampling approach to gather multiple samples per pixel rather than a single ray through the pixel center, which is known as point sampling. This adjustment aims to integrate the light falling around the pixel, approximating a more accurate representation of the continuous result. The tutorial discusses a straightforward approach, sampling a square region surrounding the pixel and averaging the resultant light values to create a smoother, more visually appealing image.
{{< rawhtml>}}
<p align="center">
  <img src="../images/antialias-before-after.png" alt="Image description" width="750" height="425" style="border-radius: 10px;"/>
</p>
<p align="center">
  <em>Figure 4: Before and after antialiasing</em>
</p>
{{< /rawhtml>}}

## Materials
In the context of implementing materials in ray tracing, the tutorial addresses critical design decisions that are aimed at enhancing code structure and readability. One fundamental choice discussed is whether to adopt a universal material type, equipped with numerous parameters, allowing individual material types to selectively use pertinent parameters. However, the tutorial leans towards an abstract material class that encapsulates unique behaviors, promoting a more organized and systematic approach. Within this design paradigm, the material class is tasked with two primary functions: determining whether to produce a scattered ray or absorb the incident ray, and specifying the degree to which the scattered ray should be attenuated. To facilitate communication and data transfer in ray-object intersections, the tutorial introduces the concept of a hit_record, a data structure designed to consolidate various information into a single class, allowing for more efficient and comprehensive data handling.

### Diffuse Materials
The tutorial starts with the introduction of diffuse materials, often referred to as matte materials. The tutorial prompts a crucial consideration: whether to tightly bind geometry and materials or allow for a more flexible, separate approach, typical in most renderers. Opting for the separate approach, it allows for the assignment of materials to multiple spheres independently, separating the material definition from the geometry itself. The concept of diffuse objects that do not emit light and instead take on the color of their surroundings is introduced. These objects modulate their inherent color onto the reflected light, and the tutorial outlines how light reflecting off a diffuse surface randomizes its direction, potentially being absorbed or reflected. Within this context, the tutorial presents the Lambertian distribution as a more accurate representation of real-world diffuse objects compared to the previous uniform scattering model. The Lambertian distribution scatters reflected rays based on the angle between the reflected ray and the surface normal, providing a more realistic reflection model.

To simulate the Lambertian distribution, the tutorial illustrates the creation of a sphere displaced from the surface at the point of intersection. Explaining the concept of two unique unit spheres tangent to any intersection point—one on each side of the surface—the tutorial elaborates on their displacement from the surface by their respective radius. Each sphere is associated with a direction—toward and away from the surface's normal—designating one as inside the surface and the other as outside. The process involves selecting the appropriate tangent unit sphere that aligns with the side of the surface as the ray origin, followed by picking a random point on this sphere to generate a ray from the hit point.
{{< rawhtml>}}
<p align="center">
  <img src="../images/lambertian.png" alt="Image description" width="750" height="425" style="border-radius: 10px;"/>
</p>
<p align="center">
  <em>Figure 5: Lambertian Sphere</em>
</p>
{{< /rawhtml>}}

### Metal 
Next the tutorial introduces metal materials, specifically addressing the behavior of polished metals in reflecting rays. Unlike diffuse materials, where rays scatter randomly, polished metals exhibit a different behavior when reflecting rays. The tutorial elucidates that for polished metals, the reflected ray direction follows a distinct pattern based on vector mathematics. It points out that the reflected ray direction is derived from the vector v (incident ray) plus twice the besector vector b.

{{< mathjax/block >}}
\[Reflected\ Ray = v + 2b\]
{{< /mathjax/block >}}

{{< rawhtml>}}
<p align="center">
  <img src="../images/reflection.jpg" alt="Image description" width="450" height="325" style="border-radius: 5px;"/>
</p>
<p align="center">
  <em>Figure 6: Ray Reflection</em>
</p>
{{< /rawhtml>}}
The design considerations involve working with unit vectors and ensuring that the length of the bisector vector b aligns with the dot product of the incident ray direction v and the surface normal n. This fundamental insight provides a foundational understanding of how rays interact with polished metal surfaces, offering a basis for simulating their reflective properties within the ray tracing environment.

{{< rawhtml>}}
<p align="center">
  <img src="../images/metal-sphere.png" alt="Image description" width="750" height="425" style="border-radius: 10px;"/>
</p>
<p align="center">
  <em>Figure 7: Metal Sphere</em>
</p>
{{< /rawhtml>}}

### Dielectric
The tutorial delves into the simulation of dielectric materials, encompassing transparent substances such as water, glass, and diamond. When light interacts with these materials, it bifurcates into a reflected ray and a refracted (transmitted) ray. To handle this interaction, the tutorial proposes a methodology wherein a random choice between reflection and refraction generates only one scattered ray per interaction. Understanding this process involves incorporating Snell's law, a fundamental principle describing the phenomenon of refraction. This law is expressed in terms of angles from the surface normal (θ and θ′) and the refractive indices (η and η′) of the materials involved—typically air (η = 1.0), glass (η = 1.3–1.7), and diamond (η = 2.4). However, one of the practical challenges arises when the ray encounters a material with a higher refractive index, leading to an absence of real solutions in Snell's law, resulting in the impossibility of refraction. In such cases, total internal reflection occurs, as indicated by the mathematical derivation of *sinθ′* and the limiting factor when *sinθ′* exceeds the value of 1.
{{< rawhtml>}}
<p align="center">
  <img src="../images/refraction.jpg" alt="Image description" width="650" height="325" style="border-radius: 5px;"/>
</p>
<p align="center">
  <em>Figure 8: Snell's Law</em>
</p>
{{< /rawhtml>}}
Moreover, the tutorial addresses the variability in the reflectivity of real glass with the angle of incidence, demonstrating that reflective properties change based on the viewing angle. While a complex equation exists to depict this variation, the tutorial introduces the Schlick approximation—a simplified polynomial function developed by Christophe Schlick. This approximation method is widely employed in practical applications due to its cost-effectiveness and surprisingly accurate representation of real-world scenarios. It accounts for the angle-dependent reflectivity of glass, specifically when observed at steep angles, where it assumes a mirror-like quality, simplifying the modeling of realistic reflective behaviors in dielectric materials.

## Positionable Camera
In this section the focus shifts towards addressing the placement and orientation of the camera within the rendered scene. To attain a flexible viewpoint, the tutorial introduces essential points—designating the position where the camera is placed as "lookfrom" and the focal point as "lookat." Moreover, it highlights the need to define the camera's roll or sideways tilt, which refers to its rotation around the axis formed by the lookat and lookfrom points. Explaining the concept of specifying an "up" vector for the camera, the tutorial emphasizes the importance of projecting this up vector onto a plane orthogonal to the view direction, resulting in a camera-relative "view up" (vup) vector. By employing a series of mathematical operations involving cross products and vector normalizations, the tutorial establishes a complete orthonormal basis (u, v, w) that defines the camera's orientation. These unit vectors (u, v, w) represent the camera's right, up, and opposite view direction, respectively, within a right-hand coordinate system, with the camera center situated at the origin. This approach lays the foundation for maneuvering and directing the camera to various positions and orientations within the rendered scene.
{{< rawhtml>}}
<p align="center">
  <img src="../images/view-distant.png" alt="Image description" width="750" height="425" style="border-radius: 10px;"/>
</p>
<p align="center">
  <em>Figure 9: Distant View</em>
</p>
{{< /rawhtml>}}
{{< rawhtml>}}
<p align="center">
  <img src="../images/zoom-in.png" alt="Image description" width="750" height="425" style="border-radius: 10px;"/>
</p>
<p align="center">
  <em>Figure 10: Zoomed In</em>
</p>
{{< /rawhtml>}}

## Defocus Blur 
The tutorial explores the concept of depth of field or defocus blur, stemming from real cameras' need for a larger aperture to gather light. It elucidates that the introduction of a lens before the film or sensor establishes a specific focus distance where subjects appear sharply in focus. This focus distance differs from the focal length, representing the distance between the camera center and the image plane. However, in the tutorial's simplified model, the focus distance aligns with the focal length, with the pixel grid situated on the focus plane.

To simulate defocus blur, the virtual camera utilizes an aperture solely to create a blurred effect, distinct from real cameras where aperture size impacts both exposure and blur. In this context, the tutorial describes the generation of multiple rays from the "lookfrom" position, where the vector connecting "lookfrom" and the pixel on the viewport defines the ray's direction. It then draws a parallel between the previously discussed antialiasing technique, where rays were scattered around a pixel, and the method employed for defocus blur. For defocus blur, the ray's origin is randomly chosen from an aperture situated at the "lookfrom" position, replicating the blurry effect observable in real cameras.
{{< rawhtml>}}
<p align="center">
  <img src="../images/defocus-blur.png" alt="Image description" width="750" height="425" style="border-radius: 10px;"/>
</p>
<p align="center">
  <em>Figure 11: Depth Field</em>
</p>
{{< /rawhtml>}}

## Motion Blur
In the tutorial's exploration of motion blur, it highlights the trade-off between visual quality and computational speed. With the initial choice to opt for a higher visual standard, employing multiple samples per pixel for rendering fuzzy reflection and defocus blur, the tutorial unveils the possibility of simulating various effects in a similar, brute-force manner. Motion blur, a key aspect of real cameras capturing moving objects while the shutter is open, can be emulated by calculating an average of what the camera perceives during that open shutter duration. This is achieved by sending a single ray at a random moment while the shutter is open, capturing an estimate of the light sensed by the camera at that particular instance.

Addressing the technicalities, the tutorial emphasizes the importance of managing time intervals for the shutter, contemplating both the duration from one shutter opening to the next and how long the shutter stays open for each frame. The tutorial outlines the necessity of setting up the camera with suitable shutter timings and considerations for animated objects, establishing a method for them to adapt their motion during specific frames. While discussing the method to handle motion blur and time intervals, the tutorial opts for a simplified model to create only a single frame, implicitly assuming a start at time = 0 and ending at time = 1. This approach involves adjustments to the camera to launch rays at random times between the start and end time, paving the way for future animated sphere creation and laying the groundwork for time management within the ray-tracing process.
{{< rawhtml>}}
<p align="center">
  <img src="../images/motion-blur.png" alt="Image description" width="750" height="425" style="border-radius: 10px;"/>
</p>
<p align="center">
  <em>Figure 12: Motion Blur</em>
</p>
{{< /rawhtml>}}

## BVH
The tutorial advances into a complex facet of the ray tracing process with its focus on Bounding Volume Hierarchies (BVH). This particular section presents a challenging yet pivotal segment in the ray tracing workflow. It aims to optimize the computational efficiency and streamline the code execution, which would ultimately enhance the overall speed of the ray tracer. By implementing BVH in this chapter, the tutorial aims to restructure the hittable component, ensuring that subsequent additions like rectangles and boxes can seamlessly integrate into the system without necessitating additional rework.

BVH effectively targets the time-intensive ray-object intersection, a critical bottleneck in the ray tracer's performance. This method seeks to transform the linear search pattern—dependent on the number of objects—into a more efficient logarithmic search process reminiscent of binary search. BVH achieves this optimization by sorting the objects into hierarchies of bounding volumes. The fundamental principle revolves around finding volumes that encase all the associated primitives. For instance, a bounding sphere encapsulating ten objects means that if a ray misses this sphere, it definitely bypasses all ten objects inside. Subsequently, BVH helps categorize objects into subsets, ensuring that each object exists within a single bounding volume, thereby considerably streamlining the search process for ray intersections.

### AABB
The integration of Axis-Aligned Bounding Boxes (AABBs) within the ray tracing system represents a crucial strategy within the BVH implementation. This key method enables the creation of effective divisions, enhancing the efficiency of the ray bounding volume intersection operations. As a fundamental requirement, these bounding volumes need to be compact and offer fast ray bounding volume intersection checks. The adopted approach primarily focuses on employing axis-aligned boxes, commonly referred to as AABBs for short, as they tend to work optimally for most models, demonstrating better performance compared to alternative strategies.

In the context of AABBs, the primary aim revolves around determining whether a ray intersects with these bounding boxes. The method does not necessarily require the precise hit points, surface normals, or other intricate details needed for visual representation of the object. The most common approach to intersect a ray with an AABB is through the "slab" method. This technique recognizes that an n-dimensional AABB is essentially the outcome of intersecting n-axis-aligned intervals often termed as "slabs." These intervals represent the points bounded between two endpoints, forming the structure of the AABB. For instance, in two dimensions, the intersection of two intervals forms a 2D AABB, namely, a rectangle.

The process for a ray to intersect one such interval involves assessing if the ray hits the boundaries of the AABB. In practical terms, this is executed through an understanding of the ray's parameter 't', t0 and t1 representing the boundary limits. It involves determining the point of intersection of the ray with the boundary plane, expressed through ray equations like *x(t) = A + t * b*, applicable to all three coordinates *(x/y/z)*. The essence of these calculations is to ascertain the parameter 't' that marks the point of contact of the ray with the plane. This methodical approach extends into 3D space, providing the means to determine the intersection of the ray with the bounding planes, offering a comprehensive understanding of the ray's behavior in relation to the boundaries.

### Constructing Bounding Box for Hittables 
By adding a new function to compute the bounding boxes of hittables, the ray tracer aims to establish a hierarchical structure of boxes over all the individual primitives, such as spheres, which are positioned at the leaves of this hierarchy. The design ensures that each aabb object starts as empty by default, providing a robust framework for managing objects with empty bounding volumes, aligning seamlessly with the established interval class. Moreover, this process accommodates animated objects by enabling them to return their bounds throughout their entire motion range, ensuring comprehensive bounding box computation from time equals zero to time equals one. Updates to the hittable_list object allow for the continual computation and refinement of the bounding box as new children are incrementally added, significantly contributing to the structural optimization and organization of the ray tracing process.

```cpp
...
#include "aabb.h"
...

class hittable_list : public hittable {
  public:
    std::vector<shared_ptr<hittable>> objects;

    ...
    void add(shared_ptr<hittable> object) {
        objects.push_back(object);
        bbox = aabb(bbox, object->bounding_box());
    }

    bool hit(const ray& r, double ray_tmin, double ray_tmax, hit_record& rec) const override {
        ...
    }

    aabb bounding_box() const override { return bbox; }

  private:
    aabb bbox;
};
```

### BVH Nodes
In this design approach, the BVH serves as a hittable entity, akin to lists of hittables, allowing it to manage the query of whether a given ray makes contact with it. The implementation revolves around a singular class design, incorporating both the tree and its nodes within one structured framework.

This newly introduced class, bvh_node, encapsulates the hierarchy of objects by defining a container for efficient handling of ray intersections. The class offers a hit function that handles the examination of whether the box associated with the node is intersected by the ray. Upon a hit, the function scrutinizes the children and resolves any intricate details.

The class is structured to encompass a left and right hierarchy, using smart pointers to manage shared instances of hittable objects. Additionally, it defines a bounding box (bbox) that encapsulates the spatiotemporal extent of the objects held within the node. This node-based approach significantly streamlines the traversal process of the bounding volume hierarchy, providing an organized and effective means for ray-object intersection testing within the ray tracer.

### Splitting BVH Volumes
The construction and division of BVH volumes represent a critical step in the ray tracer's efficiency, directly influencing its performance. In the development of the BVH, the most intricate part emerges during its construction, a process executed within the constructor itself. A fascinating aspect of BVHs is their inherent adaptability—ensuring that the hit function operates efficiently as long as the objects within a bvh_node are divided into two distinct sub-lists. Though optimized when the division minimizes the bounding box size of the children, the division itself is primarily for speed rather than accuracy. The adopted approach of the tutorial strikes a balance, splitting the list of primitives along a randomly chosen axis, utilizing a straightforward sorting technique to arrange the objects in each subtree. Even with only two elements in the list, one is assigned to each subtree, concluding the recursion. This uncomplicated yet effective strategy establishes the groundwork for smoother traversal algorithms while paving the way for potential optimizations in the method's functionality.

## Textures
Texture mapping in computer graphics plays a pivotal role in the representation of material effects on objects within a scene. The concept of texture mapping involves applying a specific effect, referred to as the "texture," to an object. This effect could encompass various material properties such as color, glossiness, or the creation of cut-out regions on a surface. The fundamental principle underlying texture mapping revolves around the process of associating a point on the object’s surface with a particular texture value. Initially, the tutorial develops procedural texture colors and focuses on creating a texture map consisting of a constant color. While various programs might organize constant RGB colors and textures differently, the tutorial leans towards a design that allows any color to be used as a texture, creating a versatile architecture. For this, texture coordinates are pivotal and are conventionally represented as (u, v). In the case of a constant texture, these coordinates might seem trivial, but they become indispensable when dealing with other texture types. The cornerstone of texture classes is the 'value()' method, which essentially retrieves the texture color based on the 3D coordinate of the point under consideration. The demonstrated 'solid_color' class in the code snippet embodies the foundational aspects of constant color textures, underscoring the simplicity and importance of associating color values with texture coordinates.

### Checkered Texture
Solid or spatial textures, associating texture with an object to coloring all points in 3D space itself. This unique approach allows objects to transition through the colors of the texture as they change their positions. However, typically, the relationship between the object and the solid texture remains constant to avoid this dynamic color shift. The tutorial introduces a spatial checker_texture class, exemplifying the concept of solid textures. This class implements a three-dimensional checker pattern, showcasing the idea that spatial textures rely solely on the position in space rather than conventional texture coordinates like 'u' and 'v'. The checker pattern is achieved by computing the floor of each component of the input point and performing a modulo operation, yielding either 0 or 1. This outcome corresponds to the even or odd color, effectively creating the iconic checkered pattern. Additionally, a scaling factor is introduced to control the checker pattern's size within the scene. This spatial checker texture opens up the possibilities for creative and visually appealing material effects in the world of ray tracing.

### Texture Coordinates for Sphere
In the domain of texturing for spheres in computer graphics, the utilization of texture coordinates, commonly referred to as u and v, plays a significant role. These coordinates are used to identify the location on a 2D source image or parameterized space. To establish these coordinates for a sphere, a method based on spherical coordinates is commonly adopted, where θ represents the angle from the bottom pole upwards (-Y direction), and ϕ represents the angle around the Y-axis, tracing from -X to +Z, back to -X. These angles are then mapped onto the texture coordinates u and v, each in the range of \[0,1\], with (u=0,v=0) corresponding to the bottom-left corner of the texture image.

The process of computing θ and ϕ for a given point on the unit sphere involves intricate mathematics. Utilizing the relationships between Cartesian coordinates x, y, and z, it involves inversing equations to solve for the angles θ and ϕ. The \<cmath\> function atan2() is instrumental in these calculations, enabling the determination of the angles ϕ and θ from the x, y, and z coordinates. Ultimately, a utility function is employed to compute u and v from the points on the unit sphere centered at the origin, establishing the texture coordinates for texturing spherical objects in the scene.

### Image Texture
In the context of texture mapping, the tutorial delves into the creation of an image texture class, which is designed to contain an image using the stb_image utility. This utility facilitates the reading of image data, structuring it into a large array of unsigned characters, with each component representing packed RGB values within the range of 0 to 255, signifying various shades from black to full white. Additionally, for added convenience in handling image files, a helper class, rtw_image, is provided to streamline the process of managing image-related tasks.

{{< rawhtml>}}
<p align="center">
  <img src="../images/earth-sphere.png" alt="Image description" width="750" height="425" style="border-radius: 10px;"/>
</p>
<p align="center">
  <em>Figure 13: Earth Mapped Sphere</em>
</p>
{{< /rawhtml>}}

## Perlin Noise
The tutorial explores the implementation of Perlin noise, a widely used technique in generating visually appealing solid textures. Named after its creator Ken Perlin, Perlin noise possesses the remarkable attribute of providing consistent, repeatable random-like values for 3D input points. The concept highlights that nearby points return similar values. To achieve this, the tutorial presents an incremental approach inspired by Andrew Kensler's explanation.

The Perlin noise generation initially involves tiling space with a 3D array of random numbers, a process that inherently produces blocky patterns with evident repetitions. To address this, the tutorial proposes a method involving a form of hashing to better scramble the pattern instead of employing straightforward tiling. The provided code snippet offers support to build a Perlin noise implementation, offering methods to create and manage randomized floating-point values, permute arrays, and compute the noise value at specific 3D points. This incremental building of the Perlin noise serves as a foundation for creating visually intricate solid textures by harnessing its repeatable, random-like behavior.
```cpp
class perlin {
  public:
    perlin() {
        ranfloat = new double[point_count];
        for (int i = 0; i < point_count; ++i) {
            ranfloat[i] = random_double();
        }

        perm_x = perlin_generate_perm();
        perm_y = perlin_generate_perm();
        perm_z = perlin_generate_perm();
    }

    ~perlin() {
        delete[] ranfloat;
        delete[] perm_x;
        delete[] perm_y;
        delete[] perm_z;
    }

    double noise(const point3& p) const {
        auto i = static_cast<int>(4*p.x()) & 255;
        auto j = static_cast<int>(4*p.y()) & 255;
        auto k = static_cast<int>(4*p.z()) & 255;

        return ranfloat[perm_x[i] ^ perm_y[j] ^ perm_z[k]];
    }

  private:
    static const int point_count = 256;
    double* ranfloat;
    int* perm_x;
    int* perm_y;
    int* perm_z;

    static int* perlin_generate_perm() {
        auto p = new int[point_count];

        for (int i = 0; i < perlin::point_count; i++)
            p[i] = i;

        permute(p, point_count);

        return p;
    }

    static void permute(int* p, int n) {
        for (int i = n-1; i > 0; i--) {
            int target = random_int(0, i);
            int tmp = p[i];
            p[i] = p[target];
            p[target] = tmp;
        }
    }
};
```

{{< rawhtml>}}
<br/>
<p align="center">
  <img src="../images/perlin-turb.png" alt="Image description" width="750" height="425" style="border-radius: 10px;"/>
</p>
<p align="center">
  <em>Figure 14: Perline Noise with Turbulence</em>
</p>
{{< /rawhtml>}}

The Perlin class extends its capabilities by moving the pattern off the lattice, using random unit vectors positioned on lattice points. Ken Perlin's pioneering method involves employing dot products to achieve this by replacing the usual floats with random vectors. This deviation off the lattice diminishes the blocky visual pattern and creates more organic, natural-looking textures. Another significant aspect introduced in this section is the notion of turbulence, a common practice in generating complex, composite noise by summing multiple frequencies. By utilizing the 'turb' function within the Perlin class, users can create complex patterns by combining various noise calls with different scales, thereby producing intricate and nuanced textures. Moreover, the tutorial introduces how to apply turbulence indirectly to create a marble-like texture. It demonstrates a basic implementation where the color is tied to a sine function, and turbulence adjusts the phase, causing the stripes to ripple and undulate. This effect gives the texture a marble-like appearance, providing a straightforward yet powerful way to create visually appealing textures in procedural texture generation.
{{< rawhtml>}}
<p align="center">
  <img src="../images/Perlin_final.png" alt="Image description" width="750" height="425" style="border-radius: 10px;"/>
</p>
<p align="center">
  <em>Figure 15: Perline Noise with Adjusted Phase</em>
</p>
{{< /rawhtml>}}

## Quadrilaterals
In this section of the tutorial, we witness the incorporation of a new geometric primitive, the quadrilateral, into the ray-tracer's capabilities. Unlike the spherical primitives used so far, a quadrilateral, or "quad", is introduced, defining a parallelogram shape with three key entities: *Q* (the lower-left corner), *u* (a vector representing one side), and *v* (a vector representing the other side). The corner opposite *Q* is derived as *Q+u+v*. Although quads exist in two dimensions, their representation employs three-dimensional values. The tutorial provides a clear example, illustrating how a quad is specified by its corner at the origin, extending along the *Z-axis*, and one unit along the *Y-axis*. The tutorial also addresses potential numerical challenges during ray intersection calculations for quads lying in different planes, introducing a padding technique through the aabb::pad() method. This ensures the bounding box has sufficient dimensions to avoid numerical problems without altering the quad's intersection characteristics. The addition of quads enhances the versatility of the ray tracer by incorporating a new primitive shape into its repertoire.

In the subsequent section, the tutorial delves into the fundamental process of determining ray-plane intersection—a crucial step for implementing ray tracing on geometric primitives, including quadrilaterals. Unlike spheres with implicit intersection formulas, planes are governed by the implicit formula {{< mathjax/inline >}}\(Ax+By+Cz=D\){{< /mathjax/inline>}}. The tutorial introduces this formula and demonstrates using the dot product to relate the plane's normal vector (n) and the position vector (v) of a point on the plane, resulting in the formula {{< mathjax/inline >}}\(n⋅v=D\){{< /mathjax/inline>}}. The tutorial systematically outlines the process of finding the intersection point with a ray {{< mathjax/inline >}}\(R(t)=P+td\){{< /mathjax/inline>}}, solving for t, and obtaining the intersection point. Notably, the tutorial addresses scenarios where the ray is parallel to the plane, providing a comprehensive foundation for handling ray intersection with various planar primitives.

In the ongoing exploration of determining the position of the intersection point relative to the quadrilateral, the tutorial delves into orienting points on the plane. While the intersection point lies on the plane containing the quadrilateral, it could exist anywhere on the plane, necessitating a test to discern if it falls inside or outside the quad. To achieve this, the tutorial constructs a coordinate frame for the plane, comprising a plane origin point Q and two basis vectors, u and v. Unlike conventional orthogonal axes, these vectors need not be perpendicular, allowing for a more versatile span of the entire space. The tutorial introduces the notion of UV coordinates for the intersection point, unveiling the intricacies of determining scalar values α and β to orient the point P on the plane. Through an insightful exploration of vector math, the tutorial demystifies the calculation of these coefficients, emphasizing their role in placing the intersection point within the quadrilateral and paving the way for subsequent steps in the ray-tracing process.
{{< rawhtml>}}
<p align="center">
  <img src="../images/quads.png" alt="Image description" width="750" height="750" style="border-radius: 10px;"/>
</p>
<p align="center">
  <em>Figure 16: Quads</em>
</p>
{{< /rawhtml>}}

## Lights
Unlike early raytracers that employed points or directions as light sources, contemporary methods involve lights with defined positions and sizes. The tutorial introduces the concept of emissive materials, which are materials capable of emitting light into the scene. To achieve this, a new function, "emitted," is incorporated into the material, providing information about the emitted color without engaging in reflection. This addition allows for the transformation of regular objects into sources of light, contributing to the realism and complexity of the ray-traced scenes.

## Cornell Box
Embarking on the exploration of light interaction among diffuse surfaces, the tutorial introduces the iconic "Cornell Box," a conceptual framework dating back to 1984. Designed to emulate the complexities of light dynamics within confined spaces, the tutorial guides us through the creation of a virtual Cornell Box. Comprising five walls and a luminous element, this digital rendition of the Cornell Box serves as an experimental arena for understanding how light behaves and interacts with different surfaces. The construction of this virtual environment sets the stage for further advancements in the tutorial, delving into the intricacies of light reflection, refraction, and shadowing within the context of this carefully crafted digital space.
{{< rawhtml>}}
<p align="center">
  <img src="../images/cornell-empty.png" alt="Image description" width="750" height="750" style="border-radius: 10px;"/>
</p>
<p align="center">
  <em>Figure 17: Cornell Box</em>
</p>
{{< /rawhtml>}}

## Volumes
The tutorial now ventures into the domain of volumes, adding an extra layer of realism to the virtual world by simulating elements like smoke, fog, and mist. These phenomena, often referred to as volumes or participating media, bring a dynamic and immersive quality to the digital environment. The tutorial introduces the concept of constant density mediums, where rays traversing through the volume may scatter within or continue their journey, influenced by factors such as the density of the medium. The denser the volume, the higher the likelihood of scattering events occurring. This section lays the foundation for incorporating sophisticated features like subsurface scattering, teasing the intricacies of making a volume a random surface. Through clever software architecture, the tutorial navigates the nuanced integration of volumes into the ray-tracing framework, opening up possibilities for the realistic depiction of atmospheric effects and internal structures within objects.
{{< rawhtml>}}
<p align="center">
  <img src="../images/cornell-smoke.png" alt="Image description" width="750" height="750" style="border-radius: 10px;"/>
</p>
<p align="center">
  <em>Figure 17: Cornell box with blocks of smoke</em>
</p>
{{< /rawhtml>}}

{{< rawhtml>}}
<br/>
<p align="center">
  <img src="../images/final-render-2.png" alt="Image description" width="750" height="750" style="border-radius: 10px;"/>
</p>
<p align="center">
  <em>Figure 18: Final Render</em>
</p>
{{< /rawhtml>}}