---
title: "Hello CUDA!"
date: 2025-05-01T23:17:00+09:00
slug: CudaBasics
category: CudaBasics
tags:
  - AI
  - GPU
  - Parallel-Programming
  - Ray-Tracing
summary:
description:
cover:
  image: "covers/CUDA.png"
  alt:
  caption:
  relative: true
showtoc: true
draft: true
---

## What is CUDA and Why Should You Care?

---

Deep learning as we know it today—whether it’s training massive language models, generating realistic images, or running complex computer vision systems—simply wouldn’t be possible without GPUs. These models often contain millions to billions of parameters, and the computational demand to train them is staggering. It’s the parallel processing power of GPUs that makes this kind of large-scale learning feasible. But to truly unlock that performance, we need tools like CUDA, which let us write code that runs natively on the GPU and harness every ounce of its capability.

### So, What is CUDA?

**CUDA** (short for **Compute Unified Device Architecture**) is a parallel computing platform and programming model developed by **NVIDIA**. It allows you to write **C or C++ code** that executes on NVIDIA GPUs, enabling massive speedups for compute-intensive workloads—not just traditional graphics tasks.

While Python doesn’t natively support raw CUDA code, libraries like **Numba**, **CuPy**, and **PyCUDA** expose parts of CUDA’s functionality. These tools let you write Python code that runs on the GPU under the hood, making CUDA more accessible to deep learning practitioners.

At its core, CUDA is about leveraging the GPU’s parallelism. CPUs are designed to handle a few complex tasks at a time. GPUs, by contrast, are built to handle thousands of simple operations simultaneously—perfect for deep learning workloads.

### Why Should You Care?

CUDA becomes essential when you need more than just high-level APIs. If you're building custom training loops, novel layer types, or performance-critical components like differentiable renderers or 3D reconstruction modules, writing CUDA code gives you fine-grained control over GPU computation.

Here are some practical areas where CUDA makes a major difference:

- Deep Learning: Accelerating forward/backward passes, or memory-bound operations
- Scientific Computing: Matrix math, simulations, optimization routines
- Computer Graphics: Ray tracing, shading, light simulation
- Image & Video Processing: Real-time filters, compression, object tracking

Even seemingly simple tasks—like adding two arrays in parallel—can showcase the dramatic performance gains made possible through CUDA.

### What You’ll Learn in This Post

In this beginner-friendly guide, we’ll build a simple CUDA program from scratch to understand the core ideas behind GPU programming. You’ll learn how to:

- Write a basic CUDA kernel (a GPU-executable function)
- Launch the kernel from the CPU (host) side
- Understand CUDA threads, blocks, and how they enable parallelism
- Transfer data between CPU (host) and GPU (device) memory

Whether you're extending deep learning frameworks or simply curious about low-level performance tuning, this post will help you get started with CUDA in a hands-on way.

## Setting Up Your CUDA Environment

---

Before we dive into writing CUDA code, let’s get your environment set up so you’re ready to run GPU programs.

Don’t worry—this part sounds more intimidating than it actually is. With the right steps, you’ll be running CUDA code in no time.

### What You’ll Need

To develop with CUDA, you’ll need a few things:

1. **A CUDA-capable NVIDIA GPU**
   Not all GPUs support CUDA, so make sure your machine has a supported NVIDIA graphics card. You can check the list of supported GPUs on NVIDIA’s website [here](https://developer.nvidia.com/cuda-gpus).

2. **Operating System Compatibility**
   - Windows, Linux, and WSL2 (on Windows) are all supported.
   - macOS is not officially supported for CUDA development anymore (unless you're using an older Intel Mac with an NVIDIA GPU, which is rare).

3. **The CUDA Toolkit**
   This is the main software package you'll install. It includes:
   - The **compiler** (`nvcc`)
   - CUDA **runtime libraries**
   - **Code samples**
   - Useful **tools** like `nvprof` and `nsight`

<!--
### 📦 Step-by-Step Installation (Windows/Linux)

#### 🔹 Step 1: Verify GPU Compatibility

* On **Windows**: Open the NVIDIA Control Panel or run `nvidia-smi` in the command prompt (if drivers are installed).
* On **Linux**: Run `lspci | grep -i nvidia` or `nvidia-smi`.

If your GPU shows up, you’re good to go.

---

#### 🔹 Step 2: Install NVIDIA Drivers

If you don’t have the latest NVIDIA drivers installed, download from the [official driver page](https://www.nvidia.com/Download/index.aspx).

* Make sure to choose the driver that matches your GPU model and OS.
* You need the **"Game Ready"** or **"Studio"** drivers — either works for CUDA development.

---

#### 🔹 Step 3: Download and Install the CUDA Toolkit

* Go to the [CUDA Toolkit download page](https://developer.nvidia.com/cuda-downloads).
* Select your OS and version (e.g., Windows 11, Ubuntu 22.04).
* Follow the guided instructions for either the **local** or **network installer**.

During installation, the default options are usually fine:

* CUDA Toolkit
* CUDA Samples
* Visual Studio Integration (on Windows)
* Nsight tools (optional for now)

> ⚠️ **Note**: CUDA versions are tied to specific driver versions. Make sure your driver supports the CUDA version you’re installing.

---

#### 🔹 Step 4: Verify the Installation

After installation:

* **Open a terminal or command prompt**.

* Run:

  ```bash
  nvcc --version
  ```

  This should display your installed CUDA version.

* You can also build and run one of the sample programs:

  ```bash
  cd ~/NVIDIA_CUDA-<version>/samples/1_Utilities/deviceQuery
  make
  ./deviceQuery
  ```

  If it prints your GPU info and says “Result = PASS,” you’re all set!

---

### 💡 Tips for Development

* **IDE Support**: You can write CUDA code in **Visual Studio** (on Windows) or use **VS Code**/**CLion**/**Vim**/**Emacs**/**JetBrains** on Linux.
* **Compiling**: You’ll use `nvcc`, the CUDA compiler, to build `.cu` files.
* **Running**: CUDA programs are run just like regular executables, but they will use your GPU under the hood.

Once you're set up, you're ready to write and run your very first CUDA program—which we’ll do in the next section!
 -->

### Step-by-Step Installation (Windows/Linux)

🔹 Step 1: Verify GPU Compatibility
On Windows: Open the NVIDIA Control Panel or run nvidia-smi in the command prompt (if drivers are installed).
On Linux: Run lspci | grep -i nvidia or nvidia-smi.
If your GPU shows up, you’re good to go.

---

🔹 Step 2: Install NVIDIA Drivers
If you don’t have the latest NVIDIA drivers installed, download them from the [official driver page](https://www.nvidia.com/Download/index.aspx)..
Make sure to choose the driver that matches your GPU model and OS.
You need the “Game Ready” or “Studio” drivers — either works for CUDA development.

---

🔹 Step 3: (Windows Only) Install Visual Studio with C++ Support
To compile CUDA code on Windows, you’ll need a C++ compiler — and Visual Studio is the officially supported option.
Here’s how to set it up:

- Download the free [Visual Studio Community Edition](https://visualstudio.microsoft.com).
- During installation, make sure to select the “Desktop development with C++” workload
- You can skip all other workloads if you only plan to use it for CUDA development

💡 This ensures that the Visual C++ compiler (cl.exe) is available, which the CUDA Toolkit integrates with during installation.

---

🔹 Step 4: Download and Install the CUDA Toolkit

- Go to the [CUDA Toolkit download page](https://developer.nvidia.com/cuda-downloads).
- Select your OS and version (e.g., Windows 11, Ubuntu 22.04)
- Follow the guided instructions for either the local or network installer

During installation, the default options are usually fine.

> ⚠️ Note: CUDA versions are tied to specific driver versions. Make sure your driver supports the CUDA version you’re installing.

---

🔹 Step 5: Verify the Installation
After installation:

Open a terminal or command prompt.
Run:

```bash
nvcc --version
```

This should display your installed CUDA version.

You can also build and run one of the sample programs:

```bash
cd \~/NVIDIA\_CUDA-<version>/samples/1\_Utilities/deviceQuery
make
./deviceQuery
```

If it prints your GPU info and says “Result = PASS,” you’re all set!

### 💡 Tips for Development

- IDE Support: On Windows, use Visual Studio; on Linux, VS Code, CLion, Vim, or JetBrains IDEs all work well.
- Compiling: Use nvcc, the CUDA compiler, to build .cu files.
- Running: CUDA programs are run just like regular executables — but with your GPU!

Once you're set up, you're ready to write and run your very first CUDA program — which we’ll do in the next section!

Gladly! Here's a detailed and beginner-friendly draft for your **“Hello, GPU! – Your First CUDA Program”** section. This is where things start getting exciting!

## Hello, GPU! – Your First CUDA Program

---

Now that your CUDA environment is ready, it’s time to write your first CUDA program! Think of this like the “Hello, World” of GPU computing.

Instead of printing to the screen, though, we’re going to write a **kernel** that runs on the GPU and shows how multiple threads can execute in parallel.

### Structure of a CUDA Program

Before jumping into code, let’s break down what a basic CUDA program looks like. There are **two major parts**:

1. **Host code** – This runs on the CPU (your normal C/C++ code).
2. **Device code (Kernel)** – This runs on the GPU and is launched from the host.

You’ll use CUDA keywords like `__global__` to tell the compiler which function should run on the GPU.

### 🧪 A Simple CUDA Program

Here’s a simple CUDA program that prints a message from the GPU. It doesn’t do much useful work, but it shows you how to define a kernel and launch it with multiple threads.

```cpp
#include <iostream>

// A kernel is a function that runs on the GPU
__global__ void helloFromGPU() {
    int idx = threadIdx.x;
    printf("Hello from GPU thread %d!\n", idx);
}

int main() {
    // Launch the kernel with 4 threads in 1 block
    helloFromGPU<<<1, 4>>>();

    // Wait for GPU to finish before accessing results on host
    cudaDeviceSynchronize();

    return 0;
}
```

### 🧪 What’s Happening Here?

Let’s break this down line by line:

- `__global__ void helloFromGPU()`: This declares a **kernel** function that will run on the GPU and be launched from the CPU.

- `threadIdx.x`: This is a built-in CUDA variable. It tells you the **index of the thread** within a block, a thread is the smallest unit of execution in CUDA. Each thread gets its own index. A block is a group of threads that work together. All threads in a block can share data through shared memory and can synchronize with each other. CUDA runs many blocks in parallel on the GPU to handle large computations efficiently.

- In short:
  - Thread : One tiny worker.
  - Block : A team of threads working together.
  - Grid : Many blocks working together.

- `helloFromGPU<<<1, 4>>>();`: This is how you **launch a kernel** in CUDA.
  - `1` is the number of **blocks**
  - `4` is the number of **threads per block**

  So here, we’re launching 4 threads in 1 block.

- `cudaDeviceSynchronize();`: This ensures that the CPU waits until all GPU threads are done running before continuing. Without this, your program might finish before the GPU prints anything!

### 🧪 Compiling and Running It

To compile CUDA programs, use `nvcc`:

```bash
nvcc hello.cu -o hello
```

Then run it:

```bash
./hello
```

You should see output like:

```
Hello from GPU thread 0!
Hello from GPU thread 1!
Hello from GPU thread 2!
Hello from GPU thread 3!
```

Each message is coming from a **different GPU thread**, running in parallel.

> ⚠️ Note: If you encounter an error like "unable to find cl.exe in PATH" when compiling your CUDA code, it likely means the Microsoft C++ compiler from Visual Studio isn’t accessible. Add the path to cl.exe (usually inside `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\<version>\bin\Hostx64\x64\`) to your system’s PATH environment variable.

### A Quick Note on `printf` in CUDA

- `printf()` from inside GPU code only works if your device supports it.
- The output might not appear in order—because threads run in parallel!

### Why This Example Matters

Even though it’s simple, this program teaches you:

- How to define a GPU kernel
- How to launch it with multiple threads
- How CUDA separates host (CPU) and device (GPU) code

Next up, we’ll do something more meaningful: using multiple GPU threads to **add two arrays in parallel**. This is where the real performance gains begin.

Absolutely! Here's a clear and beginner-friendly expansion of the **"Understanding CUDA Terminology"** section. This will help your readers build a solid mental model before they dive into more complex examples.

## Understanding CUDA Terminology

---

Before we jump into writing more powerful CUDA programs, it’s important to get comfortable with the basic **building blocks** of CUDA. These terms may sound a little abstract at first, but once you understand how they work together, everything starts to click.

Let’s break it down.

### 🖥️ Host vs. Device

This is one of the first distinctions you need to grasp in CUDA:

- **Host** = your **CPU** and its memory (RAM).
- **Device** = your **GPU** and its memory (VRAM).

Your CUDA program has parts that run on the host (normal C++ code) and parts that run on the device (the GPU). Communication between the two happens through **memory copies**, which you’ll handle manually (at least at first).

🗣️ Think of the CPU as the “manager” and the GPU as the “worker swarm.” The manager gives orders (launches kernels), and the workers do the grunt work (parallel tasks).

---

### 🚀 Kernels

A **kernel** is a special function that runs on the GPU. You define it with the `__global__` keyword.

Example:

```cpp
__global__ void add(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    result[idx] = a[idx] + b[idx];
}
```

When you launch a kernel, **many GPU threads run it at the same time**, each with its own thread index.

---

### 🧵 Threads

A **thread** in CUDA is the smallest unit of execution. Each thread runs a copy of the kernel code.

You can think of it like this:

> One thread = one worker doing one small job

Each thread has its own unique ID, which it gets from special built-in variables like:

- `threadIdx.x` – index of the thread within a block
- `blockIdx.x` – index of the block within the grid

Using these IDs, each thread can figure out **which part of the data** it should work on.

---

### 🧱 Blocks

Threads are grouped into **blocks**. Each block can have up to thousands of threads (depending on the GPU), and they **can cooperate** with each other by sharing data using **shared memory** (which we’ll talk about later).

So:

- A block = a **group of threads**
- Threads inside the same block can **sync and share memory**

CUDA gives you access to block-level indexing with:

- `blockIdx.x` – position of this block in the grid
- `blockDim.x` – how many threads are in this block

---

### 🌐 Grids

Blocks are further grouped into a **grid**. When you launch a kernel, you define:

```cpp
<<<number_of_blocks, threads_per_block>>>
```

That’s your grid layout.

So to recap:

```
Grid
 └── Block 0
      └── Thread 0
      └── Thread 1
      ...
 └── Block 1
      └── Thread 0
      └── Thread 1
      ...
```

---

### 🧮 Thread Index Calculation

To process a large dataset, you need each thread to know **which part of the data it owns**. You calculate a unique **global thread index** like this:

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

This way, each thread accesses its own index in the array and does its part of the work.

---

### 🧠 Quick Reference Summary

| Term        | Meaning                            |
| ----------- | ---------------------------------- |
| Host        | Your CPU and its memory            |
| Device      | The GPU and its memory             |
| Kernel      | A GPU function, launched from host |
| Thread      | A single execution unit on the GPU |
| Block       | A group of threads                 |
| Grid        | A group of blocks                  |
| `threadIdx` | Index of thread within a block     |
| `blockIdx`  | Index of block within the grid     |
| `blockDim`  | Number of threads in a block       |

---

### 🎨 Visual Analogy: Pizza Factory 🍕

Imagine a massive pizza factory:

- **Host (CPU)**: The manager at HQ telling everyone what pizzas to make.
- **Device (GPU)**: The factory floor full of chefs.
- **Grid**: The entire pizza factory.
- **Block**: A kitchen station with a team of chefs.
- **Thread**: A single chef making one pizza.
- **Kernel**: The recipe each chef follows.

Every chef (thread) is making a pizza (task) using the same recipe (kernel), but for a different order (data index).

---

With this terminology under your belt, you’re ready to tackle real-world parallel programming. In the next section, we’ll build on this to perform **array addition** using the GPU—your first practical CUDA program.

---

Would you like a diagram to help visualize threads, blocks, and grids? Or want to add a quick quiz or exercise to reinforce the terms?
