# Reading and Discussion List for Stanford CS348K

* Web site for most recent offering of the course: <http://cs348k.stanford.edu/spring23>

## Lecture 1: Throughput Computing Review ##

* [Lecture slides](https://gfxcourses.stanford.edu/cs348k/spring23/lecture/intro/)

__Post-Lecture Required Readings/Videos:__

* Students that *have not* taken CS149 or feel they need a refresher on basic parallel computer architecture should first watch this [pre-recorded lecture](https://www.youtube.com/watch?v=wtrR9i5zmvg) that is similar to lecture 2 in CS149. It's a full 90 minutes so feel welcome to skip/fast-forward through the parts that you know.  The technical content begins eight minutes in.
* [The Compute Architecture of Intel Processor Graphics Gen9](https://software.intel.com/sites/default/files/managed/c5/9a/The-Compute-Architecture-of-Intel-Processor-Graphics-Gen9-v1d0.pdf). Intel Corporation
  * This is not an academic paper, but a whitepaper from Intel describing the architectural geometry of a recent GPU.  I'd like you to read the whitepaper, focusing on the description of the processor in Sections 5.3-5.5. Then, given your knowledge of the concepts discussed in the prerecorded video and in lecture 1 (multi-core, SIMD, multi-threading, etc.), I'd like you to describe the organization of the processor (using terms from the lecture, not Intel terms). For example:
    * What is the basic processor building block?
    * How many times is this block replicated for additional parallelism?
    * How many hardware threads does it support?
    * What width of SIMD instructions are executed by those threads? Are there different widths supported? Why is this the case?
    * Does the core have superscalar execution capabilities?
  * Consider your favorite data-parallel programming language, such as GLSL/HLSL shading languages from graphics, [CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html), OpenCL, [ISPC](https://ispc.github.io/), NumPy, TensorFlow, or just an OpenMP #pragma parallel for. Can you think through how an "embarrassingly parallel" for loop can be mapped to this architecture? (You don't need to write this down in your writeup, but you could if you wish.)
  * Note that an update to the Gen9 architecuture is Gen11, which you can read about [here](https://www.intel.com/content/dam/develop/external/us/en/documents/the-architecture-of-intel-processor-graphics-gen11-r1new.pdf).  (We chose to have to read the Gen9 whitepaper since it's a bit more detailed on the compute sections.)
  * __For those that want to go futher, I also encourage you to read [NVIDIA's V100 (Volta) Architecture whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf), linked in the "further reading" below.__ Can you put the organization of this GPU in correspondence with the organization of the Intel GPU? You could make a table contrasting the features of a modern AVX-capable Intel CPU, Intel Integrated Graphics (Gen9), NVIDIA GPUs (Volta, Ampere) etc.  Hint: here are some diagrams from CS149: [here](https://gfxcourses.stanford.edu/cs149/fall21/lecture/multicorearch/slide_80) and [here](https://gfxcourses.stanford.edu/cs149/fall21/lecture/gpuarch/slide_46).

__Other Recommended Readings:__

* [Volta: Programmability and Performance](https://www.hotchips.org/wp-content/uploads/hc_archives/hc29/HC29.21-Monday-Pub/HC29.21.10-GPU-Gaming-Pub/HC29.21.132-Volta-Choquette-NVIDIA-Final3.pdf). Hot Chips 29 (2017)
  * This Hot Chips presentation documents features in NVIDIA Volta GPU.  Take a good look at how a chip is broken down into 80 streaming multi-processors (SMs), and that each SM can issue up to 4 warp instructions per clock, and supports up to concurrent 64 warps.  You may also want to look at the [NVIDIA Volta Whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf).
* [The Story of ISPC](https://pharr.org/matt/blog/2018/04/18/ispc-origins.html). Pharr (2018)
  * Matt Pharr's multi-part blog post is an riveting description of the history of [ISPC](https://ispc.github.io/), a simple, and quite useful, language and compiler for generating SIMD code for modern CPUs from a SPMD programming model.  ISPC was motivated by the frustration that the SPMD programming benefits of CUDA and GLSL/HLSL on GPUs could easily be realized on CPUs, provided applications were written in a simpler, constrained programming system that did not have all the analysis challenges of a language like C/C++.
* [Scalability! But at What COST?](http://www.frankmcsherry.org/assets/COST.pdf) McSherry, Isard, and Murray. HotOS 2015
  * The arguments in this paper are very consistent with the way we think about performance in the visual computing domain.  In other words, efficiency and raw performance are different than "scalable".
