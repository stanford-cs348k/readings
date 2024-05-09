# Reading and Discussion List for Stanford CS348K

This page contains discussion prompts for papers on the reading list for Stanford CS348K: Visual Computing Systems, taught by [Kayvon Fatahahalian](http://graphics.stanford.edu/~kayvonf/). You can find the web site for most recent offering of the course here: <http://cs348k.stanford.edu>.

## Lecture 1: Course Introduction ##

* [Lecture slides](https://gfxcourses.stanford.edu/cs348k/spring24/lecture/intro/)

__Post-Lecture Required Readings:__

* [What Makes a Graphics Systems Paper Beautiful](https://graphics.stanford.edu/~kayvonf/notes/systemspaper/). Fatahalian (2019)
  * A major theme of this course is "thinking like a systems architect". This short blog post discusses how systems artitects think about the intellectual merit and evaluation of systems. There are no specific reading response questions for this blog post, but please read the blog post, and if you have time, click through to some of the paper links.  These are the types of issues, and the types of systems, we will be discussing in this class.  The big reading of the night is Google HDR+ paper listed below. I want you to analyze that paper's arguments in the terms of the goals and constraints underlying the design of the camera application in Google Pixel smartphones.
* [Burst Photography for High Dynamic Range and Low-light Imaging on Mobile Cameras](https://hdrplusdata.org/). Hasinoff et al. SIGGRAPH Asia 2016
   * __A bit of background:__ This paper addresses a common concern in photography that all of us have probably encountered before: the tension between generating images that are sharp (but noisy due too low of light, or distorted by using a flash) and images that are blurry but accurately depict the lighting of a scene. If the scene is dark and the photographer uses a short exposure (to allow light to flow into the camera for only a short period of time), then the image will be sharp, but too noisy to look good. (In class 2 we talked about a few sources of noise: photon noise, sensor noise, etc.). If the photographer uses a long exposure to allow more light to enter the camera, during the exposure objects in the scene may move over the duration of the exposure (or the camera moves... e.g., due to camera shake when hand-holding a camera), and the result is a blurry image.  You can see examples of blurry images from long exposures [here](https://gfxcourses.stanford.edu/cs348k/spring23/lecture/digitalcamera1/slide_78) and [here](https://gfxcourses.stanford.edu/cs348k/spring23/lecture/digitalcamera1/slide_83).  One way to solve the problem is to use a flash, but if you turn on your camera's flash, you'll lose the original look and feel of the scene. No one likes a photo of them taken with flash!  When researchers first started studying ways to use computation to improve images, the thinking was that if you could take two pictures... a blurry long exposure image, and a sharp low exposure image, then there might be a way to combine the two images to produce a single good photograph.  Here's [one example](https://hhoppe.com/flash.pdf) of this technique, where there is a blurred long exposure to capture true colors and a second short exposure taken with flash to get a sharp version of the scene.  Another variant of this technique, called "bracketing", applies the idea to the problem of avoiding oversaturation of pixels. You can learn more about bracketing at these links.
     * <https://www.photoup.net/best-practices-for-shooting-bracketed-images-for-hdr/>
     * <https://www.youtube.com/watch?v=54JXUJhvFSs>

   * This is a *very technical* paper.  But don't worry, your job is not to understand all the technical details of the algorithms, it is to approach the paper with a systems mindset, and think about the end-to-end considerations that went into the particular choice of algorithms. In general, (after class 1) I want you to pay the most attention to Section 1, and (after class 2) Section 4.0 (you can ignore the detailed subpixel alignment in 4.1), Section 5, and Section 6. Specifically, as you read this paper, I'd like you think about the following issues:
   * (AFTER CLASS 1) *Any good system typically has a philosophy underlying its design.*  This philosophy serves as a framework for which the system architect determines whether design decisions are good or bad, consistent with principles or not, etc. Page 2 of the paper clearly lists some of the principles that underlie the philosophy taken by the creators of the camera processing pipeline at Google. For each of the four principles, given your assessment of why the principle is important for a digital camera. Do you agree with all the principles?
   * (AFTER CLASS 1) Designing a good system is about meeting design goals, subject to certain constraints. (If there were no constraints, it would be easy to use unlimited resources to meet the goals.) What are the major constraints of the system? For example are there performance constraints? Usability constraints? Etc.
   * (AFTER CLASS 2) The main technical idea of this paper is to combine a sequence of *similarly underexposed photos* to generate a high-quality single photograph, rather than attempt to combine a sequence of photos with different exposures (the latter is called "bracketing").  What are the arguments in favor of the chosen approach?  Appeal to the main system design principles. 
   * (AFTER CLASS 2) Why is the motivation for the weighted merging process described in Section 5?  Why did the authors not use the simpler approach of just adding up all the aligned images? (Can you justify the designer's decision by appealing to one of the stated design principles?)
   * (AFTER CLASS 2) Finally, take a look at the "finishing" steps in Section 6.  Many of those steps should sound familiar to you after today’s lecture (or for sure after lecture 3).

__Other Recommended Readings:__

* Students that *have not* taken CS149 or feel they need a refresher on basic parallel computer architecture should first watch this [pre-recorded lecture](https://www.youtube.com/watch?v=wtrR9i5zmvg) that is similar to lecture 2 in CS149. It's a full 90 minutes so feel welcome to skip/fast-forward through the parts that you know.  The technical content begins eight minutes in.
* [The Compute Architecture of Intel Processor Graphics Gen9](https://software.intel.com/sites/default/files/managed/c5/9a/The-Compute-Architecture-of-Intel-Processor-Graphics-Gen9-v1d0.pdf). Intel Corporation
  * This is not an academic paper, but a whitepaper from Intel describing the architectural geometry of a recent GPU.  Focus on the description of the processor in Sections 5.3-5.5. Then, given your knowledge of the concepts discussed in the prerecorded video and in lecture 1 (multi-core, SIMD, multi-threading, etc.), I'd like you to describe the organization of the processor (using terms from the lecture, not Intel terms). For example:
    * What is the basic processor building block?
    * How many times is this block replicated for additional parallelism?
    * How many hardware threads does it support?
    * What width of SIMD instructions are executed by those threads? Are there different widths supported? Why is this the case?
    * Does the core have superscalar execution capabilities?
  * Consider your favorite data-parallel programming language, such as GLSL/HLSL shading languages from graphics, [CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html), OpenCL, [ISPC](https://ispc.github.io/), NumPy, TensorFlow, or just an OpenMP #pragma parallel for. Can you think through how an "embarrassingly parallel" for loop can be mapped to this architecture? (You don't need to write this down in your writeup, but you could if you wish.)
  * Note that an update to the Gen9 architecuture is Gen11, which you can read about [here](https://www.intel.com/content/dam/develop/external/us/en/documents/the-architecture-of-intel-processor-graphics-gen11-r1new.pdf).  (We chose to have to read the Gen9 whitepaper since it's a bit more detailed on the compute sections.)
  * __For those that want to go further, I also encourage you to read [NVIDIA's V100 (Volta) Architecture whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf), linked in the "further reading" below.__ Can you put the organization of this GPU in correspondence with the organization of the Intel GPU? You could make a table contrasting the features of a modern AVX-capable Intel CPU, Intel Integrated Graphics (Gen9), NVIDIA GPUs (Volta, Ampere) etc.  Hint: here are some diagrams from CS149: [here](https://gfxcourses.stanford.edu/cs149/fall21/lecture/multicorearch/slide_80) and [here](https://gfxcourses.stanford.edu/cs149/fall21/lecture/gpuarch/slide_46).
* [Volta: Programmability and Performance](https://www.hotchips.org/wp-content/uploads/hc_archives/hc29/HC29.21-Monday-Pub/HC29.21.10-GPU-Gaming-Pub/HC29.21.132-Volta-Choquette-NVIDIA-Final3.pdf). Hot Chips 29 (2017)
  * This Hot Chips presentation documents features in NVIDIA Volta GPU.  Take a good look at how a chip is broken down into 80 streaming multi-processors (SMs), and that each SM can issue up to 4 warp instructions per clock, and supports up to concurrent 64 warps.  You may also want to look at the [NVIDIA Volta Whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf).
* [The Story of ISPC](https://pharr.org/matt/blog/2018/04/18/ispc-origins.html). Pharr (2018)
  * Matt Pharr's multi-part blog post is an riveting description of the history of [ISPC](https://ispc.github.io/), a simple, and quite useful, language and compiler for generating SIMD code for modern CPUs from a SPMD programming model.  ISPC was motivated by the frustration that the SPMD programming benefits of CUDA and GLSL/HLSL on GPUs could easily be realized on CPUs, provided applications were written in a simpler, constrained programming system that did not have all the analysis challenges of a language like C/C++.
* [Scalability! But at What COST?](http://www.frankmcsherry.org/assets/COST.pdf) McSherry, Isard, and Murray. HotOS 2015
  * The arguments in this paper are very consistent with the way we think about performance in the visual computing domain.  In other words, efficiency and raw performance are different than "scalable".
 
## Lecture 2: Digital Camera Processing Pipeline (Part I) ##

* [Lecture slides](https://gfxcourses.stanford.edu/cs348k/spring24/lecture/camera1)

__Post-Lecture Required Readings:__

* [The Frankencamera: An Experimental Platform for Computational Photography](http://graphics.stanford.edu/papers/fcam/). A. Adams et al. SIGGRAPH 2010
   * Frankencamera was a paper written right about the time mobile phone cameras were becoming "acceptable" in quality. Phones were beginning to contain a non-trivial amount of compute power, and computational photography papers were an increasingly hot topic in the computer graphics community. (Note that the Frankencamera paper predates the HDR+ paper by six years.)  At this time many compelling image processing and editing techniques were being created, and many of them revolved around generating high-quality photographs from a sequence of multiple shots or exposures. The problem was that digital cameras at the time provided a very poor API for software applications to control the camera hardware and its components. Many of the pieces were there for a programmable camera platform to be built (good processing capability and interesting algorithms to use it, high enough sensor resolutions, sufficient-quality lenses), but someone had to architect a coherent system to make these components accessible and composable by software applications. Frankencamera was an attempt to do that: It involved two things:
      * The design of an API for programming cameras (a mental model of an abstract programmable camera architecture).
      * Two implementations of that architecture: an open source camera reference design, and an implementation on a Nokia smartphone.
   * When you read the paper, we’re going to focus on the abstract "architecture" presented by a Frankencamera, not the two research implementations. Remember any computing machine consists of state (nouns), and supports a set of operations that manipulate that state (verbs).  For example, the x86 instruction set architecture defines a program's state as a set of processor registers and the contents of memory. The machine provides a set of operations (called x86 machine instructions) that manipulate the contents of registers or memory. An add instruction reads the state in two registers and modifies the contents of the output register. Now, let's think about a programmable camera as an abstract machine. Specifically I’d like you to think about the following:
      1. Please describe the major pieces of the Frankcamera abstract machine (the system’s nouns): e.g., devices, sensors, processors, etc. Give some thought to why a "sensor" is not just any other "device"? Is there anything special about the sensor? Why did the designers choose to give it special status?
      2. Describe the major operations the machine could perform (the system’s verbs).  For example, in your own words, what is a "shot"? Would you say a shot is a command to the abstract machine? Or is a shot a set of commands? What do you think about the word “timeline” as a good word to describe what a “shot” actually is?
      3. What output does executing a shot generate?  How is a "frame" different from a "shot"? Why is this distinction made by the system?
      4. One of the most interesting aspects of the F-cam design is that it adopts a "best effort" philosophy.  The programmer gives the machine a command defining an operation to perform, and the machine tries to carry out the command, but may not do it exactly. For example, imagine you told your CPU to add 2.0 to a number and instead of adding them it multiplied the number by 1.9. ("Close enough, right?") Do you think this is a good choice for F-cam? What would have been an alternative design choice in this case? Do you think the designers of F-cam made a wise decisions? (Why or why not?) This is a place to get opinionated.  
      5. Would you say that F-cam is a “programmable” camera architecture or a “configurable architecture”.  What kinds of “programs” does the abstract machine run? (Note/hint: see question 2)
      6. It's always good to establish the scope of what a system is trying to do. In this case, how would you characterize the particular type of computational photography algorithms that F-cam seeks to support/facilitate/enable?  What types of algorithms are out of scope?
   * Students may be interested that vestiges of ideas from the Frankencamera can now be seen in the Android Camera2 API:
https://developer.android.com/reference/android/hardware/camera2/package-summary

__Other Recommended Readings:__
  * The old [Stanford CS448A course notes](http://graphics.stanford.edu/courses/cs448a-10/) remains a very good reference for camera image processing pipeline algorithms and issues.
  * [Clarkvision.com](http://www.clarkvision.com/articles/index.html) has some very interesting material on cameras.
  * [Demosaicking: Color Filter Array Interpolation](http://ieeexplore.ieee.org/document/1407714/). Gunturk et al. IEEE Signal Processing Magazine, 2005
  * [Unprocessing Images for Learned Raw Denoising](https://www.timothybrooks.com/tech/unprocessing/). Brooks et al. CVPR 2019
  * [A Non-Local Algorithm for Image Denoising](http://dl.acm.org/citation.cfm?id=1069066). Buades et al. CVPR 2005
  * [A Gentle Introduction to Bilateral Filtering and its Applications](http://people.csail.mit.edu/sparis/bf_course/). Paris et al. SIGGRAPH 2008 Course Notes
  * [A Fast Approximation of the Bilateral Filter using a Signal Processing Approach](http://people.csail.mit.edu/sparis/publi/2006/tr/Paris_06_Fast_Bilateral_Filter_MIT_TR.pdf). Paris and Durand. MIT Tech Report 2006 (extends their ECCV 2006 paper)
  * [Exposure Fusion](http://ieeexplore.ieee.org/document/4392748/). Mertens et al. Computer Graphics and Applications, 2007
     * This is a great reference for how non-local tone mapping is done.    
  * [Local Laplacian Filters: Edge-aware Image Processing with a Laplacian Pyramid](https://people.csail.mit.edu/sparis/publi/2011/siggraph/). Paris et al. SIGGRAPH 2011
  * [The Laplacian Pyramid as a Compact Image Code](http://ieeexplore.ieee.org/document/1095851/). Burt and Adelson, IEEE Transactions on Communications 1983.
 
## Lecture 3: Digital Camera Processing Pipeline (Part II) ##

* [Lecture slides](https://gfxcourses.stanford.edu/cs348k/spring24/lecture/camera2)

For the required reading for the next class, please see required readings under lecture 4. During class we focused our discussion on the architecture of the Frankencamera virtual machine, and continued the lecture on algorithms used in a modern digital camera processing pipeline. For suggested going further readings, please see the list of readings given under Lecture 3. 

__Other Recommended Readings:__

* Please also see the readings under the previous lecture.
* [Synthetic Depth-of-Field with a Single-Camera Mobile Phone](http://graphics.stanford.edu/papers/portrait/wadhwa-portrait-sig18.pdf). Wadha et al. SIGGRAPH 2018.
    * This is a paper about the implementation of "Portrait Mode" in Google Pixel smartphones. It is a dense paper, similar to the HDR+ paper from 2016, but it is a detailed description of how the system works under the hood.
* [Handheld Mobile Photography in Very Low Light](https://google.github.io/night-sight/). Liba et al. SIGGRAPH Asia 2019
    * This is a paper about the implementation of "Night Sight" in Google Pixel smartphones.  

## Lectures 4 and 5: Efficiently Scheduling Image Processing Algorithms ##

* [Lecture slides](https://gfxcourses.stanford.edu/cs348k/spring24/lecture/imagescheduling/)

__Pre-Lecture Required Reading: (to read BEFORE lecture 4)__

* [Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines](http://people.csail.mit.edu/jrk/halide-pldi13.pdf). Ragan-Kelley, Adams, et al. PLDI 2013 
   * Note: Alternatively you may read the selected chapters in the Ragan-Kelley thesis linked below in recommended readings. (Or the CACM article.) The thesis chapters involve a little more reading than the paper, but in my opinion they are a more accessible explanation of the topic, so I recommend it for students.
   * In reading this paper, I want you to specifically focus on describing the _philosophy of Halide_.  Specifically, if we ignore the "autotuner" described in Section 5 of the paper (you can skip this section), what is the role of the Halide programmer, and what is the role of the Halide system/compiler?
      * Hint 1: Which of the two (the programmer or the compiler) is responsible for major optimization decisions?
      * Hint 2: Can a change to a schedule change the output of a Halide program?
   * It's useful to think of the Halide "scheduling language" (not the algorithm description language) as a language for manipulating a set of loop nests.  What does it mean (in code) to fuse one loop with another loop?  Writing some psuedocode might help you explain this.
   * Let's consider what type of programmer Halide provides the most value for. Ignoring the autoscheduler (and just considering the language algorithm expression language and scheduling language), what class of programmer do you think is the target of Halide?  Novice programmers? Experts in code optimization? Students that have taken a class like CS149? Why do you think so?
   * In your own words, in two-three sentences or less, attempt to summarize what you think is the most important idea in the design of Halide?
   * Advanced question: In my opinion, there is one major place where the core design philosophy of Halide is violated. It is described in Section 4.3 in the paper, but is more clearly described in Section 8.3 of the Ph.D. thesis. (See sliding window optimizations and storage folding).  Why do you think am I claiming this compiler optimization is a significant departure from the core principles of Halide? (There are also valid arguments against my opinion.)
      * Hint: what aspects of the program’s execution is not explicitly described in the schedule in these situations?
      
__Post-Lecture Required Reading: (to read AFTER lecture 4)__

* [Learning to Optimize Halide with Tree Search and Random Programs](https://halide-lang.org/papers/halide_autoscheduler_2019.pdf). Adams et al. SIGGRAPH 2019 
   * This paper documents the design of the modern autoscheduling algorithm that is now implemented in the Halide compiler.  This is a very technical paper, so I recommend that you adopt the "read for high-level understanding first, then dive into some details" reading strategy I suggested in class. Your goal should be to get the big points of the paper, not all the details.
   * The back-tracking tree search used in this paper is certainly not a new idea (you might have implemented algorithms like this in an introductory AI class), but what was interesting was the way the authors formulated the code scheduling problem as a sequence of choices that could be optimized using tree search. Please summarize how scheduling a Halide program is modeled as a sequence of choices. Specifically, what are the choices at each step?
      * Note: one detail you might be interested to take a closer look at is the "coarse-to-fine refinement" part of Section 3.2. This is a slight modification to a standard back tracking tree search.  
   * An optimizer's goal is to minimize a cost metric.  In the case of this paper, the cost is the runtime of the scheduled program. Why is a machine-learned model used to *predict the scheduled program's runtime*?  Why not just compile the program and run it on a real computer to measure its cost?
   * The other interesting part of this paper is the engineering of the learned cost model.  This part of the work was surprisingly difficult, perhaps the hardest part of the project. Observe that the authors do not present an approach based on end-to-end learning where the input is a Halide program DAG and the output is the estimated cost of this program. Instead they use traditional compiler analysis of the program's AST to compute a collection of program *features*, then what is learned is how to weight these features when estimating cost (See Section 4.2). For those of you with a bit of deep learning background, I'm interested in your thoughts here.  Do you like the hand-engineered features approach?  Why not go end-to-end?
      * Alternatively, in 2024, an obvious thing to try would be to provide an LLM the Halide source code and ask the LLM what it things the program's cost is?  Do you think this would work? Why or why not?
   
__Other Recommended Readings:__

* [Decoupling Algorithms from the Organization of Computation for High Performance Image Processing](http://people.csail.mit.edu/jrk/jrkthesis.pdf). Ragan-Kelley (MIT Ph.D. thesis, 2014)
   * Please read Chapters 1, 4, 5, and 6.1 of the thesis
   * You might also find this [CACM article](https://cacm.acm.org/magazines/2018/1/223877-halide/fulltext?mobile=false) on Halide from 2018 an accessible read.
* [Halide Language Website](http://halide-lang.org/) (contains documentation and many tutorials)
* Check out this useful [Youtube Video](https://www.youtube.com/watch?v=3uiEyEKji0M) on Halide scheduling
* [Efficient automatic scheduling of imaging and vision pipelines for the GPU](https://dl.acm.org/doi/abs/10.1145/3485486). Andersen et al. OOPSLA 2021.
* [Differentiable Programming for Image Processing and Deep Learning in Halide](https://people.csail.mit.edu/tzumao/gradient_halide/). Li et al. SIGGRAPH 2018
* [Searching for Fast Demosaicking Algorithms](https://dl.acm.org/doi/full/10.1145/3508461) Ma et al. SIGGRAPH 2022
* [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/system/files/osdi18-chen.pdf) Chen et al. OSDI 2018
   * [TVM](https://tvm.apache.org/) is another system that provides Halide-like scheduling functionality, but targets ML applications. (See Section 4.1 in the paper for a description of the schedule space) 
* [Learning to Optimize Tensor Programs](https://arxiv.org/abs/1805.08166). Chen et al. NIPS 2018

## Lecture 6: Efficient DNN Inference and Scheduling ##

* [Lecture slides](https://gfxcourses.stanford.edu/cs348k/spring24/lecture/dnnscheduling/)

__Post-Lecture Required Reading:__

* [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://arxiv.org/abs/1704.04760). Jouppi et al. ISCA 2017
   * Like many computer architecture papers, the TPU paper includes a lot of *facts* about details of the system.  I encourage you to understand these details, but look past all the complexity and try and look for the main lessons learned: things like motivation of the architects, key constraints they were working under, key principles in their resultign design. Here are the questions I'd like to see you address.
   * What was the motivation for Google to seriously consider the use of a custom processor for accelerating DNN computations in their datacenters, as opposed to using CPUs or GPUs? (Section 2)
   * I'd like you to resummarize how the `matrix_multiply` operation works.  More precisely, can you flesh out the details of how the TPU carries out the work described in this sentence at the bottom of page 3: "A matrix operation takes a variable-sized B*256 input, multiplies it by a 256x256 constant weight input, and produces a B*256 output, taking B pipelined cycles to complete". Don't worry if you can't, we'll talk about it in class.
   * We are going to talk about the "roofline" charts in Section 4 during class. Roofline plots are a useful tool for understanding the performance of software on a system. These graphs plot the max performance of the chip (Y axis) given a program with an arithmetic intensity (X -- ratio of math operations to data access). How are these graphs used to assess the performance of the TPU and to characterize the workloads run on the TPU? (which workloads making good use of the TPU?)
    * Section 8 (Discussion) of this paper is an outstanding example of good architectural thinking.  Make sure you understand the points in this section as we'll discuss a number of them in class.  Particularly for us in CS348K, what is the point of the bullet "Pitfall: Architects have neglected important NN tasks."?

__Other Recommended Readings:__
* [Stanford CS231: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/).
    * If you haven't taken CS231N, I recommend that you read through the lecture notes of modules 1 and 2 for very nice explanation of key topics.
* [An Introduction to different Types of Convolutions in Deep Learning](https://towardsdatascience.com/)
* [NVIDIA CUTLASS Github repo](https://github.com/NVIDIA/cutlass)
* [NVIDIA CuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/index.html)
* [Facebook Tensor Comprehensions](https://research.fb.com/announcing-tensor-comprehensions/)
    * The associated Arxiv paper is [Tensor Comprehensions: Framework-Agnostic High-Performance Machine Learning Abstractions](https://arxiv.org/abs/1802.04730), Vasilache et al. 2018.

## Lecture 7: Hardware Acceleration of DNNs ##

* [Lecture slides](https://gfxcourses.stanford.edu/cs348k/spring24/lecture/dnnhardware/)

__Recommended Readings:__
* NVIDIA Tensor Core
    * <https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf>
    * <https://www.anandtech.com/show/12673/titan-v-deep-learning-deep-dive/3>
    * <https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/>
    * <https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9926-tensor-core-performance-the-ultimate-guide.pdf>
* Google TPU v3
    * HotChips: <https://hotchips.org/assets/program/conference/day2/HotChips2020_ML_Training_Google_Norrie_Patil.v01.pdf>
    * HotChips: <https://hotchips.org/assets/program/tutorials/HC2020.Google.SameerKumarDehaoChen.v02.pdf>
    * <https://www.nextplatform.com/2018/05/10/tearing-apart-googles-tpu-3-0-ai-coprocessor/>
    * <https://cloud.google.com/tpu/docs/system-architecture>
  * Cerebras WSE
    * HotChips: <https://hotchips.org/assets/program/tutorials/HC2020.Cerebras.NataliaVassilieva.v02.pdf>
    * Corporate Whitepaper... <https://cerebras.net/resources/achieving-industry-best-ai-performance-through-a-systems-approach/>
* NVIDIA DLA (open source)
    * <http://nvdla.org>
* GraphCore IPU
    * <https://www.graphcore.ai/products/ipu>
* Microsoft Brainwave
    * <https://www.microsoft.com/en-us/research/uploads/prod/2018/03/mi0218_Chung-2018Mar25.pdf>
* SambaNova's Cardinal (very little public documentation)
    * <https://sambanova.ai/>

## Lecture 8: Generative AI for Image Creation (Part I) ##

* [Lecture slides](https://gfxcourses.stanford.edu/cs348k/spring24/lecture/generative1/)

__Recommended Readings:__

 * [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543). Zhang et al. ArXiv 2023.
 * [InstructPix2Pix: Learning to Follow Image Editing Instructions](https://www.timothybrooks.com/instruct-pix2pix/). Brooks et al. CVPR 2023.
 * [Prompt-to-Prompt Image Editing with Cross-Attention Control](https://prompt-to-prompt.github.io/). Hertz et al. ArXiv 2022.
 * [Blended Diffusion: Text-driven Editing of Natural Images](https://omriavrahami.com/blended-diffusion-page/). Avrahami et al. CVPR 2022.
 * [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://dreambooth.github.io/). Ruiz et al. CVPR 2023.
 * [Collage Diffusion](https://arxiv.org/abs/2303.00262). Sarukkai et al. ArXiv 2023.
 * [LooseControl: Lifting ControlNet for Generalized Depth Conditioning](https://shariqfarooq123.github.io/loose-control/). Bhat et al. Arxiv 2023.
 * [Block and Detail: Scaffolding Sketch-to-Image Generation](https://arxiv.org/abs/2402.18116). Sarukkai et al. Arxiv 2024
 * [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/). Blog by Lilian Weng, 2021

## Lecture 9: Generative AI for Image Creation (Part II) ##

* [Lecture slides](https://gfxcourses.stanford.edu/cs348k/spring24/lecture/generative2/)

__Post-Lecture Required Reading:__

* [Unpredictable Black Boxes are Terrible Interfaces](https://magrawala.substack.com/p/unpredictable-black-boxes-are-terrible). M. Agrawala 2023
  * This is a recent blog post by Stanford Professor Maneesh Agrawala that dives into a fundamental challenge of using recent generative AI tools to create content. I'd like you to react to the blog post in the context of the image generation task we performed together in class: using generative AI to make a poster for a Stanford dance event.  (Alternatively, feel welcome to try a free online generative AI tool like [Krea.ai](https://www.krea.ai/apps/image/realtime), [Clibdrop](https://clipdrop.co/stable-diffusion-turbo), or [Midjourney's free tier](https://www.imagine.art/) and try a design exercise of your own!  
  * Please describe the concept of "repair strategies" discussed in the blog. In your words describe the concept of repair, and give one example of a repair strategy that might be used in an image creation process.
  * What does Agrawala claim is the objective of of "establishing common ground" when working with another human, or in our case, a digital AI tool?
  * The central thesis of the blog post is that it's the unpredictability of how inputs (e.g., text strings) map to outputs (images) that causes us so much trouble using generative AI.  Specifically in the case of our class experience (or your own experience making images), how did the unpredictability of the system inhibit our ability to create the target image.  Please give specific examples in your answer.  If you are making an image on your own, a thorough answer might document a sequence of attempts to achieve a goal, discuss why and how you changed your prompts on each step.
  * I'd like to you think about controls _you wish you had_ when you were performing this task? Given some examples of __operations__ or __commands__ that you would like to have to control the system? (commands need not be text, they could be sliders, etc.) In the language of Agrawala's blog post, would you prefer to express your goals in the form of "repairs", "constraints", or in some other way?  In your answer, you make wish to skim through some of the "other recommended readings" (given in the prior lecture) that offer more advanced editing controls for image generation using generative AI.
  * Finally, pick one of the controls that you listed in the previous question, and describe the ``conceptual model" that a user (you) have while using the tool (see section in the blog post about "conceptual model" vs "true system model".  Can you think of how to reduce the problem of training an AI to have a similar conceptual model to the problem of creating paired training data for the task?  Many of the examples we discussed in class followed this pattern.   

__Other Recommended Readings:__

 * [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2112.10752.pdf). Rombach et al. CVPR 2022
 * [Cascaded Diffusion Models for High Fidelity Image Generation](https://cascaded-diffusion.github.io/). Ho et al. JMLR 2022
 * [On Distillation of Guided Diffusion Models](https://arxiv.org/abs/2210.03142). Meng et al. CVPR 2023
 * [Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise](https://arxiv.org/abs/2208.09392). Bansal et al. NeurIPS 2023 

## Lecture 10: Generating Video, Animation, 3D Geometry, Worlds and More ##

* [Lecture slides](https://gfxcourses.stanford.edu/cs348k/spring24/lecture/generative3/)

__Other Recommended Readings:__
 * [Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models](https://research.nvidia.com/labs/toronto-ai/VideoLDM/). Blattman et al. CVPR 2023
 * [Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127). Blattmann et al. 2023 
 * [DreamFusion: Text-to-3D using 2D Diffusion](https://dreamfusion3d.github.io/). Poole et al. 2002
 * [Zero-1-to-3: Zero-shot One Image to 3D Object](https://zero123.cs.columbia.edu/). Liu et al. 2023
 * [MDM: Human Motion Diffusion Model](https://guytevet.github.io/mdm-page/). Tevet et al. ICLR 2023

## Lecture 11: Creating AI Agents (Including LLM-based Problem Solving) ##

* [Lecture slides](https://gfxcourses.stanford.edu/cs348k/spring24/lecture/aiagents1)

__Pre-Lecture Required Reading:__

 * [Generative Agents: Interative Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442). Park et al. CHI 2023
   * Generating plausible agents that behave "like humans" has long been an interest of video game designers seeking to create non-playable characters. But agents that behave realistically have many other applications as well: they can serve as proxies for software testers to find bugs in games or help designers assess the playability or difficulty of game levels.  If we think more broadly, behavior that emerges from many agents performing plausible tasks over time in a simulated world can potentially give rise to global phenomenon such as organization of teams or the creation of empires (as anyone that's played games like The Sims might have experienced! :-)) This paper is about designing simulated agents that leverage queries to large-language models (e.g. ChatGPT) to produce interesting behavior without significant hand-coded logic or programmed rules. This paper touches on a number of themes from the course, and I'd like you to think about the following questions:
   * First let's start with some technical details. The paper's experiments are performed in a small "Sims"-like work called Smallville. The key subroutine used by agents in this paper is a query to a stateless large language model (LLM). For those of you that have used ChatGPT or similar systems like Google's Bard, just picture this module working like those systems. The input query is a text string of finite length (e.g., a few thousand characters), and the output of the LLM is text string response. It's easy to picture how to code-up a "bot" to operate within Smallville (use game APIs to move to place X, turn stove to "on", etc.), and it's easy to understand how one could generate prompts for an LLM and receive responses, the agents described in this paper need to translate the text string responses from the LLM to agent actions in the game. What is the mechanism for turning LLM responses into agent actions in the game? (For example, if the agent is in a bedroom and the LLM says the character should clean up the kitchen, how does the agent turn this direction into actions in the game?) This is discussed in Section 5.
   * The paper hypothesizes that this stateless module (with small, finite inputs) will be insufficient for creating characters that behave over long time scales in a consistent and rational way. Summarize the reasons for this challenge? (hint: consider continuity)
   * To address the challenge described above, the paper's solution is to "summarize" a long history of the agent into a finite-length input for the LLM.  There are two parts to this approach. The first is the "memory stream".  Describe what the memory stream's purpose is in the agent architecture.  Then describe how retrieval is used to select what data from the memory stream should be used in each query. (Why doesn't the system just provide the entire memory stream to a query?)
   * Of course, over a long simulation, enough activity happens to an agent that a memory stream grows quite long.  One way to address this might be to ask ChatGPT to generate a summary of a long text string into a shorter one.  But the authors go with a different approach that they call __reflection__. How is reflection implemented and give your thoughts on this approach, which indeed is a form of summarization of the memory stream.
   * Ideas in a paper can sometimes sound really interesting, but then you get to the evaluation section and realize that the cool ideas aren't really that helpful.  This is a particularly hard piece of work to evaluate, and I'd like you to take a detailed look at the evaluation sections (Section 6 and 7). How do the authors evaluate their work? What do you think?  Do you believe that important aspects of the agent architecture have merit?
   * BTW, code is here (https://github.com/joonspk-research/generative_agents), and it's been replicated in by the [AI Town Project](https://www.convex.dev/ai-town).

__Post-Lecture Required Reading:__

* [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://voyager.minedojo.org/)
    * This is another paper that uses pretrained language models as the key engine for creating autonomous agents for playing a game.  In this case, the focus is on an "open world" game, Minecraft. Note that after reading the paper, you might want to see the Voyager Algorithm" in Appendix A.1, and the actual prompt structure as given in Appendix A.3.4. Here are some questions to respond to:
    * In many ways the structure of the solution is similar to the examples we discussed in class:  There's an LLM tasked to emit a "plan" describing what the Minecraft character should do. The plan is expressed as Python code which makes calls to a Minecraft API which actually controls the in-game character. The LLM uses techniques such as in-context prompting and reflection to produce successful plans.  However, in this paper there's a new idea that we didn't see in prior work: the LLM is responsible for generating the next task to perform itself!  Talk about (a) how the system proposes new tasks to complete (b) why the task order matters ("curriculum"), and (c) how a successful completion of a task grows the API that the agent has access to in the future.
    * What is the "skill library" that the agent has access to?
    * When a plan does not successfully solve a task, the agent receives two forms of feedback. One might come from the python interpreter, another is from the Minecraft engine itself. What are the two forms of feedback and how does the system use that feedback to "try again" and make a new plan in the hopes of succeeding?
    * Let's turn our attention to evaluation.  What is the key metrics that the authors use as a proxy for "better"?
    * What are the key aspects of the system that you think are the most important to evaluate?  Remember, think about what ideas are proposed as "good ideas", and the evaluation should show evidence that these ideas matter in terms of improving the key metrics.

__Other Recommended Readings:__

* [ProgPrompt: Generating Situated Robot Task Plans using Large Language Models](https://progprompt.github.io/). Singh et al. ICRA 2023
* [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903.pdf). Wei et al. NeurIPS 2022
* [ViperGPT: Visual Inference via Python Execution for Reasoning](https://viper.cs.columbia.edu/). Menon et al. ICCV 2023 
* [Lil' Logs LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/). Blog by Lilian Weng 2023
* [A Survey on Large Language Model-Based Game Agents](https://github.com/git-disl/awesome-LLM-game-agent-papers). List maintained by Sihao Hu

## Lecture 12: Fast 3D World Simulation for Model Training (Part I) ##

* [Lecture slides](https://gfxcourses.stanford.edu/cs348k/spring24/lecture/trainingsim)

__Post-Lecture Required Reading:__

* [An Extensible, Data-Oriented Architecture for High-Performance, Many-World Simulation](https://madrona-engine.github.io/shacklett_siggraph23.pdf). Shacklett et al. SIGGRAPH 2023

If you were to create a computer game today, you'd probably not write the entire game from scratch. Instead, you'd choose to write your game using the APIs provided by a game engine framework, such as  e.g., [Unity](https://unity.com/), [Unreal](https://www.unrealengine.com/), or [Phaser](https://phaser.io/)) because it would be not only far more productive to do so, but also because you'd probably not be able to implement key parts of your game (like advanced rendering, physics, input collection from controllers, etc) as well as domain experts in these areas. In other words, existing engines provide valuable, well-implemented building blocks for creating a game, which allows game developers to focus on implementing the _logic and content specific to their game_ (specific game rules, creating worlds, etc.).

In this paper open the open source [Madrona Engine](https://madrona-engine.github.io/) the authors observed that there was an emerging need to create simulators that execute at very high throughput when running a "batch" of thousands of independent instances of a world. Early examples of these "batch simulators" can be found here for [Atari games](https://arxiv.org/abs/1907.08467) (github [here](https://github.com/NVlabs/cule)), [robotics physics simulations](https://arxiv.org/abs/2108.10470), and [navigation of indoor environments](https://graphics.stanford.edu/projects/bps3D/). These batch simulators were all written from scratch. This paper is based on a simple claim: it is likely that in the near future there will be a need to make many more unique simulators for training agents like computer game bots, robots, etc., and that not everyone that wants to create a simulator will an expert at high-performance programming on GPUs.  Therefore, there should be a "game framework" for high performance batch simulators.  

As you read the paper, please respond to the following questions:

* As always, make sure you read and understand the requirements and goals of the system as presented in sections 1 and 2.  The paper lists these goals clearly. Please make sure you understand all of them, but I'd like you to focus your response on the "PERFORMANCE" goal. Specifically, performance in this paper does not mean "run in parallel on the GPU", it means "efficiently run in parallel on the GPU".  For those that have taken CS149, think back to the basic principles of SIMD execution and efficient memory access. What are the key ideas in this paper that pertain specifically to the "high performance" goal?
  * Hint: GPUs perform best when they can execute the same instruction across many threads AND, when those threads access data, high-performance GPU memory performs best when adjacent threads are accessing adjacent memory addresses.     

* There are two major abstractions in the presented system: components and the computation graph (there's a good description in Section 4).  Let's focus on components first.  How are components for all worlds stored in a single table? Give at least one reason why this leads to high GPU performance.  (See Section 5.1). Note, it may be helpful to dive into the description of how Madrona implements component deletion as a way to check your understanding of this part of the paper (see the text related to Figure 2.)

* The authors choose to implement all game logic components as a computation graph (all instances use the same graph), and then execute the computation graph for ALL game instances by running a single graph node N for all instances, then moving on to the next graph node.  Give at lease one reason why this leads to high GPU performance (possible answers lie in rest of Section 5).

* This is a good paper to dig into the evaluation, so we can discuss in class what questions the evaluation is trying to answer.  There are four configurations evaluated in Section 7, (BATCH-ECS-GPU, ECS-GPU, ECS-CPU, and REF-CPU).  For each of the configurations, please provide an explanation of how the configuration is used in a comparison to make a claim about "X is better than Y".  For example, the paper provides ECS-CPU as a good high-performance C++ implementation because comparing just BATCH-ECS-GPU to REF-CPU alone doesn't prove the ideas in the paper are the reason for the observed speedups.  For example, with only that comparison, the speedup could be due to the faster speed of a GPU vs a CPU, or low performance of Python code vs. high oerformance CUDA code.   
  * Hint: the goal of a scientific paper evaluation is to show that the ideas in the paper have merit.  It is not to show that the authors are better programmers than the programmers of prior work.

* Ignoring robotics applications, are there good reasons to train AI agents for the purpose of making video games better?  What might you do with a very high performance batch simulator?

* Finally, interested students might wish to take a look at the [Madrona web site](https://madrona-engine.github.io/), or even dig into some [example game stater code](https://github.com/shacklettbp/madrona_escape_room).  We'd love it if you wanted to write your own game in Madrona! ;-)

__Other Recommended Readings:__
* [Accelerating Reinforcement Learning through GPU Atari Emulation](https://arxiv.org/abs/1907.08467). Dalton et al. NeurIPS 2020.
* [Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning](https://arxiv.org/abs/2108.10470). Makoviychuk et al. 2021
* [Large Batch Simulation for Deep Reinforcement Learning](https://graphics.stanford.edu/projects/bps3D/). Shacklett et al. ICLR 2021.
* [EnvPool Github Repo](https://github.com/sail-sg/envpool)
* [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html). GPU-accelerated MuJoCo.



