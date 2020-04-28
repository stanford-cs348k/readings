# Reading and Discussion List for Stanford CS348K

* Course web site: <http://cs348k.stanford.edu>

## Lecture 1: Throughput Computing Review ##

* [Lecture slides](http://cs348k.stanford.edu/spring20/lecture/intro)

__Post-Lecture Required Readings: (2)__

* [The Compute Architecture of Intel Processor Graphics Gen9](https://software.intel.com/sites/default/files/managed/c5/9a/The-Compute-Architecture-of-Intel-Processor-Graphics-Gen9-v1d0.pdf). Intel Corporation
  * This is not an academic paper, but a whitepaper from Intel describing the architectural geometry of a recent GPU.  I'd like you to read the whitepaper, focusing on the description of the processor in Sections 5.3-5.5. Then, given your knowledge of the concepts discussed in lecture (such as superscalar execution, multi-core, multi-threading, etc.), I'd like you to describe the organization of the processor (using terms from the lecture, not Intel terms). For example, what is the basic processor building block? How many hardware threads does it support? What width of SIMD instructions are executed by those threads? Does it have superscalar execution capabilities? How many times is this block replicated for additional parallelism?
  * Consider your favorite data-parallel programming language, such as GLSL/HLSL shading languages, [CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html), OpenCL, [ISPC](https://ispc.github.io/), or just an OpenMP #pragma parallel for. Can you think through how an embarrassingly "parallel for" loop can be mapped to this architecture. (You don't need to write this down, but you could if you wish.)
  * For those that want to go futher, I also encourage you to read [NVIDIA's V100 (Volta) Architecture whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf), linked in the "further reading" below. Can you put the organization of this GPU in correspondence with the organization of the Intel GPU? You could make a table contrasting the features of a modern AVX-capable Intel CPU, Intel Integrated Graphics (Gen9), NVIDIA GPUs, etc.
  
* [What Makes a Graphics Systems Paper Beautiful](https://graphics.stanford.edu/~kayvonf/notes/systemspaper/). Fatahalian (2019)
  * A major theme of this course is "thinking like a systems architect". This short blog post discusses how systems artitects think about the intellectual merit and evaluation of systems.  Read the blog post, and click through to some of the paper links.  These are the types of issues, and the types of systems, we will be discussing in this class.
  * If you want to read ahead, give yourself some practice with identifying "goals and constraints" by looking at sections 1 and 2 of Google's paper [Burst Photography for High Dynamic Range and Low-Light Imaging on Mobile Cameras](https://research.google/pubs/pub45586/).  What were the goals and constraints underlying the design of the camera application in Google Pixel smartphones?  

__Other Recommended Readings:__

* [Volta: Programmability and Performance](https://www.hotchips.org/wp-content/uploads/hc_archives/hc29/HC29.21-Monday-Pub/HC29.21.10-GPU-Gaming-Pub/HC29.21.132-Volta-Choquette-NVIDIA-Final3.pdf). Hot Chips 29 (2017)
  * This Hot Chips presentation documents features in NVIDIA Volta GPU.  Take a good look at how a chip is broken down into 80 streaming multi-processors (SMs), and that each SM can issue up to 4 warp instructions per clock, and supports up to concurrent 64 warps.  You may also want to look at the [NVIDIA Volta Whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf).
* [The Story of ISPC](https://pharr.org/matt/blog/2018/04/18/ispc-origins.html). Pharr (2018)
  * Matt Pharr's multi-part blog post is an riveting description of the history of [ISPC](https://ispc.github.io/), a simple, and quite useful, language and compiler for generating SIMD code for modern CPUs from a SPMD programming model.  ISPC was motivated by the frustration that the SPMD programming benefits of CUDA and GLSL/HLSL on GPUs could easily be realized on CPUs, provided applications were written in a simpler, constrained programming system that did not have all the analysis challenges of a language like C/C++.
* [Scalability! But at What COST?](http://www.frankmcsherry.org/assets/COST.pdf) McSherry, Isard, and Murray. HotOS 2015
  * The arguments in this paper are very consistent with the way we think about performance in the visual computing domain.  In other words, efficiency and raw performance are different than "scalable".
  
## Lecture 2: Digital Camera Processing Pipeline Basics ##

* [Lecture slides](http://cs348k.stanford.edu/spring20/lecture/camerabasics)

__Post-Lecture Required Readings:__

* [Burst Photography for High Dynamic Range and Low-light Imaging on Mobile Cameras](https://research.google/pubs/pub45586/). Hasinoff et al. SIGGRAPH Asia 2016
   * This is a *very technical* paper.  But don't worry, your job is not to understand all the technical details of the algorithms, it is to approach the paper with a systems mindset, and think about the end-to-end considerations that went into the particular choice of algorithms. In general, I want you to pay the most attention to Section 1, Section 4.0 (you can ignore the detailed subpixel alignment in 4.1), Section 5 (I will talk about why merging is done in the Fourier domain in class), and Section 6. Specifically, as you read this paper, I'd like you think about the following issues:
   * *Any good system typically has a philosophy underlying its design.*  This philosophy serves as a framework for which the system architect determines whether design decisions are good or bad, consistent with principles or not, etc. Page 2 of the paper clearly lists some of the principles that underlie the philosophy taken by the creators of the camera processing pipeline at Google. For each of the four principles, given an assessment of why the principle is important.
   * The main technical idea of this paper is to combine a sequence of *similarly underexposed photos*, rather than attempt to combine a sequence of photos with different exposures (the latter is called “bracketing”).  What are the arguments in favor of the chosen approach?  Appeal to the main system design principles. By the way, you can learn more about bracketing at these links.
     * <https://www.photoup.net/best-practices-for-shooting-bracketed-images-for-hdr/>
     * <https://www.youtube.com/watch?v=54JXUJhvFSs>
   * Designing a good system is about meeting design goals, subject to certain constraints. (If there were no constraints, it would be easy to use unlimited resources to meet the goals.) What are the major constraints of the system? For example are there performance constraints? Usability constraints? Etc.
   * Why is the motivation for the weighted merging process described in Section 5?  Why did the authors no use the far simpler approach of just adding up all the aligned images? (Again, can you appeal to the stated design principles?)
   * Finally, take a look at the “finishing” steps in Section 6.  Many of those steps should sound familiar to you after today’s lecture.

__Other Recommended Readings:__
* The old [Stanford CS448A course notes](http://graphics.stanford.edu/courses/cs448a-10/) remain a very good reference for camera image processing pipeline algorithms and issues.
* [Clarkvision.com](http://www.clarkvision.com/articles/index.html) has some very interesting material on cameras.
* [Demosaicking: Color Filter Array Interpolation](http://ieeexplore.ieee.org/document/1407714/). Gunturk et al. IEEE Signal Processing Magazine, 2005
* [Unprocessing Images for Learned Raw Denoising](https://www.timothybrooks.com/tech/unprocessing/). Brooks et al. CVPR 2019
* [A Non-Local Algorithm for Image Denoising](http://dl.acm.org/citation.cfm?id=1069066). Buades et al. CVPR 2005
* [A Gentle Introduction to Bilateral Filtering and its Applications](http://people.csail.mit.edu/sparis/bf_course/). Paris et al. SIGGRAPH 2008 Course Notes
* [A Fast Approximation of the Bilateral Filter using a Signal Processing Approach](http://people.csail.mit.edu/sparis/publi/2006/tr/Paris_06_Fast_Bilateral_Filter_MIT_TR.pdf). Paris and Durand. MIT Tech Report 2006 (extends their ECCV 2006 paper)

  
## Lecture 3: Digital Camera Processing Pipeline Basics ##

* [Lecture slides](http://cs348k.stanford.edu/spring20/lecture/camerapipeline2)

__Post-Lecture Required Readings:__

* [The Frankencamera: An Experimental Platform for Computational Photography](http://graphics.stanford.edu/papers/fcam/). A. Adams et al. SIGGRAPH 2010
   * Frankencamera was a paper written right about the time mobile phone cameras were becoming “acceptable” in quality, phones were beginning to contain a non-trivial amount of compute power, and computational photography papers we’re an increasingly hot topic in the SIGGRAPH community.  At this time many compelling image processing and editing techniques were being published, and many of them revolved around generating high quality photographs from a sequence of multiple shots or exposures.  However, current cameras at the time provided a very poor API to the camera hardware and its components.  In short, many of the pieces were there for a programmable camera platform to be built, but someone had to attempt to architect a coherent system to make them accessible.  Frankencamera was an attempt to do that: It involved two things:
      * The design of an API for programming cameras (a mental model of an abstract programmable camera architecture).
      * And two implementations of that architecture: an open camera reference design, and an implementation on a Nokia smartphone.
   * When you read the paper, we’re going to focus on the abstract architecture presented by a Frankencamera. Specifically I’d like you to think about the following:
      1. I’d like you to describe the major pieces of the Frankcamera abstract machine (the system’s nouns):  e.g., devices, sensors, processors, etc.
      2. Then describe the major operations the machine could perform (the system’s verbs).  In other words, would you say a “shot” is a command to the machine?  Or is a shot a set of commands?  Would you say the word “timeline” be a good word to use to describe a “shot”?
      3. What output does executing a shot generate?  How is a frame different from a shot?  Why is this distinction made by the system?
      4. Would you say that F-cam is a “programmable” camera architecture or a “configurable architecture”.  What kinds of “programs” does the abstract machine run? (Note: see question 2)
      5. How would you characterize the particular type of computational photography algorithms that F-cam seeks to support/facilitate/enable (provides value for)?
   * Students may be interested that vestiges of ideas from the Frankencamera can now be seen in the Android Camera2 API:
https://developer.android.com/reference/android/hardware/camera2/package-summary

__Other Recommended Readings:__
* [Synthetic Depth-of-Field with a Single-Camera Mobile Phone](http://graphics.stanford.edu/papers/portrait/wadhwa-portrait-sig18.pdf). Wadha et al. SIGGRAPH 2018.
   * This is a paper about the implementation of "Portrait Mode" in Google Pixel smartphones.
* [Exposure Fusion](http://ieeexplore.ieee.org/document/4392748/). Mertens et al. Computer Graphics and Applications, 2007
* [Fast Local Laplacian Filters: Theory and Applications](http://people.csail.mit.edu/sparis/publi/2014/tog/Aubry_14-Fast_Local_Laplacian_Filters.pdf). Aubry et al. Transactions on Graphics 2014
* [Local Laplacian Filters: Edge-aware Image Processing with a Laplacian Pyramid](https://people.csail.mit.edu/sparis/publi/2011/siggraph/). Paris et al. SIGGRAPH 2013
* [The Laplacian Pyramid as a Compact Image Code](http://ieeexplore.ieee.org/document/1095851/). Burt and Adelson, IEEE Transactions on Communications 1983.

## Lecture 4: Efficiently Scheduling Image Processing Algorithms ##

* [Lecture slides](http://cs348k.stanford.edu/spring20/lecture/halide)

__Post-Lecture Required Readings: (2)__
* [Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines](http://people.csail.mit.edu/jrk/halide-pldi13.pdf). Ragan-Kelley, Adams, et al. PLDI 2013 
   * Note: Alternatively you may read the selected chapters in the Ragan-Kelley thesis linked below in recommended readings.  The thesis chapters involve a little more reading than the paper, but it is a more accessible explanation of the topic, so I recommend it for students.
   * In reading this paper, I want you to specifically focus on describing the philosophy of Halide.  Specifically, if we ignore the "autotuner" described in Section 5 of the paper, what is the role of the programmer, and what is the role of the Halide system/compiler?
      * Hint 1: Which component is responsible for major optimization decisions?
      * Hint 2: Can a change to a schedule change the output of a Halide program?
   * Who do you think is the type of programmer targeted by Halide?  Novices? Experts? Etc.?
   * Advanced question: In my opinion, there is one major place where the core design philosophy of Halide is violated.  It is described in Section 4.3 in the paper, but is more clearly described in Section 8.3 of the Ph.D. thesis.  (see sliding window optimizations and storage folding).  Why do you think am I claiming this compiler optimization is a significant departure from the core principles of Halide? (there are also valid arguments against my opinion.)
      * Hint: what aspects of the program’s execution is not explicitly described in the schedule in these situations?
      
* [Learning to Optimize Halide with Tree Search and Random Programs](https://halide-lang.org/papers/halide_autoscheduler_2019.pdf). Adams et al. SIGGRAPH 2019 
   * This paper documents the design of the autoscheduling algorithm that is not implemented in the Halide compiler.  This is quite a technical paper, so I recommend that you adopt the "coarse to fine" reading structure that we discussed in class.  Your goal is to get the big points of the paper, not all the details.
   * The back-tracking tree search used in this paper is certainly not a new idea (you've probably implemented algorithms like this in an introductory AI class), but what was interesting was the way the authors formulated the scheduling problem as a sequence of choices that could be optimized using tree search.  Summarize how scheduling is modeled as a sequence of choices?
      * Note: one detail you might be interested to take a closer look at is the "coarse-to-fine refinement" part of Section 3.2. This is a slight modification to a standard backtracking tree search.  
   * An optimizer's goal is to minimize a cost.  In the case of this paper, the cost is the runtime of the scheduled program.  Why is a machine learned model used to *predict the scheduled program's runtime*?  Why not just compile the program and run it on a machine?
   * The other interesting part of this paper is the engineering of the learned cost model.  This was surprisingly difficult.  Observe that the authors do not present an approach based on end-to-end learning where the input is a Halide program DAG and the output is an estimated cost, instead they use compiler analysis to compute a collection of program features, and then what is learned is how to weight these features in estimating cost (See Section 4.2). For those of you with a bit of deep learning background, I'm interested in your thoughts here.  Do you like the hand-engineered features approach?  
   
__Other Recommended Readings:__
* [Decoupling Algorithms from the Organization of Computation for High Performance Image Processing](http://people.csail.mit.edu/jrk/jrkthesis.pdf). Ragan-Kelley (MIT Ph.D. thesis, 2014)
    * Please read Chapters 1, 4, 5, and 6.1 of the thesis
* [Differentiable Programming for Image Processing and Deep Learning in Halide](https://people.csail.mit.edu/tzumao/gradient_halide/). Li et al. SIGGRAPH 2018
* [Halide Language Website](http://halide-lang.org/) (contains documentation and many tutorials)
* Check out this useful [Youtube Video](https://www.youtube.com/watch?v=3uiEyEKji0M) on Halide scheduling
* [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/system/files/osdi18-chen.pdf) Chen et al. OSDI 2018
    * [TVM](https://tvm.apache.org/) is another system that provides Halide-like scheduling functionality, but targets ML applications. (See Section 4.1 in the paper for a description of the schedule space) 
* [Learning to Optimize Tensor Programs](https://arxiv.org/abs/1805.08166). Chen et al. NIPS 2018

## Lecture 5: Efficient DNN Inference (Software Techniques) ##

* [Lecture slides](http://cs348k.stanford.edu/spring20/lecture/dnneval)

__Post-Lecture Required Reading:__

* [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://arxiv.org/abs/1704.04760). Jouppi et al. ISCA 2017
   * Like many computer architecture papers, the TPU paper includes a lot of *facts* about details of the system.  I encourage you to understand these details, but try to look past all the complexity and try and look for the main lessons learned.  (motivation, key constraints, key principles in the design). Here are the questions I'd like to see you address.
   * What was the motivation for Google to seriously consider the use of a custom processor for accelerating DNN computations in their datacenters, as opposed to using CPUs or GPUs? (Section 2)
   * I'd like you to resummarize how the `matrix_multiply` operation works.  More precisely, can you flesh out the details of how the TPU carries out the work described in this sentence at the bottom of page 3: "A matrix operation takes a variable-sized B*256 input, multiplies it by a 256x256 constant weight input, and produces a B*256 output, taking B pipelined cycles to complete".
   * I'd like to talk about the "roofline" charts in Section 4.  These graphs plot the max performance of the chip (Y axis) given a program with an arithmetic intensity (X -- ratio of math operations to data access).  How are these graphs used to assess the performance of the TPU and to characterize the workload run on the TPU?
    * Section 8 (Discussion) of this paper is an outstanding example of good architectural thinking.  Make sure you understand the points in this section as we'll discuss a number of them in class.  Particularly for us in this class, what is the point of the bullet "Pitfall: Architects have neglected important NN tasks."?

__Strongly, Strongly Recommended Readings:__
* [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842), Szegedy et al. CVPR 2015 (this is the Inception paper).
    * You may also enjoy reading [this useful blog post](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202) about versions of the Inception network.
* [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861). Howard et al. 2017

__Other Recommended Readings:__
* [Stanford CS231: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/).
    * If you haven't taken CS231N, I recommend that you read through the lecture notes of modules 1 and 2 for very nice explanation of key topics.
* [An Introduction to different Types of Convolutions in Deep Learning](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d). by Paul-Louis Pröve (a nice little tutorial)
* [Facebook Tensor Comprehensions](https://research.fb.com/announcing-tensor-comprehensions/)
    * The associated Arxiv paper is [Tensor Comprehensions: Framework-Agnostic High-Performance Machine Learning Abstractions](https://arxiv.org/abs/1802.04730). Vasilache et al. 2018.
* [What is the State of Neural Network Pruning?](https://arxiv.org/abs/2003.03033) Blalock et al. MLSys 2020
    * This paper is a good read even if you are not interested in DNN pruning, because the paper addresses issues and common mistakes in how to compare performance-oriented academic work.
* [Progressive Neural Architecture Search](https://arxiv.org/abs/1712.00559). Liu et al. ECCV 2018

## Lecture 6: DNN Hardware Accelerators ##

* [Lecture slides](http://cs348k.stanford.edu/spring20/lecture/dnnhardware)

__Post-Lecture Required Reading:__

* There was no post-lecture required reading.  (We read the TPU paper the last time, so take a break!)

__Other Recommended Readings:__

* [SCNN: An Accelerator for Compressed-sparse Convolutional Neural Networks](https://arxiv.org/abs/1708.04485). Parashar et al. ISCA 2017
* [EIE: Efficient Inference Engine on Compressed Deep Neural Network](https://arxiv.org/abs/1602.01528), Han et al. ISCA 2016
* [vDNN: Virtualized Deep Neural Networks for Scalable, Memory-Efficient Neural Network Design](https://arxiv.org/abs/1602.08124), Rhu et al. MICRO 2016
* [Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Network](http://eyeriss.mit.edu/), Chen et al. ISCA 2016

## Lecture 7: Parallel DNN Training ##

* [Lecture slides](http://cs348k.stanford.edu/spring20/lecture/dnntrain)

__Post-Lecture Required Reading:__

* There are two required readings post Lecture 7, but they are *pre-reading* for Lecture 8, so please see the readings listed under Lecture 8.

__Other Recommended Readings:__
* [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677). Goyal et al. 2017
   * A nice description of why learning rate should scale with mini-batch size, and empirical studies of how to implement this intuition effectively.
* [ImageNet Training in Minutes](https://arxiv.org/abs/1709.05011), You et al. 2018
* [Scaling Distributed Machine Learning with the Parameter Server](https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf), Li et al. OSDI 2014
* [Deep Gradient Compression](https://arxiv.org/abs/1712.01887), Lin et al. ICLR 2018

## Lecture 8: Raising the Level of Abstraction for Model Creation ##

__Pre-Lecture Required Reading:__

* [Overton: A Data System for Monitoring and Improving Machine-Learned Products](https://arxiv.org/abs/1909.05372), Ré et al. 2019
* [Ludwig: a type-based declarative deep learning toolbox](https://arxiv.org/abs/1909.07930), Molino et al. 2019









  

