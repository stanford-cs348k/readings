# Reading and Discussion List for Stanford CS348K

This page contains discussion prompts for papers on the reading list for the Stanford course CS348K: Visual Computing Systems, taught by [Kayvon Fatahahalian](http://graphics.stanford.edu/~kayvonf/). You can find the web site for most recent offering of the course here: <http://cs348k.stanford.edu>.

## Lecture 1: Course Introduction ##

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

## Lecture 2: Digital Camera Processing Pipeline (Part I) ##

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
  * The course notes from [Stanford CS448A](http://graphics.stanford.edu/courses/cs448a-10/) remain a very good reference for camera image processing pipeline algorithms and issues. This was an old Stanford course taught by Marc Levoy.
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

For the required reading for the next class, please see required readings under lecture 4. During class we focused our discussion on the architecture of the Frankencamera virtual machine, and continued the lecture on algorithms used in a modern digital camera processing pipeline. For suggested going further readings, please see the list of readings below.

__Other Recommended Readings:__

* Please also see the recommending readings under Lecture 3 on the topic of image processing algorithms.
* [Synthetic Depth-of-Field with a Single-Camera Mobile Phone](http://graphics.stanford.edu/papers/portrait/wadhwa-portrait-sig18.pdf). Wadha et al. SIGGRAPH 2018.
    * This is a paper about the implementation of "Portrait Mode" in Google Pixel smartphones. It is a dense paper, similar to the HDR+ paper from 2016, but it is a detailed description of how the system works under the hood.
* [Handheld Mobile Photography in Very Low Light](https://google.github.io/night-sight/). Liba et al. SIGGRAPH Asia 2019
    * This is a paper about the implementation of "Night Sight" in Google Pixel smartphones.  

## Lecture 4: Efficiently Scheduling Image Processing Algorithms ##

__Pre-Lecture Required Reading: (two papers)__

* [Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines](http://people.csail.mit.edu/jrk/halide-pldi13.pdf). Ragan-Kelley, Adams, et al. PLDI 2013 
   * Note: Alternatively you may read the selected chapters in the Ragan-Kelley thesis linked below in recommended readings. (Or the CACM article.) The thesis chapters involve a little more reading than the paper, but in my opinion they are a more accessible explanation of the topic, so I recommend it for students.
   * In reading this paper, I want you to specifically focus on trying to describe the _philosophy of Halide_ in your own words. Specifically, if we ignore the "auto-tuner" described in Section 5 of the paper (you can skip this section), what is the expected role/responsibilities of the Halide programmer? and what is the role/responsibilities of the Halide system/compiler?
      * Hint 1: Which of the two (the programmer or the compiler) is responsible for major optimization decisions?
      * Hint 2: Can a change to a schedule change the output of a Halide program?
   * It's useful to think of the Halide "scheduling language" (not the algorithm description language) as a domain-specific language for manipulating a set of loop nests.  What does it mean to fuse one loop with another loop?  Writing some psuedocode might help you explain this.
   * Let's consider what type of programmer Halide provides the most value for. Ignoring the autoscheduler (and just considering the algorithm expression language and the scheduling language), what class of programmer do you think is the target of Halide?  Novice programmers? Experts in code optimization? Students that have taken a class like CS149? Why do you think so?
   * In your own words, in a few sentences or less, attempt to summarize what you think is the most important idea in the design of Halide?
   * Advanced question: In my opinion, there is one major place where the core design philosophy of Halide is violated (albiet necessarily so to get good performance). It is described in Section 4.3 in the paper, but is more clearly described in Section 8.3 of the Ph.D. thesis. (See sliding window optimizations and storage folding).  Why do you think am I claiming this compiler optimization is a departure from the core principles of Halide? (There are also valid arguments against my opinion.)
      * Hint: what aspects of the program’s execution is not explicitly described in the schedule in these situations?

* [Learning to Optimize Halide with Tree Search and Random Programs](https://halide-lang.org/papers/halide_autoscheduler_2019.pdf). Adams et al. SIGGRAPH 2019 
   * This paper documents the design of the modern autoscheduling algorithm that is now implemented in the Halide compiler.  This is a very technical paper, so I recommend that you adopt the "read for high-level understanding first, then dive into some details" reading strategy I suggested in class. Your goal should be to get the big points of the paper, not all the details.
   * The back-tracking tree search used in this paper is certainly not a new idea (you might have implemented algorithms like this in an introductory AI class), but what was interesting was the way the authors formulated the code scheduling problem as a sequence of choices that could be optimized using tree search. Please summarize how scheduling a Halide program is modeled as a sequence of choices. Specifically, what are the choices at each step?
      * Note: one detail you might be interested to take a closer look at is the "coarse-to-fine refinement" part of Section 3.2. This is a slight modification to a standard back-tracking tree search.  What is the motivation for this approach?  
   * An optimizer's goal is to minimize a cost metric.  In the case of this paper, the cost is the runtime of the scheduled program. Why is a machine-learned model used to *predict the scheduled program's runtime*?  Why not just compile the program and run it on a real computer to measure its cost?
   * The other interesting part of this paper is the engineering of the learned cost model.  This part of the work was surprisingly difficult, perhaps the hardest part of the project. Observe that the authors do not present an approach based on end-to-end learning where the input is a Halide program text string or Halide function DAG and the output is the estimated cost of this program. Instead they use traditional compiler analysis of the program's AST to compute a collection of program *features*, then learned how to weight these features when estimating cost (See Section 4.2). For those of you with a bit of deep learning background, I'm interested in your thoughts here.  Do you like the hand-engineered features approach?  Why not go end-to-end?
   * Alternatively, in 2025, an obvious thing to try would be to provide an LLM the Halide source code and ask the LLM what it things the program's cost is? Or perhaps even just ask an LLM to optimize the program. Do you think this would work? Why or why not? Note, take a look at [KernelBench](https://scalingintelligence.stanford.edu/blogs/kernelbench/), which is a benchmark for efforts to do just that.
   
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

## Lecture 5: Hardware Acceleration: What's in a Modern GPU + AI Accelerators ##
  
__Pre-Lecture Required Reading:__

* [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://arxiv.org/abs/1704.04760). Jouppi et al. ISCA 2017
   * Like many computer architecture papers, the TPU paper includes a lot of *facts* about details of the system.  I encourage you to understand these details, but look past all the complexity and try and look for the main lessons learned: things like motivation of the architects, key constraints they were working under, key principles in their resultign design. Here are the questions I'd like to see you address.
   * What was the motivation for Google to seriously consider the use of a custom processor for accelerating DNN computations in their datacenters, as opposed to using CPUs or GPUs? (Section 2)
   * I'd like you to resummarize how the `matrix_multiply` operation works.  More precisely, can you flesh out the details of how the TPU carries out the work described in this sentence at the bottom of page 3: "A matrix operation takes a variable-sized B*256 input, multiplies it by a 256x256 constant weight input, and produces a B*256 output, taking B pipelined cycles to complete". Don't worry if you can't, we'll talk about it in class.
   * We are going to talk about the "roofline" charts in Section 4 during class. Roofline plots are a useful tool for understanding the performance of software on a system. These graphs plot the max performance of the chip (Y axis) given a program with an arithmetic intensity (X -- ratio of math operations to data access). How are these graphs used to assess the performance of the TPU and to characterize the workloads run on the TPU? (which workloads making good use of the TPU?)
    * Section 8 (Discussion) of this paper is an outstanding example of good architectural thinking.  Make sure you understand the points in this section as we'll discuss a number of them in class.  Particularly for us in CS348K, what is the point of the bullet "Pitfall: Architects have neglected important NN tasks."?

__Other Recommended Readings:__

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
* [NVIDIA CUTLASS Github repo](https://github.com/NVIDIA/cutlass)
* [ThunderKittens: Simple, Fast, and Adorable AI Kernels](https://arxiv.org/abs/2410.20399) Specter et al. 2024.
* [Triton: An Intermediate Language and Compiler for
Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
* [Facebook Tensor Comprehensions](https://research.fb.com/announcing-tensor-comprehensions/)
    * The associated Arxiv paper is [Tensor Comprehensions: Framework-Agnostic High-Performance Machine Learning Abstractions](https://arxiv.org/abs/1802.04730), Vasilache et al. 2018.
