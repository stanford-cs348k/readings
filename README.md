# Reading and Discussion List for Stanford CS348K

This page contains discussion prompts for papers on the reading list for the Stanford course CS348K: Visual Computing Systems, taught by [Kayvon Fatahahalian](http://graphics.stanford.edu/~kayvonf/). You can find the web site for most recent offering of the course here: <http://cs348k.stanford.edu>.

__Please keep in mind that all reading responses should always include a response to the following prompt:__ 

Describe your top N (N < 3) takeaways from the discussions in the last class. 
 * What was the most surprising/interesting thing you learned?
 * Is there anything you feel passionate about (Agreed with, disagreed with?) that you want to react to?
 * What was your big takeaway in general?

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

## Lecture 6: Adding Controls to Generative AI Systems ##

__Pre-Lecture Required Reading:__

* [Unpredictable Black Boxes are Terrible Interfaces](https://magrawala.substack.com/p/unpredictable-black-boxes-are-terrible). M. Agrawala 2023
  * This is a recent blog post by Stanford Professor Maneesh Agrawala that dives into a fundamental challenge of using recent generative AI tools to create content. As part of reading the blog post, I'd like you to give the argument some context by using a modern generative AI tool like [Krea.ai](https://www.krea.ai/apps/image/realtime), [Clibdrop](https://clipdrop.co/stable-diffusion-turbo), or [Midjourney's free tier](https://www.imagine.art/) or Adobe's [Firefly](https://firefly.adobe.com/), and use the tool to create an image for the following hypothetical scene: *Photo of a grassy university campus scene with two men and two women playing beach volleyball and a university tower in the background to the left, and a football stadium in the background to the right, puffy white clouds in the sky. One of the women is spiking the ball down on the man's head.* (Note: that's the scene I want you to create, not necessarily the prompt I want you to use.) I encourage you to post your best image on Ed for all to view!    
  * Please describe the concept of "repair strategies" discussed in the blog. In your words describe the concept of repair, and give one example of a repair strategy that might be used in an image creation process.
  * What does Agrawala claim is the objective of of "establishing common ground" when working with another human, or in our case, a digital AI tool?
  * The central thesis of the blog post is that it is the unpredictability of how inputs (e.g., text strings) map to outputs (images) that causes us so much trouble using generative AI.  Specifically in the case of our own image creation experience (see above), did the unpredictability of the system inhibit our ability to create the target imag?.  Please give specific examples in your answer.  A thorough answer might document a sequence of attempts to achieve a goal, discuss why and how you changed your prompts on each step.
  * I'd like to you think about controls _you wish you had_ when you were performing this task? Given some examples of __operations__ or __commands__ that you would like to have to control the system? (commands need not be text, they could be sliders, etc.) In the language of Agrawala's blog post, would you prefer to express your goals in the form of "repairs", "constraints", or in some other way?  In your answer, you make wish to skim through some of the "other recommended readings" (given in the prior lecture) that offer more advanced editing controls for image generation using generative AI.
  * Finally, pick one of the controls that you listed in the previous question, and describe the ``conceptual model" that a user (you) have while using the tool (see section in the blog post about "conceptual model" vs "true system model".  Can you think of how to reduce the problem of training an AI to have a similar conceptual model to the problem of creating paired training data for the task?  Many of the examples we discussed in class followed this pattern.   

__Other Recommended Readings:__

 * [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543). Zhang et al. ArXiv 2023.
 * [InstructPix2Pix: Learning to Follow Image Editing Instructions](https://www.timothybrooks.com/instruct-pix2pix/). Brooks et al. CVPR 2023.
 * [Prompt-to-Prompt Image Editing with Cross-Attention Control](https://prompt-to-prompt.github.io/). Hertz et al. ArXiv 2022.
 * [Blended Diffusion: Text-driven Editing of Natural Images](https://omriavrahami.com/blended-diffusion-page/). Avrahami et al. CVPR 2022.
 * [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://dreambooth.github.io/). Ruiz et al. CVPR 2023.
 * [Collage Diffusion](https://arxiv.org/abs/2303.00262). Sarukkai et al. ArXiv 2023.
 * [LooseControl: Lifting ControlNet for Generalized Depth Conditioning](https://shariqfarooq123.github.io/loose-control/). Bhat et al. Arxiv 2023.
 * [Block and Detail: Scaffolding Sketch-to-Image Generation](https://arxiv.org/abs/2402.18116). Sarukkai et al. Arxiv 2024
 * [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/). Blog by Lilian Weng, 2021
    
## Lecture 7: Neurosymbolic Systems for Generating and Interpreting Content ##

__Post-Lecture Required Reading:__
 * [MoVer: Motion Verification for Motion Graphics Animations](https://mover-dsl.github.io/). Ma et al. SIGGRAPH 2025
   * As mentioned in class, MoVer follows a design pattern that is being increasingly adopted across a variety of application domains: systems that LLMs to author programs that perform a desired task (in this case editing animations), and simultaneously the LLM is asked to generate auxiliary programs that *verify* that the output of the main program matches the task description. In other words, the LLM is responsible for generating a program to perform a task, and the correctness tests for that program. *To help you understand the paper better, I highly recommend you start by watching the video on the paper's web site, as well as inspect the examples on that page. Click on the example images to see a detailed breakdown of what MoVer generates, and how the LLM behaves after each correction step.* Please consider the following in your response:
   * In the MoVer system, the system generates two outputs (in two different "languages") given a prompt. The first is just a python program written against a common graphics API. Executing this program produces a "motion graphics animation", which, in the case of this paper, is a sequence of SVG files for each animation frame. The second is a program in the motion verification language that checks that certain properties hold for the motion graphics animation. *Describe the primitives of the domain-specific motion verification language.* (In your answer make sure to differentiable between single-object predicates and multi-object predicates.) 
   * To make sure you can interpret the logical statements expressed in MoVer, please give a plain text description of all the clauses in Equation 1. (The meaning of the statement is summarized in the paper right after Equation 1, but it would be helpful to describe the clause as "there exists a motion where...".
   * Note: You might find the description of the Allen temporal interval algebra useful [here](https://ics.uci.edu/~alspaugh/cls/shr/allen.html), as well as the 2D rectangle extensions (see citations in the paper)
   * In the evaluation section, the paper reports in Table 3 that for 59% of the generated animations, all tests after the first generation try, and in 95\% of cases, all verification tests pass after multiple animation generation tries.  What is the stated reason for the increased success with multiple tries?  Those that commonly play around with LLMs might have a question for the authors about a simple baseline that was not tested. (Hint: how do we know MoVer is providing a benefit?)
   * The paper states that the results in Table 3 were computed by using "ground truth" MoVer programs, not the MoVer programs generated by the system. Why do you think the authors did this? Is it good scientific evaluation?
   * Regardless of whether ground truth or system-generated MoVer programs are used, does passing all the MoVer tests guarantee the system behaved correctly, why or why not? (HintL what is the definition of "correct"?)
   * Take a look at the prompts used in Figures 1, 6, and 7.  Give your opinion about whether these prompts are representative of input you'd expect artists to provide.  What type of user do you believe the system anticipates? 
   * One of the limitations stated in the paper (see limitations section) is that the properties that can be verified are quite simple.  Speculate about one verifier you might like to add to the system, and how might you implement that check?

__Other Recommended Readings:__
 * [Learning to Generate Programs for 3D Shape Structure Synthesis](https://rkjones4.github.io/shapeAssembly.html) Jones et al. SIGGRAPH 2020
 * [Editing Motion Graphics Video via Motion Vectorization and Transformation](https://sxzhang25.github.io/publications/motion-vectorization.html) Zhang et al. SIGGRAPH Asia 2023 
 * [ProgPrompt: Generating Situated Robot Task Plans using Large Language Models](https://progprompt.github.io/) Sing et al. ICRA 2023
 * [The Scene Language: Representing Scenes with Programs, Words, and Embeddings](https://ai.stanford.edu/~yzzhang/projects/scene-language/) Zhang et al. CVPR 2025
 * [Iterative Motion Editing with Natural Language](https://purvigoel.github.io/iterative-motion-editing/) Goel et al. SIGGRAPH 2023

## Lecture 8: Curating Training Sets: The Unsung Hero of Generative AI ##

__Post-Lecture Required Reading:__
  * [DataComp: In search of the next generation of multimodal datasets](https://www.datacomp.ai/dcclip/index.html#home). Gadre et al. NeurIPS 2023
    * This is one of a small number of academic papers that begins to look at the design of the dataset that gomes into models, instead of the design of the model itself. Please consider the following questions:
    * This is a benchmarking paper, and in this paper, the authors argue that the field should hold the DNN MODEL DESIGN constant, and "compete" on the design of training datasets, rather than hold the training set constant and compete on the accuracy of the model.  What are the authors trying to get the community to explore and understand better?
    * In your reading, just focus on the "Filtering Track" of the benchmark (ignore the "Bring Your Own Data" part.) What is the "CommonPool" created by the authors?  What processing was done to create the "CommonPool"?
    * What are the motivations for creating multiple CommonPool sizes for the community? (Table 3)
    * What is the dataset DataComp-1B compared to CommonPool?  (You can see DataComp-1B in the last row of Table 1.)
    * Note that each data example is an image/text pair.  What do you think the motivation is for each of the "basic filtering", "text filtering", and "image filtering" types of filtering?  Does your intuition suggest they can help? Were there any filtering steps that didn't make much sense to you?
    * Give me your opinion on whether CLIP score filtering makes sense, and as well as the "distance to imageNet category" aspect of text filtering. Both of these made me stop and think a bit about whether they made sense to me.
    * How did the authors evaluate the quality of a given dataset? (You may need to Google CLIP what clip embedding is. In short, CLIP allows an image and a text string to be "embedded" in spaces where you can compute the "distance" between the image and the text via a dot product. An image and a text string are "similar" if they embed to similar vectors -- e.g., the value of the dot-product is large.)
    * Figure 2 is an interesting graph.  In your own words please describe the MAIN TAKEAWAY from the graph.  I'll start with one smaller takeaway to help aid your understanding: The graph shows that when randomly filtering data from the common pool to create a dataset, retaining more data in the training set leads to better model performance.
    * Figure 3 is also an interesting graph. What is the main point from this figure? In other words, why is it notable that there is high corrolation between results from SMALL commonPool dataset construction experiments and the MEDIUM commonPool dataset construction experiments?
    * Finally, after reading the paper, if you have some background in working with LLMs and popular LLM datasets, you might enjoy reading the DataComp-LM paper in the "other recommended readings" section.  It's a more modern version of the this paper from 2024 that focuses on curating text datasets for language model training.
      
__Other Recommended Readings:__
  * [DataComp-LM: In search of the next generation of training sets for language models](https://arxiv.org/abs/2406.11794). Li et al. NeurIPS 2024
     * This is a paper about curating datasets for text generation models, but its lessons are very applicable to visual data as well. 
  * [Visual Fact Checker: Enabling High-Fidelity Detailed Caption Generation](https://research.nvidia.com/labs/dir/vfc/). Ge et al. CVPR 2024
  * [Playground v3: Improving Text-to-Image Alignment with Deep-Fusion Large Language Models](https://arxiv.org/abs/2409.10695). Liu et al 2024
  * [ShareGPT4V: Improving Large Multi-Modal Models with Better Captions](https://sharegpt4v.github.io/). Chen et al. ECCV 2024
  * [Edify 3D: Scalable High-Quality 3D Asset Generation](https://arxiv.org/abs/2411.07135). Bala et al. 2024

## Lecture 9: The Role of Virtual World Simulation in Training Agents ##

  __Post-Lecture Required Reading:__
   * [Exploring Game Space of Minimal Action Games via Parameter Tuning and Survival Analysis](https://www.nealen.net/papers/08030128.pdf). Isaksen et al. 2018
      * In this paper the goal is to use an agent (loosely crafted to model aspects of human play) to predict game difficulty for a very simple space of games. Given this difficulty measure, the authors present several ideas for how it could be used to influence game design (build games with adaptive levels of difficulty, explore the game design space, etc.) Note: I recommend you begin by taking a few moments to play a version of Flappy Bird yourself, [which you can do here](https://flappybird.io/). Then consider the following questions:
      *  The authors often refer to the "game space" of Flappy Bird.  What is a "game space"? And for the case of Flappy Bird, how is that space parameterized? (e.g., give one example of a parameter.)
      *  I'd like you to think of another genre of game (or a specific game itself) that you enjoy, and hypothesize what the game space for it might look like? What might the most important parameters be? Is it easy to parameterize?  Is the meaning of each parameter sufficiently high level that it's hard to think of how you'd implement it? In your selected game genre, what's an example of a parameter that might logically be a good parameter to have to control, but it is completely unclear what the implementation of that parameter might be?
      * In this paper, the authors create a "human-like" agent by hand-crafting a specific algorithm for solving Flappy Bird challenges, and then limiting that agent with human-inspired difficulties.  One such limitation is the number of times a key can be pressed per second. Another limitation models misjudging when the key should be pressed.  How do the authors model a novice player? A highly skilled player?
      * An interesting assumption about the player model is that a novel player can be modeled by "corrupting" the decisions of an "expert" player.  Do you think this is a good model for an novice player?  What might be a problem with this approach? (e.g. do experts and novices differ only in how they can carry out a good strategy?).  In light of your answer, reflect on why the authors might have chosen Flappy Bird for this research project? 
      * Please define the hazard function h(x), and what is the meaning of a game where h(x)=0? (Is the game very easy or very hard?)
      * In Figure 15(a), the game is designed to get more challenging with each obstacle passed (as score goes up). However, the hazard rate h(x) is roughly constant in the bottom green line.  How is this possible?
      * In Figure 12(a), the shape of the curves is different depending on player skill (note this is a plot of f(x) -- the probability of getting score x, not hazard rate h(x)). Can you explain the difference in shape between the black line (highest skill) and bluest line (lowest skill)?
      * Give me a simple algorithm for generate a random Flappy Bird game from the game space that is "playable" for a given player profile.
      * Section 8c is titled "Computational Creativity".
         * What is the roll of the "human-like" agent in enabling automated "computational creativity"?
         * Philosophical question. Do you think the system described is being creative?
      * Finally, I'd like you to comment on the scalability of the approach discussed in this paper. Do you think it will work for more complex games? Why or why not? (Consider the challenges associated with a designing a human-like agent for more complex games.) 

## Lecture 10: High-Performance Simulation for Agent Training  (Part 1) ## 

__Post-Lecture Required Reading:__

* [An Extensible, Data-Oriented Architecture for High-Performance, Many-World Simulation](https://madrona-engine.github.io/shacklett_siggraph23.pdf). Shacklett et al. SIGGRAPH 2023

If you were to create a computer game today, you'd probably not write the entire game from scratch. Instead, you'd choose to write your game using the APIs provided by a game engine framework, such as  e.g., [Unity](https://unity.com/), [Unreal](https://www.unrealengine.com/), or [Phaser](https://phaser.io/)) because it would be not only far more productive to do so, but also because you'd probably not be able to implement key parts of your game (like advanced rendering, physics, input collection from controllers, etc) as well as domain experts in these areas. In other words, existing engines provide valuable, well-implemented building blocks for creating a game, which allows game developers to focus on implementing the _logic and content specific to their game_ (specific game rules, creating worlds, etc.).

In this paper the authors observed that there was an emerging need to create simulators that execute at very high throughput when running a "batch" of thousands of independent instances of a virtual world. Early examples of these "batch simulators" can be found here for [Atari games](https://arxiv.org/abs/1907.08467) (github [here](https://github.com/NVlabs/cule)), [robotics physics simulations](https://arxiv.org/abs/2108.10470), and [navigation of indoor environments](https://graphics.stanford.edu/projects/bps3D/). These batch simulators were all written from scratch, for example in parallel programming languages CUDA or JAX. This paper is based on a simple hypothesis: there will be a need to make many more unique simulators for training agents like computer game bots, robots, etc., and that not everyone that wants to create a simulator will an expert at high-performance programming on GPUs.  Therefore, there should be a "game framework" for authoring high-performance batch simulators.  

As you read the paper, please respond to the following questions:

* As always, make sure you read and understand the requirements and goals of the system as presented in sections 1 and 2.  The paper lists these goals clearly. Please make sure you understand all of them, but I'd like you to focus your response on the "PERFORMANCE" goal. Specifically, performance in this paper does not mean "run in parallel on the GPU", it means "efficiently run in parallel on the GPU".  For those that have taken CS149, think back to the basic principles of SIMD execution and efficient memory access. What are the key ideas in this paper that pertain specifically to the "high performance" goal?
  * Hint: GPUs perform best when they can execute the same instruction across many threads AND, when those threads access data, high-performance GPU memory performs best when adjacent threads are accessing adjacent memory addresses.     

* There are two major abstractions in the presented system: components and the computation graph (there's a good description in Section 4). Let's focus on components first.  How are components for all worlds stored in a single table? Give at least one reason why this leads to high GPU performance.  (See Section 5.1). Note, it may be helpful to dive into the description of how Madrona implements component deletion as a way to check your understanding of this part of the paper (see the text related to Figure 2.)

* The authors choose to implement all game logic components as a computation graph (all instances use the same graph), and then execute the computation graph for ALL game instances by running a single graph node N for all instances, then moving on to the next graph node.  Give at least one reason why this leads to high GPU performance (possible answers lie in rest of Section 5).

* This is a good paper to dig into the evaluation, so we can discuss in class what questions the evaluation is trying to answer.  There are four configurations evaluated in Section 7, (BATCH-ECS-GPU, ECS-GPU, ECS-CPU, and REF-CPU).  For each of the configurations, please provide an explanation of how the configuration is used in a comparison to make a claim about "X is better than Y".  For example, the paper provides ECS-CPU as a good high-performance C++ implementation because comparing just BATCH-ECS-GPU to REF-CPU alone doesn't prove the ideas in the paper are the reason for the observed speedups.  For example, with only that comparison, the speedup could be due to the faster speed of a GPU vs a CPU, or low performance of Python code vs. high-performance CUDA code. (Hint: The goal of a scientific paper evaluation is to show that the *presented ideas* in the paper have merit.  It is not to show that the authors are better programmers than the programmers of prior work.)

* Finally, interested students might wish to take a look at the [Madrona web site](https://madrona-engine.github.io/), or even dig into some example game starter code on the Madrona github repo.

__Other Recommended Readings:__
* [Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning](https://arxiv.org/abs/2108.10470). Makoviychuk et al. 2021
* [Kinetix: Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks](https://kinetix-env.github.io/). Matthews et al. 2024
* [Waymax: An Accelerated, Data-Driven Simulator for Large-Scale Autonomous Driving Research](https://arxiv.org/abs/2310.08710). Gulino et al. 2023
* [GPUDrive: Data-driven, multi-agent driving simulation at 1 million FPS](https://arxiv.org/abs/2408.01584). Kazemkhani et al. 2025

## Lecture 11: High-Performance Simulation for Agent Training (Part 2) ## 

* __Please see reading listed for lecture 10 above.__

## Lecture 12: Generative Interactive "World Models" ## 

__Pre-Lecture Required Reading:__

For this lecture, there are two required readings:
  
* [Genie: Generative Interactive Environments](https://sites.google.com/view/genie-2024/home). Bruce et al. 2024
   * So far in this course we've talked about foundation models for generating images (or videos, meshes, etc.).  We've also talked about implementing virtual world simulators that can be used as training grounds for AI agents.  There are two problems with the simulators we've discussed so far: First, despite the optimization efforts we talked about last class, it can be expensive to simulate detailed worlds. Second, if the goal is to train a general problem solving agent, it's hard to find a diverse set of worlds to train that agent on. There certainly aren't games spanning the range of task we'd like AI agents to perform! And it would be prohibitive to implement a new game for each new task. In this paper, the authors attempt to skip the challenges of designing and executing world simulators, and instead seek to learn an ___interactive world model___ from just video data that can respond to user input.  Note that the questions below stem from the Genie paper listed above. The second required reading is the Genie 2 article from late December 2024.  Let's first address these technical questions... 
   * The input to the training procedure is just unlabeled video.  Please describe what is meant by the "latent action" that the method attempts to infer.  
   * Please describe the inputs and outputs of each step of the process (the video tokenzier, the latent action model, and the dynamics model).  What is the responsibility of each step? The details of how the steps are implemented with a spatiotemporal transformer architecture are not relevant to our in-class discussion. (Students without ML background can skip the details of the implementation, but I want everyone to understand how the pieces fit together.)
   * Since the point of the paper is to create a model that can "generate interactive environments", the evaluation of the proposed method should assess how well this goal was achieved. The visual quality metric is just a metric for "does the experience look real", which has nothing to do with interactivity.  However, the authors offer up a new metric for assessing "controllability" (See section 3.0).  What is the intuition behind the metric they propose?  Do you buy the metric as a measure of "goodness" of an interactive environment?  What are questions you might ask about an interactive environment to assess whether it was a good one? Perhaps you might consider things like: consistency? plausibility? Challenging to solve in a particular way?  Tests a particular skill? Fun to play?
   * IMPORTANT: In class our discussion will be a debate about whether you believe over the next 10 years, if you wanted to train the most general and capable AI agents the world has ever seen... do you feel it is more promising to push forward on the approach of learning interactive world simulators from large-scale observations, or whether it's more promising to invest in technology of faster, more detailed world simulation engines like those discussed in the Madrona class session. __You have to take a side!__  Please be prepared to argue with your classmates in favor of one side of the debate.  Here are some issues to consider:
      *  What are the pros/cons of each approach? Consider:
          * The costs of making an interactive environment for a specific situation?  (run-time per frame of output costs? Content creation costs? Simulation implementation costs?)
          * Consider procedural approaches to creating content, like [ProcThor](https://procthor.allenai.org/) or [XLand](https://deepmind.google/discover/blog/generally-capable-agents-emerge-from-open-ended-play/) or [Kinetix](https://kinetix-env.github.io/).
          * How would you specify what an interactive environment should do?  (in the Genie paper they seem to simply provide a single starting image and a text Prompt)
          * How will you generate a huge diversity of experiences and worlds?
          * What does it mean for an AI agent to be a general problem solver?
* [Genie 2: A Large-Scale Foundation World Model](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) Parker-Holder et al. 2024
   * This is a required reading, but there are no questions specific to the reading (just focus on the questions above).  However, I wasn't you do read through this article to get a sense of the capabilities of modern world models (in late 2024).

__Other Recommended Readings:__

* [DayDreamer: World Models for Physical Robot Learning](https://arxiv.org/abs/2206.14176). We et al. 2024
* [Oasis: A Universe in a Transformer](https://oasis-model.github.io/) Decart/Etched 2024
* [Diffusion for World Modeling: Visual Details Matter in Atari](https://diamond-wm.github.io/) Alonso et al. 2024

## Lecture 13: LLM-Based Problem Solving Agents (Part 1) ##

__Pre-Lecture Required Reading:__

* [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://voyager.minedojo.org/)
    * This paper uses pretrained language models for creating autonomous agents for playing a game.  In this case, the focus is on an "open world" game, Minecraft, where the player can explore a world and perform a wide range of tasks.  Note that after reading the paper, you might want to see the "Voyager Algorithm" in Appendix A.1, and the actual prompt structure as given in Appendix A.3.4. Here are some questions to respond to:
    * In many ways the structure of the Voyager agent is similar to many other agents today, and used principles discussed in the neuro-symbolic generative models lecture. There's an LLM tasked to emit a "plan" describing what the Minecraft character should do. The plan is expressed as Python code (written by the LLM) which makes calls to a predefined Minecraft API (called [Mineflayer](https://github.com/PrismarineJS/mineflayer)) which actually controls the in-game character. The LLM uses techniques which we will discuss in class, such as in-context prompting and "reflection" to produce successful plans.  However, in this paper there's a new idea that's less common in other AI agent designs: *the LLM is responsible for generating the next task to perform itself!*  Please describe (a) how the system proposes new tasks to complete (b) why the order of the tasks matters (what is a "curriculum"), and (c) how a successful completion of a task results in an expansion of the API the agent has access to in the future.
    * What is the "skill library" that the agent has access to?
    * When a plan does not successfully solve a task, the agent receives two forms of feedback. One comes from the python interpreter and the other is from the Minecraft engine itself. What are the two forms of feedback and how does the system use that feedback to "try again" and make a new plan in the hopes of succeeding?
    * Let's turn our attention to evaluation.  What are the key metrics that the authors use as a proxy for "better" than prior work?
    * What are the key aspects of the system that you think are the most important to evaluate?  Remember, think about what ideas are proposed as "good ideas", and the evaluation should show evidence that these ideas matter in terms of improving the key metrics.

__Other Recommended Readings:__

* [ProgPrompt: Generating Situated Robot Task Plans using Large Language Models](https://progprompt.github.io/). Singh et al. ICRA 2023
* [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903.pdf). Wei et al. NeurIPS 2022
* [ViperGPT: Visual Inference via Python Execution for Reasoning](https://viper.cs.columbia.edu/). Menon et al. ICCV 2023 
* [Lil' Logs LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/). Blog by Lilian Weng 2023
* [A Survey on Large Language Model-Based Game Agents](https://github.com/git-disl/awesome-LLM-game-agent-papers). List maintained by Sihao Hu

## Lecture 14: LLM-Based Problem Solving Agents (Part 2) ##

__Pre-Lecture Required Reading:__

 * [Generative Agents: Interative Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442). Park et al. CHI 2023
   * Generating plausible agents that behave "like humans" has long been an interest of video game designers seeking to create non-playable characters. But agents that behave realistically have many other applications as well: they can serve as proxies for software testers to find bugs in games or help designers assess the playability or difficulty of game levels.  If we think more broadly, behavior that emerges from many agents performing plausible tasks over time in a simulated world can potentially give rise to global phenomenon such as organization of teams or the creation of empires (as anyone that's played games like The Sims might have experienced! :-)) This paper is about designing simulated agents that leverage queries to large-language models to produce interesting behavior without significant hand-coded logic or programmed rules. This paper touches on a number of themes from the course, and I'd like you to think about the following questions:
   * First let's start with some technical details. The paper's experiments are performed in a small "Sims"-like work called Smallville. The key subroutine used by agents in this paper is a query to a stateless large language model (LLM). The input query is a text string of finite length (e.g., a few thousand characters), and the output of the LLM is text string response. It's easy to picture how to code-up a "bot" to operate within Smallville (use game APIs to move to place X, turn stove to "on", etc.), and it's easy to understand how one could generate prompts for an LLM and receive responses, the agents described in this paper need to translate the text string responses from the LLM to agent actions in the game. What is the mechanism for turning LLM responses into agent actions in the game? (For example, if the agent is in a bedroom and the LLM says the character should clean up the kitchen, how does the agent turn this direction into actions in the game?) This is discussed in Section 5.
   * The paper hypothesizes that this stateless module (with small, finite inputs) will be insufficient for creating characters that behave over long time scales in a consistent and rational way. Summarize the reasons for this challenge? (hint: consider continuity)
   * To address the challenge described above, the paper's solution is to "summarize" a long history of the agent into a finite-length input for the LLM.  There are two parts to this approach. The first is the "memory stream".  Describe what the memory stream's purpose is in the agent architecture.  Then describe how retrieval is used to select what data from the memory stream should be used in each query. (Why doesn't the system just provide the entire memory stream to a query?)
   * Of course, over a long simulation, enough activity happens to an agent that a memory stream grows quite long.  One way to address this might be to ask ChatGPT to generate a summary of a long text string into a shorter one.  But the authors go with a different approach that they call __reflection__. How is reflection implemented and give your thoughts on this approach, which indeed is a form of summarization of the memory stream.
   * Ideas in a paper can sometimes sound really interesting, but then you get to the evaluation section and realize that the cool ideas aren't really that helpful.  This is a particularly hard piece of work to evaluate, and I'd like you to take a detailed look at the evaluation sections (Section 6 and 7). How do the authors evaluate their work? What do you think?  Do you believe that important aspects of the agent architecture have merit?
   * BTW, code is here (https://github.com/joonspk-research/generative_agents), and it's been replicated in by the [AI Town Project](https://www.convex.dev/ai-town).

## Lecture 15: Video Compression + Video Conferencing Systems ##

__Recommended Readings:__
 * [Warehouse-scale video acceleration: co-design and deployment in the wild](https://dl.acm.org/doi/abs/10.1145/3445814.3446723). Ranganathan et al. ASPLOS 2021
    * This is the Google VCU paper I discussed in lecture 
 * [Overview of the H.264/AVC Video Coding Standard](https://ieeexplore.ieee.org/document/1218189). Wiegand et al. IEEE TCSVT '03
 * [vbench: Benchmarking Video Transcoding in the Cloud](http://arcade.cs.columbia.edu/vbench-asplos18.pdf). Lottarini et al. ASPLOS 18
 * [Salsify: Low-Latency Network Video through Tighter Integration between a Video Codec and a Transport Protocol](https://snr.stanford.edu/salsify/). Fouladi et al. NSDI 2018
 * [Encoding, Fast and Slow: Low-Latency Video Processing Using Thousands of Tiny Threads](https://www.usenix.org/system/files/conference/nsdi17/nsdi17-fouladi.pdf). Fouladi et al. NSDI 17
 * [Gradient-Based Pre-Processing for Intra Prediction in High Efficiency Video Coding](https://link.springer.com/article/10.1186/s13640-016-0159-9). BenHajyoussef et al. 2017
 * [Neural Adaptive Content-Aware Internet Video Delivery](https://www.usenix.org/system/files/osdi18-yeo.pdf). Yeo et al OSDI 18.
 * [Learning Binary Residual Representations for Domain-specific Video Streaming](https://arxiv.org/pdf/1712.05087.pdf). Tsai et al. AAAI 18
 * [Nonverbal Overload: A Theoretical Argument for the Causes of Zoom Fatigue](https://tmb.apaopen.org/pub/nonverbal-overload/release/2). Bailenson 2021.

## Lecture 16: The DSPy System ##

__Recommended Readings:__
 * [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714). Khattab et al. 2023
    * Please see the [DSPy site](https://dspy.ai/) for more details.   

## Lecture 17: Finishing up Video Conferencing + The Role of Differentiable Rendering for Scene Reconstruction ##

__Post-Lecture Required Reading:__
 * [SLANG.D: Fast, Modular and Differentiable Shader Programming](https://research.nvidia.com/labs/rtr/publication/bangaru2023slangd/). Bangaru et al. SIGGRAPH Asia 2023
    * Discussion prompts to come asap (late Tuesday afternoon)... you may wish to skim the [original Slang language design paper](https://dl.acm.org/doi/pdf/10.1145/3197517.3201380) or read through [some examples on github](https://github.com/shader-slang/slang) to understand the concept interfaces and associated types.    

