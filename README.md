# Reading and Discussion List for Stanford CS348K

* Web site for most recent offering of the course: <http://cs348k.stanford.edu/spring22>

## Lecture 1: Throughput Computing Review ##

* [Lecture slides](https://gfxcourses.stanford.edu/cs348k/spring22/lecture/intro/)

__Post-Lecture Required Readings/Videos: (2)__

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
  
## Lecture 2: Digital Camera Processing Pipeline (Part I) ##

* [Lecture slides](https://gfxcourses.stanford.edu/cs348k/spring22/lecture/camerapipeline1/)

__Post-Lecture Required Readings:__

* [What Makes a Graphics Systems Paper Beautiful](https://graphics.stanford.edu/~kayvonf/notes/systemspaper/). Fatahalian (2019)
  * A major theme of this course is "thinking like a systems architect". This short blog post discusses how systems artitects think about the intellectual merit and evaluation of systems.  Read the blog post, and click through to some of the paper links.  These are the types of issues, and the types of systems, we will be discussing in this class.  The "real" reading of the night is below, and I want you to analyze the author's arguments in the terms of the goals and constraints underlying the design of the camera application in Google Pixel smartphones.
* [Burst Photography for High Dynamic Range and Low-light Imaging on Mobile Cameras](https://research.google/pubs/pub45586/). Hasinoff et al. SIGGRAPH Asia 2016
   * This is a *very technical* paper.  But don't worry, your job is not to understand all the technical details of the algorithms, it is to approach the paper with a systems mindset, and think about the end-to-end considerations that went into the particular choice of algorithms. In general, I want you to pay the most attention to Section 1, Section 4.0 (you can ignore the detailed subpixel alignment in 4.1), Section 5 (I will talk about why merging is done in the Fourier domain in class), and Section 6. Specifically, as you read this paper, I'd like you think about the following issues:
   * *Any good system typically has a philosophy underlying its design.*  This philosophy serves as a framework for which the system architect determines whether design decisions are good or bad, consistent with principles or not, etc. Page 2 of the paper clearly lists some of the principles that underlie the philosophy taken by the creators of the camera processing pipeline at Google. For each of the four principles, given an assessment of why the principle is important.
   * The main technical idea of this paper is to combine a sequence of *similarly underexposed photos*, rather than attempt to combine a sequence of photos with different exposures (the latter is called “bracketing”).  What are the arguments in favor of the chosen approach?  Appeal to the main system design principles. By the way, you can learn more about bracketing at these links.
     * <https://www.photoup.net/best-practices-for-shooting-bracketed-images-for-hdr/>
     * <https://www.youtube.com/watch?v=54JXUJhvFSs>
   * Designing a good system is about meeting design goals, subject to certain constraints. (If there were no constraints, it would be easy to use unlimited resources to meet the goals.) What are the major constraints of the system? For example are there performance constraints? Usability constraints? Etc.
   * Why is the motivation for the weighted merging process described in Section 5?  Why did the authors not use the far simpler approach of just adding up all the aligned images? (Again, can you appeal to the stated design principles?)
   * Finally, take a look at the “finishing” steps in Section 6.  Many of those steps should sound familiar to you after today’s lecture.

__Other Recommended Readings:__
* The old [Stanford CS448A course notes](http://graphics.stanford.edu/courses/cs448a-10/) remain a very good reference for camera image processing pipeline algorithms and issues.
* [Clarkvision.com](http://www.clarkvision.com/articles/index.html) has some very interesting material on cameras.
* [Demosaicking: Color Filter Array Interpolation](http://ieeexplore.ieee.org/document/1407714/). Gunturk et al. IEEE Signal Processing Magazine, 2005
* [Unprocessing Images for Learned Raw Denoising](https://www.timothybrooks.com/tech/unprocessing/). Brooks et al. CVPR 2019
* [A Non-Local Algorithm for Image Denoising](http://dl.acm.org/citation.cfm?id=1069066). Buades et al. CVPR 2005
* [A Gentle Introduction to Bilateral Filtering and its Applications](http://people.csail.mit.edu/sparis/bf_course/). Paris et al. SIGGRAPH 2008 Course Notes
* [A Fast Approximation of the Bilateral Filter using a Signal Processing Approach](http://people.csail.mit.edu/sparis/publi/2006/tr/Paris_06_Fast_Bilateral_Filter_MIT_TR.pdf). Paris and Durand. MIT Tech Report 2006 (extends their ECCV 2006 paper)

  
## Lecture 3: Digital Camera Processing Pipeline (Part II) ##

* [Lecture slides](https://gfxcourses.stanford.edu/cs348k/spring22/lecture/camerapipeline2/)

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

## Lecture 4: Digital Camera Processing Pipeline (Part III) ####

* [Lecture slides](https://gfxcourses.stanford.edu/cs348k/spring22/lecture/camerapipeline3/)

Please see required readings under lecture 5.

## Lecture 5: Efficiently Scheduling Image Processing Algorithms ##

* [Lecture slides](https://gfxcourses.stanford.edu/cs348k/spring22/lecture/imagescheduling/)

__Pre-Lecture Required Reading: (to read BEFORE lecture 5)__
* [Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines](http://people.csail.mit.edu/jrk/halide-pldi13.pdf). Ragan-Kelley, Adams, et al. PLDI 2013 
   * Note: Alternatively you may read the selected chapters in the Ragan-Kelley thesis linked below in recommended readings.  The thesis chapters involve a little more reading than the paper, but it is a more accessible explanation of the topic, so I recommend it for students.
   * In reading this paper, I want you to specifically focus on describing the philosophy of Halide.  Specifically, if we ignore the "autotuner" described in Section 5 of the paper, what is the role of the programmer, and what is the role of the Halide system/compiler?
      * Hint 1: Which component is responsible for major optimization decisions?
      * Hint 2: Can a change to a schedule change the output of a Halide program?
   * Who do you think is the type of programmer targeted by Halide?  Novices? Experts? Etc.?
   * Advanced question: In my opinion, there is one major place where the core design philosophy of Halide is violated.  It is described in Section 4.3 in the paper, but is more clearly described in Section 8.3 of the Ph.D. thesis.  (see sliding window optimizations and storage folding).  Why do you think am I claiming this compiler optimization is a significant departure from the core principles of Halide? (there are also valid arguments against my opinion.)
      * Hint: what aspects of the program’s execution is not explicitly described in the schedule in these situations?
      
__Post-Lecture Required Reading: (to read AFTER lecture 5)__
* [Learning to Optimize Halide with Tree Search and Random Programs](https://halide-lang.org/papers/halide_autoscheduler_2019.pdf). Adams et al. SIGGRAPH 2019 
   * This paper documents the design of the autoscheduling algorithm that is not implemented in the Halide compiler.  This is quite a technical paper, so I recommend that you adopt the "coarse to fine" reading structure that we discussed in class.  Your goal is to get the big points of the paper, not all the details.
   * The back-tracking tree search used in this paper is certainly not a new idea (you've probably implemented algorithms like this in an introductory AI class), but what was interesting was the way the authors formulated the scheduling problem as a sequence of choices that could be optimized using tree search.  Summarize how scheduling is modeled as a sequence of choices?
      * Note: one detail you might be interested to take a closer look at is the "coarse-to-fine refinement" part of Section 3.2. This is a slight modification to a standard backtracking tree search.  
   * An optimizer's goal is to minimize a cost.  In the case of this paper, the cost is the runtime of the scheduled program? Why is a machine learned model used to *predict the scheduled program's runtime*?  Why not just compile the program and run it on a machine?
   * The other interesting part of this paper is the engineering of the learned cost model.  This part of the work was surprisingly difficult.  Observe that the authors do not present an approach based on end-to-end learning where the input is a Halide program DAG and the output is an estimated cost, instead they use compiler analysis to compute a collection of program features, and then what is learned is how to weight these features in estimating cost (See Section 4.2). For those of you with a bit of deep learning background, I'm interested in your thoughts here.  Do you like the hand-engineered features approach?  
   
__Other Recommended Readings:__
* [Decoupling Algorithms from the Organization of Computation for High Performance Image Processing](http://people.csail.mit.edu/jrk/jrkthesis.pdf). Ragan-Kelley (MIT Ph.D. thesis, 2014)
    * Please read Chapters 1, 4, 5, and 6.1 of the thesis
* [Differentiable Programming for Image Processing and Deep Learning in Halide](https://people.csail.mit.edu/tzumao/gradient_halide/). Li et al. SIGGRAPH 2018
* [Halide Language Website](http://halide-lang.org/) (contains documentation and many tutorials)
* Check out this useful [Youtube Video](https://www.youtube.com/watch?v=3uiEyEKji0M) on Halide scheduling
* [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/system/files/osdi18-chen.pdf) Chen et al. OSDI 2018
    * [TVM](https://tvm.apache.org/) is another system that provides Halide-like scheduling functionality, but targets ML applications. (See Section 4.1 in the paper for a description of the schedule space) 
* [Learning to Optimize Tensor Programs](https://arxiv.org/abs/1805.08166). Chen et al. NIPS 2018

## Lecture 6: Efficient DNN Inference (Software Techniques) ##

* [Lecture slides](http://cs348k.stanford.edu/spring22/lecture/dnninference)

__Post-Lecture Required Reading:__

* [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://arxiv.org/abs/1704.04760). Jouppi et al. ISCA 2017
   * Like many computer architecture papers, the TPU paper includes a lot of *facts* about details of the system.  I encourage you to understand these details, but try to look past all the complexity and try and look for the main lessons learned.  (motivation, key constraints, key principles in the design). Here are the questions I'd like to see you address.
   * What was the motivation for Google to seriously consider the use of a custom processor for accelerating DNN computations in their datacenters, as opposed to using CPUs or GPUs? (Section 2)
   * I'd like you to resummarize how the `matrix_multiply` operation works.  More precisely, can you flesh out the details of how the TPU carries out the work described in this sentence at the bottom of page 3: "A matrix operation takes a variable-sized B*256 input, multiplies it by a 256x256 constant weight input, and produces a B*256 output, taking B pipelined cycles to complete".
   * I'd like to talk about the "roofline" charts in Section 4.  These graphs plot the max performance of the chip (Y axis) given a program with an arithmetic intensity (X -- ratio of math operations to data access).  How are these graphs used to assess the performance of the TPU and to characterize the workload run on the TPU?
    * Section 8 (Discussion) of this paper is an outstanding example of good architectural thinking.  Make sure you understand the points in this section as we'll discuss a number of them in class.  Particularly for us in this class, what is the point of the bullet "Pitfall: Architects have neglected important NN tasks."?

__Strongly, Strongly Recommended Readings (Follow on from the lecture about inference):__
* [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842), Szegedy et al. CVPR 2015 (this is the Inception paper).
    * You may also enjoy reading [this useful blog post](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202) about versions of the Inception network.
* [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861). Howard et al. 2017

__Other Recommended Readings:__
* [Stanford CS231: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/).
    * If you haven't taken CS231N, I recommend that you read through the lecture notes of modules 1 and 2 for very nice explanation of key topics.
* [An Introduction to different Types of Convolutions in Deep Learning](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d), by Paul-Louis Pröve (a nice little tutorial)
* [Facebook Tensor Comprehensions](https://research.fb.com/announcing-tensor-comprehensions/)
    * The associated Arxiv paper is [Tensor Comprehensions: Framework-Agnostic High-Performance Machine Learning Abstractions](https://arxiv.org/abs/1802.04730), Vasilache et al. 2018.
* [What is the State of Neural Network Pruning?](https://arxiv.org/abs/2003.03033), Blalock et al. MLSys 2020
    * This paper is a good read even if you are not interested in DNN pruning, because the paper addresses issues and common mistakes in how to compare performance-oriented academic work.
* [Progressive Neural Architecture Search](https://arxiv.org/abs/1712.00559), Liu et al. ECCV 2018

## Lecture 7: DNN Hardware Accelerators ##

* [Lecture slides](http://cs348k.stanford.edu/spring22/lecture/dnnhardware)

__Post-Lecture Required Reading:__

* Following our discussion today.  Your reading exercise is to learn about and write up a summary of at least one major modern AI chip.  Information is hard to come by as some of these architectures are tightly guarded secrets, but try to comb whitepapers, blogs, hacker sites, etc. This is an open-ended writeup, but you might start with a summary on facts like:
    * Enumerating basic statistics: number of processing units, amount of on chip memory, amount of off-chip mem bandwdith.
    * How programmable are the cores?
    * How does DNN algorithms map to these resources?
    * Then I'd like to see you reflect on what is interesting to you about the architecture?
        * What concerns to you have?
        * Do you like the design?
  
 See the suggested readings below for a list of possible starting points.

* __In addition, please also see the required readings (pre-readings) for Thursday.__

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
* Apple's ML accelerator in their M12 chip
* Anything from companies on this list ;-)
    * <https://www.crn.com/slide-shows/components-peripherals/the-10-coolest-ai-chip-startups-of-2020>>
* Academic efforts, like:
    * [SCNN: An Accelerator for Compressed-sparse Convolutional Neural Networks](https://arxiv.org/abs/1708.04485), Parashar et al. ISCA 2017
    * [EIE: Efficient Inference Engine on Compressed Deep Neural Network](https://arxiv.org/abs/1602.01528), Han et al. ISCA 2016
    * [vDNN: Virtualized Deep Neural Networks for Scalable, Memory-Efficient Neural Network Design](https://arxiv.org/abs/1602.08124), Rhu et al. MICRO 2016
    * [Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Network](http://eyeriss.mit.edu/), Chen et al. ISCA 2016
* An excellent survey organizing different types of designs:
    * [Efficient Processing of Deep Neural Networks: A Tutorial and Survey](https://www.rle.mit.edu/eems/wp-content/uploads/2017/11/2017_pieee_dnn.pdf), Zhe et al. IEEE 2017

## Lecture 8: System Support for Curating Training Data ##

* [Lecture slides](http://cs348k.stanford.edu/spring22/lecture/trainingdata)

__Pre-Lecture Required Reading:__

There are two required readings for this lecture. Use the second reading to supplement the first. The prompt questions are shared across the readings.
* [Snorkel: Rapid Training Data Creation with Weak Supervision](http://www.vldb.org/pvldb/vol11/p269-ratner.pdf). Ratner et al. VLDB 2017.  
* [Snorkel DryBell: A Case Study in Deploying Weak Supervision at Industrial Scale](https://arxiv.org/abs/1812.00417). Bach et al. SIGMOD 2019
   * This is a paper about the deployment of Snorkel at Google.  Pay particular attention to the enumeration of "core principles" in Section 1 and the final "Discussion" in section 7.  *Skimming this paper in conjunction with the first required reading is recommended*.
   * You also might want to check out the [Snorkel company blog](https://www.snorkel.org/blog/) for more digestible examples.
* __Prompt questions:__
   * First let's get our terminology straight.  What is mean by "weak supervision", and how does it differ from the traditional supervised learning scenario where a training procedure is given a training set consisting of a set of data examples and a corresponding set of "ground truth" labels for each of these examples?  Since there are very few per-example labels, how does supervision get inserted into the system?
   * Like in all systems, I'd like everyone to pay particular attention to the design principles described in Section 1 of the Ratner et al. paper, as well as the principles defined in the Drybell paper. If you had to boil the entire philosophy of Snorkel down to one thing, what would you say it is?  Hint: look at principle 1 in the Ratner et al. paper. 
   * The main *abstraction* in Snorkel is the *labeling function*.  Please describe what the output interface of a labeling function is. Then, and most importantly, __what is the value__ of this abstraction.  Hint: you probably want to refer to the key principle of Snorkel in your answer.
   * What is the role of the __final model training__ part of Snorkel? (That is, training an off-the-shelf DNN architecture on the probabalistic labels produced by Snorkel.)  Why not just use the probablistic labels as the model itself?
   * One interesting aspect of Snorkel is the notion of learning from "non-servable assets". (This is more clearly articulated in the Snorkel DryBell paper, so take a look there). This is definitely not an issue that would be high on the list of academic concerns, but it is quite important. 
   * From the ML perspective, the key technical insight of Snorkel (and of most follow on Snorkel papers, see [here](https://arxiv.org/abs/1810.02840), [here](https://www.cell.com/patterns/fulltext/S2666-3899(20)30019-2), and [here](https://arxiv.org/abs/1910.09505) for some examples) is the mathematical modeling of correlations between labeling functions in order to more accurately estimate probabilistic labels.  We will not talk in detail about these generative algorithms in class, but many of you will enjoy learning about them.  
   * In general, I'd like you to reflect on Snorkel and (if time) some of the recommended papers below. (see the Rekall blog post, or the Model Assertions paper for different takes.) I'm curious about your comments. 

__Other Recommended Readings:__

* [Accelerating Machine Learning with Training Data Management](https://ajratner.github.io/assets/papers/thesis.pdf). Alex Ratner's Stanford Ph.D. Dissertation (2019)
   * This is the thesis that covers Snorkel and related systems. As was the case when we studied Halide, it can often be helpful to read Ph.D. theses, since they are written up after the original publication on a topic, and often include more discussion of the bigger picture, and also the widsom of hindsight.  I highly recommend Alex's thesis.  
* [Model Assertions for Monitoring and Improving ML Models](https://cs.stanford.edu/~matei/papers/2020/mlsys_model_assertions.pdf). Kang et al. MLSys 2020
    * A similar idea here, but with different human-provided priors about how to turn existing knowledge in a model into additional supervision.
* [Rekall: Specifying Video Events using Compositions of Spatiotemporal Labels](https://arxiv.org/abs/1910.02993), Fu et al. 2019
    * In the context of Snorkel, Rekall could be viewed as a system for writing labeling functions for learning models for detecting events in video.  Alternatively, from a databases perspective, Rekall can be viewed as a system for defining models by not learning anything at all -- and just having the query itself be the model.   
    * Blog post: <https://dawn.cs.stanford.edu/2019/10/09/rekall/>, code: <https://github.com/scanner-research/rekall>
* [Waymo's recent blog post on image retrieval systems as data-curation systems](https://blog.waymo.com/2020/02/content-search.html), Guo et al 2020.
* [Selection via Proxy: Efficient Data Selection for Deep Learning](https://cs.stanford.edu/people/matei/papers/2020/iclr_svp.pdf), Coleman et al. ICLR 2020.
* __The unsupervised or semi-supervised learning angle.__  We'd be remiss in the 2021 version of this class not to talk about the huge body of work that attempts to reduce the amount of labeled training data required using unsupervised learning techniques.
    * [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165). Brown et al. NeurIPS 2020. (The GPT-3 paper)
    * [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709). Chen et al. ICLM 2020 (The SimCLR paper)
    * [Unsupervised Learning of Visual Features by Contrasting Cluster Assignments](https://arxiv.org/abs/2006.09882). Caron et al. NeurIPS 2020. (The SwAV paper)
    * [Data Distillation: Towards Omni-Supervised Learning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Radosavovic_Data_Distillation_Towards_CVPR_2018_paper.pdf), Radosavovic et al. CVPR 2018

## Lecture 9: Raising the Level of Abstraction for Model Creation ##

* [Lecture Slides](http://cs348k.stanford.edu/spring22/lecture/dnnabstractions)

__Pre-Lecture Required Reading:__

* [Overton: A Data System for Monitoring and Improving Machine-Learned Products](https://arxiv.org/abs/1909.05372), Ré et al. 2019
* [Ludwig: a type-based declarative deep learning toolbox](https://arxiv.org/abs/1909.07930), Molino et al. 2019

Often when you hear about machine learning abstractions, we think about ML frameworks like [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/), or [MX.Net](https://mxnet.apache.org/).  Instead of having to write key model ML layers yourself, these frameworks present the abstraction of a ML graph "operator", and allow model creation by composing operators into DAGs.  However, the abstraction of designing models by wiring up data flow graphs of operators is still quite low.  One might characterize these abstractions as being targeted for an ML engineer---someone who has taken a lot of ML classes, and has experience implementing model architectures in TensorFlow, experience selecting the right model for the job, or with the know-how to adjust hyperparameters to make training successful.  
The two papers for our discussion are efforts to begin to raise the level of abstraction even higher.  These are systems that emerged out of two major companies (Overton out of Apple, and Ludwig out of Uber), and they share the underlying philosophy that some of the operational details of getting modern ML to work can be abstracted away from users that simply want to use technologies to quickly train, validate, and continue to maintain accurate models.  In short these two systems can be thought of as different takes on Karpathy’s software 2.0 argument, which you can read in this [Medium blog post](https://medium.com/@karpathy/software-2-0-a64152b37c35).  I’m curious about your thoughts on this post as well!

When reading these papers, please consider the following:

* A good system provides valuable services to the user.  So in these papers, who is the "user"? (what is their goal, what is their skillset?) What are the painful, hard, or tedious things that the systems are designed to do for the user?

* Another way we can think about these papers is that they are taking a position that existing systems are helping users with the wrong problem.  What types of problems are these systems really trying to help with (Hint: do you think they are more geared toward design of new ML model architectures, or getting the right training data into the system?)

* The following two (very similar) statements appear in the papers. First, what is the value of this separation?  Or at least what is the *future promise* of this separation?   (what system services does it enable?)
   * Overton: "Informally, the schema defines what the model computes but not how the model computes it."
   * Ludwig: "The higher level of abstraction provided by the type-based ECD architecture allows for a separation between what a model is expected to learn to do and how it actually does it."

* Following up on the previous question: Do you buy the claim that Ludwig truly separates what a model is expected to learn and how it learns it?  It seems like the user specifies a good bit about the dataflow of the solution.  What are your thoughts?  (It seems like Overton's abstractions are a lot closer to really "zero code" model design.)

* Let's specifically contrast the abstractions of Ludwig with that of a lower-level ML system like TensorFlow.  TensorFlow/MX.Net/PyTorch largely abstract ML model definition as a DAG of N-Tensor operations.  How is Ludwig different?  What are the operators and what are the data-types exchanged by operators?  What is the value of having richer types than just forcing all input/output data to be an N-D tensor?

__Other Recommended Readings:__
* Following this lecture, it would be good to take a look at a number of commercial offerings of model creating services. Examples include [Amazon Sagemaker](https://aws.amazon.com/sagemaker/), [Google AutoML](https://cloud.google.com/automl), and [Weights and Biases](https://wandb.ai/).  Or data labeling services like [Scale](https://scale.com/), [Hive Data](https://thehive.ai/hive-data), [Clarifai](https://www.clarifai.com/), etc.

## Lecture 10: Video Compression (Traditional and Learned) ##

* [Lecture Slides](http://cs348k.stanford.edu/spring22/lecture/videocompression)

__Post-Lecture Required Reading:__

* [Warehouse-scale video acceleration: co-design and deployment in the wild](https://dl.acm.org/doi/abs/10.1145/3445814.3446723). Ranganathan et al. ASPLOS 2021

This is a very recent paper about the deployment of fixed-function hardware for video encoding and decoding in Google datacenters (Youtube, Google Photos, etc).  I thought it was a great example of a systems paper describing goals, constraints, and cross-cutting issues. Please address the following in your summary:

* An interesting stat from the paper was that it takes over one CPU-hour of compute to encode 150 frames of 2160p video using VP9 (top of Section 4.5). State why companies like Google care so much about spending large amounts of computation to achieve very high quality (or similarly, very low bitrate) video encoding.  Why does more computation help? (the last part is a review from lecture).
* Please describe the reasons why Youtube must encode a video multiple times and at many resolutions/quality settings? And why is multiple output target (MOT) encoding a useful approach in this environment? Hint, consider low resolution output versions of a high-resolution input video.) 
* An interesting paragraph towards the end of Section 2 (see header "Video Usage Patterns at Scale") broke the workload down into videos that are viral and highly watched (constituting most of the watch time), videos that are modestly watched, and videos that are rarely watched at all. For each of these categories, describe whether you think the VCU is a helpful platform that that type of video.
* We all have laptops and mobile devices with hardware accelerators for video encoding, but the requirements for accelerator hardware in a datacenter are quite different.  What were some of the most interesting differences to you?  What are opportunities that the datacenter case affords that might not be there in the case of hardware for consumer devices?
* In Section 3.3 there is an interesting analysis of how much bandwidth is needed for the VCU hardware to avoid being BW-bound. Was is the role of custom data compression hardware on the VCU?
* There are key sections talking about the stateless design of the encoder. (End of 3.2). Give some reasons why, in a datacenter environment, the stateless approach is beneficial. 

__Other Recommended Readings:__

* [Overview of the H.264/AVC Video Coding Standard](https://ieeexplore.ieee.org/document/1218189). Wiegand et al. IEEE TCSVT '03
* [vbench: Benchmarking Video Transcoding in the Cloud](http://arcade.cs.columbia.edu/vbench-asplos18.pdf). Lottarini et al. ASPLOS 18
* [Encoding, Fast and Slow: Low-Latency Video Processing Using Thousands of Tiny Threads](https://www.usenix.org/system/files/conference/nsdi17/nsdi17-fouladi.pdf). Fouladi et al. NSDI 17
* [Gradient-Based Pre-Processing for Intra Prediction in High Efficiency Video Coding](https://link.springer.com/article/10.1186/s13640-016-0159-9). BenHajyoussef et al. 2017
* [The Design, Implementation, and Deployment of a System to Transparently Compress Hundreds of Petabytes of Image Files for a File-Storage Service](https://arxiv.org/abs/1704.06192). Horn et al. 2017.
* [Neural Adaptive Content-Aware Internet Video Delivery](https://www.usenix.org/system/files/osdi18-yeo.pdf). Yeo et al OSDI 18.
* [Learning Binary Residual Representations for Domain-specific Video Streaming](https://arxiv.org/pdf/1712.05087.pdf). Tsai et al. AAAI 18

## Lecture 11: The Future of Video Conferencing Systems ##

* [Lecture Slides](http://cs348k.stanford.edu/spring22/lecture/videoconferencing)

__Post-Lecture Required Reading:__
* [Beyond Being There](http://worrydream.com/refs/Hollan%20-%20Beyond%20Being%20There.pdf). Hollan and Stornetta. CHI 1992

This is classic paper from the early human-computer interaction community that challenges the notion that recreating reality in digital form should be the "north star" goal of the design of virtual environments.  We're now 30 years past the paper, and technology and our ways of communicating using technology has progressed significantly, but one might argue given all the recent talk about "The Metaverse" that technologists may still be making the same mistakes. Please address the following in your summary:

* In the section titled "Being There", the authors provide a "crutch vs. shoes" analogy.  Consider modern video-conferencing systems you might use on a daily basis, and does the analogy apply?
* I'd like you to reflect on your usage of Zoom/Teams/Meet/Slack/Discord/etc. and consider what are the features of these systems that cause you to choose to use them.  Consider the modalities of text, audio, and video. When and why do you choose to use each?  Are there situations where too much resolution or too much capture fidelity hurts the experience of communicating?
* Now think about a topic that you are all experts in: online/virtual classes at Stanford.  Can you think of examples where professors used the pandemic as an opportunity to communicate in a way that was "better than being there"?  Can you think of situations where classes tried to replicate the experience of "being there" and it fell flat? Comment on how you think the themes of "beyond being there" might apply to virtual education and teaching?
* The authors hypothesize: "What if we were able to create communication tools that were richer than face-to-face?"  But of course, this was back in 1992.  Today we have technologies and systems that arguably answer this question in the positive.  What tools, platforms, systems come to mind to you?
* The idea of __intersubjectivity__ discussed in the document is interesting.  What are ways that current tools fail in this regard and how might they be improved?  Note: one good example of a success in the intersubjectivity department is the simple animated "dot dot dot" indicating someone on the other side of a DM conversation is typing.  Perhaps one failure example is how little information we get out of the online/offline indicator on Slack.
* So is Facebook, er. I mean Meta, right?  Is a high-fidelity VR social experience going to change the way we work with others? Or will the next big innovation in how with communicate be something more like Instastram, TikTok, or WhatsApp? 

__Recommended Readings:__
* [SVE: Distributed Video Processing at Facebook Scale](https://research.fb.com/wp-content/uploads/2017/10/sosp-226-cameraready.pdf). Huang et al. SOSP 2017
* [Salsify: Low-Latency Network Video through Tighter Integration between a Video Codec and a Transport Protocol](https://snr.stanford.edu/salsify/). Fouladi et al. NSDI 2018
* [Nonverbal Overload: A Theoretical Argument for the Causes of Zoom Fatigue](https://tmb.apaopen.org/pub/nonverbal-overload/release/2). Bailenson 2021.
* [Social Translucence: An Approach to Designing Systems that Support Social Processes](https://dl.acm.org/doi/10.1145/344949.345004). Erickson and Kellogg. TOCHI 2000.

## Lecture 12: The Light Field (NeRF Preliminaries) ##

* [Lecture Slides](https://gfxcourses.stanford.edu/cs348k/spring22/lecture/lightfield/)

__Recommended Readings:__
* [Light Field Rendering](https://graphics.stanford.edu/papers/light/). Levoy and Hanrahan 1996.
* [Jump: Virtual Reality Video](https://research.google/pubs/pub45617/). Anderson et al. SIGGRAPH Asia 2016
* [Instant 3D Photography](http://visual.cs.ucl.ac.uk/pubs/instant3d/). Hedman and Kopf. SIGGRAPH 2018

## Lecture 13: NeRF Training and Rendering ##

* [Lecture Slides](https://gfxcourses.stanford.edu/cs348k/spring22/lecture/nerfrendering/)

__Recommended Readings:__
* [Representing Scenes as Neural Radiance Fields for View Synthesis](https://www.matthewtancik.com/nerf) Mildenhall et al. ECCV 2020

## Lecture 14: Scheduling the 3D Graphics Pipeline ##

* [Lecture Slides](https://gfxcourses.stanford.edu/cs348k/spring22/lecture/gfxscheduling/)

__Recommended Readings:__
* [A Trip Down the LOL Graphics Pipeline](https://engineering.riotgames.com/news/trip-down-lol-graphics-pipeline). A nice introductory blog post for Riot Games that illustrates all the different rendering passes used to construct a League of Legends scene. Note how each of these passes draws geometry under different graphics pipeline state configurations.
* [A Trip Through the Graphics Pipeline 2011](https://fgiesen.wordpress.com/2011/07/09/a-trip-through-the-graphics-pipeline-2011-index/). A much more detailed blog post by Fabian Giesen describing the Direct3D 10-class pipeline
* [Pomegranate: A Fully Scalable Graphics Architecture](http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15869-f11/www/readings/eldridge00_pomegranate.pdf). M. Eldridge et al. SIGGRAPH 2000
* [Life of a Triangle - NVIDIA's Logical Pipeline](https://developer.nvidia.com/content/life-triangle-nvidias-logical-pipeline). C. Kubisch (NVIDIA GameWorks Blog, 2015)
* [Fast Tessellated Rendering on Fermi GF100](http://attila.ac.upc.edu/wiki/images/d/db/HPG10_Hot3D_Fermi.pdf). T. Purcell (High Performance Graphics Hot3D talk)
* [A Sorting Classification of Parallel Rendering](http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15869-f11/www/readings/molnar94_sorting.pdf). S. Molnar et al. IEEE Computer Graphics and Applications, 1994.

## Lecture 15: Hardware Accelerated Real-Time Ray Tracing ##

* [Lecture Slides](https://gfxcourses.stanford.edu/cs348k/spring22/lecture/rtrt/)

__Post-Lecture Require Reading:__
* [Embree: A Kernel Framework for Efficient CPU Ray Tracing](https://www.embree.org/papers/2014-Siggraph-Embree.pdf) Wald et al. SIGGRAPH 2014.

This is a paper about modern efficient ray tracing framework for modern CPUs. (This is not a GPU hardware-accelerated ray tracer).  I chose it since some of the content in our paper tracks the lecture nicely.  Note that Section 5.2 is on BVH construction, which we did not talk about in class, so feel free to skim. When reading please consider the following:

* Embree is not a ray tracer application. It is a small library of basic primitives that is meant for developers that want to implement a high performance multi-core CPU-based ray tracer.  In your own words, can you list what the three major primitives are (what do they do)?  (Figures 2 and 3 should help). Note that in the conclusion section (Section 9) the authors note some of the drawbacks of their decision to provide a set of individual kernels instead of of a full ray tracer.

* Why do you think Intel made the decision to release a library of important kernels as opposed to a ray tracer?  You could appear to prior kernel libraries we've discussed like cuDNN or matrix multiplication libraries.

* One service provided by Embree is the contstruction of the BVH.  What is the branching factor of Embree's BVH?  

* A major concern of the implementation is making efficient use of SIMD execution units on modern Intel CPUs. (which can be 4, 8, or 16-wide depending on the chip).  The paper describes two ways of vectorizing ray-BVH intersection code: "single ray traversal" (Section 5.1.2) and "packet traversal" (5.1.3). In your own words describe the strategy on how each attempt to parallel the code.  What are the strengths/weaknessness of each approach when considering "coherent" primary rays from the camera and "incoherent" rays that result from light bouncing around the scene. In particular what motivates the "hybrid" method described in Section 5.1.4 that the paper claims is 50% faster than packet tracing alone?  (It might be helpful to refer to today's lecture slides on packet tracing.)  

* In Table 2 of the evaluation, notice the performance (rays/second) of primary rays (rays emitted from the camera) is consistently higher than ray tracing throughput achieved when doing full path tracing.  Recall path tracing will include rays needed to compute light bouncing around the scene.  Why is this the case.  You may want to refer to the packet vs single ray implementations described earlier in the paper, or consider cache locality.

__Other Recommended Readings:__
 * [OptiX: A General Purpose Ray Tracing Engine](https://research.nvidia.com/publication/optix-general-purpose-ray-tracing-engine). Parker et al. SIGGRAPH 2010
 * [Architecture Considerations for Tracing Incoherent Rays](https://research.nvidia.com/publication/architecture-considerations-tracing-incoherent-rays). Aila et al. HPG 2010
 * [An energy and bandwidth efficient ray tracing architecture](https://dl.acm.org/doi/10.1145/2492045.2492058). Kopta et al. HPG 2013 
 * [Introduction to DirectX Ray Tracing](http://intro-to-dxr.cwyman.org/). SIGGRAPH 2018 Course
 * [DirectX Ray Tracing Functional Spec](https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#rays)

## Lecture 16: Hardware Accelerated Real-Time Ray Tracing (Part 2) ##

* [Lecture Slides](https://gfxcourses.stanford.edu/cs348k/spring22/lecture/rtrt2/)

__Recommended Readings:__
* [Fast Parallel Construction of High-Quality Bounding Volume Hierarchies](https://research.nvidia.com/sites/default/files/pubs/2013-07_Fast-Parallel-Construction/karras2013hpg_paper.pdf). Karras et al. HPG 2013 
* [Spatiotemporal Reservoir resampling for real-time ray tracing
with dynamic direct lighting](https://research.nvidia.com/sites/default/files/pubs/2020-07_Spatiotemporal-reservoir-resampling/ReSTIR.pdf). Bitterli et al. SIGGRAPH 2020
* [Interactive reconstruction of Monte Carlo image sequences using a recurrent denoising autoencoder](https://research.nvidia.com/sites/default/files/publications/dnn_denoise_author.pdf). Chaitanya
et al. SIGGRAPH 2017
* [Neural Supersampling for Real-time Rendering](https://research.facebook.com/publications/neural-supersampling-for-real-time-rendering/). Xiao et al. SIGGRAPH 2020

## Lecture 17: Guest Lecture: Brian Karis (Epic) ##

* [Lecture Slides](https://advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf)

__Recommended Video:__

* [Nanite: A Deep Dive](https://www.youtube.com/watch?v=eviSykqSUUw&list=PLabw4gCouThkLIavA-dn4QDmJ_3tqH4LK). B. Karis, SIGGRAPH 2021 Talk

__Other Recommended Readings:__
* [Efficient GPU Rendering of Subdivision Surfaces using Adaptive Quadtrees](http://www.niessnerlab.org/projects/brainerd2016efficient.html). W. Brainerd et al. SIGGRAPH 2016
* [DiagSplit: Parallel, Crack-Free, Adaptive Tessellation for Micropolygon Rendering](http://graphics.stanford.edu/papers/diagsplit/). Fisher et al. SIGGRAPH Asia 2019

## Lecture 18: Rendering and Simulation for Model Training ##

* [Lecture Slides](https://gfxcourses.stanford.edu/cs348k/spring22/lecture/renderforml/)

__Recommended Readings:__
* [Large Batch Simulation for Deep Reinforcement Learning](https://graphics.stanford.edu/projects/bps3D/). Shacklett et al. ICLR 2021.
* [SEED RL: Scalable and Efficient Deep-RL with Accelerated Central Inference](https://arxiv.org/abs/1910.06591). Espeholt et al. ICLR 2020 (Also see the Google [blog post](https://ai.googleblog.com/2020/03/massively-scaling-reinforcement.html))
* [Habitat: A Platform for Embodied AI Research](https://arxiv.org/abs/1904.01201). Savva et al. ICCV 2019. ([Project site](https://aihabitat.org/))
* [RoboTHOR: An Open Simulation-to-Real Embodied AI Platform](https://arxiv.org/abs/2004.06799). Deitke et al. CVPR 2020 ([Project site](https://ai2thor.allenai.org/robothor/))
* [CARLA: An Open Urban Driving Simulator](https://arxiv.org/abs/1711.03938). Dosovitskiy et al. CoRL 2017 ([Project site](https://carla.org/))
* [Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning](https://arxiv.org/abs/2108.10470). Makoviychuk et al. 2021 ([Project site](https://developer.nvidia.com/isaac-gym))

## Lecture 19: Guest Lecture: Yong He and Teresa Foley (NVIDIA) ##

__Pre-Lecture Required Reading:__
* [Slang: Language Mechanisms for Extensible Real-time Shading Systems](http://graphics.cs.cmu.edu/projects/slang/). Y. He et al. SIGGRAPH 2018

This is a challenging paper, so your goal when reading should be to get the big points and not necessarily understand all the details.  In particular, Section 4 requires a fair amount of graphics programming knowledge.  After getting oriented with the paper, I recommend that you put effort into understanding the code is Figure 1.  On the left side of the figure are C++ classes and HLSL shader code that an engine framework like Unity or Unreal might provide. (You'll see classes and HLSL code for a library of lights, materials, etc.) On the right side of the figure is the application code that uses that library.  In particular find `main.cpp` and make sure you understand what the calls to `bindProgram()` and `setParams()` do "under the hood" in terms of triggering GPU communication or state changes.

You may also want to browse the [Slang documentation](https://shader-slang.github.io/slang/user-guide/) on the [Slang github site](https://github.com/shader-slang/slang).

In your response, please address the following questions:

* The goal of the Slang language is about providing programmers better support for negotiating the conflicting goals of writing modular, easy-to-maintain shader code and generating high-performance code for GPUs.  This tension is actually best described in Section 2.1 of the [Shader Components](http://graphics.cs.cmu.edu/projects/shadercomp/he17_shadercomp.pdf) paper listed under recommended readings, so you many want to read that section from this prior paper first.  After taking a look, please describe the two major aspects of "performant GPU" code that game developers require. Hint: one is about communication, the other about code specialization.

* Another key design idea of the Slang language is that although it is possible to use bespoke, domain-specific languages to achieve the same goals, minimal extensions to an existing general purpose language is desirable.  Do you agree with this philosophy? Why or why not?

* In your own words, describe what an "associated type" is.  This is something good to know in general. ;-)

* I'd like you to extrapolate a bit about the value of getting the right abstractions in place.  Consider what decisions the compiler/runtime system might be able to do for the programmer in the future given a code library written in terms of generics, interfaces, and associated types. Specifically, given information about what classes are linked together, would it be valid in the future of the Slang compiler/runtime to "choose" between generating and binding highly specialized code for the GPU (like it does today) or perhaps generate a single "general" shader that uses conditional control flow to dynamically switch between code paths depending on the types of classes that fill in the generic template?  In other words, can the Slang compiler choose to perform "despecialization" in the same way that is currently chooses "specialization".  

* Not a question, but a suggestion: For those wanting to go further, I recommend reading parts of [Yong He's thesis](http://graphics.cs.cmu.edu/projects/renderergenerator/yong_he_thesis.pdf) on Slang. In particular, the reflection and potential future discussed in Chapter 7 is very interesting.  

__Other Recommended Readings:__
* [Shader Components: Modular and High Performance Shader Development](http://graphics.cs.cmu.edu/projects/shadercomp/). Y. He et al. SIGGRAPH 2017
* [A Language for Shading and Lighting Calculations](http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15869-f11/www/readings/hanrahan90_rsl.pdf). P. Hanrahan and J. Lawson. SIGGRAPH 1990 
* [Cg: A System for Programming Graphics Hardware in a C-like Language](http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15869-f11/www/readings/mark03_cg.pdf). W. R. Mark et al. SIGGRAPH 2003
* [A Real-Time Procedural Shading System for Programmable Graphics Hardware](http://graphics.stanford.edu/projects/shading/pubs/sig2001/). K. Proudfoot et al. SIGGRAPH 2001
* [Shade Trees](http://graphics.pixar.com/library/ShadeTrees/paper.pdf). R. Cook. SIGGRAPH 1984
* [An Image Synthesizer](http://dl.acm.org/citation.cfm?id=325247). K. Perlin. SIGGRAPH 1985

