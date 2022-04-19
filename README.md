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

* [2021 Lecture slides](http://cs348k.stanford.edu/spring21/lecture/dnnhardware)

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

* [2021 Lecture slides](http://cs348k.stanford.edu/spring21/lecture/trainingdata)

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

