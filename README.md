# Pathwise-Optimization-for-Merchant-Energy-Production

## Abstract
We study merchant energy production modeled as a compound switching and timing option. The resulting Markov decision process is intractable. Least squares Monte Carlo combined with information relaxation and duality is a state-of-the-art reinforcement learning methodology to obtain operating policies and optimality gaps for related models. Pathwise optimization is a competing technique developed for optimal stopping settings, in which it typically provides superior results compared to this approach, albeit with a larger computational effort. We apply these procedures to merchant energy production. Employing pathwise optimization requires methodological extensions. We use principal component analysis and block coordinate descent in novel ways to respectively precondition and solve the ensuing ill-conditioned and large scale linear program, which even a cutting-edge commercial solver is unable to handle directly. Both techniques yield near optimal operating policies on realistic ethanol production instances. However, at the cost of both considerably longer run times and greater memory usage, {\color{blue}which limits the length of the horizon of the instances that it can handle}, pathwise optimization leads to substantially tighter dual bounds compared to least squares Monte Carlo, even when specified in a simple fashion, complementing it in this case. Thus, it plays a critical role in obtaining small optimality gaps. Our numerical observations on the magnitudes of these bound improvements differ from what is currently known. This research has potential relevance for other commodity merchant operations contexts and motivates additional algorithmic work in the area of pathwise optimization.

## Description
This repo provides a C++ implementation of pathwise optimization in the context of merchant energy production. It applies the PCA based preconditioning step and the block coordinate descent (BCD) method to solve the large scale pathwise linear program.

The code uses three packages: (i) Gurobi 7.5, (ii) LAPACKE 3.7.1, and (iii) dlib C++ machine learning package. Package (iii) is included in the src folder. However, the first two packages need additional installations. 

Gurobi can be found on the website https://www.gurobi.com/

LAPACKE 3.7.1 is on the website https://netlib.org/lapack/lapack-3.7.1.html. Note that the BLAS package is included in LAPACKE 3.7.1.

We upload Makefile to provide detailed information on how the code and packages are compiled on our machine.

## How to compile and run our code:
Step 1: Make sure that all packages have been properly installed

Step 2: Change the path in the Makefile based on your installation in Step (i)

Step 3: Make sure that your machine has at least 128 GB memory

Step 4: Compile and run the code
