# Least Squares Monte Carlo and Pathwise Optimization for Merchant Energy Production

The manuscript is avilable at [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3900797).

## Abstract
We study merchant energy production modeled as a compound switching and timing option. The resulting Markov decision process is intractable. Least squares Monte Carlo combined with information relaxation and duality is a state-of-the-art reinforcement learning methodology to obtain operating policies and optimality gaps for related models. Pathwise optimization is a competing technique developed for optimal stopping settings, in which it typically provides superior results compared to this approach, albeit with a larger computational effort. We apply these procedures to merchant energy production. Employing pathwise optimization requires methodological extensions. We use principal component analysis and block coordinate descent in novel ways to respectively precondition and solve the ensuing ill-conditioned and large scale linear program, which even a cutting-edge commercial solver is unable to handle directly. Both techniques yield near optimal operating policies on realistic ethanol production instances. However, at the cost of both considerably longer run times and greater memory usage, which limits the length of the horizon of the instances that it can handle, pathwise optimization leads to substantially tighter dual bounds compared to least squares Monte Carlo, even when specified in a simple fashion, complementing it in this case. Thus, it plays a critical role in obtaining small optimality gaps. Our numerical observations on the magnitudes of these bound improvements differ from what is currently known. This research has potential relevance for other commodity merchant operations contexts and motivates additional algorithmic work in the area of pathwise optimization.

## Description
This repo provides a C++ implementation of least squares Monte Carlo (LSM) and pathwise optimization (PO) for merchant energy production. The "LSM_src" and "PO_src" folders contain source files for LSM and PO, respectively. The folder "Ethanol_InputFile" includes calibrated parameters for a term structure price model used to simulate forward curves for the ethanol production instance.

The implementation uses three packages: (i) Gurobi 7.5, (ii) LAPACKE 3.7.1, and (iii) dlib C++ machine learning package. Package (iii) is included in the src folder. However, the first two packages need additional installations.

Gurobi can be found on the website https://www.gurobi.com/.

LAPACKE 3.7.1 is on the website https://netlib.org/lapack/lapack-3.7.1.html. Note that LAPACKE 3.7.1. includes the BLAS package.

Our machine uses the GCC 4.8.5 (Red Hat 4.8.5-11) compilier and the CentOS Linux 7 operating system. We upload Makefile to provide detailed information on how the code and packages are compiled on our machine.

## How to compile and run our code:
Step 1: Make sure that all packages have been properly installed

Step 2: Change the Makefile, in particular the paths for libraries and headers, based on your installation in Step 1

Step 3: Make sure that your machine has at least 128 GB memory

Step 4: Compile and run the code
