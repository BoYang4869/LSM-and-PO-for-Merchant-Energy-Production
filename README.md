# Pathwise-Optimization-for-Merchant-Energy-Production
This repo provides a C++ implementation of pathwise optimization in the context of merchant energy production. Specifically, we consider a realistic ethanol production instance modeled as a large scale pathwise linear program (PLP). The production plant converts corn and natural gas into ethanol. The owner of the plant wants to maximize the total profit whiling facing uncertain prices of the three commodities. The code applies the block coordinate descent (BCD) method to solve PLP. The output are lower and dual bounds as well as a control policy.

This code uses three packages: (i) Gurobi 7.5; (ii) LAPACKE 3.7.1; (iii) dlib C++ machine learning package.

Package (iii) is included in the src folder. However, the first two packages need additional installations. 

Gurobi can be found on the website https://www.gurobi.com/

LAPACKE 3.7.1 is on the website https://netlib.org/lapack/lapack-3.7.1.html

We provide Makefile for the readers to know how the code and packages are compiled on our machine.
