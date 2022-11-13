<p align="center">
  <img src="https://user-images.githubusercontent.com/60443339/201536468-f8b01086-c36b-434b-8a1d-0bf3ec123af3.png#gh-dark-mode-only" alt="drawing"        height="400" /> 
  <img src="https://user-images.githubusercontent.com/60443339/198055044-a2e7270c-684d-40ac-9836-98d37e77fa33.png#gh-light-mode-only" alt="drawing" height="400" />
 </p>
 
 <p align="center">
  <img src="https://user-images.githubusercontent.com/60443339/201536468-f8b01086-c36b-434b-8a1d-0bf3ec123af3.png#gh-dark-mode-only" alt="drawing"        height="400" /> 
  <img src="https://user-images.githubusercontent.com/60443339/198055044-a2e7270c-684d-40ac-9836-98d37e77fa33.png#gh-light-mode-only" alt="drawing" height="400" />
 </p>
 


This program computes the Pi number using different Spigot algorithms. 
To perform the operations it is used a floating point precision arithmetic library. PiDecimals allows you to use GMP (The GNU Multiple Precision Arithmetic Library) or MPFR (The GNU Multiple Precision Floating-Point Reliable Library) to compute the Pi number with the algorithms supported.

This version has been developed for clusters. It can be used as a benchmark to compare and test the CPU performance in cluster environments. Processes are managed with MPI and threads are managed with OpenMP.

<img src="https://user-images.githubusercontent.com/60443339/197706209-6482adfb-684a-4e26-bb21-4502c110a938.png" alt="drawing" height="100"/>

<img src="https://user-images.githubusercontent.com/60443339/195342306-1eb14b7d-ce25-41a8-87b5-545011edf172.png" alt="drawing" height="100"/>

#### Spigot Algorithms

Currently, PiDecimalsMPI allows you to compute Pi using three different algorithms:

* Bailey-Borwein-Plouffe. The expression is presented below:

<img src="https://user-images.githubusercontent.com/60443339/195336253-bf6aeeea-c255-458c-9f16-7fcc91d5b2c7.png" alt="drawing" height="85" />

* Bellard. The expression is presented as follows:

<img src="https://user-images.githubusercontent.com/60443339/195340916-7508ee10-2209-413a-b24a-92cede2aea44.png" alt="drawing" />

* Chudnovsky. The expression is shown below:

<img src="https://user-images.githubusercontent.com/60443339/195336414-27422fd3-4884-4cf4-a7b8-47bf49f5b67a.png" alt="drawing" height="85" />

#### Multiple Precision Floating Point Libraries

Currently, PiDecimals allows yoy to compute Pi using two different floating point arithmetic libraries: 

* GMP (https://gmplib.org/)

* MPFR (https://www.mpfr.org/)

## Compilation and Installation

To compile the code succesfully it is necessary to have installed MPI, OpenMP, GMP and MPFR library. 
If you are using a Linux distro it is very likely that you already have these dependencies installed.

To compile the source code use the "compile.sh" located at the root project directory. 
In the future it is expected to replace the bash script with a Make file.   

## Launch

When the source code is compiled you are ready to launch: 

```console
mpirun -np num_procs ./PiDecimalsMPI.x library algorithm precision num_threads [-csv]
```
* num_procs param is the number of processes that you want to use to perform the operations.
* library can be 'GMP' or 'MPFR'.
* algorithm is a value between 0 and X. The X value may depend on the library used.
* precision param is the value of precision you want to use to perform the operations. 
* num_threads param is the number of threads that you want to use to perform the operations.
* -csv param is optional. If this param is used the program will show the results in csv format.

En example of use could be:
```console
mpirun -np 2 ./PiDecimals.x MPFR 1 50000 4
```
And the output could be:

