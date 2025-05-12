Let's get comfortable with the some kernel basics and write our first kernel.   

We'll solve the problem of having two arrays (python lists) and we want to add them together.  

First up - we need to mentally move from sequential programming (i.e. looping) and move to leverage the core advantage of gpus, massive parallelization.  

Thus:  
making the shift to parallel programming:  
https://youtu.be/MEZ7XhzTLEg  

writing your first Triton kernel:  
https://youtu.be/8P0M-DXr774  

verifying your kernel (always important, numerical fidelity even though we have parallelized the work):  
https://youtu.be/kEGW0SemWWw  

and finally, let's compare our Triton kernel vs Pytorch in terms of speed:  
https://youtu.be/Nh5QIkGuExQ  

A self contained kernel, wrapper, and test case driver is here with extensive code comments.  (TODO - make a video walkthrough):  
[Vector addition kernel](https://github.com/gpu-mode/triton-tutorials/blob/main/kernels/vector_addition_tutorial.py)


