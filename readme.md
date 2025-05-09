Work in Progress!

General summary:
work through progressive kernels to learn Triton with:  
a - [vector add](https://github.com/gpu-mode/triton-tutorials/blob/main/kernels/vector_addition_tutorial.py)    
b - [simple matmul](https://github.com/gpu-mode/triton-tutorials/blob/main/kernels/matmul_outer_k.py) (outer k loop and tiled)   
c - fused softmax (in pytorch, then in Triton)  
d - softmax with backward  
e - flash attention 2  
f - group gemm with backwards (bf16, then fp8)    
g - MoE permute / unpermute kernels

## Getting Started:
Lessons are arranged in order, starting with lesson_1 etc. 

Some of the above are also examples in the core Triton repo...what's the difference?   
Generically and from an optionated view, the examples in the core repo tend to put too much, too soon into the examples obfuscating the core lessons one should take from it (again, biased opinion lol).   
We'll stick with simpler versions that allow the core tenets of kernel programming to shine through, bypassing autotuning, certain cache optimizations, and other aspects and pull them in with later lessons.   
In addition. goal is to have almost every line in each kernel commented with clear, concise comments to make it easy to map what the kernel code is doing. 

External contributions are very welcome - this is meant to be an evolving, iteratively improving tutorial series. 



