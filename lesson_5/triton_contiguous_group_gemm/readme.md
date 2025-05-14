Group General Matrix-Multiply (GEMM) is a key operation in Mixture-of-Experts (MoE) models. 

The Group GEMM kernel batches the execution of multiple independant GEMM problems in a single fused kernel. Each group can have different shapes. 

This tutorial will walk through how to write and optimize a Group GEMM kernel in Triton. 

(TODO - make a video)