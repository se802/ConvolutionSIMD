# Speeding up Convolution Algorithm with Streaming SIMD

Convolution is a fundamental operation in many signal and image processing applications. The convolution algorithm can be computationally expensive, especially for large input sizes. One way to speed up the convolution algorithm is to use Streaming SIMD (Single Instruction, Multiple Data) instructions that are available on modern processors. 

This project demonstrates how to implement the convolution algorithm using Streaming SIMD instructions, resulting in significant performance improvements.

## Prerequisites

To run this project, you need:

- A modern CPU that supports Streaming SIMD instructions (e.g., Intel SSE, AVX, or ARM NEON)
- A C compiler that supports the SIMD intrinsics (e.g., GCC or Clang)

## Installation

To install the project, simply clone the repository::::

## Implementation 

To see the details of our implementation, have a look at convolutionAVX.c. More precisely look at applyKernel methods and convolutionAVX. 

