cmake_minimum_required(VERSION 3.16)
project(Homework3 C)


set(CMAKE_C_STANDARD 11)

set(CMAKE_C_FLAGS " -pg -mavx2 -mfma -O3 -DDEBUG" )



add_executable(Homework3 BMP_Processor_Serial_std.c handle_image_header.h image_types.h)