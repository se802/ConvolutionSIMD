# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/clion/233/bin/cmake/linux/x64/bin/cmake

# The command to remove a file.
RM = /snap/clion/233/bin/cmake/linux/x64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/stelios/CLionProjects/ConvolutionAVX

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/stelios/CLionProjects/ConvolutionAVX

# Include any dependencies generated for this target.
include CMakeFiles/Homework3.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Homework3.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Homework3.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Homework3.dir/flags.make

CMakeFiles/Homework3.dir/BMP_Processor_Serial_std.c.o: CMakeFiles/Homework3.dir/flags.make
CMakeFiles/Homework3.dir/BMP_Processor_Serial_std.c.o: BMP_Processor_Serial_std.c
CMakeFiles/Homework3.dir/BMP_Processor_Serial_std.c.o: CMakeFiles/Homework3.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/stelios/CLionProjects/ConvolutionAVX/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/Homework3.dir/BMP_Processor_Serial_std.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/Homework3.dir/BMP_Processor_Serial_std.c.o -MF CMakeFiles/Homework3.dir/BMP_Processor_Serial_std.c.o.d -o CMakeFiles/Homework3.dir/BMP_Processor_Serial_std.c.o -c /home/stelios/CLionProjects/ConvolutionAVX/BMP_Processor_Serial_std.c

CMakeFiles/Homework3.dir/BMP_Processor_Serial_std.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/Homework3.dir/BMP_Processor_Serial_std.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/stelios/CLionProjects/ConvolutionAVX/BMP_Processor_Serial_std.c > CMakeFiles/Homework3.dir/BMP_Processor_Serial_std.c.i

CMakeFiles/Homework3.dir/BMP_Processor_Serial_std.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/Homework3.dir/BMP_Processor_Serial_std.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/stelios/CLionProjects/ConvolutionAVX/BMP_Processor_Serial_std.c -o CMakeFiles/Homework3.dir/BMP_Processor_Serial_std.c.s

# Object files for target Homework3
Homework3_OBJECTS = \
"CMakeFiles/Homework3.dir/BMP_Processor_Serial_std.c.o"

# External object files for target Homework3
Homework3_EXTERNAL_OBJECTS =

Homework3: CMakeFiles/Homework3.dir/BMP_Processor_Serial_std.c.o
Homework3: CMakeFiles/Homework3.dir/build.make
Homework3: CMakeFiles/Homework3.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/stelios/CLionProjects/ConvolutionAVX/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable Homework3"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Homework3.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Homework3.dir/build: Homework3
.PHONY : CMakeFiles/Homework3.dir/build

CMakeFiles/Homework3.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Homework3.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Homework3.dir/clean

CMakeFiles/Homework3.dir/depend:
	cd /home/stelios/CLionProjects/ConvolutionAVX && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/stelios/CLionProjects/ConvolutionAVX /home/stelios/CLionProjects/ConvolutionAVX /home/stelios/CLionProjects/ConvolutionAVX /home/stelios/CLionProjects/ConvolutionAVX /home/stelios/CLionProjects/ConvolutionAVX/CMakeFiles/Homework3.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Homework3.dir/depend

