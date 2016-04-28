# Submission 07

Updates:
	1. Fixed bugs. Now the simulation won't explode.


# Submission 06

IMPORTANT:
    1. After generating the project using cmake and running it once to generate executable,
       please copy tbb_debug.dll and Half.dll from nuparu/lib/win into where the executable resides.
	   
    2. Libraries used were compiled with VS 14 x64 compiler. They may not work with your compiler
	   if you are using a different one.
	   
	   
Min OpenGL requirement: 4.3
Visual Studio requirement: 14 (2015) x64


Updates:
	1. Implemented FLIP using GPU kernels;
	
Performance Analysis:
(20 x 20 x 20 grid):
    19,200          particles -> 12  FPS
	153,600         particles -> 8   FPS
	518,400         particles -> 4   FPS
	1,228,800       particles -> 2   FPS
	2,400,000       particles -> 1.2 FPS
	
(10^3 grid, 2,400 particles)            -> 45  FPS
(20^3 grid, 19,200 particles)           -> 12  FPS
(40^3 grid, 166,400 particles)          -> 4.8 FPS
(50^3 grid, 320,200 particles)          -> 2.7 FPS
(80^3 grid, 1,331,200 particles)        -> 0.7 FPS


Existing Problems:
	1. The simulation is unstable. Some local area will explode during the simulation.

	   
# Submission 05

IMPORTANT:
    1. After generating the project using cmake and running it once to generate executable,
       please copy tbb_debug.dll and Half.dll from nuparu/lib/win into where the executable resides.
	   
    2. Libraries used were compiled with VS 14 x64 compiler. They may not work with your compiler
	   if you are using a different one.
	   
	3. Uncomment buildSDF() in FS_Grid::render() to enable writing VDB files.
	   You may want to change the ouput directory specified in FS_Grid::buildSDF().
	   
Min OpenGL requirement: 4.3
Visual Studio requirement: 14 (2015) x64

Updates:
	1. Implemented Conjugate Gradient method with Diagonal Preconditioning using GPU kernels.
	   Current implementation is not optimal due to heavy CPU intervention but it is still
	   about 10 times faster than the CPU implementation.
	   

# Submission 04

IMPORTANT:
    1. After generating the project using cmake and running it once to generate executable,
       please copy tbb_debug.dll and Half.dll from nuparu/lib/win into where the executable resides.
	   
    2. Libraries used were compiled with VS 14 x64 compiler. They may not work with your compiler
	   if you are using a different one.
	   
	3. Uncomment buildSDF() in FS_Grid::render() to enable writing VDB files.
	   You may want to change the ouput directory specified in FS_Grid::buildSDF().
	   
Min OpenGL requirement: 4.3
Visual Studio requirement: 14 (2015) x64

Parallelization Performance Analysis (10 x 10 grid with 8 particles per cell):
    Topic                                  Serial(FPS)      Parallel(FPS)      Comparison      Reason(if negative)
    Interpolation (grid to particles)      ~= 5.0           ~= 7.0             +40%
	Interpolation (particles to grid)      ~= 7.0           ~= 6.0             -14.2%          Mutex overhead
	Extrapolation                          ~= 6.5           ~= 7.0             +7.7%

Updates:
	1. Pressure solve
	2. Parallelization
	3. Meshing using OpenVDB
	
	
# Submission 03

IMPORTANT: after generating the project using cmake and running it once to generate executable,
please copy tbb_debug.dll into where the executable resides.

1. Implemented all required features;
2. Used parallel_for in trilinear interpolation.


# Submission 02

Min OpenGL requirement: 4.3
Visual Studio requirement: 14 (2015) x64

Updates:
1. Grid and MACGrid classes;
2. Construction and initialization of MAC grid;
3. Transfer particle velocity to grid;
4. A trilinear interpolation for PIC (not used yet. Only gravity affects particles right now);
5. Debug rendering of the grid, particles, and velocity components.


# Submission 01

Basic viewer funtionality implemented.

Tested on VS 2015, Windows 10.

Inorder to compile the code, you need to change two #define's in main.hpp:
    1. SCENE_FILE_NAME: the path + name of the scene json file (e.g. "C:/path_to_scene_file/scene.json")
    2. SHADERS_DIR: the directory where the shaders locate without trailing slash (e.g. "C:/path_to_shaders")


# CIS563-FluidSolver

FluidSolver Submission guidelines:

(Borrowed from CIS565 README)

If you have modified any of the CMakeLists.txt files at all (aside from the list of CORE_SRC), you must test that your project can build. Beware of any build issues.

Open a GitHub pull request so that we can see that you have finished. The title should be "Submission: YOUR NAME".

In the body of the pull request, include a link to your repository.

Submit on canvas with  a direct link to your pull request on GitHub


And you're done!