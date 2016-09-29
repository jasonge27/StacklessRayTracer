# Ray Tracer on GPU
Implemented stackless KDTree on GPU using CUDA to accelerate ray tracing rendering algorithm. Hardware level optimization for register spills and local memory overhead. 

## Parallel Ray Tracer on GPU

Ray tracing follows every beam of light in a scene. Since we know the physical laws of reflection, refraction and scattering, when enough number of light beams are contructed, we can render the scene up to real world effects. Ray tracing is a natually parallel algorithm because physics tell us light beams do not interfere with each other.

The main calculation of ray tracing involves calculating the intersecting point of any light beam and the objects in the scene. We implemented stackless KDTree algorithm in the paper Stackless KD-Tree Traversal for High Performance GPU Ray Tracing (by Popov, Gunther et al. in 2007) and observed  100X times speed up.

|                                          | Direct Intersection Calculation | Intersection Calculation by Vanilla Kdtree | Intersection  Calculation by Stackless Kdtree |
| :--------------------------------------: | :-----------------------------: | :--------------------------------------: | :--------------------------------------: |
|            CPU rendering time            |              255s               |                 22193ms                  |                 13371ms                  |
| GPU rendering time (before hardware optimization) |             3282ms              |                  198ms                   |                   98ms                   |
|                 Speedup                  |               77                |                   112                    |                   137                    |

Experiment Details. 

CPU: Intel Core i7-2630QM, GPU: NVIDIA GeForce GT 610 with 6GB RAM. 

OS: Linux 12.04, Compiler: nvcc in CUDA 5.0

Scene: an 3996 face ant in a cube, 800*600 light beams, 3 times reflection for shallow calculation. 

Effect:  ![alt tag](https://github.com/jasonge27/RayTracingCUDA/blob/master/scene/rendered_ant.png)

## Hardware Level Optimization for GPU

