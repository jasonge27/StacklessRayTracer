# Ray Tracer on GPU
Implemented stackless KDTree on GPU using CUDA to accelerate ray tracing rendering algorithm. Hardware level optimization for register spills and local memory overhead. 

## Parallel Ray Tracer on GPU

Ray tracing follows every beam of light in a scene. Since we know the physical laws of reflection, refraction and scattering, when enough number of light beams are contructed, we can render the scene up to real world effects. Ray tracing is a natually parallel algorithm because physics tell us light beams do not interfere with each other.

The main calculation of ray tracing involves calculating the intersecting point of any light beam and the objects in the scene. We implemented stackless KDTree algorithm in the paper [1] and observed  100X times speed up.

|                                          | Direct Intersection Calculation | Intersection Calculation by Vanilla Kdtree | Intersection  Calculation by Stackless Kdtree |
| :--------------------------------------: | :-----------------------------: | :--------------------------------------: | :--------------------------------------: |
|            CPU rendering time            |              255s               |                 22.193s                  |                 13.371s                  |
| GPU rendering time (before hardware optimization) |             3.282s              |                  0.198s                  |                  0.098s                  |
|                 Speedup                  |               77                |                   112                    |                   137                    |

Experiment Details. 

CPU: Intel Core i7-2630QM, GPU: NVIDIA GeForce GT 610 with 6GB RAM. 

OS: Linux 12.04, Compiler: nvcc in CUDA 5.0

Scene: an 3996 face ant in a cube, 800*600 light beams, 3 times reflection for shallow calculation. 

Effect:  

<p align="center">
  <img src="https://github.com/jasonge27/RayTracingCUDA/blob/master/scene/renderedAnt.png" width="200"/>
</p>

## Hardware Level Optimization for GPU

### Bottleneck analysis

#### Register Spills vs Warp Occupancy Trade-off

On the one hand, if we use more registers per thread, we can reduce the local memory overhead caused by register spill. On the other hand, if we use less registers per thread, we can run more warps in parallel. We need to find the right balance in this trade-off.

- Register spills. Adding option `-Xptaxs -v` when compiling with `nvcc` gives us the register usage info:  **register usage: 63, stored spill 324B, load spill 452B** Explanation. GT 610 has compute capability 2.1 with 63 registers at most in each thread. The total size of registers available is 64KB. If a thread asks for more registers than the limit, there will be register spill.

  Visual profiler says **local memory overhead: 62.3%**

  Explanation. When register spills happen, the thread will read/write through L1 cache into local memory which is slower than directly to registers. Moreover, in the case of L1 cache miss, instructions have to be re-issued putting more pressure on the memory bandwidth. This is why we have such a large **local memory overhead**.

- Warp occupancy.  Visual profiler says **warp occupancy 33.1%** 

  Explanation. We are using blocks of dimension 16*8 in the computation. GT 610 can allow at most 48 warps to be simultanously running in hardware given the block dimension we've chosen. Warp occupancy says we're only using 16 warps on average, which is a consequence of register spill. Each block uses up to 4KB registers, with the limit of 64KB registers in total, it makes sense that we can only run 16 warps at the same time. 

  ​

#### Branch divergence

Visual profiler gives us **warp execution efficiency 45.4%**

Since the threads in one single warp share the same intruction stream, when the threads diverge in `if-then A else B` statement into thread group 1 and thread group 2, thread group 1 will excecute A with group 2 waiting, and then group 2 will execute B with group 1 watiing. The warp exec efficiency ratio says that basically our time doubled because of this phenomenon.

### Optimizations

- Optimization 1. Using compiler flags `-prec-dev=false, -prec-sqrt=false` decreases the precision of division and sqrt computation. The compiler will then optimize out some intermediate varialbes to reduce register usage.

- Optimization 2. Trade-off between register spills and warp occupancy. We use the compiler flag `maxregcount=48` to limit maximal number of registers each thread can use. 

  |                     | Time | Warp Occupancy | Local memory overhead | Spilled Store | Spilled Load |
  | :-----------------: | :--: | :------------: | :-------------------: | :-----------: | :----------: |
  | Before Optimization | 98ms |     31.1%      |         62.3%         |     324B      |     452B     |
  |   Optimization 1    | 60ms |     33.4%      |         48.8%         |     122B      |     122B     |
  |  Optimization 1+2   | 49ms |     49.6%      |         53.7%         |     240B      |     240B     |

  ​

  References

  [1] Popov, Gunther et al., Stackless KD-Tree Traversal for High Performance GPU Ray Tracing, 2007

