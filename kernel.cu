//kernelPBO.cu (Rob Farber)

#include <stdio.h>
#include "scene.hpp"
#include "kdtree.h"

#define blocksize_x 8
#define blocksize_y 6 

#define EPSILON (1e-6)

__constant__ float3 dcamPos;
__constant__ float3 dwincenterPos;

__constant__ float3 dwinup;
__constant__ float3 dwindown;
__constant__ float3 dboxup;
__constant__ float3 dboxdown;

__constant__ unsigned int dwinwidth;
__constant__ unsigned int dwinheight;

__constant__ float hx;
__constant__ float hy;

__constant__ float3 objectColor; 

__constant__ unsigned int nTri;
__constant__ unsigned int nVtx;
__constant__ unsigned int nLS;

/*__constant__ float* dVtxBuf;
__constant__ int* dTriVtxBuf;
__constant__ float* dNormal;
__constant__ float* dLS;
__constant__ float* dsandboxColor;
__constant__ unsigned int* dsandboxIsReflective;
*/

texture<float, 1 > texref_VtxBuf;
texture<int, 1 > texref_TriVtx;
texture<float, 1 > texref_Normal;
texture<float, 1 > texref_LS;
texture<float, 1 > texref_sandboxColor;
texture<unsigned int, 1 > texref_sandboxIsReflective;

// memory address in cudamemory
/*
extern "C" float* dVtxBuf = NULL;
extern "C" int* dTriVtxBuf = NULL;
extern "C" float* dNormal = NULL;
extern "C" float* dLS = NULL;
extern "C" float* dsandboxColor = NULL;
extern "C" unsigned int* dsandboxIsReflective = NULL;
*/

void checkCUDAError(const char *msg) 
{
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

__device__ inline float3 CrossProduct( float3 p1, float3 p2)
{
  float3 tmp;
  tmp.x = p1.y*p2.z - p1.z*p2.y;
  tmp.y = p1.z*p2.x - p1.x*p2.z;
//  tmp.y = p1.x*p2.z - p1.z*p2.x;
  tmp.z = p1.x*p2.y - p1.y*p2.x;
  return tmp;
}

__device__ inline  float InnerProduct( float3 p1, float3 p2)
{
  return p1.x*p2.x + p1.y*p2.y + p1.z*p2.z;
}

// return the value of t, when p+t*dir intersect triangle p0-p1-p2
__device__ inline float TestSingleIntersection( float3 p, float3 dir, float3 p0, float3 p1, float3 p2) 
{
  float3 E1;
  E1.x = p1.x - p0.x;
  E1.y = p1.y - p0.y;
  E1.z = p1.z - p0.z;

  float3 E2;
  E2.x = p2.x - p0.x;
  E2.y = p2.y - p0.y;
  E2.z = p2.z - p0.z;

  float3 T;
  T.x = p.x - p0.x;
  T.y = p.y - p0.y;
  T.z = p.z - p0.z;

  float3 P = CrossProduct( dir, E2);
  float3 Q = CrossProduct( T, E1 );
  
  float s = InnerProduct(P, E1);
 // s = 0.5; //FIXME
  if ( (s < EPSILON ) && (s>-EPSILON) ) return -1;

  float t = InnerProduct(Q, E2) / s;
  if ( t < 1e-5) return -2;

  float u = InnerProduct(P, T) / s;
  if ( (u < 0)  ) return -3;

  float v = InnerProduct(Q, dir) / s;
  if ( (v < 0 ) ) return -4;

  if ( u+v > 1 ) return -5;

//  t = 0.5; //FIXME
  return t;
}

__device__ inline bool FromLeftToRight( KdTree_rp* dtree, int idx, float3 dir)
{
  switch (dtree[idx].axis)
  {
    case 0:
      if ( dir.x > EPSILON) return true;
      else return false;
    case 1:
      if ( dir.y > EPSILON) return true;
      else return false;
    case 2:
      if ( dir.z > EPSILON) return true;
      else return false;
    default:
      printf("no such case!\n");
  }
}

__device__ inline bool TestNodeIntersection( KdTree_rp* dtree, int idx, float3 p, float3 dir)
{
  
  float tmp1, tmp2;

  // x-y
  if ( (dir.x<EPSILON) && (dir.x>-EPSILON) )
  {
    if ( (p.x>dtree[idx].xmax) || ( p.x<dtree[idx].xmin) ) return false;
  }
  else
  {
    float tx1 = (dtree[idx].xmin-p.x)/dir.x;
   // if ( tx1<EPSILON ) return false;
    tmp1 = p.y + tx1*dir.y;

    float tx2 = (dtree[idx].xmax-p.x)/dir.x;
   //if ( tx2 < EPSILON ) return false;
    tmp2 = p.y + tx2*dir.y;
  
    if ( ( (tmp1>dtree[idx].ymax) && (tmp2>dtree[idx].ymax))
      || ( (tmp1<dtree[idx].ymin) && (tmp2<dtree[idx].ymin) ) ) return false;
  }

  // y-z
  if ( (dir.y<EPSILON) && (dir.y>-EPSILON) )
  {
    if ( (p.y>dtree[idx].ymax) || (p.y<dtree[idx].ymin) ) return false;
  }
  else
  {
    float ty1 = (dtree[idx].ymin-p.y)/dir.y;
  //  if ( ty1 < EPSILON ) return false;
    tmp1 = p.z + ty1*dir.z;
  
    float ty2 = (dtree[idx].ymax-p.y)/dir.y;
   // if ( ty2 < EPSILON ) return false;
    tmp2 = p.z + ty2*dir.z;

    if ( ( (tmp1>dtree[idx].zmax) && (tmp2>dtree[idx].zmax))
      || ( (tmp1<dtree[idx].zmin) && (tmp2<dtree[idx].zmin) ) ) return false;
  }

  // x-z
  if ( (dir.z<EPSILON) && (dir.z>-EPSILON) )
  {
    if ( (p.z>dtree[idx].zmax) || (p.z<dtree[idx].zmin) ) return false;
  }
  else
  {
    float tz1 = (dtree[idx].zmin - p.z)/dir.z;
//    if ( tz1 < EPSILON ) return false;
    tmp1 = p.x + tz1*dir.x;

    float tz2 = (dtree[idx].zmax - p.z)/dir.z;
 //   if ( tz2 < EPSILON ) return false;
    tmp2 = p.x + tz2*dir.x;

    if ( ( (tmp1>dtree[idx].xmax) && (tmp2>dtree[idx].xmax))
      || ( (tmp1<dtree[idx].xmin) && (tmp2<dtree[idx].xmin) ) ) return false;
  }

  return true;
}

__device__ inline void IntersectBox( float3 p, float3 dir, KdTree_rp* dtree, int idx, float3& pt, int& face, int prevface)
{
  float t = 1e6;
  float tmp = 0;

  float ep = 0;
  float ep2 = 1e-6;

  face =  -1;

  float a,b;

  if ( (dir.x > 1e-5) || ( dir.x< -1e-5) )
  {
    if ( prevface != 0)
    {
      tmp = ( dtree[idx].xmin -  p.x) /dir.x; 
      a = p.y + tmp* dir.y;
      b = p.z + tmp* dir.z;
  
      if ( (tmp > -ep2) && (tmp<t) ) 
        if ( (a>=dtree[idx].ymin-ep) && (a<=dtree[idx].ymax+ep) && (b>=dtree[idx].zmin-ep) && (b<=dtree[idx].zmax+ep) ) 
      {
        t =tmp;
        face = 0;
      }
    }

    if ( prevface != 1)
    {
      tmp = ( dtree[idx].xmax - p.x) /dir.x;
      a = p.y + tmp* dir.y;
      b = p.z + tmp* dir.z;
      if ( (tmp > -ep2) && (tmp<t) ) 
        if ( (a>=dtree[idx].ymin-ep) && (a<=dtree[idx].ymax+ep) && (b>=dtree[idx].zmin-ep) && (b<=dtree[idx].zmax+ep) ) 
      {
        t =tmp;
        face = 1;
      }
    }
  }

  if ( (dir.y > 1e-5) || ( dir.y < -1e-5) )
  {
    if ( prevface != 2)
    {
      tmp = ( dtree[idx].ymin - p.y) / dir.y;
      a = p.x + tmp * dir.x;
      b = p.z + tmp * dir.z;
      if ( (tmp > -ep2) && ( tmp < t) )
        if ( (a>=dtree[idx].xmin-ep) && (a<=dtree[idx].xmax+ep) && (b>=dtree[idx].zmin-ep) && (b<=dtree[idx].zmax+ep) ) 
      {
          t =tmp;
          face = 2;
      }
    }

    if ( prevface != 3 )
    {
      tmp = ( dtree[idx].ymax - p.y) / dir.y;
      a = p.x + tmp * dir.x;
      b = p.z + tmp * dir.z;
      if ( (tmp > -ep2) && ( tmp < t) )
        if ( (a>=dtree[idx].xmin-ep) && (a<=dtree[idx].xmax+ep) && (b>=dtree[idx].zmin-ep) && (b<=dtree[idx].zmax+ep) ) 
      {
        t =tmp;
        face = 3;
      }
    }
  }

  if ( (dir.z > 1e-5) || ( dir.z < -1e-5) )
  {
    if ( prevface != 4 )
    {
      tmp = ( dtree[idx].zmin - p.z) / dir.z;
      a = p.x + tmp * dir.x;
      b = p.y + tmp * dir.y;
      if ( (tmp > -ep2) && ( tmp < t) )
        if ( (a>=dtree[idx].xmin-ep) && (a<=dtree[idx].xmax+ep) && (b>=dtree[idx].ymin-ep) && (b<=dtree[idx].ymax+ep) ) 
      {
        t =tmp;
        face = 4;
      }
    }

    if ( prevface != 5)
    {
      tmp = ( dtree[idx].zmax - p.z) / dir.z;
      a = p.x + tmp * dir.x;
      b = p.y + tmp * dir.y;
      if ( (tmp > -ep2) && ( tmp < t) )
        if ( (a>=dtree[idx].xmin-ep) && (a<=dtree[idx].xmax+ep) && (b>=dtree[idx].ymin-ep) && (b<=dtree[idx].ymax+ep) ) 
      {
        t =tmp;
        face = 5;
      }
    }
  }
  
  if (  face >=0 )
  {
    pt.x = p.x + t * dir.x;
    pt.y = p.y + t*dir.y;
    pt.z = p.z + t*dir.z;
  }
  
}

__device__ inline void IntersectBox2( float3 p, float3 dir, KdTree_rp* dtree, int idx, float3& pt, int& face)
{
  float t = 1e6;
  float tmp = 0;

  float ep = 0;
  float ep2 = 0;

  face =  -1;

  float a,b;

  if ( (dir.x > 1e-5) || ( dir.x< -1e-5) )
  {
      tmp = ( dtree[idx].xmin -  p.x) /dir.x; 
      a = p.y + tmp* dir.y;
      b = p.z + tmp* dir.z;
  
      if ( (tmp > ep2) && (tmp<t) ) 
        if ( (a>=dtree[idx].ymin-ep) && (a<=dtree[idx].ymax+ep) && (b>=dtree[idx].zmin-ep) && (b<=dtree[idx].zmax+ep) ) 
      {
        t =tmp;
        face = 0;
      }

      tmp = ( dtree[idx].xmax - p.x) /dir.x;
      a = p.y + tmp* dir.y;
      b = p.z + tmp* dir.z;
      if ( (tmp > ep2) && (tmp<t) ) 
        if ( (a>=dtree[idx].ymin-ep) && (a<=dtree[idx].ymax+ep) && (b>=dtree[idx].zmin-ep) && (b<=dtree[idx].zmax+ep) ) 
      {
        t =tmp;
        face = 1;
      }
  }

  if ( (dir.y > 1e-5) || ( dir.y < -1e-5) )
  {
      tmp = ( dtree[idx].ymin - p.y) / dir.y;
      a = p.x + tmp * dir.x;
      b = p.z + tmp * dir.z;
      if ( (tmp > ep2) && ( tmp < t) )
        if ( (a>=dtree[idx].xmin-ep) && (a<=dtree[idx].xmax+ep) && (b>=dtree[idx].zmin-ep) && (b<=dtree[idx].zmax+ep) ) 
      {
          t =tmp;
          face = 2;
      }

      tmp = ( dtree[idx].ymax - p.y) / dir.y;
      a = p.x + tmp * dir.x;
      b = p.z + tmp * dir.z;
      if ( (tmp > ep2) && ( tmp < t) )
        if ( (a>=dtree[idx].xmin-ep) && (a<=dtree[idx].xmax+ep) && (b>=dtree[idx].zmin-ep) && (b<=dtree[idx].zmax+ep) ) 
      {
        t =tmp;
        face = 3;
      }
  }

  if ( (dir.z > 1e-5) || ( dir.z < -1e-5) )
  {
      tmp = ( dtree[idx].zmin - p.z) / dir.z;
      a = p.x + tmp * dir.x;
      b = p.y + tmp * dir.y;
      if ( (tmp > ep2) && ( tmp < t) )
        if ( (a>=dtree[idx].xmin-ep) && (a<=dtree[idx].xmax+ep) && (b>=dtree[idx].ymin-ep) && (b<=dtree[idx].ymax+ep) ) 
      {
        t =tmp;
        face = 4;
      }

      tmp = ( dtree[idx].zmax - p.z) / dir.z;
      a = p.x + tmp * dir.x;
      b = p.y + tmp * dir.y;
      if ( (tmp > ep2) && ( tmp < t) )
        if ( (a>=dtree[idx].xmin-ep) && (a<=dtree[idx].xmax+ep) && (b>=dtree[idx].ymin-ep) && (b<=dtree[idx].ymax+ep) ) 
      {
        t =tmp;
        face = 5;
      }
  }
  
  if (  face >=0 )
  {
    pt.x = p.x + t * dir.x;
    pt.y = p.y + t*dir.y;
    pt.z = p.z + t*dir.z;
  }
  
}

__device__ inline int FindLeaf( KdTree_rp* dtree, int idx, float3 pt)
{
  while ( dtree[idx].trinum <0 )
  {
    switch (dtree[idx].axis)
    {
      case 0:
        if ( pt.x < dtree[idx].splitpos ) idx = dtree[idx].left;
        else idx = dtree[idx].right;
        break;
      case 1:
        if ( pt.y < dtree[idx].splitpos ) idx = dtree[idx].left;
        else idx = dtree[idx].right;
        break;
      case 2:
        if ( pt.z < dtree[idx].splitpos ) idx = dtree[idx].left;
        else idx = dtree[idx].right;
        break;
    }
  }

  return idx;

}

__device__ inline  void FindIntersectedTriangle( float3 p, float3 dir, int& intersectedTri, float3& intersectedPt, float* dVtxBuf, int* dTriVtxBuf, KdTree_rp* dtree)
{
  float tmin = 1e6;
  float t = tmin;
  intersectedTri = -1;
  float3 p0;
  float3 p1;
  float3 p2;
  float3 pt;
  int idx0;
 
//  int stack[2*TREEDEPTH];
  
  int idx = 0;
  int face = -1;
  if ( (p.x>dtree[0].xmax) || ( p.y > dtree[0].ymax) || (p.z > dtree[0].zmax) 
    || ( p.x<dtree[0].xmin) || (p.y<dtree[0].ymin) || (p.z < dtree[0].zmin) )
    IntersectBox(p, dir, dtree, idx, pt, face, -1); // p is out of dtree[0]
  else // p is in dtree[0], find the leaf node containing p 
  {
    /*
    while (dtree[idx].trinum < 0 )
    {
      switch (dtree[idx].axis)
      {
        case 0:
          if ( p.x < dtree[idx].splitpos) idx = dtree[idx].left;
          else idx = dtree[idx].right;
        case 1:
          if ( p.y < dtree[idx].splitpos) idx = dtree[idx].left;
          else idx = dtree[idx].right;
        case 2:
          if ( p.z < dtree[idx].splitpos) idx = dtree[idx].left;
          else idx = dtree[idx].right;
      }
    }

      for ( int i = 0; i < dtree[idx].trinum; i++ )
      {
       // printf("%d ", dtree[idx].tri[i]);

        idx0 =  dTriVtxBuf[3*dtree[idx].tri[i]];

        p0.x = dVtxBuf[3*idx0];
        p0.y = dVtxBuf[3*idx0+1];
        p0.z = dVtxBuf[3*idx0+2];

        idx0  = dTriVtxBuf[3*dtree[idx].tri[i]+1];
        p1.x = dVtxBuf[3*idx0];
        p1.y = dVtxBuf[3*idx0+1];
        p1.z = dVtxBuf[3*idx0+2];

        idx0 = dTriVtxBuf[3*dtree[idx].tri[i]+2];
        p2.x = dVtxBuf[3*idx0];
        p2.y = dVtxBuf[3*idx0+1];
        p2.z = dVtxBuf[3*idx0+2];

        t = TestSingleIntersection( p, dir ,p0, p1, p2); 
    
        if ( (t > 1e-5) && ( t < tmin ) )
        {
          intersectedTri = idx;
          intersectedPt.x = p.x + t*dir.x;
          intersectedPt.y = p.y + t*dir.y;
          intersectedPt.z = p.z + t*dir.z;
          tmin = t;
        }
        
      }

    if ( intersectedTri >= 0 ) return;

    IntersectBox2(p, dir, dtree, idx, pt, face );
    idx = dtree[idx].rope[face];
    if ( (face==1) || (face==3) || (face==5) ) face--;
    else face++;
    */

    face = -2;
    pt = p;
  }
      
  if ( (face >= 0) || (face==-2) )
  while ( (intersectedTri<0) && (idx>=0) ) 
  {
    idx =  FindLeaf( dtree, idx, pt ); 

//    printf("%d\n",idx);
    if (dtree[idx].trinum>0) // leaf node
    {
//      printf("leaf node:%d\n", idx);
      for ( int i = 0; i < dtree[idx].trinum; i++ )
      {
       // printf("%d ", dtree[idx].tri[i]);

        idx0 =  dTriVtxBuf[3*dtree[idx].tri[i]];

        p0.x = dVtxBuf[3*idx0];
        p0.y = dVtxBuf[3*idx0+1];
        p0.z = dVtxBuf[3*idx0+2];

        idx0  = dTriVtxBuf[3*dtree[idx].tri[i]+1];
        p1.x = dVtxBuf[3*idx0];
        p1.y = dVtxBuf[3*idx0+1];
        p1.z = dVtxBuf[3*idx0+2];

        idx0 = dTriVtxBuf[3*dtree[idx].tri[i]+2];
        p2.x = dVtxBuf[3*idx0];
        p2.y = dVtxBuf[3*idx0+1];
        p2.z = dVtxBuf[3*idx0+2];

        t = TestSingleIntersection( p, dir ,p0, p1, p2); 
    
        if ( (t > 1e-5) && ( t < tmin ) )
        {
          intersectedTri = dtree[idx].tri[i];
          intersectedPt.x = p.x + t*dir.x;
          intersectedPt.y = p.y + t*dir.y;
          intersectedPt.z = p.z + t*dir.z;
          tmin = t;
        }
        
      }
//      printf("\n");
    }

    if ( intersectedTri >= 0 ) break;

    // second intersected point of the current leaf node
    // face: the intersected face of the current leaf ndoe
    IntersectBox(pt, dir, dtree, idx, pt, face, face);  

    // next node adjacent to the leaf
    idx = dtree[idx].rope[face];
    if ( (face==1) || (face==3) || (face==5) ) face --;
    else face++;
  }
}

__device__ inline uchar4 computeColor( int depth, float3 p, float3 dir, float* dVtxBuf, int* dTriVtxBuf, float* dNormal, float* dLS, KdTree_rp* dtree  )
{
  int tri = -1;
  float3 intersectPt;
  int triOnTheWay = -1;
  float3 intersectPtOnTheWay;

  float3 tempDir; 

  
  FindIntersectedTriangle(p,dir, tri, intersectPt, dVtxBuf, dTriVtxBuf, dtree); 

  uchar4 color;
  color.w= 0;
  color.x=0; color.y = 0; color.z= 0;

  int i =0;

  float l;
  l = sqrt( dir.x * dir.x + dir.y*dir.y + dir.z*dir.z);
  dir.x = dir.x /l;
  dir.y = dir.y / l;
  dir.z = dir.z /l;
    
  if ( tri < 0) // no intersection, return sandbox color
  {
    int face = -1;

    float3 pt;
    if ( fabs(dir.y)>0)
    {
      l = (-1.0 -p.y )/ dir.y;
      pt.x = p.x + l*dir.x;
      pt.y = -1.0;
      pt.z = p.z + l * dir.z;
      if ( (pt.x >=-1.0) && (pt.x<=1.0) && (pt.z>=-1.0) && (pt.z<=1.0) )
      {
        face = 2; // ymin 
      if ( depth == 0) 
        {
        tempDir.x = dir.x;
        tempDir.y = -dir.y;
        tempDir.z = dir.z;
        color = computeColor(depth+1,pt, tempDir, dVtxBuf, dTriVtxBuf, dNormal, dLS, dtree); 
        color.x = color.x * 0.8;
        color.y = color.y * 0.8;
        color.z = color.z * 0.8;
        }
      }
    }

    if ( fabs(dir.y)>0)
    {
      l = (1.0 -p.y )/ dir.y;
      pt.x = p.x + l*dir.x;
      pt.y = 1.0;
      pt.z = p.z + l * dir.z;
      if ( (pt.x >=-1.0) && (pt.x<=1.0) && (pt.z>=-1.0) && (pt.z<=1.0) )
        face = 3; // ymax
    }

    if ( fabs(dir.x)>0)
    {
      l = (-1.0 - p.x ) /dir.x;
      pt.x = -1.0;
      pt.y = p.y + l * dir.y;
      pt.z = p.z + l * dir.z;
      if ( (pt.y>=-1.0) && (pt.y<=1.0) && (pt.z>=-1.0) && (pt.z<=1.0))
        face = 0; // xmin
    }

    if ( fabs(dir.x)>0)
    {
      l = (1.0 - p.x ) /dir.x;
      pt.x = 1.0;
      pt.y = p.y + l * dir.y;
      pt.z = p.z + l * dir.z;
      if ( (pt.y>=-1.0) && (pt.y<=1.0) && (pt.z>=-1.0) && (pt.z<=1.0))
        face = 1; // xmax
    }

    if ( fabs(dir.z)>0)
    {
      l = (1.0 - p.z ) /dir.z;
      pt.x = p.x + l*dir.x;
      pt.y = p.y + l*dir.y;
      pt.z = 1.0;
      if ( (pt.x>=-1.0) && (pt.x<=1.0) && (pt.y>=-1.0) && (pt.y<=1.0) )
        face = 5; // zmax
    }

    switch (face)
    {
      case 2:
        break;
      case 3:
        color.x = 0;
        color.y = 0;
        color.z = 0;
        break;
      case 0:
        color.x = 10;
        color.y = 100;
        color.z = 10;
        break;
      case 5:
        color.x = 100;
        color.y = 10;
        color.z = 10;
        break;
      case 1:
        color.x = 10;
        color.y = 10;
        color.z =100;
        break;
      default:
        break;
    }

  }
  else  // intersected with a triangle
  {
    // check shadow or illumination
    triOnTheWay = -1;
    for ( i = 0; i < nLS; i++) // for every light source
    {

      tempDir.x = dLS[3*i] - intersectPt.x;
      tempDir.y = dLS[3*i+1] - intersectPt.y;
      tempDir.z = dLS[3*i+2] - intersectPt.z;
      
      l = sqrt(tempDir.x*tempDir.x + tempDir.y * tempDir.y + tempDir.z * tempDir.z);
      tempDir.x = tempDir.x / l;
      tempDir.y = tempDir.y / l;
      tempDir.z = tempDir.z / l;
      

      l = tempDir.x*dNormal[3*tri] + tempDir.y*dNormal[3*tri+1] + tempDir.z*dNormal[3*tri+2];
      
      if ( l < -1e-5) continue;

      FindIntersectedTriangle( intersectPt, tempDir, triOnTheWay, intersectPtOnTheWay, dVtxBuf, dTriVtxBuf, dtree);

      if ( triOnTheWay >= 0 ) // in a shadow of triangle "triOnTheWay"
      {
        /*
        color.w=0;
        color.x = 100;
        color.y = 0;
        color.z = 0;
        */
      }
      else  // directly illuminated by the current light source
      {
        color.w = 0;

        if ( i == 0) l = 0.3;
        else if ( i == 1) l =  0.1;

        color.x = color.x+ l* objectColor.x ;
        color.y = color.y+ l* objectColor.y ; 
        color.z = color.z+ l *objectColor.z ;
      }
    }

    l =  0.5*fabs( dir.x*dNormal[3*tri] + dir.y *dNormal[3*tri+1] + dir.z*dNormal[3*tri+2] ); 
    color.x = color.x+ l * objectColor.x;
    color.y = color.y + l* objectColor.y;
    color.z = color.z + l*objectColor.z;
   /* 
    color.x =  objectColor.x;
    color.y =  objectColor.y;
    color.z = objectColor.z;
    */
  } 
  

  return color;
}

__global__ void TracingKernel( uchar4* pos, float* dVtxBuf, int* dTriVtxBuf, float* dNormal, float* dLS, KdTree_rp* dtree ) 
{
  int pixelx = blockIdx.x*blocksize_x + threadIdx.x;
  int pixely = blockIdx.y*blocksize_y + threadIdx.y;

  float3 dir;
//  dir.x = dwindown.x + hx*pixelx - dcamPos.x;
//  dir.y = dwindown.y + hy*pixely - dcamPos.y;
  dir.x = dwinup.x - hx*pixelx - dcamPos.x;
  dir.y = dwinup.y- hy*pixely - dcamPos.y;
  dir.z = dwindown.z - dcamPos.z;
  
  pos[ pixely*dwinwidth + pixelx ] = computeColor(0, dcamPos, dir, dVtxBuf, dTriVtxBuf, dNormal, dLS, dtree);

  __syncthreads();
}

__global__ void TracingKernel_test( uchar4* pos, float* dVtxBuf, int* dTriVtxBuf, float* dNormal, float* dLS, KdTree_rp* dtree ) 
{
  int pixelx = blockIdx.x*blocksize_x + threadIdx.x;
  int pixely = blockIdx.y*blocksize_y + threadIdx.y;

  float3 dir;
//  dir.x = dwindown.x + hx*pixelx - dcamPos.x;
 // dir.y = dwindown.y + hy*pixely - dcamPos.y;
  dir.x =  -0.1;
  dir.y = -1;
  dir.z = 1;
  
  pos[ pixely*dwinwidth + pixelx ] = computeColor(0, dcamPos, dir, dVtxBuf, dTriVtxBuf, dNormal, dLS, dtree);

  __syncthreads();
}

// Be sure to launch after setting const. and tex. memory
extern "C" void launch_kernel( uchar4* pos, CScene& scene )
{
  dim3 dimBlock(8,8);

  dim3 dimGrid;
  dimGrid.x = scene.m_winwidth / dimBlock.x;
  dimGrid.y = scene.m_winheight / dimBlock.y;

  TracingKernel<<<dimGrid,dimBlock>>>(pos, scene.dVtxBuf, scene.dTriVtxBuf, scene.dNormal, scene.dLS, scene.dtree); 

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess )
  {
    printf("Cuda Error:%s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

extern "C" void launch_kernel_test( uchar4* pos, CScene& scene )
{
  dim3 dimBlock(blocksize_x,blocksize_y);
//  dim3 dimBlock(1);
  

  dim3 dimGrid;
  dimGrid.x = scene.m_winwidth / dimBlock.x;
  dimGrid.y = scene.m_winheight / dimBlock.y;

  uchar4* dpos;

  cudaMalloc( (void**) &dpos, sizeof(uchar4)* scene.m_winwidth*scene.m_winheight );
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess )
  {
    printf("Cuda Error:%s\n", cudaGetErrorString(error));
    exit(-1);
  }

  cudaMemcpy( dpos, pos, sizeof(uchar4)*scene.m_winwidth*scene.m_winheight, cudaMemcpyHostToDevice);
  error = cudaGetLastError();
  if (error != cudaSuccess )
  {
    printf("Cuda Error:%s\n", cudaGetErrorString(error));
    exit(-1);
  }

  printf("Tracing...\n");
  float GPU_time = 0;
  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );
  cudaEventRecord( start, 0 );

  TracingKernel<<<dimGrid,dimBlock>>>(dpos, scene.dVtxBuf, scene.dTriVtxBuf, scene.dNormal, scene.dLS, scene.dtree); 
 // TracingKernel_test<<<dimGrid,dimBlock>>>(dpos, scene.dVtxBuf, scene.dTriVtxBuf, scene.dNormal, scene.dLS, scene.dtree); 

  cudaEventRecord( stop, 0);
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &GPU_time, start, stop);
  printf("GPU time:%f\n", GPU_time);

  printf("%d %d\n", scene.m_winwidth, scene.m_winheight);
  printf("DeviceToHost..\n");
  cudaMemcpy( pos, dpos, sizeof(uchar4)*scene.m_winwidth*scene.m_winheight, cudaMemcpyDeviceToHost);
  printf("DeviceToHostDone.\n");

  printf("Free...\n");
  cudaFree(dpos);
  printf("Free done.\n");

  error = cudaGetLastError();
  if (error != cudaSuccess )
  {
    printf("Cuda Error:%s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

extern "C" void cudaSetConstantMem( CScene& scene )
{
  int a = scene.TriVtxBuf[0][0];  
  float3 camPos;
  camPos.x = scene.cameraPos.x;
  camPos.y = scene.cameraPos.y;
  camPos.z = scene.cameraPos.z;
  cudaMemcpyToSymbol(dcamPos, &camPos, sizeof(float3));

  float3 wincenterPos;
  wincenterPos.x = scene.windowCenter.x;
  wincenterPos.y = scene.windowCenter.y;
  wincenterPos.z = scene.windowCenter.z;
  cudaMemcpyToSymbol(dwincenterPos, &wincenterPos, sizeof(float3));

  float3 winup;
  winup.x = scene.window_diagup.x;
  winup.y = scene.window_diagup.y;
  winup.z = scene.window_diagup.z;
  cudaMemcpyToSymbol(dwinup, &winup, sizeof(float3)); 

  float3 windown;
  windown.x = scene.window_diagdown.x;
  windown.y = scene.window_diagdown.y;
  windown.z = scene.window_diagdown.z;
  cudaMemcpyToSymbol(dwindown, &windown, sizeof(float3));

  float3 boxup;
  boxup.x = scene.m_sandbox.m_diagup.x;
  boxup.y = scene.m_sandbox.m_diagup.y;
  boxup.z = scene.m_sandbox.m_diagup.z;
  cudaMemcpyToSymbol(dboxup, &boxup, sizeof(float3));

  float3 boxdown;
  boxdown.x = scene.m_sandbox.m_diagdown.x;
  boxdown.y = scene.m_sandbox.m_diagdown.y;
  boxdown.z = scene.m_sandbox.m_diagdown.z;
  cudaMemcpyToSymbol(dboxdown, &boxdown, sizeof(float3));

  float3 objColor;
  objColor.x = scene.objectColor.r;
  objColor.y = scene.objectColor.g;
  objColor.z = scene.objectColor.b;
  cudaMemcpyToSymbol(objectColor, &objColor, sizeof(float3));

  float hosthx = (winup.x-windown.x) / scene.m_winwidth;
  float hosthy = (winup.y - windown.y) / scene.m_winheight;
  cudaMemcpyToSymbol(hx, &hosthx, sizeof(hosthx));
  cudaMemcpyToSymbol(hy, &hosthy, sizeof(hosthy));

  cudaMemcpyToSymbol(dwinwidth, &scene.m_winwidth, sizeof(scene.m_winwidth));
  cudaMemcpyToSymbol(dwinheight, &scene.m_winheight, sizeof(scene.m_winheight));

  cudaMemcpyToSymbol(nTri, &scene.nTri, sizeof(unsigned int) );
  cudaMemcpyToSymbol(nVtx, &scene.nVtx, sizeof(&scene.nVtx) );
  cudaMemcpyToSymbol(nLS, &scene.nLightSource, sizeof(unsigned int) );

/*
  cudaMemcpyToSymbol( dVtxBuf, &scene.dVtxBuf, sizeof( scene.dVtxBuf ) ); 
  cudaMemcpyToSymbol( dTriVtxBuf, &scene.dTriVtxBuf, sizeof( scene.dTriVtxBuf ) );
  cudaMemcpyToSymbol( dNormal, &scene.dNormal, sizeof(scene.dNormal) );
  cudaMemcpyToSymbol( dLS, &scene.dLS, sizeof(scene.dLS) );
  cudaMemcpyToSymbol( dsandboxColor, &scene.dsandboxColor, sizeof(scene.dsandboxColor) );
  cudaMemcpyToSymbol( dsandboxIsReflective, &scene.dsandboxIsReflective, sizeof(scene.dsandboxIsReflective) );
  */
}

extern "C" void cudaSceneMalloc( CScene& scene )
{
  cudaMalloc( (void**) & scene.dVtxBuf, sizeof(float)*scene.nVtx*3);
  cudaMalloc( (void**) & scene.dTriVtxBuf, sizeof(int)*scene.nTri*3);
  cudaMalloc((void**) &scene.dNormal, sizeof(float)*scene.nTri*3);
  cudaMalloc((void**) &scene.dLS, sizeof(float)*3*scene.nLightSource);
  cudaMalloc((void**) &scene.dsandboxColor, sizeof(float)*3*5);
  cudaMalloc((void**) &scene.dsandboxIsReflective, sizeof(unsigned int)*5);
  cudaMalloc((void**) &scene.dtree, sizeof(KdTree_rp)*scene.treesize);
}

extern "C" void cudaBindToTexture( unsigned int nVtx, unsigned int nTri, unsigned int nLS, CScene& scene )
{
  cudaBindTexture(0, texref_VtxBuf, scene.dVtxBuf, sizeof(float)*nVtx*3);
  cudaBindTexture(0, texref_TriVtx, scene.dTriVtxBuf, sizeof(int)*nTri*3);
  cudaBindTexture(0, texref_Normal, scene.dNormal, sizeof(float)*3*nTri);
  cudaBindTexture(0, texref_LS, scene.dLS, sizeof(float)*3*nLS); 
  cudaBindTexture(0, texref_sandboxColor, scene.dsandboxColor, sizeof(float)*3*5);
  cudaBindTexture(0, texref_sandboxIsReflective, scene.dsandboxColor, sizeof(unsigned int)*5);
}

extern "C" void cudaPassSceneToGlobalMem( CScene& scene,  float* pVtxBuf, int* pTriVtxBuf, float* pNormal, float* pLS, float* psandboxColor, unsigned int* psandboxIsReflective, KdTree_rp* ptree)
{
  if (scene.dVtxBuf)  cudaMemcpy( scene.dVtxBuf, pVtxBuf, sizeof(float)*scene.nVtx*3, cudaMemcpyHostToDevice);

  if (scene.dTriVtxBuf) cudaMemcpy( scene.dTriVtxBuf, pTriVtxBuf, sizeof(int)*scene.nTri*3, cudaMemcpyHostToDevice);

  if (scene.dNormal) cudaMemcpy( scene.dNormal, pNormal, sizeof(float)*scene.nTri*3, cudaMemcpyHostToDevice);

  if (scene.dLS) cudaMemcpy( scene.dLS, pLS, sizeof(float)*3* scene.nLightSource, cudaMemcpyHostToDevice);

  if (scene.dsandboxColor) cudaMemcpy( scene.dsandboxColor, psandboxColor, sizeof(float3)*5, cudaMemcpyHostToDevice );

  if (scene.dsandboxIsReflective)  cudaMemcpy( scene.dsandboxIsReflective, psandboxIsReflective, sizeof(unsigned int)*5, cudaMemcpyHostToDevice);

  if (scene.dtree) cudaMemcpy( scene.dtree, ptree, sizeof(KdTree_rp)*scene.treesize, cudaMemcpyHostToDevice );
}

// called when entire application ends
extern "C" void cudaFreeTextureResources()
{
  cudaUnbindTexture(texref_VtxBuf);
  cudaUnbindTexture(texref_Normal);
  cudaUnbindTexture(texref_TriVtx);
  cudaUnbindTexture(texref_LS);
}

extern "C" void cudaFreeGlobalMemory( CScene& scene)
{
  if ( scene.dVtxBuf ) cudaFree(scene.dVtxBuf);
  if ( scene.dTriVtxBuf )  cudaFree(scene.dTriVtxBuf);
  if ( scene.dNormal )  cudaFree(scene.dNormal);
  if ( scene.dLS )  cudaFree(scene.dLS);
  if ( scene.dsandboxColor ) cudaFree(scene.dsandboxColor); 
  if ( scene.dsandboxIsReflective )  cudaFree(scene.dsandboxIsReflective);
}

  
    


   
   
