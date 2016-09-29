#include <stdlib.h>
#include "scene.hpp"
#include <string>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <algorithm>

#include "kdtree.h"

using namespace std;


using namespace std;

extern "C" void cudaSceneMalloc( CScene& );

extern "C" void cudaBindToTexture( int nVtx, float* pVtxBuf, int nTri, int* pTriVtxBuf, float* pNormal, int nLS, float* pLS, float* psandboxColor, unsigned int* psandboxIsReflective);

extern "C" void cudaPassSceneToGlobalMem( CScene&, float* pVtxBuf, int* pTriVtxBuf, float* pNormal, float* pLS, float* psandboxColor, unsigned int* psandboxIsReflective, KdTree_rp* ptree);

extern "C" void cudaSetConstantMem( CScene & );

extern "C" void cudaFreeTextureResources( );

extern "C" void cudaFreeGlobalMemory( CScene& );

CPoint3D CPoint3D::operator- (CPoint3D param)
{
  CPoint3D tmp;
  tmp.x = x - param.x;
  tmp.y = y - param.y;
  tmp.z = z - param.z;
  return tmp;
}

CPoint3D CPoint3D::operator+ (CPoint3D param)
{
  CPoint3D tmp;
  tmp.x = x + param.x;
  tmp.y = y + param.y;
  tmp.z = z + param.z;
  return tmp;
}

float CPoint3D::operator* (CPoint3D param)
{
  return x*param.x + y*param.y + z*param.z;
}

CPoint3D CPoint3D::operator^ (CPoint3D param)
{
  CPoint3D tmp;
  tmp.x = y*param.z - z*param.y;
  tmp.y = x*param.z - z*param.x;
  tmp.z = x*param.y - y*param.x;
  return tmp;
}

void CPoint3D::normalize()
{
  float l = sqrt(x*x+y*y+z*z);
  x = x/l;
  y = y/l;
  z = z/l;
}

CScene::CScene()
{
  dVtxBuf = NULL;
  dTriVtxBuf = NULL;
  dLS = NULL;
  dNormal = NULL;
  dsandboxColor = NULL;
  dsandboxIsReflective = NULL;

  int scale = 1;
  // grey color for the object
  objectColor.r = 100;
  objectColor.g = 100;
  objectColor.b = 100;

  cameraPos.x = 0;
  cameraPos.y = 0;
  cameraPos.z = -2*scale;

  windowCenter.x = 0;
  windowCenter.y = 0;
  windowCenter.z = -1*scale;
  
  window_diagup.x = 1*scale;
  window_diagup.y = 1*scale;
  window_diagup.z = -1*scale; 

  window_diagdown.x = -1*scale;
  window_diagdown.y= -1*scale;
  window_diagdown.z = -1*scale;

  m_sandbox.m_diagup.x = -1*scale;
  m_sandbox.m_diagup.y = -1*scale;
  m_sandbox.m_diagup.z = -1*scale;

  m_sandbox.m_diagup.x = 1*scale;
  m_sandbox.m_diagup.y = 1*scale;
  m_sandbox.m_diagup.z = 1*scale;

  m_winwidth = 800;
  m_winheight = 600;

  CPoint3D LS;
  LS.x = 1*scale;
  LS.y = -1*scale;
  LS.z = -1*scale;
  lightSource.push_back(LS);

  LS.x = -1*scale;
  LS.y = -1*scale;
  LS.z = -1*scale;
  lightSource.push_back(LS);

  nLightSource = lightSource.size();
}

void CScene::LoadSandBox( int argc, char** argv )
{
}

void CScene::CalculateNormal()
{
  NormalBuf.resize(nTri);
  for ( int i = 0; i < nTri; i++ )
  {
    CPoint3D a = VtxBuf[TriVtxBuf[i][1]] - VtxBuf[TriVtxBuf[i][0]];
    a.x = VtxBuf[TriVtxBuf[i][1]].x - VtxBuf[TriVtxBuf[i][0]].x;
    a.y = VtxBuf[TriVtxBuf[i][1]].y- VtxBuf[TriVtxBuf[i][0]].y;
    a.z = VtxBuf[TriVtxBuf[i][1]].z - VtxBuf[TriVtxBuf[i][0]].z;

    CPoint3D b = VtxBuf[TriVtxBuf[i][2]] - VtxBuf[TriVtxBuf[i][0]];
    b.x = VtxBuf[TriVtxBuf[i][2]].x - VtxBuf[TriVtxBuf[i][0]].x;
    b.y = VtxBuf[TriVtxBuf[i][2]].y - VtxBuf[TriVtxBuf[i][0]].y;
    b.z = VtxBuf[TriVtxBuf[i][2]].z - VtxBuf[TriVtxBuf[i][0]].z;
    // if 0,1,2 are counter-clockwise labeled looking from outside
    // then the normal vector is outward pointing
    NormalBuf[i].x  = a.y*b.z - a.z*b.y;
    NormalBuf[i].y = a.z*b.x - a.x *b.z;
    NormalBuf[i].z = a.x*b.y - a.y*b.x;

    NormalBuf[i].normalize();
  }
}
    
void CScene::CalcBoundingBox()
{
  xmin = 1e6;
  ymin = 1e6;
  zmin = 1e6;
  xmax = -1e6;
  ymax= -1e6;
  zmax = -1e6;
  for ( int i = 0; i < nVtx; i++)
  {
    if (VtxBuf[i].x < xmin) xmin = VtxBuf[i].x;
    if (VtxBuf[i].x > xmax) xmax = VtxBuf[i].x;
    if (VtxBuf[i].y < ymin) ymin = VtxBuf[i].y;
    if (VtxBuf[i].y > ymax) ymax = VtxBuf[i].y;
    if (VtxBuf[i].z < zmin) zmin = VtxBuf[i].z;
    if (VtxBuf[i].z > zmax) zmax = VtxBuf[i].z;
  }
  printf("x:%.2f~%.2f\n", xmin, xmax);
  printf("y:%.2f~%.2f\n", ymin, ymax);
  printf("Z:%.2f~%.2f\n", zmin, zmax);
}

int CScene::CalcTreeSize(KdTree* node)
{
  if ( !node ) return 0;
  else return 1+CalcTreeSize(node->pleft)+CalcTreeSize(node->pright);
}

/*
bool CScene::cmpx( int i, int j )
{
  CPoint3D p1 = VtxBuf[i];
  CPoint3D p2 = Vtxbuf[j];
  if (p1.x < p2.x ) return true;
  else return false;
}

bool CScene::cmpy( int i, int j )
{
  CPoint3D p1 = VtxBuf[i];
  CPoint3D p2 = Vtxbuf[j];
  if ( p1.y < p2.y ) return true;
  else return false;
}

bool CScene::cmpz( int i, int j )
{
  CPoint3D p1 = VtxBuf[i];
  CPoint3D p2 = Vtxbuf[j];
  if ( p1.z < p2.z ) return true;
  else return false;
}

vector<int> ChooseTriSTLVersion( vector<int> plist, float mid, int axis) 
{
  switch axis:
    case 0:
      sort(plist.begin(), plist.end(), cmpx );
      vector<int>::iterator it = upper_bound( plist.begin(), plist.end(),  cmpx(axis) );
    case 1:
      sort(plist.begin(), plist.end(), cmpy );
      vector<int>::iterator it = upper_bound( plist.begin(), plist.end(),  cmpy(axis) );
    case 2:
      sort(plist.begin(), plist.end(), cmpz );
      vector<int>::iterator it = upper_bound( plist.begin(), plist.end(),  cmpz(axis) );

  vector<int> chosen;
  chosen.resize( it - plist.begin()+1);
  copy( plist.begin(), it, chosen.begin() );

  return chosen;
}
*/

bool IsInBox( CPoint3D p, float xmin, float xmax, float ymin, float ymax, float zmin, float zmax )
{
  if ( (p.x>=xmin) && (p.x<=xmax) && (p.y>=ymin) && (p.y<=ymax) && (p.z>=zmin) && (p.z<=zmax) )
    return true;
  else return false;
}

vector<int> CScene::ChooseTri( vector<int> trilist, float xmin, float xmax, float ymin, float ymax, float zmin, float zmax)
{
  vector<int> chosen;
  chosen.clear();

  for ( int i = 0; i < trilist.size(); i++)
  {
    int idx0 = TriVtxBuf[ trilist[i] ][0];
    int idx1 = TriVtxBuf[ trilist[i] ][1];
    int idx2 = TriVtxBuf[ trilist[i] ][2];

    CPoint3D p0 = VtxBuf[ idx0 ];
    CPoint3D p1 = VtxBuf[ idx1 ];
    CPoint3D p2 = VtxBuf[ idx2 ];

    if ( IsInBox(p0, xmin, xmax, ymin, ymax, zmin, zmax) || 
      IsInBox(p1, xmin, xmax, ymin, ymax, zmin, zmax) || 
      IsInBox(p2, xmin, xmax, ymin, ymax, zmin, zmax) )
      chosen.push_back( trilist[i] );
  }

  return chosen;
}

KdTree* CScene::ConstructKdTree(int axis, int depth, vector<int>trilist, float xmin, float xmax, float ymin, float ymax, float zmin, float zmax)
{
//  if (trilist.size() == 0 ) return NULL;

  KdTree* node= new KdTree;
  node->xmin = xmin;
  node->xmax = xmax;
  node->ymin = ymin;
  node->ymax = ymax;
  node->zmin = zmin;
  node->zmax = zmax;

//  for ( int i =0; i< 6; i++) node->rope[i] = NULL;
  
  if ( (trilist.size() <= LEAFSIZE)  )
  {
    node->trinum = trilist.size();
//    node->tri = (int*)malloc( sizeof(int)*trilist.size() );
    for ( int i = 0; i < trilist.size(); i++) 
      node->tri[i] = trilist[i];
    node->axis = axis;
    node->splitpos = -1;
    node->pleft = NULL;
    node->pright = NULL;
  }
  else
  {
    node->trinum = -1;
 //   node->tri = NULL;
    float mid;
    vector<int> leftptlist;
    leftptlist.clear();
    vector<int> rightptlist;
    rightptlist.clear();
    switch (axis)
    {
      case 0:
        mid = (xmin+xmax)/2;
        leftptlist = ChooseTri( trilist, xmin, mid, ymin, ymax, zmin, zmax);
        rightptlist = ChooseTri( trilist, mid, xmax, ymin, ymax, zmin, zmax);

        node->pleft = ConstructKdTree( (axis+1)%3, depth+1, leftptlist, xmin,mid, ymin, ymax, zmin, zmax); 
        node->pright = ConstructKdTree( (axis+1)%3, depth+1, rightptlist, mid, xmax, ymin, ymax, zmin, zmax);
        break;
      case 1:
        mid = (ymin+ymax)/2;
        leftptlist = ChooseTri( trilist, xmin, xmax, ymin, mid, zmin, zmax);
        rightptlist = ChooseTri( trilist, xmin, xmax, mid, ymax, zmin, zmax);

        node->pleft = ConstructKdTree( (axis+1)%3, depth+1, leftptlist, xmin,xmax, ymin, mid, zmin, zmax); 
        node->pright = ConstructKdTree( (axis+1)%3, depth+1, rightptlist, xmin, xmax, mid, ymax, zmin, zmax);
        break;
      case 2:
        mid = (zmin+zmax)/2;
        leftptlist = ChooseTri( trilist, xmin, xmax, ymin, ymax, zmin, mid);
        rightptlist = ChooseTri( trilist, xmin, xmax, ymin, ymax, mid, zmax);

        node->pleft = ConstructKdTree( (axis+1)%3, depth+1, leftptlist, xmin, xmax, ymin, ymax, zmin, mid);
        node->pright = ConstructKdTree( (axis+1)%3, depth+1, rightptlist, xmin, xmax, ymin, ymax, mid, zmax);
        break;
      default:
        break;
    }
    node->axis = axis;
    node->splitpos = mid; 
  }

  return node;
}

bool CScene::SplitPlaneAboveBox( int axis, float splitpos, int idx)
{
  switch (axis)
  {
    case 0:
      if ( splitpos >= tree_rp[idx].xmax) return true;
      else return false;
    case 1:
      if ( splitpos >= tree_rp[idx].ymax) return true;
      else return false;
    case 2:
      if ( splitpos >= tree_rp[idx].zmax) return true;
      else return false;
    default:
      return true;
  }
}

bool CScene::SplitPlaneBelowBox( int axis, float splitpos, int idx)
{
  switch (axis)
  {
    case 0:
      if ( splitpos <= tree_rp[idx].xmin) return true;
      else return false;
    case 1:
      if ( splitpos <= tree_rp[idx].ymin) return true;
      else return false;
    case 2:
      if ( splitpos <= tree_rp[idx].zmin) return true;
      else return false;
    default:
      return true;
  }
}

void CScene::OptimizeRope( int idx, int &rp, int face)
{
 /* 
  while ( tree_rp[rp].trinum < 0) // not leaf
  {
    if ( face/2 == tree_rp[rp].axis ) break; 
    
    if ( SplitPlaneAboveBox( tree_rp[rp].axis, tree_rp[rp].splitpos, idx) )
      rp = tree_rp[rp].left;
    else if ( SplitPlaneBelowBox( tree_rp[rp].axis, tree_rp[rp].splitpos, idx) )
      rp = tree_rp[rp].right;
    else break;

    if ( rp<0) break;

  }
  */
  
}

void CScene::BuildRope( int idx, int rope[6] )
{
  for ( int i =0; i < 6; i++ )
      if ( abs(rope[i]) > 4000 )
      { printf("rope out of range\n"); exit(-1); return;}

  if ( tree_rp[idx].trinum >= 0) // is leaf    
  {
    for ( int i = 0; i < 6; i++ )
      tree_rp[idx].rope[i] = rope[i];
  }
  else
  {
    for ( int i =0 ; i< 6; i++ )
    {
      if ( rope[i]>=0 ) OptimizeRope( idx, rope[i], i );
    }

    if ( tree_rp[idx].axis < 0) printf("Line 381\n");

    int sl = 2* tree_rp[idx].axis;
    int sr = 2* tree_rp[idx].axis + 1;

    int ropeL[6];
    int ropeR[6];
    for ( int i = 0; i< 6; i++ )
    {
      ropeL[i] = rope[i];
      ropeR[i] = rope[i];
    }
    ropeL[sr] = tree_rp[idx].right;
    ropeR[sl] = tree_rp[idx].left;

    if ( tree_rp[idx].right >0) BuildRope( tree_rp[idx].right, ropeR);
    if ( tree_rp[idx].left > 0) BuildRope( tree_rp[idx].left, ropeL );
  }
}

void CScene::RpKdTreePrint(int argc)
{
  for ( int i =0; i < treesize; i++)
  {
    printf("node%d: left%d, right%d, axis%d, splitpos%.2f\n", i, tree_rp[i].left, tree_rp[i].right, tree_rp[i].axis, tree_rp[i].splitpos );
    if (tree_rp[i].trinum >= 0)
    {
      printf("  Leaf node containing:");
      for ( int j = 0; j < tree_rp[i].trinum ; j++)
   //     printf("%d ", tree_rp[i].tri[j]);
          printf(". ");
      printf("\n");
      printf("  Rope: ");
      for ( int j=0; j<6; j++ )
        printf("%d ", tree_rp[i].rope[j]);
      printf("\n");
    }

      printf("  xmin %.3f, xmax %.3f, ymin %.3f, ymax % .3f, zmin %.3f, zmax %.3f\n", tree_rp[i].xmin,tree_rp[i].xmax, tree_rp[i].ymin, tree_rp[i].ymax, tree_rp[i].zmin, tree_rp[i].zmax);

    
  }

  return;
}

void CScene::ConstructRpKdTree( int idx, KdTree* node)
{
  if (!node) return;

  static int count = 0;
  
  tree_rp[idx].xmin = node->xmin;
  tree_rp[idx].xmax = node->xmax;
  tree_rp[idx].ymin = node->ymin;
  tree_rp[idx].ymax = node->ymax;
  tree_rp[idx].zmin = node->zmin;
  tree_rp[idx].zmax = node->zmax;
  
  tree_rp[idx].axis = node->axis;
  tree_rp[idx].splitpos = node->splitpos;

  tree_rp[idx].trinum =  node->trinum;
  if ( node->trinum>= 0)
  {
//    tree_rp[idx].tri= (int*) malloc( node->trinum* sizeof(int) );

    for ( int i =0; i< node->trinum; i++)
      tree_rp[idx].tri[i] = node->tri[i];
  }
//  else tree_rp[idx].trinum = NULL;

  if ( node->pleft )
  {
    count++;
    tree_rp[idx].left = count;
    ConstructRpKdTree( count, node->pleft);
  }
  else tree_rp[idx].left = -1;

  if ( node->pright )
  {
    count++;
    tree_rp[idx].right = count;
    ConstructRpKdTree( count, node->pright);
  }
  else tree_rp[idx].right = -1;
}

void CScene::LoadObjects( int argc, char** argv )
{
  FILE* file = fopen( argv[1], "r");
  if (!file) 
  {
    cerr<<"Cannot open obj file"<<endl;
    return;
  }

  char buf[128];
  
  CPoint3D tmp;
  vector<int> idx;
  idx.resize(3);

  VtxBuf.clear();
  TriVtxBuf.clear();

  while ( fscanf(file, "%s", buf)!=EOF )
  {
    switch( buf[0] )
    {
      case '#':
        // eat up rest of the line
        fgets(buf,sizeof(buf), file);
        break; 
      case 'v':
        fscanf(file, "%f %f %f\n", &tmp.x, &tmp.y, &tmp.z);
        VtxBuf.push_back(tmp);
        // eat up rest of the line
       // fgets(buf,sizeof(buf), file);
        break;
      case 'f':
        fscanf(file, "%d %d %d\n", &idx[0], &idx[1], &idx[2]);
        idx[0] = idx[0]-1; idx[1] = idx[1]-1, idx[2] =idx[2]-1;
        TriVtxBuf.push_back(idx);
        // eat up rest of the line
      //  fgets(buf,sizeof(buf), file);
        break;
      default:
        // eat up rest of the line
        fgets(buf, sizeof(buf), file);
        break;
    }
  }

  nVtx = VtxBuf.size();
  nTri = TriVtxBuf.size();
  printf("Obj file loaded. Vertex number:%d. Face number:%d\n", nVtx, nTri);

  fclose(file);

  printf("Calculating normal vector...\n");
  CalculateNormal();

  printf("Calculating bounding box...\n");
  CalcBoundingBox();

  vector<int> trilist;
  trilist.resize( nTri );
  for ( int i = 0; i < nTri; i++ ) trilist[i] = i;

  printf("Building kdtree...\n");
  tree = ConstructKdTree( 0, 0, trilist, xmin, xmax, ymin, ymax, zmin, zmax );

  treesize = CalcTreeSize(tree);
  tree_rp = (KdTree_rp*)malloc( treesize * sizeof(KdTree_rp) );
  printf("Convert to indexed kdtree...\n");
  ConstructRpKdTree(0,tree);


  printf("Building rope...\n");
  for ( int i =0; i< treesize; i++) 
    for ( int j =0; j<6; j++) tree_rp[i].rope[j] = -10;
  int rope[6];
  for ( int i =0; i < 6; i++ ) rope[i] = -1;
  BuildRope( 0, rope );
}

void CScene::WriteObjects()
{
  FILE* fout = fopen("scene.obj", "w");
  for ( int i =0; i< nVtx; i++)
    fprintf(fout, "v %f %f %f\n", VtxBuf[i].x, VtxBuf[i].y, VtxBuf[i].z);
  for ( int i = 0; i<nTri; i++)
    fprintf(fout, "f %d %d %d\n", TriVtxBuf[i][0]+1, TriVtxBuf[i][1]+1, TriVtxBuf[i][2]+1);
//  for ( int i = 0; i<nTri; i++)
 //   fprintf(fout, "fn %f %f %f\n", NormalBuf[i].x, NormalBuf[i].y, NormalBuf[i].z);
  fclose(fout);
}

void CScene::BindToTexture()
{
  // position of points
  float* pVtxBuf = (float*)malloc(nVtx*3*sizeof(float));
  for ( int i= 0; i < nVtx; i++ )
  {
    pVtxBuf[3*i] = VtxBuf[i].x;
    pVtxBuf[3*i+1] = VtxBuf[i].y;
    pVtxBuf[3*i+2] = VtxBuf[i].z;
  }

  // index of three vertices of every triangle
  int* pTriVtxBuf = (int*)malloc(nTri*3*sizeof(int));
  for ( int i = 0; i < nTri; i++ )
  {
    pTriVtxBuf[3*i] = TriVtxBuf[i][0];
    pTriVtxBuf[3*i+1] = TriVtxBuf[i][1];
    pTriVtxBuf[3*i+2] = TriVtxBuf[i][2];
    if ( i <2)
      printf("%d %d %d\n", TriVtxBuf[i][0], TriVtxBuf[i][1], TriVtxBuf[i][2]);
  }

  // normal vector of every triangle
  float* pNormal = (float*)malloc(nTri*3*sizeof(float));
  for ( int i =0; i < nTri; i++)
  {
    pNormal[3*i] = NormalBuf[i].x;
    pNormal[3*i+1] = NormalBuf[i].y;
    pNormal[3*i+2] = NormalBuf[i].z;
  }

  // position of light sources
  float* pLightSource = (float*)malloc( lightSource.size()*3*sizeof(float));
  for ( int i =0; i < lightSource.size(); i++ )
  {
    pLightSource[3*i] = lightSource[i].x;
    pLightSource[3*i+1] = lightSource[i].y;
    pLightSource[3*i+2] = lightSource[i].z;
  }

  float* psandboxColor = (float*)malloc( sizeof(float)*3*5);
  for ( int i=0; i <5; i++)
  {
    psandboxColor[3*i] = m_sandbox.facecolor[i].r;
    psandboxColor[3*i+1] = m_sandbox.facecolor[i].g;
    psandboxColor[3*i+2] = m_sandbox.facecolor[i].b;
  }

  unsigned int* psandboxIsReflective = (unsigned int*)malloc( sizeof(unsigned int)*5);
  for ( int i =0; i< 5; i++)
    psandboxIsReflective[i] = m_sandbox.isReflective[i];

  cudaBindToTexture(nVtx, pVtxBuf, nTri, pTriVtxBuf, pNormal, lightSource.size(), pLightSource, psandboxColor, psandboxIsReflective); 

  free(pVtxBuf);
  free(pTriVtxBuf);
  free(pNormal);
  free(psandboxColor);
  free(psandboxIsReflective);
}

void CScene::PassSceneToGlobalMem()
{
  // position of points
  float* pVtxBuf = (float*)malloc(nVtx*3*sizeof(float));
  for ( int i= 0; i < nVtx; i++ )
  {
    pVtxBuf[3*i] = VtxBuf[i].x;
    pVtxBuf[3*i+1] = VtxBuf[i].y;
    pVtxBuf[3*i+2] = VtxBuf[i].z;
  }

  // index of three vertices of every triangle
  int* pTriVtxBuf = (int*)malloc(nTri*3*sizeof(int));
  for ( int i = 0; i < nTri; i++ )
  {
    pTriVtxBuf[3*i] = TriVtxBuf[i][0];
    pTriVtxBuf[3*i+1] = TriVtxBuf[i][1];
    pTriVtxBuf[3*i+2] = TriVtxBuf[i][2];
    if ( i <2)
      printf("%d %d %d\n", TriVtxBuf[i][0], TriVtxBuf[i][1], TriVtxBuf[i][2]);
  }

  // normal vector of every triangle
  float* pNormal = (float*)malloc(nTri*3*sizeof(float));
  for ( int i =0; i < nTri; i++)
  {
    pNormal[3*i] = NormalBuf[i].x;
    pNormal[3*i+1] = NormalBuf[i].y;
    pNormal[3*i+2] = NormalBuf[i].z;
  }

  // position of light sources
  float* pLightSource = (float*)malloc( lightSource.size()*3*sizeof(float));
  for ( int i =0; i < lightSource.size(); i++ )
  {
    pLightSource[3*i] = lightSource[i].x;
    pLightSource[3*i+1] = lightSource[i].y;
    pLightSource[3*i+2] = lightSource[i].z;
  }

  float* psandboxColor = (float*)malloc( sizeof(float)*3*5);
  for ( int i=0; i <5; i++)
  {
    psandboxColor[3*i] = m_sandbox.facecolor[i].r;
    psandboxColor[3*i+1] = m_sandbox.facecolor[i].g;
    psandboxColor[3*i+2] = m_sandbox.facecolor[i].b;
  }

  unsigned int* psandboxIsReflective = (unsigned int*)malloc( sizeof(unsigned int)*5);
  for ( int i =0; i< 5; i++)
    psandboxIsReflective[i] = m_sandbox.isReflective[i];

  cudaPassSceneToGlobalMem(*this,  pVtxBuf, pTriVtxBuf, pNormal,  pLightSource, psandboxColor, psandboxIsReflective, tree_rp); 

  free(pVtxBuf);
  free(pTriVtxBuf);
  free(pNormal);
  free(pLightSource);
  free(psandboxColor);
  free(psandboxIsReflective);
}

void CScene::SetConstantMem()
{
  cudaSetConstantMem( *this );
}

void CScene::FreeTextureResources()
{
  cudaFreeTextureResources();
}

void CScene::FreeGlobalMemory()
{
  cudaFreeGlobalMemory( *this);
}

void CScene::SceneMalloc()
{
  // can not use cudaMalloc in cpp file
  // call cudaMalloc in cu file
  cudaSceneMalloc( *this );
}


