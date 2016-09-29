#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include "kdtree.h"

class CPoint3D
{
  public:
    float x;
    float y;
    float z;
  
    CPoint3D operator- (CPoint3D);
    CPoint3D operator+ (CPoint3D);
    float operator* (CPoint3D); // inner dot product between vectors 
    CPoint3D operator^ (CPoint3D); // outer dot product between vectors

    void normalize();
};

class RGB
{
  public:
    unsigned int r;
    unsigned int g;
    unsigned int b;
};

class CSandBox 
{
  public:
    CPoint3D m_diagdown; //x,y,z small 
    CPoint3D m_diagup; // x,y,z large

    // The faces are labeled as: 
    // 0: zmax, 1:xmax, 2:ymax, 3:xmin, 4:ymin
    // xmin face is the window face
    RGB facecolor[5];
    unsigned int isReflective[5];

    CSandBox()
    {
      facecolor[1].r = 200;
      facecolor[1].g = 0;
      facecolor[1].b = 0;

      facecolor[2].r = 0;
      facecolor[2].g = 200;
      facecolor[2].b = 0;

      facecolor[3].r = 0;
      facecolor[3].g = 0;
      facecolor[3].b = 200;

      facecolor[4].r = 250;
      facecolor[4].g = 250;
      facecolor[4].b = 250;

      isReflective[0] =1;
      isReflective[1] =0;
      isReflective[2] =0;
      isReflective[3] =0;
      isReflective[4] =0;
    }
};

// Passing to texture memory is done by calling member function PassSceneToTexture()
// Passing to constant memory is done when calling member functiSetConstantMem()
class CScene // can only manage one object
{
  public:
    CSandBox m_sandbox; // pass to const. & tex. memory

    unsigned int m_winwidth ; // pass to constant memory
    unsigned int m_winheight ; // pass to const. mem 

    unsigned int nVtx; // pass to const. mem
    std::vector<CPoint3D> VtxBuf; // pass to tex. mem

    unsigned int nTri; // pass to const. mem
    std::vector<std::vector<int> > TriVtxBuf; // pass to tex. mem

    RGB objectColor; // pass to const. mem

    std::vector<CPoint3D> NormalBuf; // pass to tex. mem

    unsigned int nLightSource; // pass to const. mem
    std::vector<CPoint3D> lightSource; // pass to tex. mem

    CPoint3D cameraPos; // pass to const. mem

    // pass to const. mem
    CPoint3D windowCenter; 
    CPoint3D window_diagup;
    CPoint3D window_diagdown;
    
    void CalcBoundingBox();

  public:
    // The Bounding box and KdTree
    float xmin;
    float xmax;
    float ymin;
    float ymax;
    float zmin;
    float zmax;

    // list of Kdtree
    KdTree* tree;
    KdTree* ConstructKdTree( int axis, int depth, std::vector<int> trilist, 
    
    float xmin, float xmax, float ymin, float ymax, float zmin, float zmax);
    void OptimizeRope( int idx, int & rp, int face);
    void BuildRope( int idx, int rope[6] );
    bool SplitPlaneBelowBox( int, float, int );
    bool SplitPlaneAboveBox( int , float ,int );

    std::vector<int> ChooseTri( std::vector<int>, float, float, float, float, float, float);

    int treesize;
    int CalcTreeSize( KdTree* node );
    // Dynamic aray of kdtree
    KdTree_rp* tree_rp;
    void ConstructRpKdTree( int idx, KdTree* node);

  public:
    void RpKdTreePrint( int argc);

  public:
    float* dVtxBuf;
    int* dTriVtxBuf;
    float* dNormal;
    float* dLS;
    float* dsandboxColor;
    unsigned int* dsandboxIsReflective;
    KdTree_rp* dtree;

  private:
    void CalculateNormal();

  public:
    CScene();

  public:
    // callback
    void keyboard( unsigned char key, int x, int y)
    {
    }

    void LoadSandBox( int argc, char** argv );
    void LoadObjects( int argc, char** argv );
    void WriteObjects( );

    //The first thing always
    void SceneMalloc();

    // Make sure to call SceneMalloc before
    void PassSceneToGlobalMem();

    // Bind global memory to texture
    void BindToTexture();

    void SetConstantMem();
    
    // when entire application ends
    // Don't forget to call in the main 
    void FreeTextureResources();
    void FreeGlobalMemory();

//TODO    
//void TransformObject( );
//
};

#endif 
