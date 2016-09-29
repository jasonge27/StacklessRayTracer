#include <stdlib.h>
#include <stdio.h>
#include "scene.hpp"

#include <opencv.hpp>
#include <highgui.h>

using namespace cv;

CScene scene;

extern "C" void launch_kernel_test( char*, CScene& scene );
extern "C" void cudaFreeTextureResources(); 

int main( int argc,char** argv )
{
  scene.LoadObjects(argc, argv);
  scene.WriteObjects();

  printf("malloc...\n");
  scene.SceneMalloc();
  printf("malloc done.\n");

  printf("copy to global mem...\n");
  scene.PassSceneToGlobalMem();
  printf("copy done.\n");

  printf("constant mem...\n");
  scene.SetConstantMem();
  printf("constant mem copy done.\n");

  char* pos = (char*)malloc( sizeof(char)*4* scene.m_winwidth * scene.m_winheight ); 
  for ( int i = 0; i < scene.m_winwidth; i++ )
    for ( int j = 0; j < scene.m_winheight; j++ )
      for ( int k =0; k < 4; k++)
        pos[ (i*scene.m_winheight + j)*4 + k ] = 0;

  launch_kernel_test( pos, scene );

  //free the global memory associated with scene
  scene.FreeGlobalMemory( );

  cudaFreeTextureResources();

  Mat image( scene.m_winheight, scene.m_winwidth, CV_8UC3, Scalar(0,0,0));

  for ( int i = 0; i < scene.m_winheight; i++ )
    for ( int j = 0; j < scene.m_winwidth; j++ )
      {
        image.at<cv::Vec3b>(i,j)[0] = pos[ (i*scene.m_winwidth+ j)*4 + 2 ];
        image.at<cv::Vec3b>(i,j)[1] = pos[ (i*scene.m_winwidth+ j)*4 + 1 ];
        image.at<cv::Vec3b>(i,j)[2] = pos[ (i*scene.m_winwidth + j)*4 + 0 ];
      }

  imwrite("im_testK.png", image);
  free(pos);
  
  return 0;
}

