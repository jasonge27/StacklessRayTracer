#ifndef KD_TREE_H
#define KD_TREE_H

#define LEAFSIZE 15 
//#define TREEDEPTH 15 

typedef struct node 
{
  struct node* pleft;
  struct node* pright;

  int axis;
  float splitpos;

  int trinum;
  int tri[LEAFSIZE];

//  struct node* rope[6];

  float xmin;
  float xmax;
  float ymin;
  float ymax;
  float zmin;
  float zmax;
}KdTree;

typedef struct node_rp
{
  int left;
  int right;

  int axis;
  float splitpos;

  int trinum;
  int tri[LEAFSIZE];

  int rope[6];

  float xmin;
  float xmax;
  float ymin;
  float ymax;
  float zmin;
  float zmax;
}KdTree_rp;

#endif
