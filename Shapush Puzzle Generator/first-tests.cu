#include <iostream>
#include <string.h>

#define MAX_BLOCKS 6
#define WIDTH 5

int map[] = {
  0,0,1,0,0,
  0,1,0,1,0,
  1,0,0,0,1,
  1,1,1,1,1,
  1,0,0,0,1,
};

struct board{
  unsigned int playerX;
  unsigned int playerY;
  unsigned int depth;
  unsigned int* board;
};

void perm(int *m){
  int l = WIDTH*WIDTH;
  int onv = 2;
  for(int i = 0; i < l; i++){
    if(m[i] == 1){
      m[i] = onv;
      if(i % WIDTH > 0 && m[i-1] == 1){m[i-1] = onv;}
      if(i % WIDTH < WIDTH-1 && m[i+1] == 1){m[i+1] = onv;}
      onv++;
      i = max(i-WIDTH*2,0);
    }
    else if(m[i]>1){
      if(i%WIDTH > 0 && m[i-1] == 1){
        m[i-1] = m[i];
      }
      if(i%WIDTH <WIDTH-1 && m[i+1] == 1){
        m[i+1] = m[i];
      }
      if(i/WIDTH>>0 < WIDTH && m[i+WIDTH] > 1 && m[i+WIDTH] != m[i]){
        int tv = m[i+WIDTH];
        for(int j = 0; j < l; j++){
          if(m[j] == tv){m[j] = m[i];}
        }
      }
    }
  }

  int totalNums = 1;
  for(int i = 0; i < l; i++){
    if(m[i] > totalNums){
      int v = m[i];
      for(int j = i; j < l; j++){
        if(m[j] == v){
          m[j] = totalNums;
        }
      }
      totalNums++;
    }
  }

  std::cout << "    regions: " << totalNums << std::endl;
}

struct board* getSearchSpace(){

  perm(map);

  int spots = 0;
  int mapSize = sizeof(map)/sizeof(map[0]);
  for(int i = 0; i < mapSize ; i++){if(map[i]){spots++;}}

  long int puzzles = mapSize*spots;//player*flag *blockCols*blockPorts

  //print basic info
  std::cout << "      spots: " << spots << std::endl;
  std::cout << "   map size: " << mapSize << std::endl;
  std::cout << " max blocks: " << MAX_BLOCKS << std::endl;
  std::cout << "    puzzles: " << puzzles << std::endl << std::endl;

  //print board
  for(int i = 0; i < WIDTH; i++){
    for(int j = 0; j < WIDTH; j++){
      if(map[i*WIDTH+j]){
        std::cout << " " << map[i*WIDTH+j];
      }
      else{
        std::cout << " .";
      }
    }
    std::cout << std::endl << std::endl;
  }

  struct board* ans = (board*)malloc((9*9) * sizeof(struct board));

  int at = 0;
  for(int px = 0; px < 3 ; px++){
    for(int py = 0; py < 3 ; py++){
      for(int fx = 0; fx < 3 ; fx++){
        for(int fy = 0; fy < 3 ; fy++){

//          for(int bp = 0; bp < 9 ; bp++){
//            for(int h = 0; h < 9 ; h++){
//              for(int p = 0; p < 9 ; p++){

                if(px==fx && py == fy){
                  continue;
                }//player starts at the flag

                unsigned int theBoard[] = {
                  0,0, 0,0, 0,0,
                  0,0, 0,0, 0,0,
                  0,0, 0,0, 0,0
                };

                ans[at].playerX = px;
                ans[at].playerY = py;
                ans[at].depth = 0;
                ans[at++].board = theBoard;
//              }
//            }
//          }
        }
      }
    }
  }
  return ans;
}

__global__ void searchSpace(){
  int id = threadIdx.x;
}

__global__ void solveBoard(){

}

int main(void){

  struct board* searchSpace = getSearchSpace();

  //std::cout << searchSpace[0].playerX << "," << searchSpace[0].playerY << std::endl;

  //std::cout << searchSpace[30].playerX << "," << searchSpace[30].playerY << std::endl;


  //std::cout << (sizeof(&searchSpace)) << std::endl;
  //std::cout << (sizeof(&searchSpace)/sizeof(searchSpace[0])) << std::endl;

  return 0;
}
