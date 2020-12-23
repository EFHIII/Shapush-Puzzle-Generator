#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <algorithm>
#include <unordered_set>

#define MAX_BLOCKS 5
#define MAX_BLOCK_SIZE 4
#define WIDTH 5
#define HEIGHT WIDTH

#define BOARD_SIZE WIDTH * HEIGHT * 2
#define DATA_SIZE BOARD_SIZE + 3

#define PLAYER_X 2
#define PLAYER_Y 2

int map[] = {
  1, 1, 5, 5, 0,
  1, 0, 0, 0, 0,
  1, 0, 0, 4, 4,
  2, 0, 0, 0, 4,
  2, 2, 3, 3, 0,
};

int factorial(int a){
  int ans = 1;
  for(int i = a; i>1; i--){
    ans = ans * i;
  }
  return ans;
}

std::vector<int *> getSearchSpace(){

  int mapSize = sizeof(map)/sizeof(map[0]);

  int mx = 0;//max block number
  for(int i = 0; i < mapSize; i++){
    if(map[i] > mx) { mx = map[i]; }
  }

  std::vector<int> mxa;
  int *blockAts = new int[mx];
  const int bl = mx * MAX_BLOCK_SIZE;
  int *blockPos = new int[bl];
  for(int i = 0; i < mx; i++) {
    mxa.push_back(i);
    blockAts[i] = 0;
  }

  //get block positions
  for(int i = 0; i < mapSize; i++) {
    if(map[i] > 0) {
      blockPos[MAX_BLOCK_SIZE * (map[i] - 1) + blockAts[map[i] - 1]++] = i;
    }
  }

  // print board
  for(int i = 0; i < HEIGHT; i++) {
    for(int j = 0; j < WIDTH; j++) {
      if(map[WIDTH * i + j] > 0){
        std::cout << map[WIDTH * i + j] << " ";
      }
      else{
        std::cout << ". ";
      }
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  //initialize permutations
  const int perms = factorial(mx);
  const int allPerms = perms * mx;
  int *perm = new int[allPerms];

  //permute
  int at = 0;
  do{
    for(int i = 0; i < mx; i++){
      perm[at++] = mxa[i];
    }
  } while(std::next_permutation(mxa.begin(),mxa.end()));

  //all boards
  std::vector<int *> ans;

  for(int i = 0; i < perms; i++){
    int *at = new int[mx];
    for(int j = 0; j < mx; j++) {
      at[j] = 0;
    }
    bool done = false;

    while(!done){
      int *t = new int[mapSize * 2];
      int atT = 0;
      int v;
      for(int j = 0; j < mapSize; j++) {
        v = map[j] > 0 ? perm[i * mx + map[j] - 1] + 1 : 0;
        t[atT++] = v;
        t[atT++] = v == mx ? -2 : -1;
      }

      for(int j = 0; j < mx; j++){
        t[blockPos[j * MAX_BLOCK_SIZE + at[j]] * 2 + 1] = perm[i * mx + j];
      }

      //append to ans
      ans.push_back(t);

      int on = 0;
      while(on < mx && at[on] >= blockAts[on] - 1){
        at[on++] = 0;
      }
      if(on >= mx){
        done = true;
      } else{
        at[on]++;
      }

      //done = true;
    }
  }

  std::cout << "puzzles:" << ans.size() << std::endl << std::endl;
  delete[] blockAts;
  delete[] blockPos;
  delete[] perm;

  return ans;
}

void printBoard(int *b){
  std::cout << "[" << std::endl;
  for(int j = 0; j < WIDTH; j++) {
    for(int i = 0; i < HEIGHT; i++) {
      if(i > 0){
        std::cout << ",";
      }
      else{
        std::cout << "  [";
      }
      if(b[(i + j * WIDTH) * 2 + 1] < 0){
        std::cout << "[" << b[(i + j * WIDTH) * 2] << "," << b[(i + j * WIDTH) * 2 + 1] << "]";
      }
      else{
        std::cout << "[" << b[(i + j * WIDTH) * 2] << ", " << b[(i + j * WIDTH) * 2 + 1] << "]";
      }
    }
    std::cout << "]," << std::endl;
  }
  std::cout << "]";
}

#include <windows.h>
//const int COLS[] = {2, 8, 8, 4, 6, 3, 5, 1, 7};
const int COLS[] = {10, 8, 8, 12, 14, 11, 13, 9, 15};
//const int COLS2[] = {10, 0, 0, 12, 14, 11, 13, 9, 15};
const int COLS2[] = {2, 0, 0, 4, 6, 3, 5, 1, 7};
HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

void color(int id, int id2){
  if(id == -1){return;}
  SetConsoleTextAttribute(hConsole, COLS[id + 2] + 16 * COLS2[id2 + 2]);
}

void prettyPrintBoard(int *b){
  for(int j = 0; j < WIDTH; j++) {
    for(int i = 0; i < HEIGHT; i++) {
      color(b[(i + j * WIDTH) * 2], b[(i + j * WIDTH) * 2]);
      std::cout << "[";
      color(b[(i + j * WIDTH) * 2 + 1], b[(i + j * WIDTH) * 2 + 1]);
      if(i == b[BOARD_SIZE] && j == b[BOARD_SIZE + 1]){
        switch(b[BOARD_SIZE + 2]){
          case(0): std::cout << "V";break;
          case(1): std::cout << "<";break;
          case(2): std::cout << "^";break;
          case(3): std::cout << ">";break;
        }
      }
      else if(b[(i + j * WIDTH) * 2] == 0){
        std::cout << " ";
      }
      else{
        std::cout << "@";
      }
      color(b[(i + j * WIDTH) * 2], b[(i + j * WIDTH) * 2]);
      std::cout << "]";
    }
    SetConsoleTextAttribute(hConsole, 15);
    std::cout << std::endl;
  }
}

std::string stringify(int *a){
  std::string str;
  for (int i = 0; i < DATA_SIZE; i++) {
    str += std::to_string(a[i] + 2);
    //str += ",";
  }
  return str;
}

bool winning(int *board){
  //if(true){
  //  return true;
  //}
  if(board[(board[BOARD_SIZE] + board[BOARD_SIZE + 1] * WIDTH) * 2 + 1] == -2){
    return true;
  }
  return false;
}

__forceinline__ __device__ bool moveBlock(int *board, int block, int x, int y){
  int ar[MAX_BLOCK_SIZE * 3];
  int at = 0;
  for(int i = 0; i < WIDTH; i++){
    for(int j = 0; j < HEIGHT; j++){
      if(board[(i + j * WIDTH) * 2] == block){
        if(i + x < 0 || j + y < 0 || i + x >= WIDTH || j + y >= HEIGHT ||
          board[(i + x + (j + y) * WIDTH) * 2] > 0 && board[(i + x + (j + y) * WIDTH) * 2] != block ){
          return false;
        }
        else{
          ar[at * 3] = i;
          ar[at * 3 + 1] = j;
          ar[at * 3 + 2] = board[(i + j * WIDTH) * 2 + 1];
          at++;
        }
      }
    }
  }

  for(int i = 0; i < at; i++){
    board[(ar[i * 3] + ar[i * 3 + 1] * WIDTH) * 2] = 0;
    board[(ar[i * 3] + ar[i * 3 + 1] * WIDTH) * 2 + 1] = -1;
  }
  for(int i = 0; i < at; i++){
    board[(ar[i * 3] + x + (ar[i * 3 + 1] + y) * WIDTH) * 2] = block;
    board[(ar[i * 3] + x + (ar[i * 3 + 1] + y) * WIDTH) * 2 + 1] = ar[i * 3 + 2];
  }

  return true;
}

__forceinline__ __device__ void finish(int *newQueue, int *board, int X, int Y, bool grabbing, int facing, int dir, int to){
  if(!grabbing){
    facing = dir;
  }

  for(int i = 0; i < BOARD_SIZE; i++){
    newQueue[to + i] = board[i];
  }
  newQueue[to + BOARD_SIZE    ] = X;
  newQueue[to + BOARD_SIZE + 1] = Y;
  newQueue[to + BOARD_SIZE + 2] = facing;
}

__global__ void makeMoves(int *queue, int *newQueue, int max){
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx >= max){return;}

  const int from =  ((idx) / 8) * (DATA_SIZE);
  const int to = (idx) * (DATA_SIZE);

  const int dir = (idx) % 4;
  const bool grabbing = ((idx) / 4) % 2 == 1;

  //printf("%d: from %d\n", idx, from);
  //printf("%d - %d %d\n", (idx) % 8, dir, grabbing?1:0);

  int X = queue[from + BOARD_SIZE];
  int Y = queue[from + BOARD_SIZE + 1];
  int facing = queue[from + BOARD_SIZE + 2];
  //printf("%d: from %d  %d,%d %d\n", idx, from, X, Y, facing);
  int board[BOARD_SIZE];

  for(int i = 0; i < BOARD_SIZE; i++){
    board[i] = queue[from + i];
  }

  int x = 0;
  int y = 0;

  switch(dir){
    case(0): y = 1;break;
    case(1): x =-1;break;
    case(2): y =-1;break;
    case(3): x = 1;break;
  }

  int curs0 = (int)board[(X + Y * WIDTH) * 2];
  int curs1 = (int)board[(X + Y * WIDTH) * 2 + 1];

  if(X + x >= 0 && X + x < WIDTH && Y + y >= 0 && Y + y < HEIGHT){
    int nexs0 = (int)board[(X + x + (Y + y) * WIDTH) * 2];
    int nexs1 = (int)board[(X + x + (Y + y) * WIDTH) * 2 + 1];
    int fc0 = 0;
    int fc1 = 0;

    switch(facing){
      case(0): fc1 = 1;break;
      case(1): fc0 =-1;break;
      case(2): fc1 =-1;break;
      case(3): fc0 = 1;break;
    }

    int fcn0 = 0;
    if(X + fc0 >= 0 && X + fc0 < WIDTH && Y + fc1 >= 0 && Y + fc1 < HEIGHT){
      fcn0 = board[(X + fc0 + (Y + fc1) * WIDTH) * 2];
    }

    bool rmp = curs1 > 0;
    if(grabbing && (X + fc0 < 0 || Y + fc1 < 0 || X + fc0 >= WIDTH || Y + fc1 >= HEIGHT) ){}
    else if(grabbing && (curs0 == nexs0 || (fcn0 == nexs0 && (rmp || curs0 == 0 || curs1 == 0)) || nexs0 == 0 && (rmp || curs0 == 0 || curs1 == 0) ) ){
      if(rmp && ((fc0 == x && fc1 == y && curs0 != 0) || nexs0 == 0) ){
        finish(newQueue, board, X, Y, grabbing, facing, dir, to);
        return;
      }
      //if(X + fc0 >= 0 && X + fc0 < WIDTH && Y + fc1 >= 0 && Y + fc1 < HEIGHT){
        if(fcn0 > 0 && fcn0 != curs0 && moveBlock(board,fcn0,x,y) ){
          X += x;
          Y += y;
          finish(newQueue, board, X, Y, grabbing, facing, dir, to);
          return;
        }
        if(fcn0 != 0 && fcn0 != curs0){
          finish(newQueue, board, X, Y, grabbing, facing, dir, to);
          return;
        }
      //}
    }
    else if(grabbing && nexs0 > 0 && nexs1 == curs0){
      if(fcn0 > 0 && (x != fc0 || y != fc1) && fcn0 != curs0 && moveBlock(board,fcn0,x,y) ){
        X += x;
        Y += y;
        finish(newQueue, board, X, Y, grabbing, facing, dir, to);
        return;
      }
      else if(fcn0 != curs0 && fcn0 != 0){
        finish(newQueue, board, X, Y, grabbing, facing, dir, to);
        return;
      }
    }

    else if(grabbing && curs0 > 0 && nexs0 == curs1){
      if(fcn0 > 0 && fcn0 != nexs0 && moveBlock(board,fcn0,x,y) ){
        if(fcn0 == curs0){
          moveBlock(board,fcn0,-x,-y);
        }
        X += x;
        Y += y;
        finish(newQueue, board, X, Y, grabbing, facing, dir, to);
        return;
      }
      else if(fcn0 != nexs0 && fcn0 != 0){
        if(fcn0 == curs0){
          X += x;
          Y += y;
        }
        finish(newQueue, board, X, Y, grabbing, facing, dir, to);
        return;
      }
    }

    if(X + fc0 < 0 || Y + fc1 < 0 || X + fc0 >= WIDTH || Y + fc1 >= HEIGHT){
      finish(newQueue, board, X, Y, grabbing, facing, dir, to);
      return;
    }
    if(curs0 == nexs0){
      X += x;
      Y += y;
      finish(newQueue, board, X, Y, grabbing, facing, dir, to);
      return;
    }
    if(nexs0 > 0 && nexs1 == curs0){
      X += x;
      Y += y;
      finish(newQueue, board, X, Y, grabbing, facing, dir, to);
      return;
    }
    if(curs0 > 0 && nexs0 == curs1){
      X += x;
      Y += y;
      finish(newQueue, board, X, Y, grabbing, facing, dir, to);
      return;
    }
    if(curs1 > 0 && nexs1 == curs1){
      X += x;
      Y += y;
      finish(newQueue, board, X, Y, grabbing, facing, dir, to);
      return;
    }
  }
  finish(newQueue, board, X, Y, grabbing, facing, dir, to);
  return;
}

int main(){
  SetConsoleTextAttribute(hConsole, 15);
  std::vector<int *> searchSpace = getSearchSpace();
  const int SZ = searchSpace.size();
  int best = 0;
  int bestIndex = 0;

  for(int puz = 0; puz < SZ; puz++){
    std::unordered_set<std::string> allBoards;

    int newBoard[DATA_SIZE];
    newBoard[BOARD_SIZE + 0] = PLAYER_X;
    newBoard[BOARD_SIZE + 1] = PLAYER_Y;
    newBoard[BOARD_SIZE + 2] = 0;
    for(int i = 0; i < BOARD_SIZE; i++){
      newBoard[i] = searchSpace[puz][i];
    }
    //std::cout << "puzzle " <<  puz << "/" << SZ << "    " << puz * 100 / SZ << "%" << std::endl << stringify(newBoard) << std::endl;
    std::cout << "puzzle " <<  puz << "/" << SZ << "    " << puz * 100 / SZ << "%" << std::endl;
    allBoards.insert(stringify(newBoard));

    std::vector<int *> queue = {newBoard};

    bool done = false;
    int moves = 0;
    while(!done && queue.size() > 0){
      moves++;
      std::vector<int> h_queue;
      for(int i = 0; i < queue.size(); i++){
        //std::cout << "queue " << i << std::endl;
        //prettyPrintBoard(queue[i]);
        //std::cout << std::endl << std::endl;
        for(int j = 0; j < DATA_SIZE; j++){
          h_queue.push_back(queue[i][j]);
        }
      }

      int *d_queue;
      int *d_newQueue;

      cudaMalloc(&d_queue, h_queue.size() * sizeof(int));
      cudaMalloc(&d_newQueue, h_queue.size() * sizeof(int) * 8);

      cudaMemcpy( d_queue, h_queue.data(), h_queue.size() * sizeof(int), cudaMemcpyHostToDevice);

      int blocks = queue.size() / 128 + 1;
      //std::cout << "blocks: " << blocks << std::endl;

      int h_max = queue.size() * 8;

      queue.clear();

      std::cout << "move " << moves << " threads: " << h_max << std::endl;

      makeMoves<<< blocks, 1024 >>>(d_queue, d_newQueue, h_max);
      cudaDeviceSynchronize();

      int *h_newQueue = new int[h_queue.size() * 8];

      cudaMemcpy( h_newQueue, d_newQueue, h_queue.size() * 8 * sizeof(int), cudaMemcpyDeviceToHost);

      //std::vector<int *> newQueue;

      for(int i = 0; i < h_queue.size() * 8; i+= DATA_SIZE){
        int *temp = new int[DATA_SIZE];
        for(int j = 0; j < DATA_SIZE; j++){
          temp[j] = h_newQueue[i + j];
        }

        //std::cout << temp[BOARD_SIZE] << ", " << temp[BOARD_SIZE + 1] << ", " << temp[BOARD_SIZE + 2] << std::endl;
        //std::cout << moves << " - " << ((i) / ((DATA_SIZE) * (8))) << " - " <<  ((((i) / (DATA_SIZE)) % 8) + 1)  << std::endl;
        //prettyPrintBoard(temp);
        //std::cout << std::endl << std::endl;

        if(allBoards.find(stringify(temp)) == allBoards.end()){
          if(winning(temp)){
            std::cout << "Solved in " << moves << " moves; best = " << best << std::endl;
            if(moves >= best){
              best = moves;
              bestIndex = puz;
              prettyPrintBoard(temp);
              printBoard(searchSpace[puz]);
            }
            done = true;
            i = h_queue.size() * 8;
          }
          queue.push_back(temp);
          allBoards.insert(stringify(temp));

          //prettyPrintBoard(temp);
          //std::cout << std::endl << std::endl;
        }
      }

    }

  }

  std::cout << "best took " << best << " moves" << std::endl;
  printBoard(searchSpace[bestIndex]);

  return 0;
}
