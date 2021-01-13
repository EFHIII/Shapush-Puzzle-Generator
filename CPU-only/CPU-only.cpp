#define PLAYER_X 2
#define PLAYER_Y 2

// 0 = nothing, # = parts of a block and it's initial position

int map[] = {
  1, 1, 5, 5, 0,
  1, 0, 0, 0, 0,
  2, 0, 0, 4, 4,
  2, 0, 0, 0, 4,
  2, 3, 3, 3, 0,
};

int topFew = 40;
int n = 1;

//  . .#. . .
//  .#. .#. .
// #. . . .#.
// #.#.#.#.#.
// #. . . .#.

#define MAX_BLOCKS 5// largest number in map[]
#define MAX_BLOCK_SIZE 3// most instances of the same digit > 0 in map[]

#define WIDTH 5// width of map[]
#define HEIGHT WIDTH// height of map[]

// -- end of main parameters --

#define THREAD_COUNT 0// 0 to detect hardware_concurrency (default: 8)

#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <algorithm>
#include <thread>

#include <mutex>

#include <windows.h>

std::string title = "G-";

#define BOARD_SIZE WIDTH * HEIGHT * 2
#define DATA_SIZE BOARD_SIZE + 3

int factorial(int a){
  int ans = 1;
  for(int i = a; i > 1; i--){
    ans = ans * i;
  }
  return ans;
}

// generates all the puzzles that will be checked
// Moidify this to search for different/more/fewer puzzle permutations
std::vector<int *> getSearchSpace(){

  int mapSize = WIDTH * HEIGHT;

  int mx = 0;//max block number
  for(int i = 0; i < mapSize; i++){
    if(map[i] > mx)
      mx = map[i];
  }

  std::vector<int> mxa;
  int *blockAts = new int[mx];
  const int bl = mx * MAX_BLOCK_SIZE;
  int *blockPos = new int[bl];
  for(int i = 0; i < mx; i++){
    mxa.push_back(i);
    blockAts[i] = 0;
  }

  //get block positions
  for(int i = 0; i < mapSize; i++){
    if(map[i] > 0){
      blockPos[MAX_BLOCK_SIZE * (map[i] - 1) + blockAts[map[i] - 1]++] = i;
    }
  }

  // print board
  for(int i = 0; i < HEIGHT; i++){
    for(int j = 0; j < WIDTH; j++){
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
    for(int j = 0; j < mx; j++){
      at[j] = 0;
    }
    bool done = false;

    while(!done){
      int *t = new int[BOARD_SIZE];
      int atT = 0;
      int v;
      for(int j = 0; j < mapSize; j++){
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

  std::cout << "puzzles:" << (int)ans.size() << std::endl << std::endl;
  delete[] blockAts;
  delete[] blockPos;
  delete[] perm;

  return ans;
}

void printBoard(int *b){
  std::cout << " [" << std::endl;
  for(int i = 0; i < HEIGHT; i++){
    for(int j = 0; j < WIDTH; j++){
      if(j > 0){
        std::cout << ",";
      }
      else{
        std::cout << "        [";
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
  std::cout << "      ]";
}

// pretty, colorful, console output
const int COLS[] = {10, 8, 8, 12, 14, 11, 13, 9, 15};
const int COLS2[] = {2, 0, 0, 4, 6, 3, 5, 1, 7};
HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

void color(int id, int id2){
  if(id == -1){return;}
  SetConsoleTextAttribute(hConsole, COLS[id + 2] + 16 * COLS2[id2 + 2]);
}

void prettyPrintBoard(int *b){
  for(int j = 0; j < WIDTH; j++){
    for(int i = 0; i < HEIGHT; i++){
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

// win condition
bool winning(int *board){
  return board[(board[BOARD_SIZE] + board[BOARD_SIZE + 1] * WIDTH) * 2 + 1] == -2;
}

bool moveBlock(int *board, int block, int x, int y){
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

void finish(int *newQueue, int *board, int X, int Y, bool grabbing, int facing, int dir, int to){
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

void makeMoves(int *queue, int *newQueue, int to, int dir, bool grabbing){
  //const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  //if(idx >= max){return;}

  //const int to = (idx) * (DATA_SIZE);

  //const int dir = (idx) % 4;
  //const bool grabbing = ((idx) / 4) % 2 == 1;

  int X = queue[BOARD_SIZE];
  int Y = queue[BOARD_SIZE + 1];
  int facing = queue[BOARD_SIZE + 2];
  int board[BOARD_SIZE];

  for(int i = 0; i < BOARD_SIZE; i++){
    board[i] = queue[i];
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

    if(X + x < 0 || Y + y < 0 || X + x >= WIDTH || Y + y >= HEIGHT){
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

//trie stuff
#define ALPHABET_SIZE MAX_BLOCKS + 3

struct TrieNode {
  struct TrieNode *children[ALPHABET_SIZE];
};

struct TrieNode *getNode(void){
  struct TrieNode *pNode =  new TrieNode;

  for (int i = 0; i < ALPHABET_SIZE; i++)
    pNode->children[i] = NULL;

  return pNode;
}

bool trieSearch(struct TrieNode *pCrawl, int *key){
  bool ret = true;
  for (int i = 0; i < DATA_SIZE; i++){
    int index = key[i] + 2;
    if (!pCrawl->children[index]){
      pCrawl->children[index] = getNode();
      ret = false;
    }

    pCrawl = pCrawl->children[index];
  }

  return ret;
}

// free all trie memory to avoid a memory leak
void trieDelete(struct TrieNode *root){
  for (int i = 0; i < ALPHABET_SIZE; i++){
    if(root->children[i]){
      trieDelete(root->children[i]);
      delete root->children[i];
    }
  }
}

// run puzzles (multithreaded)
std::mutex mu;
int cooldown = 0;
int lastb = 0;
void runPuzzles(int id, int threadCount, std::vector<int *> searchSpace, int SZ, int &best, int &bestIndex){
  // check every nth puzzle, where n = # of threads
  for(int puz = id; puz < SZ; puz += threadCount){
    // log puzzle starting
    mu.lock();
    //
    //std::cout << (SZ - puz) << " ";
    if(n == 1){
      std::cout << "thread " << (id + 1) << ": puzzle " <<  (puz/threadCount) << "/" << (SZ/threadCount) << " " << puz * 100 / SZ << "%" << " best: " << best << std::endl;
    }
    mu.unlock();

    // initialize trie of all board states already searched
    struct TrieNode *allBoards = getNode();

    // preparing input board
    int *newBoard = new int[DATA_SIZE];
    newBoard[BOARD_SIZE + 0] = PLAYER_X;
    newBoard[BOARD_SIZE + 1] = PLAYER_Y;
    newBoard[BOARD_SIZE + 2] = 0;
    for(int i = 0; i < BOARD_SIZE; i++){
      newBoard[i] = searchSpace[puz][i];
    }

    // add initial board to the list of previously searched boards
    trieSearch(allBoards, newBoard);

    // prepare queue of board states to be tree-searched
    std::vector<int *> queue = {newBoard};

    // perform tree-search
    bool done = false;
    int moves = 0;
    while(!done && (int)queue.size() > 0){
      moves++;
      int *newQueue = new int[(int)queue.size() * 8 * (DATA_SIZE)];

      int S = (int)queue.size();
      int SS = S * 8 * (DATA_SIZE);
      for(int i = 0; i < S; i++){
        for(int j = 0; j < 4; j++){
          makeMoves(queue[i], newQueue, (DATA_SIZE) * (i * 8 + j), j, false);
          makeMoves(queue[i], newQueue, (DATA_SIZE) * (i * 8 + j + 4), j, true);
        }
      }

      //makeMoves<<< blocks, 512 >>>(queue, newQueue, h_max);

      queue.clear();

      // check generated states, add new ones to the queue

      int *temp;
      for(int i = 0; i < SS; i+= DATA_SIZE){
        temp = &newQueue[i];

        // debug log
        //mu.lock();
        //std::cout << moves << " - " << ((i) / ((DATA_SIZE) * (8))) << " - " <<  ((((i) / (DATA_SIZE)) % 8) + 1)  << std::endl;
        //prettyPrintBoard(temp);
        //mu.unlock();
        //std::cout << std::endl;

        // if the state hasn't been searched yet:
        if(!trieSearch(allBoards, temp)){
          // win state condition
          if(winning(temp)){
            //std::cout << "s";
            //mu.lock();
            //std::cout << "thread " << (id + 1) << ": Solved " << (puz/threadCount) << " in " << moves << " moves; best = " << best << std::endl;
            //prettyPrintBoard(temp);
            //mu.unlock();

            if(moves >= best){
              best = moves;
              bestIndex = puz;
            }
            if(cooldown >= 0){cooldown--;}
            if(moves >= topFew && (cooldown <= 0 || moves > lastb)){
              cooldown = 80;
              lastb = moves + 10;
              topFew += 10;
              mu.lock();
              std::cout << std::endl;
              //std::cout << "thread " << (id + 1) << ": Solved " << (puz/threadCount) << " in " << moves << " moves; best = " << best << std::endl;
              //prettyPrintBoard(temp);
              std::cout << "    {" << std::endl << "      title: \"" << title << n++ << "\"," << std::endl <<
                "      size: { width: " << WIDTH << ", height: "<< HEIGHT << " }," << std::endl <<
                "      start: { x: " << PLAYER_X << ", y: " << PLAYER_Y << " }," << std::endl <<
                "      board:";
              printBoard(searchSpace[puz]);
              std::cout << "," << std::endl << "      stepGoals: [" << moves << ", 0, 0, 0]," << std::endl <<
                "      best: 0" << std::endl << "    },";
              mu.unlock();
            }
            done = true;
            i = SS;
          }

          // add to queue
          int *temp2 = new int[DATA_SIZE];
          for(int i = 0; i < DATA_SIZE; i++){
            temp2[i] = temp[i];
          }

          queue.push_back(temp2);
        }
      }

      // free memory
      delete[] newQueue;
    }

    // free memory
    queue.clear();
    queue.shrink_to_fit();
    trieDelete(allBoards);
    delete allBoards;
  }
}

// main function
int main(){
  int threadCount = THREAD_COUNT;
  if(threadCount == 0){
    threadCount = std::thread::hardware_concurrency();
    if(threadCount == 0){
      threadCount = 8;
    }
  }

  std::cout << "Running " << threadCount << " threads" << std::endl;
  SetConsoleTextAttribute(hConsole, 15);// set console colors to white on black
  std::vector<int *> searchSpace = getSearchSpace();// all the puzzles to be tested
  const int SZ = (int)searchSpace.size();// length of searchSpace

  // keeps track of the ideal puzzle and its score
  int best = 0;
  int bestIndex = 0;

  // multi-thread
  std::vector<std::thread> threads;

  int id = 0;
  for(int i = 0; i < threadCount; i++){
      threads.emplace_back([&](){runPuzzles(id++, threadCount, searchSpace, SZ, best, bestIndex);});
  }

  for( auto & t : threads ){
    t.join();
  }

  // print final result
  std::cout << "best took " << best << " moves" << std::endl;
  printBoard(searchSpace[bestIndex]);

  return 0;
}
