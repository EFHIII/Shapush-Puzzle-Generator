# Shapush Puzzle Generator
To generate Shapush puzzles, you can open the `Shapush Puzzle Generator.sln` file in VS, compile, and run it. To generate different puzzles, you can change the parameters in the first 20 lines of `first-tests.cu`. map is a 1-dimensional array that represents the 2 dimensional board's starting block positions. The function `getSearchSpace` uses this to generate potentially interesting permutations of puzzles with those block placements and the given player starting position. By default, it searches for the puzzles that require the most moves to solve when solved perfectly, but this can be changed; the metric is defined after the line
```
  // win state condition
```
Because of the use of CUDA, the code will probably only work on systems with an NVIDIA GPU that has a compute capability > 2.0. There's also use of CPU multithreading, so computers with more CPU cores will likely see better results.

Puzzles generated with this can be directly used with https://github.com/EFHIII/shapush

# TODO
- make non-CUDA fallback
