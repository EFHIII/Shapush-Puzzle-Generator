# Steps
- Host defines the problem (like hardest 3x3 puzzle)
- Host generates the puzzle-space (the collection of all puzzles to be searched)
- Host sends the device a sequence of many puzzles
- Device solves each puzzle in parallel
  - Device further splits each puzzle computing each 'next move' in parallel
- device sends the host:
  - if puzzle was solvable
  - minimum moves to solve each puzzle
  - size of space searched
  - solved board state
- host evaluates the result sent from the device to give each board a score
- host reports back the top scoring boards to the user

# idea 2
- CPU works on each puzzle in sequence
- CPU sends GPU work on a per-layer basis
- GPU generates next 8 possible states
- check states for if they already exist

# TODO
- fix minor bug in pathing (my guess is it's when the player is grabbing inside a block without moving a block)
