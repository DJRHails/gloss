You are playing FreeCell. Win the game by moving all 52 cards to the foundations.

$rules

How to play here:

- Submit moves with the `play` tool. Its `moves` argument is a space-separated sequence
  of move codes, applied in order (e.g. "3h 27 1a"). You may send one move or many.
- Moves are applied until the first illegal one: the legal prefix STAYS applied, the
  rest of the sequence is discarded, and the tool result tells you which move failed and
  why. Plan sequences you are confident in.
- The tool result confirms each applied move$feedback_clause. Track the board state
  yourself in your reasoning — think carefully about where every card is before you
  move.

Play deliberately: read the board, consider candidate lines, check them for blockers,
then commit. If the game becomes unwinnable, keep making the best progress you can.
