FreeCell rules and notation used throughout:

- Board: 8 cascades (columns, numbered 1-8, listed root-first so the LAST card shown is
  the one on top and available), 4 free cells (named a-d, each holds one card), and 4
  foundations ("home", one per suit C/D/H/S, built up A -> K).
- Cards are two characters, rank then suit: A 2 3 4 5 6 7 8 9 T J Q K and C D H S
  (T = ten). D and H are red; C and S are black.
- Tableau building: a card may be placed on a cascade card one rank higher of the
  opposite colour (e.g. 8H on 9S). Any card may go to an empty cascade.
- A move is two characters, source then destination: cascades `1`-`8`, free cells
  `a`-`d`, `f` = first empty free cell (destination only), `h` = home/foundation
  (destination only; the suit is implied by the moved card). Examples: `27` (cascade 2
  to cascade 7), `3f` (top of cascade 3 to a free cell), `ah` (free cell a to home),
  `5h` (top of cascade 5 to home).
- A cascade-to-cascade move transfers the LONGEST properly-ordered run that legally
  fits, limited by capacity (1 + empty free cells) x 2^(empty cascades); an empty
  destination does not count toward the multiplier. There is no automatic play to the
  foundations: every card sent home is an explicit `h` move.
- The game is won when all four foundations reach K.
