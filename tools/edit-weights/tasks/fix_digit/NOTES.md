# fix_digit (harness notes — not shown to Claude)

Planted bug: fold output-row 7 into row 1, then zero row 7 / set b7=-10.
Ideal repair: restore W_out[7], b_out[7], W_out[1], b_out[1] toward the
pre-bug values (answer_key.bug.restored_hint), or otherwise re-separate
class 7 from class 1 with minimal edits.
