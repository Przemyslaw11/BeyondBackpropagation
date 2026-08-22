# Refactoring notes

This refactor is intentionally incremental. The canonical package owns new
contracts and boundaries, while legacy scripts and modules remain available to
protect existing research workflows. The decision register in
`docs/refactoring-decisions.md` records every intentional behavior correction
and its scientific impact.

Published values and the dataset protocol are not regenerated or rewritten by
the migration. Reproduced runs should be identified by their structured run
metadata and compared only when their measurement definitions match.
