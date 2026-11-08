#!/bin/sh
# Build-time check of the runtime image: dReach must answer delta-sat (exit 51) on a reachable
# toy model and unsat (exit 52) on an unreachable one, and /usr/bin/time must exist (the runner uses it).
set -u
test -x /usr/bin/time || { echo "dReach self-test: /usr/bin/time missing" >&2; exit 1; }
dir=$(mktemp -d)
for case in reach:51 unreach:52; do
    model=${case%%:*}
    want=${case##*:}
    cp "/opt/dreach/selftest/$model.drh" "$dir/"
    (cd "$dir" && dReach -k 0 -z "$model.drh" --precision 0.001 > "$model.log" 2>&1)
    got=$?
    if [ "$got" -ne "$want" ]; then
        echo "dReach self-test $model: exit $got, expected $want" >&2
        cat "$dir/$model.log" >&2
        exit 1
    fi
    echo "dReach self-test $model: ok (exit $got)"
done
rm -rf "$dir"
