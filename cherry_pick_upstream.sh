#!/bin/bash
# cherry_pick_upstream.sh — 밀린 upstream/master 커밋을 오래된 순서로 cherry-pick.
#
# - 충돌없는 커밋: 반영 (git cherry-pick -x, 원본 커밋 해시 기록)
# - empty(이미 반영된 패치): skip
# - 충돌: abort 후 $OUT/cp_conflicts.txt + $OUT/cp_conflict_files.txt 에 기록
#
# 사용법:
#   git checkout -b <sync-브랜치> master
#   ./cherry_pick_upstream.sh [베이스 브랜치] [출력 디렉터리]
#   (기본: 베이스=master, 출력=/tmp)
#
# 결과 예: DONE ok=251 empty=5 conflicts=11 errors=0
set -euo pipefail
cd "$(dirname "$0")"
BASE="${1:-master}"
OUT="${2:-/tmp}"
UPSTREAM="upstream/master"

cur=$(git branch --show-current)
if [ -z "$cur" ] || [ "$cur" = "$BASE" ]; then
  echo "현재 브랜치('$cur')에서 실행 불가 — sync 브랜치를 checkout한 뒤 실행하세요" >&2
  exit 1
fi

git rev-list --reverse --format='%H' "$BASE".."$UPSTREAM" 2>/dev/null > "$OUT/behind_ordered.txt"
: > "$OUT/cp_conflicts.txt"
: > "$OUT/cp_conflict_files.txt"
ok=0; empty=0; conf=0; err=0
while read -r c; do
  if git cherry-pick -x "$c" >"$OUT/cp_out.txt" 2>&1; then
    ok=$((ok+1))
  else
    if [ -f .git/CHERRY_PICK_HEAD ]; then
      unmerged=$(git diff --name-only --diff-filter=U 2>/dev/null || true)
      if [ -n "$unmerged" ]; then
        conf=$((conf+1))
        echo "$c" >> "$OUT/cp_conflicts.txt"
        { echo "@@$c"; echo "$unmerged"; } >> "$OUT/cp_conflict_files.txt"
        git cherry-pick --abort >/dev/null 2>&1
      else
        empty=$((empty+1))
        git cherry-pick --skip >/dev/null 2>&1 || git cherry-pick --quit >/dev/null 2>&1
      fi
    else
      err=$((err+1))
      echo "ERROR $c" >> "$OUT/cp_conflicts.txt"
      git cherry-pick --abort >/dev/null 2>&1 || git cherry-pick --quit >/dev/null 2>&1
    fi
  fi
done < "$OUT/behind_ordered.txt"
echo "DONE ok=$ok empty=$empty conflicts=$conf errors=$err"
echo "conflict log: $OUT/cp_conflicts.txt / $OUT/cp_conflict_files.txt"
