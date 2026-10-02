#!/usr/bin/env bash
# Claude 턴이 끝날 때 바뀐 것을 자동으로 커밋한다 (2026-10-02 사용자 요청).
# .claude/settings.json 의 Stop 훅이 부른다. 바뀐 게 없으면 아무것도 하지 않는다.
#
# - 무엇을 담을지는 .gitignore 가 정한다 (data/·models/·weights/ 등은 빠진다).
# - 5MB 가 넘는 파일은 담지 않고 알린다 — 한 번 커밋하면 기록에서 지우기 어렵다.
# - 머지·리베이스가 진행 중이면 건드리지 않는다.
# - 훅이 실패해도 Claude 작업을 막지 않도록 늘 0 으로 끝낸다.

set -u
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}" 2>/dev/null || exit 0
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0

gd=$(git rev-parse --git-dir)
if [ -e "$gd/MERGE_HEAD" ] || [ -d "$gd/rebase-merge" ] || [ -d "$gd/rebase-apply" ]; then
    exit 0
fi

git add -A >/dev/null 2>&1 || exit 0

skipped=0
while IFS= read -r -d '' f; do
    [ -f "$f" ] || continue
    if [ "$(wc -c <"$f")" -gt 5000000 ]; then
        git reset -q -- "$f"
        skipped=$((skipped + 1))
    fi
done < <(git diff --cached --name-only -z --diff-filter=AM)

if git diff --cached --quiet; then
    [ "$skipped" -gt 0 ] && echo "{\"systemMessage\": \"자동 커밋: 5MB 넘는 파일 ${skipped}개는 담지 않았습니다 (git status 로 확인)\"}"
    exit 0
fi

n=$(git diff --cached --name-only | wc -l | tr -d ' ')
names=$(git diff --cached --name-only | awk -F/ 'NR<=4 {printf "%s%s", (NR>1 ? ", " : ""), $NF}')
[ "$n" -gt 4 ] && names="$names 외 $((n - 4))개"

git commit -q -m "자동 커밋: $names" \
    -m "Claude Code 턴이 끝날 때 .claude/hooks/autocommit.sh 가 남긴 커밋." \
    -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" >/dev/null 2>&1 || exit 0

msg="자동 커밋 $(git log -1 --format=%h) (파일 ${n}개)"
[ "$skipped" -gt 0 ] && msg="$msg — 5MB 넘는 파일 ${skipped}개는 뺐습니다"
echo "{\"systemMessage\": \"$msg\"}"
exit 0
