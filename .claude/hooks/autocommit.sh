#!/usr/bin/env bash
# Claude 턴이 끝날 때 바뀐 것을 자동으로 커밋한다 (2026-10-02 사용자 요청).
# .claude/settings.json 의 Stop 훅이 부른다. 바뀐 게 없으면 아무것도 하지 않는다.
#
# - 무엇을 담을지는 .gitignore 가 정한다 (data/·models/·weights/ 등은 빠진다).
# - 5MB 가 넘는 파일은 담지 않고 알린다 — 한 번 커밋하면 기록에서 지우기 어렵다.
# - 머지·리베이스·cherry-pick·revert 가 진행 중이거나, HEAD 가 브랜치에 있지 않거나, 충돌이 남아 있으면
#   알리고 건너뛴다 (2026-10-31). stash pop 충돌은 MERGE_HEAD 같은 표식을 남기지 않아서, 옛 훅은
#   충돌 표시(<<<<<<<)째 커밋했다 — 시험 저장소에서 재현했다. 색인에서만 풀린 충돌(git restore --staged·
#   git reset 뒤)도 파일에 남은 표시로 거른다.
# - push 는 하지 않는다 (원격 GitHub 은 2026-10-31 에 붙였다). add·commit 이 실패하면 알린다.
# - 훅이 실패해도 Claude 작업을 막지 않도록 늘 0 으로 끝낸다.

set -u
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}" 2>/dev/null || exit 0
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0

say() { echo "{\"systemMessage\": \"$1\"}"; }
skip() { say "자동 커밋 건너뜀: $1 (git status 로 확인)"; exit 0; }
fail() { say "자동 커밋 실패: $1 (git status 로 확인)"; exit 0; }

# CHERRY_PICK_HEAD·REVERT_HEAD 는 reftable 저장소에서 파일이 아니라서 ref 로 묻는다.
for ref in MERGE_HEAD CHERRY_PICK_HEAD REVERT_HEAD; do
    git rev-parse -q --verify "$ref" >/dev/null 2>&1 && skip "머지·리베이스·cherry-pick·revert 가 진행 중입니다"
done
gd=$(git rev-parse --git-dir)
if [ -d "$gd/rebase-merge" ] || [ -d "$gd/rebase-apply" ] || [ -d "$gd/sequencer" ]; then
    skip "머지·리베이스·cherry-pick·revert 가 진행 중입니다"
fi
git symbolic-ref -q HEAD >/dev/null 2>&1 || skip "HEAD 가 브랜치에 있지 않습니다"
[ -z "$(git ls-files -u)" ] || skip "풀리지 않은 충돌이 있습니다"
if git rev-parse -q --verify HEAD >/dev/null 2>&1 &&
    git diff HEAD -U0 --no-color --no-ext-diff --no-textconv | grep -qE '^\+(<{7}|>{7}) '; then
    skip "충돌 표시가 남은 파일이 있습니다"
fi

git add -A >/dev/null 2>&1 ||
    fail "git add -A 가 실패했습니다 — 다른 프로그램이 파일을 잡고 있거나 .git/index.lock 이 남았을 수 있습니다"

skipped=0
while IFS= read -r -d '' f; do
    [ -f "$f" ] || continue
    if [ "$(wc -c <"$f")" -gt 5000000 ]; then
        git reset -q -- "$f"
        skipped=$((skipped + 1))
    fi
done < <(git diff --cached --name-only -z --diff-filter=AM)

if git diff --cached --quiet; then
    [ "$skipped" -gt 0 ] && say "자동 커밋: 5MB 넘는 파일 ${skipped}개는 담지 않았습니다 (git status 로 확인)"
    exit 0
fi

# 한글 이름이 \354\204... 로 깨지지 않게 core.quotepath=off
n=$(git -c core.quotepath=off diff --cached --name-only | wc -l | tr -d ' ')
names=$(git -c core.quotepath=off diff --cached --name-only | awk -F/ 'NR<=4 {printf "%s%s", (NR>1 ? ", " : ""), $NF}')
[ "$n" -gt 4 ] && names="$names 외 $((n - 4))개"

git commit -q -m "자동 커밋: $names" \
    -m "Claude Code 턴이 끝날 때 .claude/hooks/autocommit.sh 가 남긴 커밋." \
    -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" >/dev/null 2>&1 ||
    fail "변경은 스테이지만 해 두었습니다 — 새 PC 라면 git config user.name·user.email 부터"

msg="자동 커밋 $(git log -1 --format=%h) (파일 ${n}개)"
[ "$skipped" -gt 0 ] && msg="$msg — 5MB 넘는 파일 ${skipped}개는 뺐습니다"
say "$msg"
exit 0
