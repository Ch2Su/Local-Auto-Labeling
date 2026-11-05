#!/usr/bin/env bash
# Claude 턴이 끝날 때 바뀐 것을 자동으로 커밋한다 (2026-10-02 사용자 요청).
# .claude/settings.json 의 Stop 훅이 부른다. 바뀐 게 없으면 아무것도 하지 않는다.
#
# - 무엇을 담을지는 .gitignore 가 정한다 (data/·models/·weights/ 등은 빠진다).
# - 5MB 가 넘는 파일은 담지 않고 알린다 — 한 번 커밋하면 기록에서 지우기 어렵다.
# - 이럴 때는 알리고 건너뛴다 (2026-10-31, 11-01 보강): 머지·리베이스·cherry-pick·revert 진행 중,
#   squash 머지·cherry-pick -n 결과가 스테이지돼 커밋을 기다림, HEAD 가 브랜치에 있지 않음, 풀리지 않은 충돌,
#   담을 내용에 충돌 표시(<<<<<<< ·>>>>>>> 로 시작하는 줄)나 비밀 키처럼 보이는 줄이 있음, 바뀐 파일이
#   200개 넘음(내보내기·압축 풀기 같은 것 — 그만큼 담으면 60초 제한에 걸린다).
#   stash pop 충돌은 MERGE_HEAD 같은 표식을 남기지 않아 옛 훅은 충돌 표시째 커밋했다(시험 저장소에서 재현).
#   충돌 표시는 스테이지한 결과에서 찾는다 — 이름이 바뀐 채 충돌한 파일은 git reset 뒤 새 파일이 되어서다.
# - 담을 것은 임시 색인(GIT_INDEX_FILE)에서 고른다. 건너뛰거나 커밋이 실패해도 실제 색인은 그대로다
#   (11-01 전에는 실패하면 git add -A 결과가 스테이지된 채 남아, 뒤이은 git reset --hard 가 새 파일을 지웠다).
# - 세션이 이 저장소의 다른 워크트리(.claude/worktrees/*)에서 일하면 그 워크트리를 커밋한다. Claude Code 는
#   워크트리로 옮겨도 CLAUDE_PROJECT_DIR 를 처음 체크아웃에 두고, 지금 위치는 stdin JSON 의 cwd 로만 준다.
#   다른 저장소로 cd 한 경우는 따라가지 않는다(같은 .git 을 쓰는 체크아웃만).
# - push 는 하지 않는다 (원격 GitHub 은 2026-10-31 에 붙였다). 실패하면 git 오류의 첫 줄과 함께 알린다.
# - 훅이 실패해도 Claude 작업을 막지 않도록 늘 0 으로 끝낸다.

set -u
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}" 2>/dev/null || exit 0
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0

# 메시지는 JSON 한 줄이다 — \ 와 " 는 이스케이프하고, 섞여 들어온 git 출력의 줄바꿈·탭은 공백으로.
say() {
    local s=$1
    s=${s//\\/\\\\}; s=${s//\"/\\\"}; s=${s//$'\r'/}; s=${s//$'\n'/ }; s=${s//$'\t'/ }
    printf '{"systemMessage": "%s"}\n' "$s"
}
skip() { say "자동 커밋 건너뜀: $1"; exit 0; }
fail() { say "자동 커밋 실패: $1"; exit 0; }

# 워크트리: stdin 의 cwd 가 같은 .git 을 쓰는 다른 체크아웃이면 그쪽으로 옮긴다.
input=
[ -t 0 ] || IFS= read -r -t 2 -d '' input
re='"cwd"[[:space:]]*:[[:space:]]*"(([^"\\]|\\.)*)"'
if [[ $input =~ $re ]]; then
    cwd=${BASH_REMATCH[1]}
    cwd=${cwd//\\\\/\\}
    common=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)
    top=$(cd "$cwd" 2>/dev/null && git rev-parse --show-toplevel 2>/dev/null)
    if [ -n "$common" ] && [ -n "$top" ] &&
        [ "$(cd "$top" && git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" = "$common" ]; then
        cd "$top" || exit 0
    fi
fi
cd "$(git rev-parse --show-toplevel)" 2>/dev/null || exit 0

gd=$(git rev-parse --absolute-git-dir) || exit 0
tmp="$gd/autocommit.$$"
trap 'rm -f "$tmp".*' EXIT
# git 이 남긴 오류 가운데 원인을 말하는 줄 (error:/fatal: 이 없으면 마지막 줄 — 훅 출력 등)
why() {
    local l
    l=$(grep -m1 -E '^(error|fatal):' "$tmp.err" 2>/dev/null) ||
        l=$(grep -v '^[[:space:]]*$' "$tmp.err" 2>/dev/null | tail -n1)
    printf '%s' "${l:-git 이 이유를 남기지 않았습니다}"
}

# 바뀐 게 없으면 여기서 끝낸다 — 아래 검사들은 git 을 여러 번 불러 턴마다 0.5초 넘게 든다(11-01 실측).
st=$(git status --porcelain --untracked-files=all 2>"$tmp.err") || fail "git status 가 실패했습니다 — $(why)"
[ -n "$st" ] || exit 0

# MERGE_HEAD 는 비거나 깨져도 파일이 남아 파일로 본다. CHERRY_PICK_HEAD·REVERT_HEAD 는 reftable 저장소에서
# 파일이 아니라서 ref 로 묻는다.
if [ -e "$gd/MERGE_HEAD" ] || [ -d "$gd/rebase-merge" ] || [ -d "$gd/rebase-apply" ] || [ -d "$gd/sequencer" ] ||
    git rev-parse -q --verify CHERRY_PICK_HEAD >/dev/null 2>&1 ||
    git rev-parse -q --verify REVERT_HEAD >/dev/null 2>&1; then
    skip "머지·리베이스·cherry-pick·revert 가 진행 중입니다 (git status 로 확인)"
fi
git symbolic-ref -q HEAD >/dev/null 2>&1 || skip "HEAD 가 브랜치에 있지 않습니다 (git status 로 확인)"
[ -z "$(git ls-files -u)" ] || skip "풀리지 않은 충돌이 있습니다 (git status 로 확인)"
# merge --squash 는 SQUASH_MSG, cherry-pick -n 은 MERGE_MSG 만 남긴다. 그 결과가 스테이지돼 있으면 사람이
# 메시지를 붙여 커밋할 몫이다. 메시지 파일만 보면 작업을 버린 뒤에도 남아 영영 건너뛰니 스테이지된 변경과 함께 본다.
for m in SQUASH_MSG MERGE_MSG; do
    if [ -e "$gd/$m" ] && ! git diff --cached --quiet 2>/dev/null; then
        skip "squash 머지나 cherry-pick -n 결과가 커밋을 기다립니다 — 끝난 일이면 직접 커밋하세요 (.git/$m)"
    fi
done
nchg=$(printf '%s\n' "$st" | wc -l | tr -d ' ')
[ "$nchg" -le 200 ] ||
    skip "바뀐 파일이 ${nchg}개입니다 — 내보내기·압축 풀기처럼 담지 않을 것이면 .gitignore 를, 담을 것이면 직접 커밋하세요"

idx=$(git rev-parse --git-path index)
[ -f "$idx" ] && cp "$idx" "$tmp.index"
gi() { GIT_INDEX_FILE="$tmp.index" git "$@"; }
gi add -A >/dev/null 2>"$tmp.err" || fail "git add -A 가 실패했습니다 — $(why)"

skipped=0
while IFS= read -r -d '' f; do
    [ -f "$f" ] || continue
    if [ "$(wc -c <"$f")" -gt 5000000 ]; then
        # :(literal) — 이름의 [ ] * ? 를 glob 으로 읽어 다른 파일까지 빼지 않게
        gi reset -q -- ":(literal)$f" 2>"$tmp.err" || fail "5MB 넘는 파일을 빼지 못했습니다 — $(why)"
        skipped=$((skipped + 1))
    fi
done < <(gi diff --cached --name-only -z --no-renames --diff-filter=AMT)

# 담을 내용(추가된 줄)을 "파일<TAB>줄" 로 모아 충돌 표시와 비밀 키를 찾는다. 걸리면 커밋 전체를 건너뛴다 —
# 그 파일만 빼면 짝을 이루는 변경(이름이 바뀐 파일의 옛 이름 삭제 등)만 커밋된다.
gi -c core.quotepath=off diff --cached -U0 --no-color --no-ext-diff --no-textconv --no-renames \
    --src-prefix=a/ --dst-prefix=b/ 2>/dev/null |
    awk '/^\+\+\+ /{f=substr($0,7); next} /^\+/{print f "\t" substr($0,2)}' >"$tmp.diff"
hits() { cut -f1 | sort -u | awk 'NR<=3 {printf "%s%s", (NR>1 ? ", " : ""), $0} END {if (NR>3) printf " 외 %d개", NR-3}'; }
marks=$(awk -F'\t' 'index($2, "<<<<<<< ") == 1 || index($2, ">>>>>>> ") == 1' "$tmp.diff" | hits)
[ -z "$marks" ] || skip "충돌 표시(<<<<<<< ·>>>>>>>)로 시작하는 줄이 있습니다: $marks — 진짜 충돌이 아니면 직접 커밋하세요"
keys='-----BEGIN [A-Z ]*PRIVATE KEY-----|sk-ant-[A-Za-z0-9_-]{20,}|rpa_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{30,}'
keys="$keys|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}|hf_[A-Za-z0-9]{30,}|AIza[0-9A-Za-z_-]{35}"
secrets=$(grep -E -e "$(printf '\t').*($keys)" "$tmp.diff" | hits)
[ -z "$secrets" ] || skip "비밀 키처럼 보이는 줄이 있습니다: $secrets — 키는 지우고 환경 변수로 (잘못 걸린 것이면 직접 커밋)"

if gi diff --cached --quiet; then
    [ "$skipped" -gt 0 ] && say "자동 커밋: 5MB 넘는 파일 ${skipped}개는 담지 않았습니다 (git status 로 확인)"
    exit 0
fi

# 한글 이름이 \354\204... 로 깨지지 않게 core.quotepath=off
files=$(gi -c core.quotepath=off diff --cached --name-only)
n=$(printf '%s\n' "$files" | wc -l | tr -d ' ')
names=$(printf '%s\n' "$files" | awk -F/ 'NR<=4 {printf "%s%s", (NR>1 ? ", " : ""), $NF}')
[ "$n" -gt 4 ] && names="$names 외 $((n - 4))개"

gi commit -q -m "자동 커밋: $names" \
    -m "Claude Code 턴이 끝날 때 .claude/hooks/autocommit.sh 가 남긴 커밋." \
    -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" >/dev/null 2>"$tmp.err" ||
    fail "커밋하지 못했습니다(실제 색인은 그대로) — $(why)"

# 임시 색인으로 커밋했으니 실제 색인을 새 HEAD 에 맞춘다. log -1 대신 rev-parse — log.showSignature 가 켜진
# PC 에서는 log 가 서명 확인 줄까지 내보낸다.
# 실제 .git/index.lock 을 잡는 것은 이 reset 하나뿐이다. 그 순간 다른 git 이 잠금을 쥐고 있으면 HEAD 만 나가고 색인이
# 뒤처지므로, 0.2초 간격으로 다섯 번 다시 해 본 뒤에만 알린다 (2026-11-05).
head=$(git rev-parse --short HEAD)
tries=0
until git reset -q 2>"$tmp.err"; do
    tries=$((tries + 1))
    [ "$tries" -le 5 ] ||
        skip "커밋 $head 은 됐지만 색인을 새 HEAD 에 맞추지 못했습니다 — git reset 을 한 번 돌리세요 ($(why))"
    sleep 0.2
done

msg="자동 커밋 $head (파일 ${n}개)"
[ "$skipped" -gt 0 ] && msg="$msg — 5MB 넘는 파일 ${skipped}개는 뺐습니다"
say "$msg"
exit 0
