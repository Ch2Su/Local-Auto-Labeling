"""저장해 둔 목록끼리 주소 첫 칸(번호)이 같은 카메라를 가리키는지 본다. API 를 부르지 않는다."""
import json
import os
import sys
import urllib.parse


def load(path):
    with open(path, 'rb') as f:
        obj = json.loads(f.read().decode('utf-8', 'replace'))
    return obj['response']['data']


def first_seg(u):
    return ([s for s in urllib.parse.urlsplit(str(u or '')).path.split('/') if s] or [''])[0]


def table(items):
    return {first_seg(it.get('cctvurl')): (str(it.get('cctvname')), str(it.get('coordx')), str(it.get('coordy')))
            for it in items}


def main():
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding='utf-8', errors='replace')
    d = sys.argv[1]
    names = sys.argv[2:]
    tabs = {n: table(load(os.path.join(d, n))) for n in names}
    base = names[0]
    for n in names[1:]:
        a, b = tabs[base], tabs[n]
        common = set(a) & set(b)
        same = sum(1 for k in common if a[k] == b[k])
        same_name = sum(1 for k in common if a[k][0] == b[k][0])
        print('%s vs %s: ids %d / %d, common %d, same (name, x, y) %d, same name %d, only in first %d, only in second %d' % (
            base, n, len(a), len(b), len(common), same, same_name, len(set(a) - set(b)), len(set(b) - set(a))))
    # 토큰이 목록마다 다른가(같은 카메라의 주소 꼬리)
    raw = {n: {first_seg(it.get('cctvurl')): str(it.get('cctvurl')) for it in load(os.path.join(d, n))} for n in names}
    for n in names[1:]:
        common = set(raw[base]) & set(raw[n])
        tail = lambda u: urllib.parse.urlsplit(u).path.split('/', 2)[-1]
        print('%s vs %s: same token tail %d / %d' % (base, n, sum(1 for k in common if tail(raw[base][k]) == tail(raw[n][k])), len(common)))
    # 이름이 겹치는 카메라, 좌표가 겹치는 카메라
    a = tabs[base]
    by_name, by_xy = {}, {}
    for k, (nm, x, y) in a.items():
        by_name.setdefault(nm, []).append(k)
        by_xy.setdefault((x, y), []).append(k)
    print('%s: names shared by 2+ ids: %d names;  coords shared by 2+ ids: %d coords (max %d at one point)' % (
        base, sum(1 for v in by_name.values() if len(v) > 1), sum(1 for v in by_xy.values() if len(v) > 1),
        max(len(v) for v in by_xy.values())))


if __name__ == '__main__':
    main()
