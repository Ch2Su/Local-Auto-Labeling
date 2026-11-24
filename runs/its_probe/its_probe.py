"""ITS(국가교통정보센터) CCTV Open API 를 조금만 불러 응답 구조·해상도를 잰다.

키는 환경 변수 ITS_API_KEY(프로세스에 없으면 Windows 사용자 환경 변수)에서 읽고 어디에도 찍지 않는다.
받은 응답(24시간짜리 영상 주소·실제 지명이 들어 있다)은 --out 에만 쓴다 - 저장소 밖에 둘 것.

    its_probe.py list --type ex --cctv-type 3 --out DIR      API 1회 - 목록과 응답 구조
    its_probe.py stills --in FILE --n 200                    정지 영상 머리만 받아 크기 분포
    its_probe.py hls --in FILE --n 3                         재생 목록·비트레이트·프레임 크기
    its_probe.py mp4 --in FILE --n 3                         동영상(mp4) 주소가 무엇인지
    its_probe.py soak --in FILE --seconds 300                한 대를 이어서 받아 끊기는지
    its_probe.py concurrent --in FILE --k 4 --seconds 30     여러 대를 한꺼번에
"""
import argparse
import collections
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = 'https://openapi.its.go.kr:9443/cctvInfo'
UA = 'Mozilla/5.0 (compatible; cctv-probe)'
BOX_KR = (124.0, 132.0, 33.0, 39.0)   # minX maxX minY maxY - 전국

_KEY = ''


def api_key():
    k = os.environ.get('ITS_API_KEY', '')
    if not k and sys.platform == 'win32':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as h:
                k = winreg.QueryValueEx(h, 'ITS_API_KEY')[0]
        except OSError:
            k = ''
    k = (k or '').strip()
    if not k:
        sys.exit('ITS_API_KEY 가 없습니다')
    return k


def scrub(text):
    text = str(text)
    if _KEY:
        for s in {_KEY, urllib.parse.quote(_KEY, safe='')}:
            text = text.replace(s, '***')
    return text


def _hook(tp, val, tb):
    import traceback
    sys.stderr.write(scrub(''.join(traceback.format_exception(tp, val, tb))))


def url_shape(u):
    p = urllib.parse.urlsplit(u)
    segs = [s for s in p.path.split('/') if s]
    return '%s://%s/%s%s' % (p.scheme, p.netloc,
                             '/'.join(('<%d>' % len(s)) if len(s) > 12 else s for s in segs),
                             '?<q>' if p.query else '')


def url_kind(u):
    p = urllib.parse.urlsplit(u)
    segs = [s for s in p.path.split('/') if s]
    return '%s://%s segs=%d%s' % (p.scheme, p.netloc, len(segs), ' +query' if p.query else '')


def load_items(path):
    with open(path, 'rb') as f:
        obj = json.loads(f.read().decode('utf-8', 'replace'))
    resp = obj.get('response', obj) if isinstance(obj, dict) else obj
    data = resp.get('data') if isinstance(resp, dict) else None
    if isinstance(data, dict):
        data = [data]
    return obj, (data or [])


def pick(items, n, seed):
    items = [it for it in items if str(it.get('cctvurl') or '').strip()]
    rnd = random.Random(seed)
    if len(items) <= n:
        return list(items)
    return rnd.sample(items, n)


# ---------------------------------------------------------------- list
def cmd_list(a):
    global _KEY
    _KEY = api_key()
    box = a.box or BOX_KR
    q = urllib.parse.urlencode({'apiKey': _KEY, 'type': a.type, 'cctvType': a.cctv_type,
                                'minX': box[0], 'maxX': box[1], 'minY': box[2], 'maxY': box[3],
                                'getType': 'json'})
    req = urllib.request.Request(API + '?' + q, headers={'User-Agent': UA, 'Accept': 'application/json'})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            status, headers, body = r.status, list(r.headers.items()), r.read()
    except urllib.error.HTTPError as e:
        status, headers, body = e.code, list(e.headers.items()), e.read()
    except Exception as e:
        sys.exit('API 호출 실패: %s: %s' % (type(e).__name__, scrub(e)))
    dt = time.time() - t0
    print('HTTP %s  %.2fs  %d bytes  type=%s cctvType=%s box=%s' % (status, dt, len(body), a.type, a.cctv_type, box))
    print('headers:', ', '.join('%s=%s' % (k, scrub(v)[:60]) for k, v in headers
                                if k.lower() not in ('set-cookie',)))
    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, 'list_%s_%s.json' % (a.type, a.cctv_type))
    with open(dst, 'wb') as f:
        f.write(body)
    print('saved:', dst)
    try:
        obj = json.loads(body.decode('utf-8', 'replace'))
    except ValueError:
        print('JSON 아님. 머리:', scrub(body[:400].decode('utf-8', 'replace')))
        return
    summarize(obj)


def summarize(obj):
    if isinstance(obj, dict):
        print('top keys:', list(obj.keys()))
    resp = obj.get('response', obj) if isinstance(obj, dict) else obj
    if isinstance(resp, dict):
        print('response keys:', {k: (v if not isinstance(v, (list, dict)) else type(v).__name__)
                                 for k, v in resp.items()})
    data = resp.get('data') if isinstance(resp, dict) else None
    if isinstance(data, dict):
        data = [data]
    if not data:
        print('data 없음. 본문 머리:', scrub(json.dumps(obj, ensure_ascii=False)[:500]))
        return
    n = len(data)
    keys = collections.Counter(k for it in data for k in it.keys())
    print('items:', n, ' item keys:', dict(keys))
    filled = {k: sum(1 for it in data if str(it.get(k, '') if it.get(k) is not None else '').strip())
              for k in keys}
    print('filled:', filled)
    for k in ('cctvresolution', 'cctvformat', 'cctvtype', 'coordtype', 'filecreatetime', 'roadsectionid'):
        c = collections.Counter(str(it.get(k)) for it in data)
        if len(c) <= 12:
            print('  %s: %s' % (k, dict(c)))
        else:
            print('  %s: %d distinct, e.g. %s' % (k, len(c), [x for x, _ in c.most_common(4)]))
    kinds = collections.Counter(url_kind(str(it.get('cctvurl') or '')) for it in data)
    print('url kinds:', dict(kinds))
    print('url shape e.g.:', [url_shape(str(it.get('cctvurl') or '')) for it in data[:3]])
    if any('cctvurl2' in it for it in data):
        k2 = collections.Counter(url_kind(str(it.get('cctvurl2') or '')).split(' segs=')[0] for it in data)
        print('cctvurl2 kinds:', dict(k2))
        print('cctvurl2 shape e.g.:', [url_shape(str(it.get('cctvurl2') or '')) for it in data[:3]])
        same_first = sum(1 for it in data if
                         ([s for s in urllib.parse.urlsplit(str(it.get('cctvurl') or '')).path.split('/') if s] or [''])[0] ==
                         ([s for s in urllib.parse.urlsplit(str(it.get('cctvurl2') or '')).path.split('/') if s] or [''])[0])
        print('cctvurl2 first seg == cctvurl first seg: %d / %d;  identical urls: %d' % (
            same_first, len(data), sum(1 for it in data if it.get('cctvurl') == it.get('cctvurl2'))))
    ids = []
    for it in data:
        s = ([x for x in urllib.parse.urlsplit(str(it.get('cctvurl') or '')).path.split('/') if x] or [''])[0]
        if s.isdigit():
            ids.append(int(s))
    if ids:
        print('numeric first seg: %d of %d, range %d..%d' % (len(ids), len(data), min(ids), max(ids)))
    urls = [str(it.get('cctvurl') or '') for it in data]
    firsts = [([s for s in urllib.parse.urlsplit(u).path.split('/') if s] or [''])[0] for u in urls]
    print('distinct urls: %d  distinct first path seg: %d  distinct names: %d  distinct coords: %d' % (
        len(set(urls)), len(set(firsts)), len(set(str(it.get('cctvname')) for it in data)),
        len(set((str(it.get('coordx')), str(it.get('coordy'))) for it in data))))
    xs, ys = [], []
    for it in data:
        try:
            xs.append(float(it.get('coordx')))
            ys.append(float(it.get('coordy')))
        except (TypeError, ValueError):
            pass
    if xs:
        print('coord range: x %.4f..%.4f  y %.4f..%.4f  (%d parsed)' % (min(xs), max(xs), min(ys), max(ys), len(xs)))
    pre = collections.Counter((str(it.get('cctvname') or '').split(']')[0] + ']')
                              if str(it.get('cctvname') or '').startswith('[') else '(no bracket)' for it in data)
    print('name prefixes: %d distinct, top: %s' % (len(pre), pre.most_common(6)))
    odd = [str(it.get('cctvname')) for it in data if ';' in str(it.get('cctvname') or '')][:3]
    if odd:
        print('names with ";":', odd)


# ---------------------------------------------------------------- stills
class Pace:
    """초당 rps 번을 넘지 않게 간격을 둔다(0 이면 그대로)."""

    def __init__(self, rps):
        self.gap = 1.0 / rps if rps else 0.0
        self.lock = threading.Lock()
        self.next = 0.0

    def wait(self):
        if not self.gap:
            return
        with self.lock:
            now = time.time()
            t = max(now, self.next)
            self.next = t + self.gap
        if t > now:
            time.sleep(t - now)


def fetch_size(url, cap=600000, timeout=20, rng=0, pace=None):
    """그림 머리만 읽어 크기를 안다. rng > 0 이면 앞 rng 바이트만 달라고 하고, 모자라면 통째로 다시."""
    from PIL import ImageFile
    out = {'ok': False}
    if pace:
        pace.wait()
    t0 = time.time()
    try:
        hdr = {'User-Agent': UA}
        if rng:
            hdr['Range'] = 'bytes=0-%d' % (rng - 1)
        req = urllib.request.Request(url, headers=hdr)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out['status'] = r.status
            out['ctype'] = (r.headers.get('Content-Type') or '').split(';')[0]
            out['clen'] = r.headers.get('Content-Length')
            out['crange'] = r.headers.get('Content-Range')
            out['final'] = url_kind(r.geturl())
            p = ImageFile.Parser()
            n, head = 0, b''
            while n < cap:
                chunk = r.read(8192)
                if not chunk:
                    break
                if len(head) < 8:
                    head += chunk[:8 - len(head)]
                n += len(chunk)
                try:
                    p.feed(chunk)
                except Exception as e:      # 그림이 아닌 본문
                    out['err'] = 'parse ' + type(e).__name__
                    break
                if p.image is not None:
                    out['size'] = '%dx%d' % p.image.size
                    out['fmt'] = p.image.format
                    out['ok'] = True
                    break
            out['bytes'] = n
            out['magic'] = head[:4].hex()
    except urllib.error.HTTPError as e:
        out['err'] = 'HTTP %s' % e.code
    except Exception as e:
        out['err'] = type(e).__name__ + ': ' + str(e)[:60]
    out['sec'] = round(time.time() - t0, 2)
    if rng and not out['ok'] and not str(out.get('err', '')).startswith('HTTP'):
        full = fetch_size(url, cap=cap, timeout=timeout, rng=0, pace=pace)
        full['refetched_full'] = True
        return full
    return out


def first_seg(u):
    return ([s for s in urllib.parse.urlsplit(str(u or '')).path.split('/') if s] or [''])[0]


def cmd_stills(a):
    _, items = load_items(a.inp)
    chosen = pick(items, a.n if a.n > 0 else len(items) + 1, a.seed)
    pace = Pace(a.rps)
    print('items %d, sampled %d, workers %d, rps %s, range %s' % (len(items), len(chosen), a.workers, a.rps, a.range))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(lambda it: fetch_size(str(it['cctvurl']), rng=a.range, pace=pace), chosen))
    print('elapsed %.1fs' % (time.time() - t0))
    ok = [r for r in res if r['ok']]
    print('ok %d / %d' % (len(ok), len(res)))
    print('status:', dict(collections.Counter(r.get('status') for r in res)),
          ' refetched full:', sum(1 for r in res if r.get('refetched_full')),
          ' bytes read total: %d' % sum(r.get('bytes') or 0 for r in res))
    print('sizes:', dict(collections.Counter(r['size'] for r in ok).most_common()))
    print('formats:', dict(collections.Counter(r.get('fmt') for r in ok)))
    print('ctype:', dict(collections.Counter(r.get('ctype') for r in res)))
    print('final url kinds:', dict(collections.Counter(r.get('final') for r in res)))
    print('errors:', dict(collections.Counter(r.get('err') for r in res if not r['ok'])))
    print('magic of failures:', dict(collections.Counter(r.get('magic') for r in res if not r['ok'])))
    cl = sorted(int(r['clen']) for r in ok if r.get('clen') and str(r['clen']).isdigit())
    if cl:
        print('content-length: n=%d min=%d median=%d max=%d' % (len(cl), cl[0], cl[len(cl) // 2], cl[-1]))
    by = collections.defaultdict(list)
    for r in ok:
        if r.get('clen') and str(r['clen']).isdigit():
            by[r['size']].append(int(r['clen']))
    for k, v in by.items():
        v.sort()
        print('  %s: n=%d median bytes=%d' % (k, len(v), v[len(v) // 2]))
    secs = sorted(r['sec'] for r in res)
    print('fetch sec: median %.2f max %.2f' % (secs[len(secs) // 2], secs[-1]))
    if a.out:
        rows = [{'id': first_seg(it.get('cctvurl')), 'name': it.get('cctvname'), 'x': it.get('coordx'),
                 'y': it.get('coordy'), 'size': r.get('size'), 'err': r.get('err'), 'clen': r.get('clen'),
                 'crange': r.get('crange')}
                for it, r in zip(chosen, res)]
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(rows, f, ensure_ascii=False, indent=0)
        print('saved:', a.out)


# ---------------------------------------------------------------- hls
def get(url, timeout=20, cap=24000000):
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.geturl(), r.status, (r.headers.get('Content-Type') or ''), r.read(cap)


def m3u8_lines(text):
    return [l.strip() for l in text.splitlines() if l.strip()]


def show_tags(lines, limit=14):
    import re
    tags = [re.sub(r'URI="[^"]*"', 'URI="..."', l)[:150] for l in lines if l.startswith('#')]
    seen, out = set(), []
    for t in tags:
        k = t.split(':')[0]
        if k == '#EXTINF' and k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out[:limit]


def probe_hls(url):
    out = {}
    try:
        final, status, ctype, body = get(url)
    except Exception as e:
        return {'err': 'playlist ' + type(e).__name__ + ': ' + str(e)[:80]}
    text = body.decode('utf-8', 'replace')
    lines = m3u8_lines(text)
    out['playlist'] = {'status': status, 'ctype': ctype.split(';')[0], 'final': url_kind(final),
                       'redirected': final != url, 'bytes': len(body), 'is_m3u8': text.lstrip().startswith('#EXTM3U')}
    if not out['playlist']['is_m3u8']:
        out['head'] = text[:160]
        return out
    out['tags'] = show_tags(lines)
    variants = []
    for i, l in enumerate(lines):
        if l.startswith('#EXT-X-STREAM-INF'):
            uri = next((x for x in lines[i + 1:] if not x.startswith('#')), None)
            variants.append((l.split(':', 1)[1][:120], uri))
    media_url, media_lines = final, lines
    if variants:
        out['variants'] = [v[0] for v in variants]
        try:
            media_url, _, _, b2 = get(urllib.parse.urljoin(final, variants[0][1]))
            media_lines = m3u8_lines(b2.decode('utf-8', 'replace'))
            out['media_tags'] = show_tags(media_lines)
        except Exception as e:
            out['err'] = 'media playlist ' + type(e).__name__ + ': ' + str(e)[:80]
            return out
    segs, dur = [], None
    for l in media_lines:
        if l.startswith('#EXTINF'):
            try:
                dur = float(l.split(':', 1)[1].split(',')[0])
            except ValueError:
                dur = None
        elif not l.startswith('#'):
            segs.append((dur, l))
            dur = None
    out['segments'] = len(segs)
    durs = [d for d, _ in segs if d]
    if durs:
        out['seg_dur'] = {'min': min(durs), 'max': max(durs), 'sum': round(sum(durs), 1)}
    out['endlist'] = any(l.startswith('#EXT-X-ENDLIST') for l in media_lines)
    out['seg_ext'] = dict(collections.Counter(
        (urllib.parse.urlsplit(u).path.rsplit('.', 1)[-1][:6] if '.' in urllib.parse.urlsplit(u).path.rsplit('/', 1)[-1]
         else 'noext') for _, u in segs))
    tail = [s for s in segs if s[0]][-2:]
    tot_b, tot_d = 0, 0.0
    for d, u in tail:
        try:
            _, _, _, b = get(urllib.parse.urljoin(media_url, u))
            tot_b += len(b)
            tot_d += d
        except Exception as e:
            out['seg_err'] = type(e).__name__ + ': ' + str(e)[:60]
    if tot_d:
        out['bitrate_kbps'] = round(tot_b * 8 / tot_d / 1000)
        out['seg_bytes'] = tot_b
    return out


def cv_open(url, open_ms=20000, read_ms=20000):
    import cv2
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG,
                            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, open_ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, read_ms])


def cv_probe(url, frames=40):
    import cv2
    t0 = time.time()
    cap = cv_open(url)
    out = {'opened': bool(cap.isOpened()), 'open_sec': round(time.time() - t0, 1)}
    if not out['opened']:
        return out
    out['w'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out['h'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out['fps_prop'] = round(cap.get(cv2.CAP_PROP_FPS), 2)
    out['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out['sar'] = '%d:%d' % (int(cap.get(cv2.CAP_PROP_SAR_NUM)), int(cap.get(cv2.CAP_PROP_SAR_DEN)))
    fc = int(cap.get(cv2.CAP_PROP_FOURCC))
    out['fourcc'] = ''.join(chr((fc >> (8 * i)) & 0xFF) for i in range(4)) if fc > 0 else str(fc)
    n, shape = 0, None
    t1 = time.time()
    while n < frames:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        shape = fr.shape
    out['read'] = n
    out['read_sec'] = round(time.time() - t1, 1)
    out['shape'] = shape
    cap.release()
    return out


def by_ids(items, ids):
    """주소 첫 칸(카메라 번호)으로 고른다 - 정지 영상으로 크기를 안 카메라의 영상을 볼 때."""
    want = [s.strip() for s in ids.split(',') if s.strip()]
    tab = {first_seg(it.get('cctvurl')): it for it in items}
    return [tab[s] for s in want if s in tab]


def size_ids(path, size, n, seed):
    """stills --out 결과에서 그 크기인 카메라 번호 n개."""
    with open(path, encoding='utf-8') as f:
        rows = [r['id'] for r in json.load(f) if r.get('size') == size]
    random.Random(seed).shuffle(rows)
    return ','.join(rows[:n])


def cmd_hls(a):
    _, items = load_items(a.inp)
    if a.size_from:
        a.ids = size_ids(a.size_from, a.size, a.n, a.seed)
    chosen = by_ids(items, a.ids) if a.ids else pick(items, a.n, a.seed)
    for i, it in enumerate(chosen):
        u = str(it['cctvurl'])
        print('--- [%d] %s' % (i, url_kind(u)))
        print(' hls:', json.dumps(probe_hls(u), ensure_ascii=False))
        if not a.no_cv:
            print(' cv :', json.dumps(cv_probe(u), ensure_ascii=False))


# ---------------------------------------------------------------- mp4
def cmd_mp4(a):
    _, items = load_items(a.inp)
    chosen = pick(items, a.n, a.seed)
    for i, it in enumerate(chosen):
        u = str(it['cctvurl'])
        print('--- [%d] %s  filecreatetime=%s' % (i, url_kind(u), it.get('filecreatetime')))
        info = {}
        try:
            req = urllib.request.Request(u, headers={'User-Agent': UA, 'Range': 'bytes=0-1023'})
            with urllib.request.urlopen(req, timeout=20) as r:
                head = r.read(1024)
                info = {'status': r.status, 'ctype': (r.headers.get('Content-Type') or '').split(';')[0],
                        'clen': r.headers.get('Content-Length'), 'crange': r.headers.get('Content-Range'),
                        'accept_ranges': r.headers.get('Accept-Ranges'), 'final': url_kind(r.geturl()),
                        'magic': head[:12].hex(), 'ftyp': head[4:12].decode('latin-1', 'replace')}
        except urllib.error.HTTPError as e:
            info = {'err': 'HTTP %s' % e.code}
        except Exception as e:
            info = {'err': type(e).__name__ + ': ' + str(e)[:80]}
        print(' http:', json.dumps(info, ensure_ascii=False))
        if not a.no_cv:
            r = cv_probe(u, frames=20)
            if r.get('opened') and r.get('fps_prop') and r.get('frame_count', 0) > 0:
                r['duration_sec'] = round(r['frame_count'] / r['fps_prop'], 1)
            print(' cv  :', json.dumps(r, ensure_ascii=False))


# ---------------------------------------------------------------- still freshness
def cmd_fresh(a):
    """같은 정지 영상 주소를 몇 번 받아 그림이 얼마나 자주 바뀌는지 본다(메모리에서 해시만)."""
    import hashlib
    _, items = load_items(a.inp)
    chosen = pick(items, a.n, a.seed)
    rows = [[] for _ in chosen]
    for rnd in range(a.times):
        if rnd:
            time.sleep(a.gap)
        for i, it in enumerate(chosen):
            try:
                req = urllib.request.Request(str(it['cctvurl']), headers={'User-Agent': UA})
                with urllib.request.urlopen(req, timeout=20) as r:
                    b = r.read(4000000)
                    rows[i].append({'t': round(time.time() % 1000, 1), 'sha': hashlib.sha1(b).hexdigest()[:8],
                                    'bytes': len(b), 'lm': r.headers.get('Last-Modified'),
                                    'etag': (r.headers.get('ETag') or '')[:16], 'age': r.headers.get('Age'),
                                    'cc': r.headers.get('Cache-Control'), 'date': r.headers.get('Date')})
            except Exception as e:
                rows[i].append({'err': type(e).__name__ + ': ' + str(e)[:50]})
    for i, rr in enumerate(rows):
        shas = [x.get('sha') for x in rr]
        print('[%d] distinct images %d / %d fetches (gap %ss): %s' % (i, len(set(shas)), len(shas), a.gap, shas))
        print('    headers first:', {k: rr[0].get(k) for k in ('lm', 'etag', 'age', 'cc', 'date', 'bytes')})
        print('    headers last :', {k: rr[-1].get(k) for k in ('lm', 'etag', 'age', 'cc', 'date', 'bytes')})


# ---------------------------------------------------------------- events
EVENT_API = 'https://openapi.its.go.kr:9443/eventInfo'


def find_list(o):
    if isinstance(o, list) and o and all(isinstance(x, dict) for x in o):
        return o
    if isinstance(o, dict):
        for v in o.values():
            r = find_list(v)
            if r is not None:
                return r
    return None


def cmd_events(a):
    global _KEY
    _KEY = api_key()
    box = a.box or BOX_KR
    q = urllib.parse.urlencode({'apiKey': _KEY, 'type': 'all', 'eventType': 'all',
                                'minX': box[0], 'maxX': box[1], 'minY': box[2], 'maxY': box[3],
                                'getType': 'json'})
    req = urllib.request.Request(EVENT_API + '?' + q, headers={'User-Agent': UA, 'Accept': 'application/json'})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            status, body = r.status, r.read()
    except urllib.error.HTTPError as e:
        status, body = e.code, e.read()
    except Exception as e:
        sys.exit('API 호출 실패: %s: %s' % (type(e).__name__, scrub(e)))
    print('HTTP %s  %.2fs  %d bytes' % (status, time.time() - t0, len(body)))
    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, 'events_all.json')
    with open(dst, 'wb') as f:
        f.write(body)
    print('saved:', dst)
    try:
        obj = json.loads(body.decode('utf-8', 'replace'))
    except ValueError:
        print('JSON 아님. 머리:', scrub(body[:400].decode('utf-8', 'replace')))
        return
    summarize_events(obj)


def summarize_events(obj):
    def walk(o, path=''):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, (dict, list)):
                    walk(v, path + '/' + k)
                else:
                    print('  %s/%s = %s' % (path, k, scrub(str(v))[:80]))
        elif isinstance(o, list):
            print('  %s = list(%d)' % (path, len(o)))
    walk({k: v for k, v in obj.items()} if isinstance(obj, dict) else obj)
    items = find_list(obj) or []
    print('events:', len(items))
    if not items:
        print('본문 머리:', scrub(json.dumps(obj, ensure_ascii=False)[:400]))
        return
    print('keys:', sorted(set(k for it in items for k in it)))
    for k in ('type', 'eventType', 'eventDetailType', 'lanesBlockType', 'roadDrcType'):
        print('  %s: %s' % (k, collections.Counter(str(it.get(k)) for it in items).most_common(12)))
    print('  cross type x eventType:', collections.Counter((str(it.get('type')), str(it.get('eventType'))) for it in items).most_common(14))
    for k in ('coordX', 'coordY', 'startDate', 'endDate', 'lanesBlocked', 'message', 'linkId', 'roadName'):
        print('  filled %s: %d' % (k, sum(1 for it in items if str(it.get(k) or '').strip())))
    sd = sorted(str(it.get('startDate') or '') for it in items if str(it.get('startDate') or '').strip())
    if sd:
        print('  startDate range: %s .. %s ; started today: %d' % (sd[0], sd[-1], sum(1 for s in sd if s.startswith(time.strftime('%Y%m%d')))))
    for et in ('교통사고', '공사', '기타돌발'):
        ms = [str(it.get('message') or '')[:60] for it in items if str(it.get('eventType')) == et][:4]
        print('  e.g. %s: %s' % (et, ms))


def cmd_near(a):
    """돌발 지점마다 가장 가까운 CCTV 까지의 거리(저장해 둔 목록끼리 - API 를 부르지 않는다)."""
    import math
    with open(a.events, 'rb') as f:
        events = find_list(json.loads(f.read().decode('utf-8', 'replace'))) or []
    cams = []
    for p in a.lists:
        for it in load_items(p)[1]:
            try:
                cams.append((float(it['coordx']), float(it['coordy'])))
            except (KeyError, TypeError, ValueError):
                pass
    print('events %d, cameras %d' % (len(events), len(cams)))
    grid = collections.defaultdict(list)
    for x, y in cams:
        grid[(int(x * 50), int(y * 50))].append((x, y))      # 0.02도 칸(약 2km)

    def nearest(x, y):
        best, cnt500, cnt1k = 1e9, 0, 0
        gx, gy = int(x * 50), int(y * 50)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for cx, cy in grid.get((gx + dx, gy + dy), ()):
                    d = math.hypot((cx - x) * 111320 * math.cos(math.radians(y)), (cy - y) * 110540)
                    best = min(best, d)
                    cnt500 += d <= 500
                    cnt1k += d <= 1000
        return best, cnt500, cnt1k

    rows = collections.defaultdict(list)
    for ev in events:
        try:
            x, y = float(ev.get('coordX')), float(ev.get('coordY'))
        except (TypeError, ValueError):
            continue
        d, c5, c1 = nearest(x, y)
        rows[(str(ev.get('type')), str(ev.get('eventType')))].append((d, c5, c1))
    print('%-22s %5s %8s %8s %8s %8s' % ('도로 x 유형', 'n', '<=300m', '<=500m', '<=1km', '중앙값m'))
    for k in sorted(rows, key=lambda k: -len(rows[k])):
        v = rows[k]
        ds = sorted(d for d, _, _ in v)
        med = ds[len(ds) // 2]
        print('%-22s %5d %8d %8d %8d %8s' % ('/'.join(k), len(v), sum(d <= 300 for d in ds), sum(d <= 500 for d in ds),
                                             sum(d <= 1000 for d in ds), ('%d' % med) if med < 1e8 else '>2km'))


# ---------------------------------------------------------------- soak / concurrent
def read_for(url, seconds, label, results):
    t0 = time.time()
    cap = cv_open(url)
    r = {'label': label, 'opened': bool(cap.isOpened()), 'frames': 0, 'fails': []}
    if not r['opened']:
        r['open_sec'] = round(time.time() - t0, 1)
        results.append(r)
        return
    last = time.time()
    gaps = []
    while time.time() - t0 < seconds:
        ok, _ = cap.read()
        now = time.time()
        if ok:
            r['frames'] += 1
            if now - last > 3:
                gaps.append((round(last - t0, 1), round(now - last, 1)))
            last = now
        else:
            r['fails'].append(round(now - t0, 1))
            cap.release()
            if len(r['fails']) >= 6:
                break
            time.sleep(1.0)
            cap = cv_open(url)
            if not cap.isOpened():
                r['reopen_failed_at'] = round(time.time() - t0, 1)
                break
            last = time.time()
    r['elapsed'] = round(time.time() - t0, 1)
    r['stalls_over_3s'] = gaps[:8]
    cap.release()
    results.append(r)


def cmd_soak(a):
    _, items = load_items(a.inp)
    it = pick(items, 1, a.seed)[0]
    res = []
    read_for(str(it['cctvurl']), a.seconds, 'soak', res)
    print(json.dumps(res[0], ensure_ascii=False))


def cmd_concurrent(a):
    _, items = load_items(a.inp)
    chosen = pick(items, a.k, a.seed)
    res, ths = [], []
    for i, it in enumerate(chosen):
        th = threading.Thread(target=read_for, args=(str(it['cctvurl']), a.seconds, 's%d' % i, res))
        th.start()
        ths.append(th)
    for th in ths:
        th.join()
    for r in sorted(res, key=lambda r: r['label']):
        print(json.dumps(r, ensure_ascii=False))


def main():
    sys.excepthook = _hook
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('list')
    p.add_argument('--type', required=True, choices=['ex', 'its'])
    p.add_argument('--cctv-type', required=True, choices=['1', '2', '3', '4', '5'])
    p.add_argument('--box', nargs=4, type=float)
    p.add_argument('--out', required=True)
    p.set_defaults(fn=cmd_list)
    p = sub.add_parser('summary')
    p.add_argument('--in', dest='inp', required=True)
    p.set_defaults(fn=lambda a: summarize(load_items(a.inp)[0]))
    p = sub.add_parser('stills')
    p.add_argument('--in', dest='inp', required=True)
    p.add_argument('--n', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--rps', type=float, default=0, help='초당 요청 수 한도(0 = 한도 없음)')
    p.add_argument('--range', type=int, default=0, help='앞 N 바이트만 받기(0 = 통째 요청)')
    p.add_argument('--out')
    p.set_defaults(fn=cmd_stills)
    for name, fn in (('hls', cmd_hls), ('mp4', cmd_mp4)):
        p = sub.add_parser(name)
        p.add_argument('--in', dest='inp', required=True)
        p.add_argument('--n', type=int, default=3)
        p.add_argument('--seed', type=int, default=0)
        p.add_argument('--no-cv', action='store_true')
        if name == 'hls':
            p.add_argument('--ids', help='카메라 번호(주소 첫 칸)를 쉼표로')
            p.add_argument('--size-from', help='stills --out 결과 - 여기서 --size 인 카메라를 고른다')
            p.add_argument('--size', default='1920x1080')
        p.set_defaults(fn=fn)
    p = sub.add_parser('fresh')
    p.add_argument('--in', dest='inp', required=True)
    p.add_argument('--n', type=int, default=3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--times', type=int, default=4)
    p.add_argument('--gap', type=float, default=10)
    p.set_defaults(fn=cmd_fresh)
    p = sub.add_parser('events')
    p.add_argument('--box', nargs=4, type=float)
    p.add_argument('--out', required=True)
    p.set_defaults(fn=cmd_events)
    p = sub.add_parser('events-summary')
    p.add_argument('--in', dest='inp', required=True)
    p.set_defaults(fn=lambda a: summarize_events(json.loads(open(a.inp, 'rb').read().decode('utf-8', 'replace'))))
    p = sub.add_parser('near')
    p.add_argument('--events', required=True)
    p.add_argument('--lists', nargs='+', required=True)
    p.set_defaults(fn=cmd_near)
    p = sub.add_parser('soak')
    p.add_argument('--in', dest='inp', required=True)
    p.add_argument('--seconds', type=int, default=300)
    p.add_argument('--seed', type=int, default=0)
    p.set_defaults(fn=cmd_soak)
    p = sub.add_parser('concurrent')
    p.add_argument('--in', dest='inp', required=True)
    p.add_argument('--k', type=int, default=4)
    p.add_argument('--seconds', type=int, default=30)
    p.add_argument('--seed', type=int, default=0)
    p.set_defaults(fn=cmd_concurrent)
    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
