import sys, zipfile, collections, os

for p in sys.argv[1:]:
    print("#", p, os.path.getsize(p))
    try:
        z = zipfile.ZipFile(p)
    except Exception as e:
        print("  OPEN FAILED:", type(e).__name__, e)
        continue
    infos = z.infolist()
    files = [i for i in infos if not i.is_dir()]
    print("  entries", len(infos), "files", len(files), "sum_uncompressed_GB", round(sum(i.file_size for i in files) / 1e9, 2))
    dirs = collections.Counter()
    exts = collections.Counter()
    for i in files:
        name = i.filename
        try:
            name = name.encode("cp437").decode("cp949")
        except Exception:
            pass
        dirs[os.path.dirname(name)] += 1
        exts[os.path.splitext(name)[1].lower()] += 1
    print("  exts", dict(exts))
    for d, n in sorted(dirs.items()):
        print("   %7d  %s" % (n, d))
    names = [i.filename for i in files[:3]]
    print("  first", names)
