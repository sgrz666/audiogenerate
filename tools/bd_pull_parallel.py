"""百度盘单文件下载：小块 + 短超时 + 并发 + 断点续传 + MD5/CRC 双校验后原子落位。

实测特征（2026-11-11 node01）：
  - 每连接有初始突发额度，64KB 约 0.3s 完成；1MB 会被整形到 20KB/s 需 52s。
  - 约 2/3 请求返回 206 头后响应体被掐断，恒定卡 30.3s 得 0 字节。
  => 用 64KB 小块吃突发额度，6s 截止时间提前放弃卡顿请求，并发提高命中数。
只写自己的 staging 目录，不碰其他进程的 .part。
"""
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile

FILES_JSON = "/tmp/bd_files.json"
JAR = "/tmp/bdjar.txt"
UA = "LogStatistic"
# 实测：netdisk;P2SP;* 系 UA 会被整形（206 后响应体掐断，恒卡 30.3s 得 0 字节），
# 大区间还会 403 error_code=31326 hitcode:104。LogStatistic 稳定跑满 ~81KB/s，
# 256KB 小块可达 324KB/s，8MB 区间也正常。
CHUNK = 64 * 1024
WORKERS = int(os.environ.get("BD_WORKERS", "4"))
REQ_DEADLINE = float(os.environ.get("BD_DEADLINE", "6"))
MAX_RETRY = 200

DEST = "/data/shangguanrz/projectsss/speakgenerate/data"
STAGE = "/data/shangguanrz/projectsss/speakgenerate/.tmp/staging07"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def fresh_dlink(pan_dlink):
    """第一跳：客户端 UA + cookie + referer，拿 302 的新直链。"""
    try:
        p = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-D", "-", "--max-time", "20",
             "-A", UA, "-b", JAR, "-b", "PANWEB=1",
             "-e", "https://pan.baidu.com/disk/home", pan_dlink],
            capture_output=True, text=True, timeout=40,
        )
    except Exception:
        return None
    for line in p.stdout.splitlines():
        if line.lower().startswith("location:"):
            return line.split(":", 1)[1].strip()
    return None


def fetch_range(url, start, end, deadline):
    """第二跳：客户端 UA + Range，不带 cookie。按截止时间增量读，保留已到手字节。

    返回 (status, data, cr_start)。cr_start 用于核对服务端实际回的区间。
    """
    want = end - start + 1
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Range": f"bytes={start}-{end}",
    })
    try:
        r = urllib.request.urlopen(req, timeout=REQ_DEADLINE)
    except urllib.error.HTTPError as e:
        return e.code, b"", None
    except Exception:
        return 0, b"", None
    with r:
        if r.status != 206:
            return r.status, b"", None
        cr_start = None
        cr = r.headers.get("Content-Range", "")
        if cr.startswith("bytes "):
            try:
                cr_start = int(cr[6:].split("-")[0])
            except Exception:
                cr_start = None
        buf = bytearray()
        while len(buf) < want and time.time() < deadline:
            try:
                piece = r.read(min(65536, want - len(buf)))
            except (socket.timeout, OSError):
                break
            except Exception:
                break
            if not piece:
                break
            buf.extend(piece)
        return 206, bytes(buf), cr_start


class Job:
    def __init__(self, name, pan_dlink, total, seed=None):
        self.name = name
        self.pan_dlink = pan_dlink
        self.total = total
        self.stage = os.path.join(STAGE, name + ".staging")
        self.manifest = self.stage + ".done.json"
        self.nchunk = (total + CHUNK - 1) // CHUNK
        self.lock = threading.Lock()
        self.done = bytearray(self.nchunk)
        self.bytes_done = 0
        self.stop = False
        self.t0 = time.time()
        self.stat = {"ok": 0, "stall": 0, "http": 0, "err": 0}

        os.makedirs(STAGE, exist_ok=True)
        loaded = False
        if os.path.exists(self.manifest) and os.path.exists(self.stage):
            try:
                s = json.load(open(self.manifest, encoding="utf-8"))
                if s.get("total") == total and s.get("chunk") == CHUNK \
                        and os.path.getsize(self.stage) == total:
                    for i in s["done"]:
                        self.done[i] = 1
                    loaded = True
                    log(f"[续传] 清单命中 {sum(self.done)}/{self.nchunk} 块")
            except Exception as e:
                log(f"[警告] 清单不可用 {type(e).__name__}，重新建暂存")
        if not loaded:
            with open(self.stage, "wb") as f:
                f.truncate(total)
            if seed and os.path.exists(seed):
                n = os.path.getsize(seed)
                n = min(n, total)
                full = n // CHUNK
                if full:
                    with open(seed, "rb") as src, open(self.stage, "r+b") as dst:
                        left = full * CHUNK
                        while left > 0:
                            b = src.read(min(1024 * 1024, left))
                            if not b:
                                break
                            dst.write(b)
                            left -= len(b)
                    for i in range(full):
                        self.done[i] = 1
                    log(f"[复用] 从旧 .part 接管 {full * CHUNK / 1024 ** 2:.1f} MB "
                        f"({full} 块)，整体 MD5 终检兜底")
        self.bytes_done = sum(self.chunk_len(i) for i in range(self.nchunk) if self.done[i])
        self.base = self.bytes_done
        self.fd = os.open(self.stage, os.O_RDWR)

    def chunk_len(self, i):
        s = i * CHUNK
        return min(s + CHUNK, self.total) - s

    def save(self):
        tmp = self.manifest + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"total": self.total, "chunk": CHUNK,
                       "done": [i for i in range(self.nchunk) if self.done[i]]}, f)
        os.replace(tmp, self.manifest)

    def worker(self, wid, queue):
        url = None
        while not self.stop:
            with self.lock:
                if not queue:
                    return
                i = queue.pop(0)
            pos = i * CHUNK
            end = pos + self.chunk_len(i) - 1
            retry = 0
            while pos <= end and not self.stop:
                if url is None:
                    url = fresh_dlink(self.pan_dlink)
                    if not url:
                        retry += 1
                        if retry > MAX_RETRY:
                            break
                        time.sleep(min(1.0 * retry, 10))
                        continue
                deadline = time.time() + REQ_DEADLINE
                status, data, cr = fetch_range(url, pos, end, deadline)
                if status == 206 and data and (cr is None or cr == pos):
                    os.pwrite(self.fd, data, pos)
                    pos += len(data)
                    retry = 0
                    with self.lock:
                        self.bytes_done += len(data)
                        self.stat["ok"] += 1
                        if pos > end:
                            self.done[i] = 1
                    continue
                retry += 1
                # 实测同一条 302 链可复用；只有 403/404 或连续失败过多才换新链，
                # 否则每次失败都跑一趟 curl，开销压过收益。
                if status in (401, 403, 404) or retry % 30 == 0:
                    url = None
                with self.lock:
                    if status == 206:
                        self.stat["stall"] += 1
                    elif status == 0:
                        self.stat["err"] += 1
                    else:
                        self.stat["http"] += 1
                        if self.stat["http"] <= 12:
                            log(f"[w{wid}] 块{i} http={status}")
                if retry > MAX_RETRY:
                    log(f"[w{wid} 放弃] 块{i} 重试耗尽 pos={pos}")
                    with self.lock:
                        self.stop = True
                    return
                if retry % 20 == 0:
                    time.sleep(min(0.5 * (retry // 20), 5))
            else:
                continue
            with self.lock:
                self.stop = True
            return

    def run(self):
        queue = [i for i in range(self.nchunk) if not self.done[i]]
        log(f"[开始] {self.name} {self.total / 1024 ** 2:.1f} MB, "
            f"{self.nchunk} 块 x {CHUNK // 1024} KB, {WORKERS} 并发, "
            f"截止 {REQ_DEADLINE}s, 待下 {len(queue)} 块")
        threads = [threading.Thread(target=self.worker, args=(w, queue), daemon=True)
                   for w in range(WORKERS)]
        for t in threads:
            t.start()
        last = time.time()
        while any(t.is_alive() for t in threads):
            time.sleep(5)
            now = time.time()
            if now - last >= 20:
                last = now
                with self.lock:
                    b = self.bytes_done
                    n = sum(self.done)
                    st = dict(self.stat)
                    self.save()
                sp = (b - self.base) / max(now - self.t0, 1e-6)
                eta = (self.total - b) / sp / 60 if sp > 0 else -1
                log(f"  {n}/{self.nchunk} 块 {b / 1024 ** 2:.1f}/{self.total / 1024 ** 2:.1f} MB "
                    f"({b / self.total * 100:.1f}%) {sp / 1024:.0f} KB/s ETA {eta:.1f}min "
                    f"ok={st['ok']} stall={st['stall']} err={st['err']} http={st['http']}")
        for t in threads:
            t.join()
        with self.lock:
            self.save()
        os.close(self.fd)
        return not self.stop and all(self.done)


def verify(path, total, expect_md5):
    if os.path.getsize(path) != total:
        log(f"[失败] 大小 {os.path.getsize(path)} != {total}")
        return False
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(4 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    got = h.hexdigest()
    log(f"[校验] MD5 {got} 期望 {expect_md5}")
    if expect_md5 and got != expect_md5:
        log("[失败] MD5 不符，保留暂存")
        return False
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
            entries = len(z.namelist())
    except Exception as e:
        log(f"[失败] zip 无法解析 {type(e).__name__}: {e}")
        return False
    if bad is not None:
        log(f"[失败] CRC 坏块 {bad}")
        return False
    log(f"[校验] ZIP_OK entries={entries}")
    return True


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "07_"
    expect_md5 = (sys.argv[2] if len(sys.argv) > 2 else "").strip().lower()
    files = json.load(open(FILES_JSON, encoding="utf-8"))
    hit = [x for x in files if x["name"].startswith(prefix)]
    if len(hit) != 1:
        log(f"[错误] 匹配 {len(hit)} 个，前缀 {prefix}")
        return 2
    m = hit[0]
    name, total, pan = m["name"], m["size"], m["dlink"]
    final = os.path.join(DEST, name)
    seed = final + ".part"
    if not (os.path.exists(final) and os.path.getsize(final) == total):
        job = Job(name, pan, total, seed=seed)
        if not job.run():
            log(f"[失败] 未完成 {sum(job.done)}/{job.nchunk} 块，暂存保留可续传")
            return 1
        if not verify(job.stage, total, expect_md5):
            return 1
        os.replace(job.stage, final)
        if os.path.exists(job.manifest):
            os.remove(job.manifest)
        log(f"[落位] {final}")
    ok = verify(final, total, expect_md5)
    log(f"[终检] {name} size={os.path.getsize(final)} "
        f"md5={'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())