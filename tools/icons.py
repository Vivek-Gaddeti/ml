"""Draws the app icons (assets/icon-192.png, assets/icon-512.png): a white "AI" on the site's pink, with a black frame line.
No libraries needed.  Usage: python tools/icons.py"""
import struct, zlib, os

def png(path, n):
    BG, W, K = (214, 36, 95), (255, 255, 255), (17, 17, 17)
    u = n / 16.0                                   # the icon is designed on a 16 x 16 grid
    def inA(gx, gy):                               # letter A: two slanted legs and a bar
        if not (2.2 <= gy <= 12.8): return False
        t = (gy - 2.2) / 10.6                      # 0 at the top, 1 at the bottom
        cx = 5.4
        half = 0.5 + t * 2.7
        leg = 1.05
        if abs(gx - (cx - half)) < leg / 2 + 0.3 or abs(gx - (cx + half)) < leg / 2 + 0.3: return True
        return 8.6 <= gy <= 9.9 and abs(gx - cx) <= half
    def inI(gx, gy):                               # letter I: bar with serifs
        if 2.2 <= gy < 3.6 and 10.2 <= gx < 14: return True
        if 11.4 <= gy < 12.8 and 10.2 <= gx < 14: return True
        return 11.3 <= gx < 12.9 and 2.2 <= gy < 12.8
    def colour(x, y):
        gx, gy = x / u, y / u
        if inA(gx, gy) or inI(gx, gy): return W
        if min(gx, gy, 16 - gx, 16 - gy) < 0.35: return K
        return BG
    raw = b"".join(b"\x00" + bytes(c for x in range(n) for c in colour(x + .5, y + .5)) for y in range(n))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", n, n, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))

here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")
for size in (192, 512):
    png(os.path.join(here, "icon-%d.png" % size), size)
print("icons written")
