"""Draws the app icons (assets/icon-192.png, assets/icon-512.png): a black "DC" and a coral dot on yellow (same palette as the Patterns icons).
No libraries needed.  Usage: python tools/icons.py"""
import struct, zlib, os

def png(path, n):
    Y, K, P = (255, 210, 63), (17, 17, 17), (255, 107, 107)
    u = n / 16.0                                   # the icon is designed on a 16 x 16 grid
    T = 1.7                                        # stroke thickness
    RO, CY = 3.6, 7.4                              # letter radius and vertical centre
    def inD(gx, gy):                               # D: stem + rounded bowl
        if 1.2 <= gx < 1.2 + T and CY - RO <= gy < CY + RO: return True
        if 1.2 <= gx < 4.2 and (CY - RO <= gy < CY - RO + T or CY + RO - T <= gy < CY + RO): return True
        if gx >= 4.2:
            d = ((gx - 4.2) ** 2 + (gy - CY) ** 2) ** .5
            return RO - T <= d <= RO
        return False
    def inC(gx, gy):                               # C: ring open on the right
        cx = 11.8
        d = ((gx - cx) ** 2 + (gy - CY) ** 2) ** .5
        return RO - T <= d <= RO and not (gx > cx + 1.0 and abs(gy - CY) < 2.2)
    def colour(x, y):
        gx, gy = x / u, y / u
        if inD(gx, gy) or inC(gx, gy): return K
        if 13.2 <= gx < 14.8 and 12.0 <= gy < 13.6: return P        # the dot
        return Y
    raw = b"".join(b"\x00" + bytes(c for x in range(n) for c in colour(x + .5, y + .5)) for y in range(n))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", n, n, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))

here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")
for size in (192, 512):
    png(os.path.join(here, "icon-%d.png" % size), size)
print("icons written")
