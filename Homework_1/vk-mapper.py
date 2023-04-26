#!/usr/bin/python3

import sys

ck, mk, vk = 0, 0, 0

for line in sys.stdin:
    line = line.strip()
    ck += 1
    mk += int(line)
    vk += int(line) ** 2
mk /= ck
vk /= ck
vk -= mk ** 2
print("%s\t%s\t%s" % (ck, mk, vk))
