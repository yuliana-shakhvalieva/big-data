#!/usr/bin/python3

import sys

ck, mk = 0, 0

for line in sys.stdin:
    line = line.strip()
    ck += 1
    mk += int(line)
mk /= ck
print("%s\t%s" % (ck, mk))
