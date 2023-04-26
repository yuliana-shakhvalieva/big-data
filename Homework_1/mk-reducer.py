#!/usr/bin/python3

import sys

ck_global, mk_global = 0, 0

for line in sys.stdin:
    line = line.strip()
    ck, mk = line.split('\t')
    try:
        ck = int(ck)
        mk = float(mk)
    except ValueError:
        continue

    mk_global = ((mk_global * ck_global) + (mk * ck)) / (ck_global + ck)
    ck_global += ck


print(mk_global)
