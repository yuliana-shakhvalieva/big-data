#!/usr/bin/python3

import sys

ck_global, mk_global, vk_global = 0, 0, 0

for line in sys.stdin:
    line = line.strip()
    ck, mk, vk = line.split('\t')
    try:
        ck = int(ck)
        mk = float(mk)
        vk = float(vk)
    except ValueError:
        continue

    vk_global = ((vk_global * ck_global) + (vk * ck)) / (ck_global + ck) + \
                        ck_global * ck * ((mk_global - mk)/(ck + ck_global)) ** 2
    mk_global = ((mk_global * ck_global) + (mk * ck)) / (ck_global + ck)
    ck_global += ck

print(vk_global)
