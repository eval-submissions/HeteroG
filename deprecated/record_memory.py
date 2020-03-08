import subprocess
import time
import re

out = open("memory.txt", 'w')
while True:
    s = subprocess.check_output("nvidia-smi --format=csv --query-gpu=utilization.gpu,memory.used".split())
    u1, m1, u2, m2 = re.findall(r"\n(\d.*) %, (\d.*) MiB\n(\d.*) %, (\d.*) MiB", str(s, 'utf8'))[0]
    t = time.time() * 1000
    print(t, u1, m1, u2, m2, file=out, flush=True)
