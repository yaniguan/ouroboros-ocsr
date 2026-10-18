# Colab access verified — 2026-09-30

Existing signed-in Colab Pro session accessed through Chrome UI.
Notebook: https://colab.research.google.com/drive/1I5HwAk544O9GeOccSDTyMx4A5izYNz2P?authuser=1

At 11:57 AM local time, executed only the existing first hardware probe cell:

```python
import subprocess, time
started = time.monotonic()
print(subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], text=True))
```

Observed fresh execution count `[1]`, output:

```text
NVIDIA A100-SXM4-40GB, 40960 MiB
```

Resource panel: Colab Pro; available 45.27 compute units; approximately
5.3 units/hour; one active session. RAM 1.6/83.5 GB, GPU RAM 0.0/40.0 GB,
disk 47.4/235.7 GB. No purchases or upgrades.

This verifies current browser access and execution on an allocated A100 only.
The notebook contains legacy project cells; they were not rerun. Current-code
environment, tests, data audit and training are not established by this probe.
