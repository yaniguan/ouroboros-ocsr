#!/bin/bash
set -euo pipefail
trap 'status=$?; printf "ERROR container repair line=%s exit=%s\n" "$LINENO" "$status" >&2' ERR
export LC_ALL=C LANG=C OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
: "${OURO_ROOT:?}"
source /u/local/Modules/default/init/modules.sh
module load apptainer/1.2.2
export CONDA_PKGS_DIRS="$OURO_ROOT/cache/conda"
mkdir -p "$OURO_ROOT/container-bin" "$OURO_ROOT/cache/apptainer" "$OURO_ROOT/reports"
if [ ! -x "$OURO_ROOT/container-tools/bin/unsquashfs" ]; then
  /u/home/y/yaniguan/miniconda3/bin/conda create -y --no-default-packages \
    --override-channels -c conda-forge -p "$OURO_ROOT/container-tools" squashfs-tools
fi
cat > "$OURO_ROOT/container-bin/unsquashfs" <<'WRAPPER'
#!/bin/bash
exec "$OURO_ROOT/container-tools/bin/unsquashfs" -processors "${NSLOTS:-4}" "$@"
WRAPPER
chmod +x "$OURO_ROOT/container-bin/unsquashfs"
export PATH="$OURO_ROOT/container-bin:$PATH"
export APPTAINER_CACHEDIR="$OURO_ROOT/cache/apptainer"
export APPTAINER_TMPDIR="${TMPDIR:-$OURO_ROOT/cache/apptainer}"
selected=
for image in \
  /u/local/apps/apptainer/H2_containers/rpylab_rpylab-R4.3.3-python-3.10.10-oneapi-gpu.sif \
  /u/local/apps/apptainer/H2_containers/ubuntu-24.04_wpython.sif \
  /u/local/apps/apptainer/H2_containers/ubuntu_22.04.sif; do
  if apptainer exec --cleanenv "$image" /bin/bash -c \
    'set -e; ldd --version; command -v gfortran; python3 -c "import platform; assert tuple(map(int,platform.libc_ver()[1].split(\".\"))) >= (2,28)"'; then
    selected=$image
    break
  fi
done
test -n "$selected"
printf '%s\n' "$selected" > "$OURO_ROOT/reports/container-image.txt"
sha256sum "$selected" > "$OURO_ROOT/reports/container-image.sha256"
if [ ! -x "$OURO_ROOT/bootstrap/bin/uv" ]; then
  /u/home/y/yaniguan/miniconda3/bin/python -m pip install --target "$OURO_ROOT/bootstrap" uv
fi
apptainer exec --cleanenv --bind "$OURO_ROOT:$OURO_ROOT" \
  --env "OURO_ROOT=$OURO_ROOT" "$selected" /bin/bash "$OURO_ROOT/code/scripts/hpc/setup.sh"
touch "$OURO_ROOT/reports/container-setup.ok"
