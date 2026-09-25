#!/usr/bin/env bash
# Install the data guard as the git pre-commit hook of this clone.
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
hook="$root/.git/hooks/pre-commit"
cat > "$hook" <<'HOOK'
#!/usr/bin/env bash
exec python "$(git rev-parse --show-toplevel)/scripts/check_no_data.py"
HOOK
chmod +x "$hook"
echo "installed $hook"
