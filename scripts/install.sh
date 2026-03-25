#!/bin/sh
# Trevec installer — https://trevec.dev
# Usage: curl -fsSL dl.trevec.dev/install.sh | sh
#        curl -fsSL dl.trevec.dev/install.sh | sh -s -- --version v1.1.0
set -eu

BOLD="\033[1m"
GREEN="\033[32m"
RED="\033[31m"
RESET="\033[0m"

TREVEC_BASE_URL="${TREVEC_BASE_URL:-https://dl.trevec.dev}"
TREVEC_INSTALL="${TREVEC_INSTALL:-$HOME/.trevec}"

info() { printf "${BOLD}%s${RESET}\n" "$1"; }
success() { printf "${GREEN}%s${RESET}\n" "$1"; }
error() { printf "${RED}error: %s${RESET}\n" "$1" >&2; exit 1; }

# Detect OS
detect_os() {
  case "$(uname -s)" in
    Darwin) echo "darwin" ;;
    Linux)  echo "linux" ;;
    *)      error "Unsupported operating system: $(uname -s). Use the PowerShell installer on Windows." ;;
  esac
}

# Detect architecture
detect_arch() {
  case "$(uname -m)" in
    x86_64|amd64)       echo "x86_64" ;;
    arm64|aarch64)      echo "aarch64" ;;
    *)                  error "Unsupported architecture: $(uname -m)" ;;
  esac
}

# Build target triple
get_target() {
  os="$1"
  arch="$2"
  case "$os" in
    darwin) echo "${arch}-apple-darwin" ;;
    linux)  echo "${arch}-unknown-linux-gnu" ;;
  esac
}

# Check for required commands
check_deps() {
  if ! command -v curl > /dev/null 2>&1; then
    error "curl is required but not found. Install curl and try again."
  fi
}

# Resolve version
resolve_version() {
  if [ -n "${TREVEC_VERSION:-}" ]; then
    echo "$TREVEC_VERSION"
    return
  fi
  version=$(curl -fsSL "${TREVEC_BASE_URL}/version.txt") || error "Failed to fetch latest version"
  echo "$version"
}

# Verify SHA256 checksum
verify_checksum() {
  local checkfile="$1"
  local expected_file="$2"

  local expected=$(awk '{print $1}' "$expected_file")

  if command -v shasum > /dev/null 2>&1; then
    local actual=$(shasum -a 256 "$checkfile" | awk '{print $1}')
  elif command -v sha256sum > /dev/null 2>&1; then
    local actual=$(sha256sum "$checkfile" | awk '{print $1}')
  else
    info "Warning: no SHA256 tool found, skipping checksum verification"
    return 0
  fi

  if [ "$actual" != "$expected" ]; then
    error "Checksum verification failed.\n  Expected: ${expected}\n  Got:      ${actual}"
  fi
}

# Update shell profile with PATH
update_path() {
  bin_dir="$1"
  path_entry="export PATH=\"${bin_dir}:\$PATH\""

  # Find the right profile file
  profile=""
  if [ -n "${ZSH_VERSION:-}" ] || [ "$(basename "${SHELL:-}")" = "zsh" ]; then
    profile="$HOME/.zshrc"
  elif [ -f "$HOME/.bashrc" ]; then
    profile="$HOME/.bashrc"
  elif [ -f "$HOME/.bash_profile" ]; then
    profile="$HOME/.bash_profile"
  elif [ -f "$HOME/.profile" ]; then
    profile="$HOME/.profile"
  fi

  if [ -z "$profile" ]; then
    info "Could not detect shell profile. Add this to your shell config:"
    info "  $path_entry"
    return
  fi

  if grep -q ".trevec/bin" "$profile" 2>/dev/null; then
    return
  fi

  printf '\n# Trevec\n%s\n' "$path_entry" >> "$profile"
  info "Added trevec to PATH in $profile"
}

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --version)
        [ $# -lt 2 ] && error "--version requires a value (e.g. --version v1.1.0)"
        TREVEC_VERSION="$2"
        shift 2
        ;;
      --help|-h)
        echo "Usage: install.sh [--version VERSION]"
        echo ""
        echo "Options:"
        echo "  --version VERSION  Install a specific version (e.g. v1.1.0)"
        echo ""
        echo "Environment variables:"
        echo "  TREVEC_VERSION     Same as --version (flag takes precedence)"
        echo "  TREVEC_INSTALL     Install directory (default: ~/.trevec)"
        echo "  TREVEC_BASE_URL    Download base URL (default: https://dl.trevec.dev)"
        exit 0
        ;;
      *)
        error "Unknown option: $1. Use --help for usage."
        ;;
    esac
  done
}

main() {
  parse_args "$@"

  info "Installing trevec..."
  echo ""

  check_deps

  os=$(detect_os)
  arch=$(detect_arch)
  target=$(get_target "$os" "$arch")
  version=$(resolve_version)

  info "  Version:  $version"
  info "  Platform: ${os}/${arch}"
  echo ""

  archive="trevec-${target}.tar.gz"
  url="${TREVEC_BASE_URL}/releases/${version}/${archive}"
  checksum_url="${url}.sha256"

  # Create temp directory
  tmp_dir=$(mktemp -d)
  trap 'rm -rf "$tmp_dir"' EXIT

  # Download archive and checksum
  info "Downloading ${archive}..."
  curl -fsSL -o "${tmp_dir}/${archive}" "$url" || error "Failed to download ${url}"
  curl -fsSL -o "${tmp_dir}/${archive}.sha256" "$checksum_url" || error "Failed to download checksum"

  # Verify checksum
  verify_checksum "${tmp_dir}/${archive}" "${tmp_dir}/${archive}.sha256"

  # Extract
  bin_dir="${TREVEC_INSTALL}/bin"
  mkdir -p "$bin_dir"
  tar xzf "${tmp_dir}/${archive}" -C "$bin_dir"
  chmod +x "${bin_dir}/trevec"

  # macOS: remove quarantine attribute
  if [ "$os" = "darwin" ]; then
    xattr -d com.apple.quarantine "${bin_dir}/trevec" 2>/dev/null || true
    # Also clear quarantine on bundled dylibs (Intel Mac build bundles libonnxruntime)
    for dylib in "${bin_dir}"/*.dylib; do
      [ -f "$dylib" ] && xattr -d com.apple.quarantine "$dylib" 2>/dev/null || true
    done
  fi

  # Update PATH
  update_path "$bin_dir"

  echo ""
  success "trevec ${version} installed successfully!"
  echo ""
  info "Run 'trevec --help' to get started."
  info "You may need to restart your shell or run:"
  info "  export PATH=\"${bin_dir}:\$PATH\""
}

main "$@"
