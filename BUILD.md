# Trevec — Build Guide

How to build all Trevec binaries and packages for distribution.

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Rust | 1.91+ | Core engine (pinned via `rust-toolchain.toml`) |
| Python | 3.12+ | Python SDK bindings (PyO3) |
| Node.js | 22 LTS | npm SDK, napi-rs native bindings |
| maturin | 1.x | Python wheel builder |
| patchelf | latest | Linux wheel packaging |
| WSL2 Ubuntu | latest | Building Linux binaries on Windows |

## Quick Reference

```bash
# Python wheel (Windows)
cd crates/trevec-python
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 python -m maturin build --release

# Python wheel (Linux via WSL2)
./build-linux-wheel.sh

# npm SDK (pure TypeScript)
cd sdks/node
npm install && npm run build && npm publish

# napi-rs native (Windows)
cd sdks/node-native
npx napi build --platform --release --manifest-path ../../crates/trevec-node/Cargo.toml

# napi-rs native (Linux via WSL2)
# See "Linux Native Binary" section below

# Rust CLI binary
cargo build --release -p trevec

# Full workspace check
cargo check --release
```

---

## 1. Python SDK (PyPI)

### Windows wheel

```bash
cd crates/trevec-python
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
python -m maturin build --release
# Output: target/wheels/trevec-X.Y.Z-cpXXX-cpXXX-win_amd64.whl
```

### Linux wheel (via WSL2)

```bash
# From Windows, run the build script:
./build-linux-wheel.sh

# Or manually in WSL2:
wsl -d Ubuntu
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:/usr/bin:/bin"
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
cd /mnt/c/Users/<username>/Apps/trevec/crates/trevec-python
python3 -m maturin build --release -o /mnt/c/Users/<username>/Apps/trevec/target/wheels/
# Output: target/wheels/trevec-X.Y.Z-cpXXX-cpXXX-manylinux_2_38_x86_64.whl
```

**Requirements in WSL2:**
```bash
pip3 install maturin patchelf --break-system-packages
rustup default stable
```

### macOS wheel (requires macOS machine or CI)

```bash
# On a Mac with Rust + Python + maturin installed:
cd crates/trevec-python
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin build --release
# Produces: trevec-X.Y.Z-cpXXX-cpXXX-macosx_XX_X_arm64.whl
```

### Publishing to PyPI

```bash
# Upload all wheels at once
source .env  # needs PYPI_API_TOKEN
python -m twine upload target/wheels/trevec-*.whl -u __token__ -p "$PYPI_API_TOKEN"
```

---

## 2. npm SDK (TypeScript — pure, no native deps)

```bash
cd sdks/node
npm install
npm run build    # compiles TypeScript
npm publish      # publishes to npm
```

**Size:** ~7KB. Zero native dependencies. Works on all platforms.

---

## 3. npm Native Bindings (napi-rs)

**IMPORTANT:** Use an isolated `CARGO_TARGET_DIR` to avoid Python symbol contamination
from `trevec-python` in the same workspace. This is required for the `.node` binary to load.

### Windows native binary

```bash
cd crates/trevec-node
CARGO_TARGET_DIR=../../target-node cargo build --release
# Output: target-node/release/trevec_node.dll
# Copy to: sdks/node-native/trevec.win32-x64-msvc.node
cp ../../target-node/release/trevec_node.dll ../../sdks/node-native/trevec.win32-x64-msvc.node
```

**Requires Node 22 LTS** (Node 24 has a compatibility issue).

### Linux native binary (via WSL2)

```bash
wsl -d Ubuntu
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:/usr/bin:/bin"
cd /mnt/c/Users/<username>/Apps/trevec/crates/trevec-node
CARGO_TARGET_DIR=/tmp/trevec-node-build cargo build --release
# Output: /tmp/trevec-node-build/release/libtrevec_node.so
cp /tmp/trevec-node-build/release/libtrevec_node.so /mnt/c/Users/<username>/Apps/trevec/sdks/node-native/trevec.linux-x64-gnu.node
```

**Requirements in WSL2:**
```bash
# Node.js 22 LTS
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs

# Rust (already installed from Python wheel build)
rustup default stable
```

### macOS native binary (requires macOS or cross-compilation)

```bash
# On a Mac:
cd sdks/node-native
npm install
npx napi build --platform --release --manifest-path ../../crates/trevec-node/Cargo.toml
# Output: trevec.darwin-arm64.node
```

### Testing native binaries

```bash
cd sdks/node-native
node -e "
const { Trevec } = require('./index');
const tv = new Trevec();
tv.add('test', 'user1');
console.log(tv.search('test', 'user1'));
console.log('OK');
"
```

**Known issue:** Node.js 24 has a compatibility issue ("Module did not self-register"). Use Node 22 LTS for testing native bindings.

### Publishing native package

The native package should be published as `@trevec/native` with platform-specific sub-packages. This requires the napi-rs prepublish workflow:

```bash
npx napi prepublish -t npm
npm publish --access public
```

---

## 4. Rust CLI Binary

```bash
cargo build --release -p trevec
# Output: target/release/trevec.exe (Windows) or target/release/trevec (Linux/macOS)
```

For Linux binary via WSL2:
```bash
wsl -d Ubuntu
export PATH="$HOME/.cargo/bin:$PATH"
cd /mnt/c/Users/<username>/Apps/trevec
cargo build --release -p trevec
# Output: target/release/trevec
```

---

## 5. Full Workspace

### Check everything compiles

```bash
cargo check --release    # all Rust crates
cargo test --release -p trevec-core -p trevec-brain -p trevec-sdk   # run tests
```

### Run all tests

```bash
# Rust tests (182 passing)
cargo test -p trevec-core -p trevec-brain -p trevec-sdk --release

# Python end-to-end
pip install target/wheels/trevec-*.whl --force-reinstall
python -c "from trevec import Trevec; tv = Trevec(); tv.add('test', user_id='u1'); print(tv.search('test', user_id='u1'))"

# npm end-to-end
cd sdks/node && npm run build
node -e "const {Trevec} = require('./dist'); const tv = new Trevec(); tv.add('test', {userId:'u1'}); console.log(tv.search('test', {userId:'u1'}))"
```

---

## Platform Build Matrix

| Artifact | Windows | Linux (WSL2) | macOS | CI |
|----------|---------|-------------|-------|-----|
| Python wheel | ✅ Local | ✅ WSL2 | Needs Mac/CI | GitHub Actions |
| npm SDK (TS) | ✅ Local | ✅ Local | ✅ Local | Any |
| npm native | ✅ Local | ✅ WSL2 | Needs Mac/CI | GitHub Actions |
| CLI binary | ✅ Local | ✅ WSL2 | Needs Mac/CI | GitHub Actions |

### GitHub Actions CI (future)

For automated multi-platform builds, create `.github/workflows/release.yml`:

```yaml
name: Release
on:
  push:
    tags: ['v*']

jobs:
  build-python:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install maturin
      - run: cd crates/trevec-python && maturin build --release
      - uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}
          path: target/wheels/*.whl

  build-node-native:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 22
      - run: cd sdks/node-native && npm install && npx napi build --platform --release --manifest-path ../../crates/trevec-node/Cargo.toml
      - uses: actions/upload-artifact@v4
        with:
          name: native-${{ matrix.os }}
          path: sdks/node-native/*.node

  publish-pypi:
    needs: build-python
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
      - run: pip install twine && twine upload wheels-*/*.whl
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}

  publish-npm:
    needs: build-node-native
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
      - run: cd sdks/node-native && npx napi prepublish -t npm && npm publish --access public
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
```
