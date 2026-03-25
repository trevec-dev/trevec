/**
 * Native binary loader. Tries to load the platform-specific .node file.
 * Returns null if no binary is available for this platform.
 */

const { existsSync } = require('fs');
const { join } = require('path');
const { platform, arch } = process;

const platformMap = {
  'win32-x64': 'trevec.win32-x64-msvc.node',
  'linux-x64': 'trevec.linux-x64-gnu.node',
  'darwin-arm64': 'trevec.darwin-arm64.node',
  'darwin-x64': 'trevec.darwin-x64.node',
  'linux-arm64': 'trevec.linux-arm64-gnu.node',
};

function loadNative() {
  const key = `${platform}-${arch}`;
  const fileName = platformMap[key];
  if (!fileName) return null;

  const filePath = join(__dirname, fileName);
  if (!existsSync(filePath)) return null;

  try {
    return require(filePath);
  } catch {
    return null;
  }
}

module.exports = { loadNative };
