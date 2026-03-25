const { existsSync, readFileSync } = require('fs');
const { join } = require('path');

const { platform, arch } = process;

let nativeBinding = null;

const platformMap = {
  'win32-x64': 'trevec.win32-x64-msvc.node',
  'darwin-x64': 'trevec.darwin-x64.node',
  'darwin-arm64': 'trevec.darwin-arm64.node',
  'linux-x64': 'trevec.linux-x64-gnu.node',
  'linux-arm64': 'trevec.linux-arm64-gnu.node',
};

const key = `${platform}-${arch}`;
const fileName = platformMap[key];

if (!fileName) {
  throw new Error(`Unsupported platform: ${platform}-${arch}`);
}

const filePath = join(__dirname, fileName);

if (existsSync(filePath)) {
  nativeBinding = require(filePath);
} else {
  throw new Error(
    `Native binding not found: ${filePath}\n` +
    `Platform: ${platform}-${arch}\n` +
    `Install the correct platform package or build from source.`
  );
}

module.exports = nativeBinding;
module.exports.Trevec = nativeBinding.Trevec;
