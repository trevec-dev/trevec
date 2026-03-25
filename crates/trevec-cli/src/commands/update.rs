use anyhow::{bail, Context, Result};
use std::fs;
use std::io::Read;
use std::path::Path;
use std::process::Command;

const BASE_URL: &str = "https://dl.trevec.dev";
const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");

pub async fn run(check: bool) -> Result<()> {
    let latest = fetch_latest_version()?;
    let current = format!("v{CURRENT_VERSION}");

    if check {
        print_version_status(&current, &latest);
        return Ok(());
    }

    if current == latest {
        eprintln!("Already up to date ({current}).");
        return Ok(());
    }

    eprintln!("Updating trevec {current} → {latest}...");
    eprintln!();

    let target = detect_target()?;
    let archive_name = format!("trevec-{target}.tar.gz");
    let archive_url = format!("{BASE_URL}/releases/{latest}/{archive_name}");
    let checksum_url = format!("{archive_url}.sha256");

    let tmp_dir = tempfile::tempdir().context("Failed to create temp directory")?;
    let archive_path = tmp_dir.path().join(&archive_name);
    let checksum_path = tmp_dir.path().join(format!("{archive_name}.sha256"));

    eprintln!("Downloading {archive_name}...");
    curl_download(&archive_url, &archive_path)?;
    curl_download(&checksum_url, &checksum_path)?;

    verify_checksum(&archive_path, &checksum_path)?;

    // Extract to temp dir
    let extract_dir = tmp_dir.path().join("extract");
    fs::create_dir_all(&extract_dir)?;
    let status = Command::new("tar")
        .args(["xzf", &archive_path.to_string_lossy(), "-C", &extract_dir.to_string_lossy()])
        .status()
        .context("Failed to run tar")?;
    if !status.success() {
        bail!("tar extraction failed");
    }

    let extracted_binary = extract_dir.join("trevec");
    if !extracted_binary.exists() {
        bail!("Extracted archive does not contain 'trevec' binary");
    }

    // Replace current binary atomically
    let current_exe = std::env::current_exe().context("Failed to determine current executable path")?;
    let current_dir = current_exe.parent().context("Current executable has no parent directory")?;
    let tmp_binary = current_dir.join(".trevec-update-tmp");

    fs::copy(&extracted_binary, &tmp_binary)
        .context("Failed to copy new binary next to current executable")?;

    // Set executable permissions
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&tmp_binary, fs::Permissions::from_mode(0o755))?;
    }

    fs::rename(&tmp_binary, &current_exe)
        .context("Failed to replace current binary (atomic rename)")?;

    // macOS: remove quarantine xattr
    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("xattr")
            .args(["-d", "com.apple.quarantine", &current_exe.to_string_lossy()])
            .status();
    }

    eprintln!();
    eprintln!("Updated trevec {current} → {latest}");

    Ok(())
}

fn fetch_latest_version() -> Result<String> {
    let url = format!("{BASE_URL}/version.txt");
    let output = Command::new("curl")
        .args(["-fsSL", &url])
        .output()
        .context("Failed to run curl. Is curl installed?")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("Failed to fetch latest version from {url}: {stderr}");
    }

    let version = String::from_utf8(output.stdout)
        .context("Invalid UTF-8 in version response")?
        .trim()
        .to_string();

    if version.is_empty() {
        bail!("Empty version response from {url}");
    }

    Ok(version)
}

fn print_version_status(current: &str, latest: &str) {
    if current == latest {
        eprintln!("trevec {current} (up to date)");
    } else {
        eprintln!("trevec {current} (update available: {latest})");
    }
}

fn detect_target() -> Result<String> {
    let os = detect_os()?;
    let arch = detect_arch()?;
    let target = match os.as_str() {
        "darwin" => format!("{arch}-apple-darwin"),
        "linux" => format!("{arch}-unknown-linux-gnu"),
        _ => bail!("Unsupported OS: {os}"),
    };
    Ok(target)
}

fn detect_os() -> Result<String> {
    let output = Command::new("uname")
        .arg("-s")
        .output()
        .context("Failed to run uname -s")?;
    let os_name = String::from_utf8_lossy(&output.stdout).trim().to_string();
    match os_name.as_str() {
        "Darwin" => Ok("darwin".to_string()),
        "Linux" => Ok("linux".to_string()),
        _ => bail!("Unsupported operating system: {os_name}"),
    }
}

fn detect_arch() -> Result<String> {
    let output = Command::new("uname")
        .arg("-m")
        .output()
        .context("Failed to run uname -m")?;
    let arch = String::from_utf8_lossy(&output.stdout).trim().to_string();
    match arch.as_str() {
        "x86_64" | "amd64" => Ok("x86_64".to_string()),
        "arm64" | "aarch64" => Ok("aarch64".to_string()),
        _ => bail!("Unsupported architecture: {arch}"),
    }
}

fn curl_download(url: &str, dest: &Path) -> Result<()> {
    let status = Command::new("curl")
        .args(["-fsSL", "-o", &dest.to_string_lossy(), url])
        .status()
        .context("Failed to run curl")?;

    if !status.success() {
        bail!("Failed to download {url}");
    }
    Ok(())
}

fn verify_checksum(archive: &Path, checksum_file: &Path) -> Result<()> {
    let mut expected_content = String::new();
    fs::File::open(checksum_file)
        .context("Failed to open checksum file")?
        .read_to_string(&mut expected_content)
        .context("Failed to read checksum file")?;

    let expected = expected_content
        .split_whitespace()
        .next()
        .context("Empty checksum file")?;

    // Try shasum first (macOS), then sha256sum (Linux)
    let actual = if let Ok(output) = Command::new("shasum").args(["-a", "256", &archive.to_string_lossy()]).output() {
        if output.status.success() {
            String::from_utf8_lossy(&output.stdout)
                .split_whitespace()
                .next()
                .unwrap_or("")
                .to_string()
        } else {
            try_sha256sum(archive)?
        }
    } else {
        try_sha256sum(archive)?
    };

    if actual != expected {
        bail!(
            "Checksum verification failed.\n  Expected: {expected}\n  Got:      {actual}"
        );
    }

    eprintln!("Checksum verified.");
    Ok(())
}

fn try_sha256sum(archive: &Path) -> Result<String> {
    let output = Command::new("sha256sum")
        .arg(&archive.to_string_lossy().to_string())
        .output()
        .context("No SHA256 tool found (tried shasum and sha256sum)")?;

    if !output.status.success() {
        bail!("sha256sum failed");
    }

    Ok(String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_target() {
        let target = detect_target().unwrap();
        let valid_targets = [
            "x86_64-apple-darwin",
            "aarch64-apple-darwin",
            "x86_64-unknown-linux-gnu",
            "aarch64-unknown-linux-gnu",
        ];
        assert!(
            valid_targets.contains(&target.as_str()),
            "Unexpected target: {target}"
        );
    }

    #[test]
    fn test_version_comparison() {
        let current = format!("v{CURRENT_VERSION}");

        // Same version
        assert_eq!(current, current);

        // Different versions are not equal
        assert_ne!(current, "v0.0.0");
        assert_ne!(current, "v99.99.99");
    }

    #[test]
    fn test_current_version_format() {
        // CARGO_PKG_VERSION should be semver without 'v' prefix
        assert!(
            !CURRENT_VERSION.starts_with('v'),
            "CARGO_PKG_VERSION should not start with 'v'"
        );
        let parts: Vec<&str> = CURRENT_VERSION.split('.').collect();
        assert_eq!(parts.len(), 3, "Version should have 3 parts: {CURRENT_VERSION}");
        for part in &parts {
            part.parse::<u32>()
                .unwrap_or_else(|_| panic!("Version part '{part}' is not a number"));
        }
    }
}
