//! `flowd init` -- scaffold integrations for external tools.
//!
//! Currently ships a single target: Cursor's `mcp.json`. The handler
//! builds the canonical flowd server stanza pointing at the currently
//! running binary (`std::env::current_exe`), deep-merges it into the
//! destination file (preserving every other key), and performs an
//! atomic tmp-file + rename. Re-running against an already-correct
//! file is a no-op.

use std::fs::{self, File};
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};

use crate::cli::InitTarget;

// kept async to match the dispatch contract in main.rs; init may grow async work later
#[allow(clippy::unused_async)]
pub async fn run(target: InitTarget) -> Result<()> {
    match target {
        InitTarget::Cursor { global, project } => cursor(global, project),
    }
}

fn cursor(global: bool, project: Option<PathBuf>) -> Result<()> {
    let dest = resolve_cursor_dest(global, project)?;

    let exe = std::env::current_exe().context("determine path of the running flowd binary")?;
    let exe_str = exe
        .to_str()
        .ok_or_else(|| anyhow!("flowd binary path is not valid UTF-8: {}", exe.display()))?
        .to_owned();

    let canonical_entry = json!({
        "command": exe_str,
        "args": ["start", "--mcp"],
    });

    let mut merged = load_json_object(&dest)?;
    {
        let root = merged
            .as_object_mut()
            .expect("load_json_object guarantees object");
        let servers = root
            .entry("mcpServers".to_owned())
            .or_insert_with(|| json!({}));
        if !servers.is_object() {
            bail!(
                "`mcpServers` in {} is not a JSON object (found {}); refusing to overwrite",
                dest.display(),
                kind(servers),
            );
        }
        servers
            .as_object_mut()
            .expect("checked above")
            .insert("flowd".to_owned(), canonical_entry);
    }

    let mut merged_bytes =
        serde_json::to_vec_pretty(&merged).context("serialize merged mcp.json")?;
    merged_bytes.push(b'\n');

    if dest.exists() {
        let on_disk = fs::read(&dest).with_context(|| format!("read {}", dest.display()))?;
        if on_disk == merged_bytes {
            println!(
                "flowd MCP entry already present at {} (no changes)",
                dest.display()
            );
            return Ok(());
        }
    }

    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    write_atomic(&dest, &merged_bytes)?;

    println!("wrote flowd MCP entry to {}", dest.display());
    println!("restart Cursor for the new MCP server to be picked up");
    Ok(())
}

fn resolve_cursor_dest(global: bool, project: Option<PathBuf>) -> Result<PathBuf> {
    match (global, project) {
        (true, _) => {
            let home = std::env::var("HOME")
                .context("$HOME is not set; required to locate ~/.cursor/mcp.json")?;
            Ok(PathBuf::from(home).join(".cursor").join("mcp.json"))
        }
        (false, Some(p)) => Ok(p.join(".cursor").join("mcp.json")),
        (false, None) => bail!(
            "pick a scope: pass --global for ~/.cursor/mcp.json or --project <path> for <path>/.cursor/mcp.json"
        ),
    }
}

/// Read `path` as JSON and return its root as an object `Value`. Missing
/// and empty files collapse to `{}`. Non-object roots are rejected so the
/// merge can't silently clobber unexpected data.
fn load_json_object(path: &Path) -> Result<Value> {
    if !path.exists() {
        return Ok(json!({}));
    }
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.iter().all(u8::is_ascii_whitespace) {
        return Ok(json!({}));
    }
    let value: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse JSON in {}", path.display()))?;
    if !value.is_object() {
        bail!(
            "{} is not a JSON object (found {}); refusing to overwrite",
            path.display(),
            kind(&value),
        );
    }
    Ok(value)
}

fn write_atomic(dest: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = tmp_path(dest);
    {
        let mut f = File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?;
        f.write_all(bytes)
            .with_context(|| format!("write {}", tmp.display()))?;
        f.sync_all()
            .with_context(|| format!("fsync {}", tmp.display()))?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&tmp, fs::Permissions::from_mode(0o600))
            .with_context(|| format!("chmod 0600 {}", tmp.display()))?;
    }
    fs::rename(&tmp, dest)
        .with_context(|| format!("rename {} -> {}", tmp.display(), dest.display()))?;
    Ok(())
}

fn tmp_path(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().map_or_else(
        || std::ffi::OsString::from("mcp.json"),
        std::ffi::OsStr::to_os_string,
    );
    name.push(".tmp");
    dest.with_file_name(name)
}

fn kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write(path: &Path, body: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, body).unwrap();
    }

    #[test]
    fn load_json_object_accepts_missing_file() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("missing.json");
        assert_eq!(load_json_object(&p).unwrap(), json!({}));
    }

    #[test]
    fn load_json_object_accepts_empty_file() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("empty.json");
        write(&p, "");
        assert_eq!(load_json_object(&p).unwrap(), json!({}));
    }

    #[test]
    fn load_json_object_rejects_array_root() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("arr.json");
        write(&p, "[]");
        assert!(load_json_object(&p).is_err());
    }

    #[test]
    fn resolve_requires_scope() {
        assert!(resolve_cursor_dest(false, None).is_err());
    }

    #[test]
    fn resolve_project_appends_cursor_dir() {
        let got = resolve_cursor_dest(false, Some(PathBuf::from("/tmp/proj"))).unwrap();
        assert_eq!(got, PathBuf::from("/tmp/proj/.cursor/mcp.json"));
    }

    #[test]
    fn tmp_path_is_sibling_with_tmp_suffix() {
        let dest = PathBuf::from("/tmp/a/b/mcp.json");
        assert_eq!(tmp_path(&dest), PathBuf::from("/tmp/a/b/mcp.json.tmp"));
    }

    #[tokio::test]
    async fn deep_merge_preserves_existing_servers() {
        let tmp = TempDir::new().unwrap();
        let cursor_dir = tmp.path().join(".cursor");
        fs::create_dir_all(&cursor_dir).unwrap();
        fs::write(
            cursor_dir.join("mcp.json"),
            r#"{"mcpServers":{"foo":{"command":"f"}}}"#,
        )
        .unwrap();

        run(InitTarget::Cursor {
            global: false,
            project: Some(tmp.path().to_path_buf()),
        })
        .await
        .unwrap();

        let bytes = fs::read(cursor_dir.join("mcp.json")).unwrap();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        let servers = v
            .get("mcpServers")
            .and_then(Value::as_object)
            .expect("mcpServers is an object");
        assert!(servers.contains_key("foo"), "foo entry was dropped");
        assert!(servers.contains_key("flowd"), "flowd entry missing");
    }

    #[tokio::test]
    async fn second_run_is_byte_identical() {
        let tmp = TempDir::new().unwrap();
        let target = InitTarget::Cursor {
            global: false,
            project: Some(tmp.path().to_path_buf()),
        };
        run(InitTarget::Cursor {
            global: false,
            project: Some(tmp.path().to_path_buf()),
        })
        .await
        .unwrap();
        let dest = tmp.path().join(".cursor").join("mcp.json");
        let first = fs::read(&dest).unwrap();
        run(target).await.unwrap();
        let second = fs::read(&dest).unwrap();
        assert_eq!(first, second);
    }

    #[tokio::test]
    async fn no_tmp_leftover_after_write() {
        let tmp = TempDir::new().unwrap();
        run(InitTarget::Cursor {
            global: false,
            project: Some(tmp.path().to_path_buf()),
        })
        .await
        .unwrap();
        let leftover = tmp.path().join(".cursor").join("mcp.json.tmp");
        assert!(!leftover.exists(), "{} still exists", leftover.display());
    }

    #[tokio::test]
    async fn missing_cursor_dir_is_created() {
        let tmp = TempDir::new().unwrap();
        assert!(!tmp.path().join(".cursor").exists());
        run(InitTarget::Cursor {
            global: false,
            project: Some(tmp.path().to_path_buf()),
        })
        .await
        .unwrap();
        assert!(tmp.path().join(".cursor").is_dir());
        assert!(tmp.path().join(".cursor").join("mcp.json").exists());
    }

    #[test]
    fn clap_rejects_global_with_project() {
        use crate::cli::Cli;
        use clap::Parser;
        let result =
            Cli::try_parse_from(["flowd", "init", "cursor", "--global", "--project", "/x"]);
        assert!(result.is_err(), "mutually exclusive flags accepted");
    }
}
