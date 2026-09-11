//! PRAW.app launcher: starts the bundled backend if it isn't already
//! healthy, then replaces this process with the GUI.
//!
//! This is a real Mach-O binary (not a script) because launchd cannot spawn
//! shell scripts as a bundle's `CFBundleExecutable`.

// std::process::Command is fine here: this launcher is macOS-only (the PRAW
// package is a .app bundle), so the Windows terminal-flash concern behind the
// workspace lint does not apply.
#![allow(clippy::disallowed_types)]

use std::io::{Read as _, Write as _};
use std::net::TcpStream;
use std::os::unix::process::CommandExt as _;
use std::process::Command;
use std::time::{Duration, Instant};

// 8080 is the most collision-prone dev port there is (and MLX-LM's default),
// so the app pairs the backend and GUI on an uncommon one instead.
const DEFAULT_BACKEND_PORT: u16 = 48080;
const MAX_LOG_BYTES: u64 = 10 * 1024 * 1024;

fn main() {
    let dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|parent| parent.to_path_buf()))
        .expect("launcher must live inside the app bundle");
    let port = std::env::var("PRAW_BACKEND_PORT")
        .ok()
        .and_then(|port| port.parse().ok())
        .unwrap_or(DEFAULT_BACKEND_PORT);

    if !backend_healthy(port) {
        start_backend(&dir, port);
    }

    let gui = dir.join("warp-oss");
    let error = Command::new(gui).exec();
    panic!("failed to exec the GUI: {error}");
}

fn backend_healthy(port: u16) -> bool {
    let address = format!("127.0.0.1:{port}")
        .parse()
        .expect("valid socket address");
    let Ok(mut stream) = TcpStream::connect_timeout(&address, Duration::from_millis(750)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(750)));
    let request = format!("GET /healthz HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\n\r\n");
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = [0u8; 128];
    let read = stream.read(&mut response).unwrap_or(0);
    // Only an answer from the backend itself counts; some other service that
    // merely accepts TCP on the port must not make the launcher skip startup.
    let text = String::from_utf8_lossy(&response[..read]);
    text.starts_with("HTTP/1.") && text.contains(" 200 ")
}

fn start_backend(dir: &std::path::Path, port: u16) {
    let backend = dir.join("selfhost_server");
    let log_path = backend_log_path();
    rotate_log_if_large(&log_path);
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .expect("failed to open backend log file");
    let err = log
        .try_clone()
        .expect("failed to clone backend log file handle");
    let spawned = Command::new(&backend)
        .arg("--bind")
        .arg(format!("127.0.0.1:{port}"))
        .stdout(log)
        .stderr(err)
        .process_group(0)
        .spawn();
    match spawned {
        Ok(_child) => {
            if !wait_for_backend(port, Duration::from_secs(10)) {
                warn_backend_failed(
                    "PRAW's agent backend did not become healthy, so agent mode will not work. \
                     The backend log is PRAW-backend.log in ~/Library/Logs.",
                    &log_path,
                );
            }
        }
        Err(error) => {
            eprintln!("failed to spawn {}: {error}", backend.display());
            warn_backend_failed(
                "PRAW's agent backend could not be launched. If macOS removed the app's tools \
                 (unsigned build), re-copy PRAW.app from the DMG and run Install PRAW.command. \
                 The backend log is PRAW-backend.log in ~/Library/Logs.",
                &log_path,
            );
        }
    }
}

fn backend_log_path() -> std::path::PathBuf {
    std::env::var("HOME")
        .map(|home| {
            std::path::Path::new(&home)
                .join("Library/Logs/PRAW-backend.log")
                .into_os_string()
        })
        .map(std::path::PathBuf::from)
        .unwrap_or_default()
}

fn rotate_log_if_large(log_path: &std::path::Path) {
    if std::fs::metadata(log_path).is_ok_and(|meta| meta.len() > MAX_LOG_BYTES)
        && let Err(error) = std::fs::rename(log_path, log_path.with_extension("log.old"))
    {
        eprintln!("failed to rotate backend log: {error}");
    }
}

/// Non-fatal: the GUI still opens so the user can see client-side state, but
/// the dialog explains that agent mode needs the backend.
fn warn_backend_failed(message: &str, log_path: &std::path::Path) {
    let script = format!(
        "display dialog \"{message}\" with title \"PRAW\" \
         buttons {{\"Open Log\", \"Continue\"}} default button \"Continue\" with icon caution"
    );
    let chosen = Command::new("osascript")
        .arg("-e")
        .arg(&script)
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
        .unwrap_or_default();
    if chosen.contains("Open Log") {
        let _ = Command::new("open").arg("-R").arg(log_path).status();
    }
}

fn wait_for_backend(port: u16, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if backend_healthy(port) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    false
}

#[cfg(test)]
mod tests {
    use std::io::{Read as _, Write as _};
    use std::net::TcpListener;

    use super::backend_healthy;

    /// Accepts one connection, drains the request, and answers with
    /// `response` (or nothing when `None`), like a one-shot HTTP service.
    fn serve_once(response: Option<&'static [u8]>) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let port = listener.local_addr().expect("local address").port();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut request = [0u8; 256];
            let _ = stream.read(&mut request);
            if let Some(response) = response {
                let _ = stream.write_all(response);
            }
        });
        port
    }

    #[test]
    fn health_probe_accepts_our_backend() {
        let port = serve_once(Some(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"status\":\"ok\"}",
        ));
        assert!(backend_healthy(port));
    }

    #[test]
    fn health_probe_rejects_wrong_status() {
        let port = serve_once(Some(b"HTTP/1.1 503 Service Unavailable\r\n\r\n"));
        assert!(!backend_healthy(port));
    }

    #[test]
    fn health_probe_rejects_non_http_listener() {
        let port = serve_once(Some(b"not-an-http-service"));
        assert!(!backend_healthy(port));
    }

    #[test]
    fn health_probe_rejects_silent_listener() {
        let port = serve_once(None);
        assert!(!backend_healthy(port));
    }
}
