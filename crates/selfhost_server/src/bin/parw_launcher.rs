//! PRAW.app launcher: starts the bundled backend if it isn't already
//! healthy, then replaces this process with the GUI.
//!
//! This is a real Mach-O binary (not a script) because launchd cannot spawn
//! shell scripts as a bundle's `CFBundleExecutable`.

// std::process::Command is fine here: this launcher is macOS-only (the PARW
// package is a .app bundle), so the Windows terminal-flash concern behind the
// workspace lint does not apply.
#![allow(clippy::disallowed_types)]

use std::net::TcpStream;
use std::os::unix::process::CommandExt as _;
use std::process::Command;
use std::time::{Duration, Instant};

const BACKEND_PORT: u16 = 8080;

fn main() {
    let dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|parent| parent.to_path_buf()))
        .expect("launcher must live inside the app bundle");

    if !backend_healthy() {
        let backend = dir.join("selfhost_server");
        // Backend logs land in the user's log dir for diagnostics.
        let log_path = std::env::var("HOME")
            .map(|home| {
                std::path::Path::new(&home)
                    .join("Library/Logs/PRAW-backend.log")
                    .into_os_string()
            })
            .unwrap_or_default();
        let log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .expect("failed to open backend log file");
        let err = log
            .try_clone()
            .expect("failed to clone backend log file handle");
        let _ = Command::new(&backend)
            .arg("--bind")
            .arg(format!("127.0.0.1:{BACKEND_PORT}"))
            .stdout(log)
            .stderr(err)
            .process_group(0)
            .spawn();
        wait_for_backend(Duration::from_secs(10));
    }

    let gui = dir.join("warp-oss");
    let error = Command::new(gui).exec();
    panic!("failed to exec the GUI: {error}");
}

fn backend_healthy() -> bool {
    TcpStream::connect_timeout(
        &"127.0.0.1:8080".parse().expect("valid socket address"),
        Duration::from_millis(500),
    )
    .is_ok()
}

fn wait_for_backend(timeout: Duration) {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if backend_healthy() {
            return;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}
