//! PARW.app launcher: starts the bundled backend if it isn't already
//! healthy, then replaces this process with the GUI.
//!
//! This is a real Mach-O binary (not a script) because launchd cannot spawn
//! shell scripts as a bundle's `CFBundleExecutable`.

use std::net::TcpStream;
use std::os::unix::process::CommandExt as _;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const BACKEND_PORT: u16 = 8080;

fn main() {
    let dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|parent| parent.to_path_buf()))
        .expect("launcher must live inside the app bundle");

    if !backend_healthy() {
        let backend = dir.join("selfhost_server");
        let _ = Command::new(&backend)
            .arg("--bind")
            .arg(format!("127.0.0.1:{BACKEND_PORT}"))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
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
