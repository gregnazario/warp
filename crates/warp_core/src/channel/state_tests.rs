use super::{ChannelState, derive_http_origin_from_ws_url};

/// Tests that mutate the global channel state hold this lock so they don't
/// interfere with each other.
static STATE_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

#[test]
fn wss_becomes_https_and_strips_path() {
    let got = derive_http_origin_from_ws_url("wss://rtc.app.warp.dev/graphql/v2");
    assert_eq!(got.as_deref(), Some("https://rtc.app.warp.dev"));
}

#[test]
fn ws_becomes_http_and_preserves_port() {
    let got = derive_http_origin_from_ws_url("ws://localhost:8080/graphql/v2");
    assert_eq!(got.as_deref(), Some("http://localhost:8080"));
}

#[test]
fn unparseable_input_returns_none() {
    assert!(derive_http_origin_from_ws_url("not a url").is_none());
    assert!(derive_http_origin_from_ws_url("https://app.warp.dev").is_none());
}

#[test]
fn self_hosted_mode_redirects_all_server_urls_and_disables_external_services() {
    let _guard = STATE_LOCK.lock();
    ChannelState::enable_self_hosted("http://127.0.0.1:8080".to_owned()).unwrap();

    assert!(ChannelState::is_self_hosted());
    assert_eq!(ChannelState::server_root_url(), "http://127.0.0.1:8080");
    assert_eq!(
        ChannelState::ws_server_url(),
        "ws://127.0.0.1:8080/graphql/v2"
    );
    assert_eq!(ChannelState::rtc_http_url(), "http://127.0.0.1:8080");
    assert_eq!(
        ChannelState::session_sharing_server_url().as_deref(),
        Some("ws://127.0.0.1:8080")
    );
    assert_eq!(ChannelState::oz_root_url(), "http://127.0.0.1:8080");
    assert!(!ChannelState::is_telemetry_available());
    assert!(!ChannelState::is_crash_reporting_available());
    assert_eq!(ChannelState::releases_base_url(), "");
    assert_eq!(ChannelState::firebase_api_key(), "");

    // Restore production state for other tests in this binary.
    ChannelState::set(ChannelState::init());
}

#[test]
fn self_hosted_mode_rejects_non_http_scheme() {
    let _guard = STATE_LOCK.lock();
    assert!(ChannelState::enable_self_hosted("ftp://127.0.0.1:8080".to_owned()).is_err());
    assert!(!ChannelState::is_self_hosted());
    ChannelState::set(ChannelState::init());
}

#[test]
fn self_hosted_mode_rejects_unparseable_url() {
    let _guard = STATE_LOCK.lock();
    assert!(ChannelState::enable_self_hosted("not a url".to_owned()).is_err());
    assert!(!ChannelState::is_self_hosted());
    ChannelState::set(ChannelState::init());
}
