//! Warn-log throttling: suppress repeated warning lines keyed by a content
//! signature so a stream of bad requests cannot flood stderr.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Upper bound on tracked keys before stale entries are pruned.
const MAX_KEYS: usize = 1024;

/// Per-key suppression window state.
pub struct WarnThrottle {
    window: Duration,
    seen: Mutex<HashMap<String, Instant>>,
}

/// New throttle that suppresses a key for `window` after each emission.
pub fn new_throttle(window: Duration) -> WarnThrottle {
    WarnThrottle {
        window,
        seen: Mutex::new(HashMap::new()),
    }
}

/// Pure decision step: return whether `key` may emit now, recording the
/// sighting when it does. Prunes expired entries once the map grows past
/// [`MAX_KEYS`].
pub fn allow(
    seen: &mut HashMap<String, Instant>,
    window: Duration,
    key: &str,
    now: Instant,
) -> bool {
    if seen.len() > MAX_KEYS {
        seen.retain(|_, last| now.duration_since(*last) < window);
    }
    match seen.get(key) {
        Some(last) if now.duration_since(*last) < window => false,
        _ => {
            seen.insert(key.to_string(), now);
            true
        }
    }
}

/// Lock the shared map and apply [`allow`] at the current instant.
pub fn warn_allowed(throttle: &WarnThrottle, key: &str) -> bool {
    let mut seen = throttle
        .seen
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    allow(&mut seen, throttle.window, key, Instant::now())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_sight_allowed_immediate_repeat_suppressed() {
        let t0 = Instant::now();
        let mut seen = HashMap::new();
        assert!(allow(&mut seen, Duration::from_secs(60), "k", t0));
        assert!(!allow(&mut seen, Duration::from_secs(60), "k", t0));
        assert!(!allow(
            &mut seen,
            Duration::from_secs(60),
            "k",
            t0 + Duration::from_secs(1)
        ));
    }

    #[test]
    fn distinct_keys_are_independent() {
        let t0 = Instant::now();
        let mut seen = HashMap::new();
        assert!(allow(&mut seen, Duration::from_secs(60), "a", t0));
        assert!(allow(&mut seen, Duration::from_secs(60), "b", t0));
        assert!(!allow(&mut seen, Duration::from_secs(60), "a", t0));
        assert!(!allow(&mut seen, Duration::from_secs(60), "b", t0));
    }

    #[test]
    fn expired_window_allows_again() {
        let t0 = Instant::now();
        let window = Duration::from_secs(60);
        let mut seen = HashMap::new();
        assert!(allow(&mut seen, window, "k", t0));
        let t1 = t0 + window;
        assert!(allow(&mut seen, window, "k", t1));
    }

    #[test]
    fn warn_allowed_uses_shared_state() {
        let throttle = new_throttle(Duration::from_secs(3600));
        let key = "model|missing=[\"x\"]";
        assert!(warn_allowed(&throttle, key));
        assert!(!warn_allowed(&throttle, key));
        assert!(warn_allowed(&throttle, "other"));
    }

    #[test]
    fn oversize_map_is_pruned() {
        let t0 = Instant::now();
        let window = Duration::from_secs(60);
        let mut seen = HashMap::new();
        let stale = t0 - window - Duration::from_secs(1);
        for i in 0..(MAX_KEYS + 2) {
            seen.insert(format!("old-{i}"), stale);
        }
        assert!(seen.len() > MAX_KEYS);

        assert!(allow(&mut seen, window, "fresh", t0));
        assert_eq!(seen.len(), 1, "stale entries pruned, fresh kept");
        assert_eq!(seen.get("fresh"), Some(&t0));
    }
}
