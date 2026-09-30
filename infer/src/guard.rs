//! Admission control for `/predict`: a token-bucket rate limiter plus an
//! in-flight concurrency cap. Pure data records and free functions only;
//! all state flows through arguments and return values.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// Classic token bucket: starts full and refills linearly up to `capacity`.
#[derive(Debug)]
pub struct TokenBucket {
    capacity: f64,
    tokens: f64,
    refill_per_sec: f64,
    last: Instant,
}

/// New bucket that starts full: `burst` tokens available, refilling at
/// `rate_per_sec` tokens per second.
pub fn new_bucket(rate_per_sec: f64, burst: f64, now: Instant) -> TokenBucket {
    TokenBucket {
        capacity: burst,
        tokens: burst,
        refill_per_sec: rate_per_sec,
        last: now,
    }
}

/// Credit the bucket with elapsed time, capped at capacity. `last` only
/// moves forward when time actually advances.
pub fn refill(bucket: &mut TokenBucket, now: Instant) {
    let elapsed = now.saturating_duration_since(bucket.last).as_secs_f64();
    if elapsed > 0.0 {
        bucket.tokens = (bucket.tokens + elapsed * bucket.refill_per_sec).min(bucket.capacity);
        bucket.last = now;
    }
}

/// Refill, then consume `cost` tokens if the bucket holds enough.
pub fn try_acquire(bucket: &mut TokenBucket, now: Instant, cost: f64) -> bool {
    refill(bucket, now);
    if bucket.tokens >= cost {
        bucket.tokens -= cost;
        true
    } else {
        false
    }
}

/// RAII guard that releases one in-flight slot when dropped.
pub struct InflightGuard<'a> {
    counter: &'a AtomicUsize,
}

impl Drop for InflightGuard<'_> {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Count of concurrently executing requests with a hard ceiling.
pub struct Inflight {
    current: AtomicUsize,
    max: usize,
}

/// New in-flight gate admitting at most `max` concurrent requests.
pub fn new_inflight(max: usize) -> Inflight {
    Inflight {
        current: AtomicUsize::new(0),
        max,
    }
}

/// Reserve one in-flight slot; `None` means the gate is full.
pub fn try_enter(gate: &Inflight) -> Option<InflightGuard<'_>> {
    let current = gate.current.fetch_add(1, Ordering::AcqRel);
    if current >= gate.max {
        gate.current.fetch_sub(1, Ordering::AcqRel);
        return None;
    }
    Some(InflightGuard {
        counter: &gate.current,
    })
}

/// Combined `/predict` admission policy. A `None` layer is disabled.
pub struct Admission {
    pub bucket: Option<Mutex<TokenBucket>>,
    pub inflight: Option<Inflight>,
}

/// Why a request was rejected at the door.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionDenied {
    RateLimited,
    Overloaded,
}

/// Build an admission policy from optional limits; disabled layers are `None`.
pub fn admission_from(
    rate: Option<f64>,
    burst: f64,
    max_inflight: Option<usize>,
    now: Instant,
) -> Admission {
    Admission {
        bucket: rate.map(|r| Mutex::new(new_bucket(r, burst, now))),
        inflight: max_inflight.map(new_inflight),
    }
}

/// Consume one rate-limit token, then one in-flight slot.
///
/// A request shed for overload may still have consumed a token. That is
/// intentional conservative accounting: under overload the token level no
/// longer matters, and hiding sheds behind spare tokens would understate
/// load.
pub fn try_admit(
    adm: &Admission,
    now: Instant,
) -> Result<Option<InflightGuard<'_>>, AdmissionDenied> {
    if let Some(bucket) = &adm.bucket {
        let mut bucket = bucket
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !try_acquire(&mut bucket, now, 1.0) {
            return Err(AdmissionDenied::RateLimited);
        }
    }
    match &adm.inflight {
        None => Ok(None),
        Some(gate) => match try_enter(gate) {
            Some(guard) => Ok(Some(guard)),
            None => Err(AdmissionDenied::Overloaded),
        },
    }
}

/// Truthy environment value: `1`, `true`, `yes`, or `on` (trimmed,
/// case-insensitive). Anything else is off.
pub fn parse_bool_flag(raw: &str) -> bool {
    matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Finite float strictly greater than zero; anything else is `None`.
pub fn parse_positive_f64(raw: &str) -> Option<f64> {
    raw.trim()
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite() && *v > 0.0)
}

/// Integer of at least 1; anything else is `None`.
pub fn parse_max(raw: &str) -> Option<usize> {
    raw.trim().parse::<usize>().ok().filter(|v| *v >= 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-9,
            "actual {actual} != expected {expected}"
        );
    }

    #[test]
    fn refill_caps_at_capacity() {
        let t0 = Instant::now();
        let mut bucket = new_bucket(10.0, 5.0, t0);
        bucket.tokens = 0.0;
        refill(&mut bucket, t0 + Duration::from_secs(2));
        assert_close(bucket.tokens, 5.0);
        assert_eq!(bucket.last, t0 + Duration::from_secs(2));
    }

    #[test]
    fn refill_ignores_zero_elapsed() {
        let t0 = Instant::now();
        let mut bucket = new_bucket(10.0, 5.0, t0);
        bucket.tokens = 0.0;
        refill(&mut bucket, t0);
        assert_close(bucket.tokens, 0.0);
        assert_eq!(bucket.last, t0);
    }

    #[test]
    fn drain_then_deny_then_recover() {
        let t0 = Instant::now();
        let mut bucket = new_bucket(2.0, 2.0, t0);
        assert!(try_acquire(&mut bucket, t0, 1.0));
        assert!(try_acquire(&mut bucket, t0, 1.0));
        assert!(!try_acquire(&mut bucket, t0, 1.0), "bucket is drained");

        // half a second at 2/sec refills exactly one token
        let t1 = t0 + Duration::from_millis(500);
        assert!(try_acquire(&mut bucket, t1, 1.0));
        assert!(!try_acquire(&mut bucket, t1, 1.0));
    }

    #[test]
    fn try_acquire_partial_cost_denied() {
        let t0 = Instant::now();
        let mut bucket = new_bucket(0.0, 2.5, t0);
        assert!(try_acquire(&mut bucket, t0, 2.5));
        assert!(!try_acquire(&mut bucket, t0 + Duration::from_secs(10), 1.0));
    }

    #[test]
    fn inflight_cap_admits_drops_and_frees() {
        let gate = new_inflight(2);
        let first = try_enter(&gate).expect("first admit");
        let second = try_enter(&gate).expect("second admit");
        assert!(try_enter(&gate).is_none(), "third admit hits the cap");

        drop(second);
        let third = try_enter(&gate).expect("freed slot reused");
        drop(first);
        drop(third);

        let refilled = try_enter(&gate).expect("all slots free again");
        drop(refilled);
    }

    #[test]
    fn try_admit_without_layers_always_ok() {
        let adm = admission_from(None, 1.0, None, Instant::now());
        assert!(try_admit(&adm, Instant::now()).unwrap().is_none());
    }

    #[test]
    fn try_admit_reports_rate_limited_and_overloaded() {
        let t0 = Instant::now();

        // drained bucket denies with RateLimited even though the gate is open
        let adm = admission_from(Some(1.0), 1.0, Some(4), t0);
        let guard = try_admit(&adm, t0).unwrap().expect("one slot");
        assert!(matches!(
            try_admit(&adm, t0),
            Err(AdmissionDenied::RateLimited)
        ));
        drop(guard);

        // full gate denies with Overloaded while tokens remain
        let adm = admission_from(Some(1.0), 4.0, Some(1), t0);
        let guard = try_admit(&adm, t0).unwrap().expect("one slot");
        assert!(matches!(
            try_admit(&adm, t0),
            Err(AdmissionDenied::Overloaded)
        ));
        drop(guard);
    }

    #[test]
    fn parse_bool_flag_accepts_truthy_spellings() {
        assert!(parse_bool_flag("1"));
        assert!(parse_bool_flag("TRUE"));
        assert!(parse_bool_flag(" yes "));
        assert!(parse_bool_flag("On"));
        assert!(!parse_bool_flag(""));
        assert!(!parse_bool_flag("0"));
        assert!(!parse_bool_flag("off"));
        assert!(!parse_bool_flag("abc"));
    }

    #[test]
    fn parse_positive_f64_accepts_only_finite_positive() {
        assert_eq!(parse_positive_f64(" 10.5 "), Some(10.5));
        assert_eq!(parse_positive_f64("0"), None);
        assert_eq!(parse_positive_f64("-1"), None);
        assert_eq!(parse_positive_f64("abc"), None);
        assert_eq!(parse_positive_f64(""), None);
        assert_eq!(parse_positive_f64("inf"), None);
        assert_eq!(parse_positive_f64("nan"), None);
    }

    #[test]
    fn parse_max_requires_at_least_one() {
        assert_eq!(parse_max("8"), Some(8));
        assert_eq!(parse_max("1"), Some(1));
        assert_eq!(parse_max("0"), None);
        assert_eq!(parse_max("-1"), None);
        assert_eq!(parse_max("abc"), None);
        assert_eq!(parse_max(""), None);
    }
}
