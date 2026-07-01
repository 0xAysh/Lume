//! Observable indexing progress (M1 Slice 3 acceptance: "progress is
//! observable during a run, not just after completion"). A plain atomic
//! counter pair is enough — no channel/callback machinery earns its keep yet.

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Default, Debug)]
pub struct Progress {
    done: AtomicU64,
    total: AtomicU64,
}

impl Progress {
    pub fn set_total(&self, total: u64) {
        self.total.store(total, Ordering::Relaxed);
    }

    pub fn add_done(&self, n: u64) {
        self.done.fetch_add(n, Ordering::Relaxed);
    }

    /// `(done, total)` as of this call — safe to poll from another thread
    /// while `Indexer::run` is in flight.
    pub fn snapshot(&self) -> (u64, u64) {
        (
            self.done.load(Ordering::Relaxed),
            self.total.load(Ordering::Relaxed),
        )
    }
}
