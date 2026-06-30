//! # lume-platform (cross-cutting Platform seam)
//!
//! The single macOS adapter for [`lume_core::Platform`]. Everything OS-specific
//! — AC-power detection, thermal pressure, `~/.lume` paths, FSEvents watching —
//! lives here and *only* here, so porting to Windows/Linux is "add one adapter,"
//! never a sprinkle of `#[cfg(target_os)]` across the codebase (DESIGN §19, §21).

// TODO(M5): MacPlatform impl — IOKit power source, thermal pressure
//           (NSProcessInfo), data_dir = ~/.lume, FSEvents via `notify`.
