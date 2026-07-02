//! # lume-platform (cross-cutting Platform seam)
//!
//! The single macOS adapter for [`lume_core::Platform`]. Everything OS-specific
//! — AC-power detection, thermal pressure, `~/.lume` paths, FSEvents watching —
//! lives here and *only* here, so porting to Windows/Linux is "add one adapter,"
//! never a sprinkle of `#[cfg(target_os)]` across the codebase (DESIGN §19, §21).

use std::{path::PathBuf, sync::Mutex};

use lume_core::{EventSink, FsEvent, LumeError, Platform, ThermalLevel};
use notify::{
    event::{ModifyKind, RenameMode},
    Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher,
};

/// Concrete macOS platform adapter.
///
/// The `notify` [`RecommendedWatcher`] is FSEvents-backed on macOS. Watchers must
/// stay alive for callbacks to continue, so this adapter owns each watcher for
/// the lifetime of the platform value.
#[derive(Default)]
pub struct MacPlatform {
    watchers: Mutex<Vec<RecommendedWatcher>>,
}

impl MacPlatform {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Platform for MacPlatform {
    fn on_ac_power(&self) -> bool {
        // Native IOKit power-source plumbing is outside this watcher slice.
        // Returning true keeps M2 delta watching usable without incorrectly
        // pausing indexing until the M5 power policy work lands.
        true
    }

    fn thermal_pressure(&self) -> ThermalLevel {
        // Native NSProcessInfo thermal-state plumbing lands with the full M5
        // power/thermal policy. This slice only needs a safe nominal default.
        ThermalLevel::Nominal
    }

    fn data_dir(&self) -> PathBuf {
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".lume")
    }

    fn watch(&self, roots: &[PathBuf], sink: EventSink) -> Result<(), LumeError> {
        let mut watcher = RecommendedWatcher::new(
            move |result: notify::Result<Event>| {
                let Ok(event) = result else {
                    return;
                };

                for fs_event in fs_events_from_notify_event(&event) {
                    sink(fs_event);
                }
            },
            Config::default(),
        )
        .map_err(notify_error)?;

        for root in roots {
            watcher
                .watch(root, RecursiveMode::Recursive)
                .map_err(notify_error)?;
        }

        self.watchers
            .lock()
            .map_err(|err| LumeError::Io(format!("platform watcher lock poisoned: {err}")))?
            .push(watcher);

        Ok(())
    }
}

/// Convert a raw `notify` event into Lume's smaller watcher vocabulary.
///
/// `FsEvent` deliberately has no rename variant. Rename pairs therefore become
/// a removal of the old path followed by creation of the new path; downstream
/// reconciliation can use persisted hashes to recognize moves without
/// re-embedding.
pub fn fs_events_from_notify_event(event: &Event) -> Vec<FsEvent> {
    match &event.kind {
        EventKind::Create(_) => event.paths.iter().cloned().map(FsEvent::Created).collect(),
        EventKind::Modify(ModifyKind::Name(RenameMode::From)) => {
            event.paths.iter().cloned().map(FsEvent::Removed).collect()
        }
        EventKind::Modify(ModifyKind::Name(RenameMode::To)) => {
            event.paths.iter().cloned().map(FsEvent::Created).collect()
        }
        EventKind::Modify(ModifyKind::Name(RenameMode::Both)) if event.paths.len() >= 2 => {
            vec![
                FsEvent::Removed(event.paths[0].clone()),
                FsEvent::Created(event.paths[1].clone()),
            ]
        }
        EventKind::Modify(_) => event.paths.iter().cloned().map(FsEvent::Modified).collect(),
        EventKind::Remove(_) => event.paths.iter().cloned().map(FsEvent::Removed).collect(),
        EventKind::Any | EventKind::Other => {
            event.paths.iter().cloned().map(FsEvent::Modified).collect()
        }
        EventKind::Access(_) => Vec::new(),
    }
}

fn notify_error(err: notify::Error) -> LumeError {
    LumeError::Io(format!("platform watcher error: {err}"))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use lume_core::{FsEvent, Platform, ThermalLevel};
    use notify::{
        event::{CreateKind, ModifyKind, RemoveKind, RenameMode},
        Event, EventKind,
    };

    use crate::{fs_events_from_notify_event, MacPlatform};

    #[test]
    fn maps_notify_create_modify_remove_events() {
        let path = PathBuf::from("/tmp/lume/image.jpg");

        let created = fs_events_from_notify_event(
            &Event::new(EventKind::Create(CreateKind::File)).add_path(path.clone()),
        );
        assert_eq!(created, vec![FsEvent::Created(path.clone())]);

        let modified = fs_events_from_notify_event(
            &Event::new(EventKind::Modify(ModifyKind::Data(
                notify::event::DataChange::Content,
            )))
            .add_path(path.clone()),
        );
        assert_eq!(modified, vec![FsEvent::Modified(path.clone())]);

        let removed = fs_events_from_notify_event(
            &Event::new(EventKind::Remove(RemoveKind::File)).add_path(path.clone()),
        );
        assert_eq!(removed, vec![FsEvent::Removed(path)]);
    }

    #[test]
    fn maps_rename_pair_as_remove_then_create() {
        let old_path = PathBuf::from("/tmp/lume/old.jpg");
        let new_path = PathBuf::from("/tmp/lume/new.jpg");

        let events = fs_events_from_notify_event(
            &Event::new(EventKind::Modify(ModifyKind::Name(RenameMode::Both)))
                .add_path(old_path.clone())
                .add_path(new_path.clone()),
        );

        assert_eq!(
            events,
            vec![FsEvent::Removed(old_path), FsEvent::Created(new_path)]
        );
    }

    #[test]
    fn data_dir_uses_home_lume_directory() {
        let platform = MacPlatform::new();
        let old_home = std::env::var_os("HOME");
        std::env::set_var("HOME", "/tmp/lume-home");

        let data_dir = platform.data_dir();

        match old_home {
            Some(value) => std::env::set_var("HOME", value),
            None => std::env::remove_var("HOME"),
        }

        assert_eq!(data_dir, PathBuf::from("/tmp/lume-home/.lume"));
    }

    #[test]
    fn conservative_power_and_thermal_defaults_are_available() {
        let platform = MacPlatform::new();

        assert!(platform.on_ac_power());
        assert_eq!(platform.thermal_pressure(), ThermalLevel::Nominal);
    }
}
