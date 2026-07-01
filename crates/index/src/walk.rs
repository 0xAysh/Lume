//! Recursive folder walk, filtered to the M1 media types (DESIGN §5's
//! basename-pairing pass, FSEvents, and reconcile are all M2 — this is just
//! "find the JPEG/PNG files under a root").

use std::path::{Path, PathBuf};

use walkdir::WalkDir;

/// Recursively find every `.jpg`/`.jpeg`/`.png` file under `root`
/// (case-insensitive extension match), sorted by path for deterministic
/// batching. Everything else — other extensions, directories, symlink loops
/// `walkdir` already guards against — is silently skipped.
pub fn walk_folder(root: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = WalkDir::new(root)
        .into_iter()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.into_path())
        .filter(|path| is_supported_image(path))
        .collect();
    paths.sort();
    paths
}

fn is_supported_image(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn finds_supported_images_case_insensitively_and_ignores_the_rest() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("a.jpg"), b"").unwrap();
        fs::write(root.join("B.JPEG"), b"").unwrap();
        fs::write(root.join("c.png"), b"").unwrap();
        fs::write(root.join("d.gif"), b"").unwrap();
        fs::write(root.join("notes.txt"), b"").unwrap();
        fs::create_dir(root.join("sub")).unwrap();
        fs::write(root.join("sub/e.PNG"), b"").unwrap();

        let found = walk_folder(root);
        let names: Vec<_> = found
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();

        assert_eq!(names.len(), 4);
        assert!(names.contains(&"a.jpg".to_string()));
        assert!(names.contains(&"B.JPEG".to_string()));
        assert!(names.contains(&"c.png".to_string()));
        assert!(names.contains(&"e.PNG".to_string()));
    }

    #[test]
    fn empty_folder_yields_no_paths() {
        let dir = tempfile::tempdir().unwrap();
        assert!(walk_folder(dir.path()).is_empty());
    }
}
