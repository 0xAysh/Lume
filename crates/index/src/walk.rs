//! Recursive folder walk, filtered to supported still-image Files. DESIGN §5's
//! basename-pairing pass, FSEvents, and reconcile are separate M2 slices.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use walkdir::WalkDir;

/// Recursively find every supported still-image file under `root`
/// (case-insensitive extension match), sorted by path for deterministic
/// batching. Everything else — other extensions, directories, symlink loops
/// `walkdir` already guards against — is silently skipped.
pub fn walk_folder(root: &Path) -> Vec<PathBuf> {
    let paths: Vec<PathBuf> = WalkDir::new(root)
        .into_iter()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.into_path())
        .filter(|path| is_supported_image(path))
        .collect();
    primary_items(paths)
}

fn primary_items(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut display_groups: BTreeMap<(PathBuf, String), BTreeSet<PathBuf>> = BTreeMap::new();
    for path in &paths {
        if is_display_image(path) {
            display_groups
                .entry(group_key(path))
                .or_default()
                .insert(path.clone());
        }
    }

    let mut primaries = Vec::new();
    for path in paths {
        if is_raw_image(&path) && display_groups.contains_key(&group_key(&path)) {
            continue;
        }
        primaries.push(path);
    }
    primaries.sort();
    primaries
}

fn is_supported_image(path: &Path) -> bool {
    is_display_image(path) || is_raw_image(path)
}

fn is_display_image(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "jpg" | "jpeg" | "png" | "heic" | "heif" | "heics" | "heifs" | "hif"
    )
}

fn is_raw_image(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "3fr"
            | "arw"
            | "cr2"
            | "cr3"
            | "dcr"
            | "dng"
            | "erf"
            | "fff"
            | "gpr"
            | "iiq"
            | "k25"
            | "kdc"
            | "mrw"
            | "nef"
            | "nrw"
            | "orf"
            | "pef"
            | "raf"
            | "raw"
            | "rw2"
            | "sr2"
            | "srf"
            | "srw"
            | "x3f"
    )
}

fn group_key(path: &Path) -> (PathBuf, String) {
    (
        path.parent().unwrap_or_else(|| Path::new("")).to_path_buf(),
        path.file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or_default()
            .to_string(),
    )
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
        fs::write(root.join("apple.HEIC"), b"").unwrap();
        fs::write(root.join("raw.CR2"), b"").unwrap();
        fs::write(root.join("negative.dng"), b"").unwrap();
        fs::write(root.join("d.gif"), b"").unwrap();
        fs::write(root.join("notes.txt"), b"").unwrap();
        fs::create_dir(root.join("sub")).unwrap();
        fs::write(root.join("sub/e.PNG"), b"").unwrap();

        let found = walk_folder(root);
        let names: Vec<_> = found
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();

        assert_eq!(names.len(), 7);
        assert!(names.contains(&"a.jpg".to_string()));
        assert!(names.contains(&"B.JPEG".to_string()));
        assert!(names.contains(&"c.png".to_string()));
        assert!(names.contains(&"apple.HEIC".to_string()));
        assert!(names.contains(&"raw.CR2".to_string()));
        assert!(names.contains(&"negative.dng".to_string()));
        assert!(names.contains(&"e.PNG".to_string()));
    }

    #[test]
    fn empty_folder_yields_no_paths() {
        let dir = tempfile::tempdir().unwrap();
        assert!(walk_folder(dir.path()).is_empty());
    }

    #[test]
    fn raw_display_pair_returns_display_item_only() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("IMG_0001.CR2"), b"").unwrap();
        fs::write(root.join("IMG_0001.JPG"), b"").unwrap();
        fs::write(root.join("RAW_ONLY.NEF"), b"").unwrap();

        let found = walk_folder(root);
        let names: Vec<_> = found
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();

        assert_eq!(names.len(), 2);
        assert!(names.contains(&"IMG_0001.JPG".to_string()));
        assert!(names.contains(&"RAW_ONLY.NEF".to_string()));
        assert!(!names.contains(&"IMG_0001.CR2".to_string()));
    }

    #[test]
    fn live_photo_pair_returns_still_item_only() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("IMG_0002.HEIC"), b"").unwrap();
        fs::write(root.join("IMG_0002.MOV"), b"").unwrap();

        let found = walk_folder(root);
        let names: Vec<_> = found
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();

        assert_eq!(names, vec!["IMG_0002.HEIC".to_string()]);
    }

    #[test]
    fn display_files_and_suffix_variants_stay_separate() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("IMG_0003.jpg"), b"").unwrap();
        fs::write(root.join("IMG_0003.png"), b"").unwrap();
        fs::write(root.join("IMG_0004.CR2"), b"").unwrap();
        fs::write(root.join("IMG_0004_edited.JPG"), b"").unwrap();
        fs::write(root.join("IMG_0005.CR2"), b"").unwrap();
        fs::write(root.join("IMG_0005-2.JPG"), b"").unwrap();

        let found = walk_folder(root);
        let names: Vec<_> = found
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();

        assert_eq!(names.len(), 6);
        assert!(names.contains(&"IMG_0003.jpg".to_string()));
        assert!(names.contains(&"IMG_0003.png".to_string()));
        assert!(names.contains(&"IMG_0004.CR2".to_string()));
        assert!(names.contains(&"IMG_0004_edited.JPG".to_string()));
        assert!(names.contains(&"IMG_0005.CR2".to_string()));
        assert!(names.contains(&"IMG_0005-2.JPG".to_string()));
    }
}
