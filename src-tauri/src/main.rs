// Prevents an extra console window on Windows in release (Tauri convention).
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    lume_app_lib::run();
}
