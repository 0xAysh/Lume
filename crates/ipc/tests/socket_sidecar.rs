use std::io::Read;
use std::os::unix::net::UnixListener;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

use half::f16;
use lume_core::{EmbedUnit, Embedding, Sidecar};
use lume_ipc::protocol::{EmbedOneResponse, ServerMessage};
use lume_ipc::{read_frame, write_frame, SocketSidecar};

fn temp_socket(name: &str) -> PathBuf {
    let mut path = PathBuf::from("/tmp");
    path.push(format!("lume-{name}-{}.sock", std::process::id()));
    let _ = std::fs::remove_file(&path);
    path
}

#[test]
fn length_prefixed_json_frame_round_trips() {
    let (mut left, mut right) = std::os::unix::net::UnixStream::pair().unwrap();
    let sent = ServerMessage::EmbedOneResponse(EmbedOneResponse {
        emb_fp16: vec![0, 60, 0, 64],
    });

    write_frame(&mut left, &sent).unwrap();
    let received: ServerMessage = read_frame(&mut right).unwrap();

    assert_eq!(received, sent);
}

#[test]
fn socket_sidecar_embeds_text_over_unix_socket() {
    let socket = temp_socket("embed-text");
    let listener = UnixListener::bind(&socket).unwrap();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut len = [0_u8; 4];
        stream.read_exact(&mut len).unwrap();
        let frame_len = u32::from_be_bytes(len) as usize;
        let mut buf = vec![0_u8; frame_len];
        stream.read_exact(&mut buf).unwrap();
        assert!(String::from_utf8(buf).unwrap().contains("\"embed_text\""));

        write_frame(
            &mut stream,
            &ServerMessage::EmbedOneResponse(EmbedOneResponse {
                emb_fp16: vec![0, 60, 0, 64],
            }),
        )
        .unwrap();
    });

    let emb = SocketSidecar::new(socket.clone(), 400)
        .embed_text("girl riding a bicycle")
        .unwrap();

    assert_eq!(emb, Embedding(vec![f16::from_f32(1.0), f16::from_f32(2.0)]));
    server.join().unwrap();
    let _ = std::fs::remove_file(socket);
}

#[test]
fn socket_sidecar_waits_for_the_server_to_bind() {
    let socket = temp_socket("startup-race");
    let server_socket = socket.clone();
    let server = thread::spawn(move || {
        thread::sleep(Duration::from_millis(75));
        let listener = UnixListener::bind(&server_socket).unwrap();
        let (mut stream, _) = listener.accept().unwrap();
        let _: lume_ipc::protocol::ClientMessage = read_frame(&mut stream).unwrap();

        write_frame(
            &mut stream,
            &ServerMessage::EmbedOneResponse(EmbedOneResponse {
                emb_fp16: vec![0, 60],
            }),
        )
        .unwrap();
    });

    let emb = SocketSidecar::new(socket.clone(), 400)
        .embed_text("startup race")
        .unwrap();

    assert_eq!(emb, Embedding(vec![f16::from_f32(1.0)]));
    server.join().unwrap();
    let _ = std::fs::remove_file(socket);
}

#[test]
fn socket_sidecar_realigns_batch_results_by_unit_index() {
    let socket = temp_socket("embed-batch");
    let listener = UnixListener::bind(&socket).unwrap();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let _: lume_ipc::protocol::ClientMessage = read_frame(&mut stream).unwrap();
        write_frame(
            &mut stream,
            &ServerMessage::EmbedResponse(lume_ipc::protocol::EmbedResponse {
                batch_id: 0,
                items: vec![
                    lume_ipc::protocol::BatchItem {
                        unit_idx: 1,
                        result: lume_ipc::protocol::UnitResult::Ok {
                            emb_fp16: vec![0, 64],
                            thumb_jpeg: vec![0xFF, 0xD8],
                        },
                    },
                    lume_ipc::protocol::BatchItem {
                        unit_idx: 0,
                        result: lume_ipc::protocol::UnitResult::Ok {
                            emb_fp16: vec![0, 60],
                            thumb_jpeg: vec![0xFF, 0xD8, 0xFF],
                        },
                    },
                ],
            }),
        )
        .unwrap();
    });

    let outcomes = SocketSidecar::new(socket.clone(), 400)
        .embed(&[
            EmbedUnit {
                file: 10,
                path: "/tmp/a.jpg".into(),
                frame_ts: None,
            },
            EmbedUnit {
                file: 11,
                path: "/tmp/b.jpg".into(),
                frame_ts: None,
            },
        ])
        .unwrap();

    match &outcomes[0] {
        lume_core::EmbedOutcome::Ok {
            emb,
            thumbnail_jpeg,
        } => {
            assert_eq!(emb, &Embedding(vec![f16::from_f32(1.0)]));
            assert_eq!(thumbnail_jpeg, &[0xFF, 0xD8, 0xFF]);
        }
        _ => panic!("expected ok outcome"),
    }
    match &outcomes[1] {
        lume_core::EmbedOutcome::Ok { emb, .. } => {
            assert_eq!(emb, &Embedding(vec![f16::from_f32(2.0)]));
        }
        _ => panic!("expected ok outcome"),
    }

    server.join().unwrap();
    let _ = std::fs::remove_file(socket);
}

#[test]
fn socket_sidecar_realigns_out_of_order_ok_and_failed_batch_results() {
    let socket = temp_socket("embed-batch-mixed");
    let listener = UnixListener::bind(&socket).unwrap();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let _: lume_ipc::protocol::ClientMessage = read_frame(&mut stream).unwrap();
        write_frame(
            &mut stream,
            &ServerMessage::EmbedResponse(lume_ipc::protocol::EmbedResponse {
                batch_id: 0,
                items: vec![
                    lume_ipc::protocol::BatchItem {
                        unit_idx: 2,
                        result: lume_ipc::protocol::UnitResult::Ok {
                            emb_fp16: vec![0, 66],
                            thumb_jpeg: vec![0xFF, 0xD8, 0x02],
                        },
                    },
                    lume_ipc::protocol::BatchItem {
                        unit_idx: 0,
                        result: lume_ipc::protocol::UnitResult::Ok {
                            emb_fp16: vec![0, 60],
                            thumb_jpeg: vec![0xFF, 0xD8, 0x00],
                        },
                    },
                    lume_ipc::protocol::BatchItem {
                        unit_idx: 1,
                        result: lume_ipc::protocol::UnitResult::Failed {
                            reason: "decode failed after out-of-order completion".into(),
                        },
                    },
                ],
            }),
        )
        .unwrap();
    });

    let outcomes = SocketSidecar::new(socket.clone(), 400)
        .embed(&[
            EmbedUnit {
                file: 10,
                path: "/tmp/a.jpg".into(),
                frame_ts: None,
            },
            EmbedUnit {
                file: 11,
                path: "/tmp/b.jpg".into(),
                frame_ts: None,
            },
            EmbedUnit {
                file: 12,
                path: "/tmp/c.jpg".into(),
                frame_ts: None,
            },
        ])
        .unwrap();

    assert!(matches!(
        &outcomes[0],
        lume_core::EmbedOutcome::Ok {
            emb,
            thumbnail_jpeg
        } if emb == &Embedding(vec![f16::from_f32(1.0)]) && thumbnail_jpeg == &[0xFF, 0xD8, 0x00]
    ));
    assert!(matches!(
        &outcomes[1],
        lume_core::EmbedOutcome::Failed { reason } if reason.contains("decode failed")
    ));
    assert!(matches!(
        &outcomes[2],
        lume_core::EmbedOutcome::Ok { emb, .. } if emb == &Embedding(vec![f16::from_f32(3.0)])
    ));

    server.join().unwrap();
    let _ = std::fs::remove_file(socket);
}
