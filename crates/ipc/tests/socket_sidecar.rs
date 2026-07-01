use std::io::Read;
use std::os::unix::net::UnixListener;
use std::path::PathBuf;
use std::thread;

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
