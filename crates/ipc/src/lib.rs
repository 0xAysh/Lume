//! # lume-ipc (L2 inference seam)
//!
//! The Rust side of the [`lume_core::Sidecar`] black box: the wire [`protocol`]
//! and (soon) the socket adapter that streams paths to Python and consumes
//! vectors + thumbnails. The Python process owns all decode, preprocess, and
//! embedding; this crate only frames bytes (DESIGN §6, §9).
//!
//! The wire contract is scaffolded now because it is one of the two seams that
//! "must be correct from commit one" (BUILD.md). The transport is not.

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;

use half::f16;
use lume_core::{EmbedOutcome, EmbedUnit, Embedding, LumeError, Sidecar as SidecarTrait};
use serde::{de::DeserializeOwned, Serialize};

use crate::protocol::{
    ClientMessage, EmbedOneRequest, EmbedRequest, EmbedTextRequest, RequestUnit, ServerMessage,
    UnitResult,
};

pub mod protocol;

const MAX_FRAME_BYTES: usize = 32 * 1024 * 1024;

/// Write one length-prefixed JSON frame.
///
/// M0 deliberately keeps the payload inspectable. If binary payload pressure
/// shows up later, this function is the single transport Module to deepen.
pub fn write_frame<W, T>(writer: &mut W, value: &T) -> Result<(), LumeError>
where
    W: Write,
    T: Serialize,
{
    let json = serde_json::to_vec(value).map_err(|e| LumeError::Sidecar(e.to_string()))?;
    if json.len() > MAX_FRAME_BYTES {
        return Err(LumeError::Sidecar(format!(
            "frame too large: {} bytes",
            json.len()
        )));
    }
    writer.write_all(&(json.len() as u32).to_be_bytes())?;
    writer.write_all(&json)?;
    Ok(())
}

/// Read one length-prefixed JSON frame.
pub fn read_frame<R, T>(reader: &mut R) -> Result<T, LumeError>
where
    R: Read,
    T: DeserializeOwned,
{
    let mut len = [0_u8; 4];
    reader.read_exact(&mut len)?;
    let len = u32::from_be_bytes(len) as usize;
    if len > MAX_FRAME_BYTES {
        return Err(LumeError::Sidecar(format!("frame too large: {len} bytes")));
    }

    let mut buf = vec![0_u8; len];
    reader.read_exact(&mut buf)?;
    serde_json::from_slice(&buf).map_err(|e| LumeError::Sidecar(e.to_string()))
}

/// Concrete L2 adapter for the Python Sidecar socket.
///
/// The adapter opens a fresh Unix socket per request for the M0 spike. M1 can
/// make connection pooling/resurrection deeper behind this same interface.
#[derive(Clone, Debug)]
pub struct SocketSidecar {
    socket_path: PathBuf,
}

impl SocketSidecar {
    pub fn new(socket_path: impl Into<PathBuf>) -> Self {
        Self {
            socket_path: socket_path.into(),
        }
    }

    fn request(&self, message: &ClientMessage) -> Result<ServerMessage, LumeError> {
        let mut stream = UnixStream::connect(&self.socket_path)?;
        write_frame(&mut stream, message)?;
        read_frame(&mut stream)
    }
}

impl SidecarTrait for SocketSidecar {
    fn embed(&self, units: &[EmbedUnit]) -> Result<Vec<EmbedOutcome>, LumeError> {
        let req = EmbedRequest {
            batch_id: 0,
            units: units
                .iter()
                .enumerate()
                .map(|(idx, unit)| RequestUnit {
                    unit_idx: idx as u32,
                    path: unit.path.to_string_lossy().into_owned(),
                    frame_ts: unit.frame_ts,
                })
                .collect(),
        };

        match self.request(&ClientMessage::Embed(req))? {
            ServerMessage::EmbedResponse(resp) => {
                let mut outcomes = vec![
                    EmbedOutcome::Failed {
                        reason: "missing sidecar result".into(),
                    };
                    units.len()
                ];
                for item in resp.items {
                    let idx = item.unit_idx as usize;
                    if idx >= outcomes.len() {
                        return Err(LumeError::Sidecar(format!(
                            "sidecar returned out-of-range unit_idx {}",
                            item.unit_idx
                        )));
                    }
                    outcomes[idx] = match item.result {
                        UnitResult::Ok {
                            emb_fp16,
                            thumb_jpeg,
                        } => EmbedOutcome::Ok {
                            emb: embedding_from_fp16_bytes(&emb_fp16)?,
                            thumbnail_jpeg: thumb_jpeg,
                        },
                        UnitResult::Failed { reason } => EmbedOutcome::Failed { reason },
                    };
                }
                Ok(outcomes)
            }
            ServerMessage::Error { message } => Err(LumeError::Sidecar(message)),
            other => Err(LumeError::Sidecar(format!(
                "unexpected sidecar response: {other:?}"
            ))),
        }
    }

    fn embed_one(&self, image: &[u8]) -> Result<Embedding, LumeError> {
        match self.request(&ClientMessage::EmbedOne(EmbedOneRequest {
            image_bytes: image.to_vec(),
        }))? {
            ServerMessage::EmbedOneResponse(resp) => embedding_from_fp16_bytes(&resp.emb_fp16),
            ServerMessage::Error { message } => Err(LumeError::Sidecar(message)),
            other => Err(LumeError::Sidecar(format!(
                "unexpected sidecar response: {other:?}"
            ))),
        }
    }

    fn embed_text(&self, query: &str) -> Result<Embedding, LumeError> {
        match self.request(&ClientMessage::EmbedText(EmbedTextRequest {
            text: query.into(),
        }))? {
            ServerMessage::EmbedOneResponse(resp) => embedding_from_fp16_bytes(&resp.emb_fp16),
            ServerMessage::Error { message } => Err(LumeError::Sidecar(message)),
            other => Err(LumeError::Sidecar(format!(
                "unexpected sidecar response: {other:?}"
            ))),
        }
    }
}

fn embedding_from_fp16_bytes(bytes: &[u8]) -> Result<Embedding, LumeError> {
    if bytes.len() % 2 != 0 {
        return Err(LumeError::Sidecar(format!(
            "fp16 embedding has odd byte length {}",
            bytes.len()
        )));
    }

    Ok(Embedding(
        bytes
            .chunks_exact(2)
            .map(|pair| f16::from_bits(u16::from_le_bytes([pair[0], pair[1]])))
            .collect(),
    ))
}
