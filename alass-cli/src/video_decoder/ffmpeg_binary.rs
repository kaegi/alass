use failure::{Backtrace, Context, Fail, ResultExt};
use std::ffi::OsString;
use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Child;
use std::process::{ChildStdout, Command, Output, Stdio};
use std::str::from_utf8;

use byteorder::ByteOrder;
use serde::{Deserialize, Deserializer};

use crate::define_error;

#[derive(Debug, PartialEq, Eq)]
pub enum CodecType {
    Audio,
    Video,
    Subtitle,
    Other(String),
}

impl<'de> Deserialize<'de> for CodecType {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        match &s[..] {
            "audio" => Ok(CodecType::Audio),
            "video" => Ok(CodecType::Video),
            "subtitle" => Ok(CodecType::Subtitle),
            s => Ok(CodecType::Other(s.to_owned())),
        }
    }
}

#[derive(Debug, Deserialize)]
struct Stream {
    pub index: usize,
    pub codec_long_name: String,
    pub channels: Option<usize>,
    /// `.mkv` does not store the duration in the streams; we have to use `format -> duration` instead
    pub duration: Option<String>,
    pub codec_type: CodecType,
}

#[derive(Debug, Deserialize)]
struct Format {
    pub duration: Option<String>,
}

/// Metadata associated with a video.
#[derive(Debug, Deserialize)]
struct Metadata {
    streams: Vec<Stream>,
    format: Option<Format>,
}

define_error!(DecoderError, DecoderErrorKind);

#[derive(Debug, Fail)]
pub enum DecoderErrorKind {
    FailedToDecodeVideoStreamInfo,
    ExtractingMetadataFailed {
        cmd_path: PathBuf,
        file_path: PathBuf,
        args: Vec<OsString>,
    },
    NoAudioStream {
        path: PathBuf,
    },
    FailedExtractingAudio {
        file_path: PathBuf,
        cmd_path: PathBuf,
        args: Vec<OsString>,
    },
    FailedSpawningSubprocess {
        path: PathBuf,
        args: Vec<OsString>,
    },
    WaitingForProcessFailed {
        cmd_path: PathBuf,
    },
    ProcessErrorCode {
        cmd_path: PathBuf,
        code: Option<i32>,
    },
    ProcessErrorMessage {
        msg: String,
    },
    DeserializingMetadataFailed {
        path: PathBuf,
    },
    ReadError,
    FailedToParseDuration {
        s: String,
    },
    AudioSegmentProcessingFailed,
    NoDurationInformation,
}

fn format_cmd(cmd_path: &PathBuf, args: &[OsString]) -> String {
    let args_string: String = args
        .iter()
        .map(|x| format!("{}", x.to_string_lossy()))
        .collect::<Vec<String>>()
        .join(" ");
    format!("{} {}", cmd_path.display(), args_string)
}

impl fmt::Display for DecoderErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecoderErrorKind::FailedToDecodeVideoStreamInfo => write!(f, "failed to decode video stream info"),
            DecoderErrorKind::DeserializingMetadataFailed { path } => {
                write!(f, "failed to deserialize metadata of file '{}'", path.display())
            }
            DecoderErrorKind::NoAudioStream { path } => write!(f, "no audio stream in file '{}'", path.display()),
            DecoderErrorKind::FailedExtractingAudio {
                file_path,
                cmd_path,
                args,
            } => write!(
                f,
                "failed to extract audio from '{}' with '{}' ",
                file_path.display(),
                format_cmd(cmd_path, args)
            ),
            DecoderErrorKind::FailedSpawningSubprocess { path, args } => {
                write!(f, "failed to spawn subprocess '{}' ", format_cmd(path, args))
            }
            DecoderErrorKind::WaitingForProcessFailed { cmd_path } => {
                write!(f, "failed to check status of subprocess '{}'", cmd_path.display())
            }
            DecoderErrorKind::ProcessErrorCode { cmd_path, code } => write!(
                f,
                "process '{}' returned error code '{}'",
                cmd_path.display(),
                code.map(|x| x.to_string())
                    .unwrap_or_else(|| String::from("interrupted?"))
            ),
            DecoderErrorKind::ProcessErrorMessage { msg } => write!(f, "stderr: {}", msg),
            DecoderErrorKind::ExtractingMetadataFailed {
                file_path,
                cmd_path,
                args,
            } => write!(
                f,
                "failed to extract metadata from '{}' using command '{}'",
                file_path.display(),
                format_cmd(cmd_path, args)
            ),
            DecoderErrorKind::ReadError => write!(f, "error while reading stdout"),
            DecoderErrorKind::FailedToParseDuration { s } => {
                write!(f, "failed to parse duration string '{}' from metadata", s)
            }
            DecoderErrorKind::AudioSegmentProcessingFailed => write!(f, "processing audio segment failed"),
            DecoderErrorKind::NoDurationInformation => write!(f, "no audio duration information found"),
        }
    }
}

trait IntoOk<T> {
    fn into_ok<I>(self) -> Result<T, I>;
}
impl<T> IntoOk<T> for T {
    fn into_ok<I>(self) -> Result<T, I> {
        Ok(self)
    }
}

pub struct VideoDecoderFFmpegBinary {}

static PROGRESS_PRESCALER: i64 = 200;

impl VideoDecoderFFmpegBinary {
    /// Samples are pushed in 8kHz mono/single-channel format.
    pub fn decode<T>(
        file_path: impl AsRef<Path>,
        receiver: impl super::AudioReceiver<Output = T>,
        mut progress_handler: impl super::ProgressHandler,
    ) -> Result<T, DecoderError> {
        let file_path_buf: PathBuf = file_path.as_ref().into();

        let args = vec![
            OsString::from("-v"),
            OsString::from("error"),
            OsString::from("-show_entries"),
            OsString::from("format=duration:stream=index,codec_long_name,channels,duration,codec_type"),
            OsString::from("-of"),
            OsString::from("json"),
            OsString::from(file_path.as_ref()),
        ];

        let ffprobe_path: PathBuf = std::env::var_os("ALASS_FFPROBE_PATH")
            .unwrap_or(OsString::from("ffprobe"))
            .into();

        let metadata: Metadata =
            Self::get_metadata(file_path_buf.clone(), ffprobe_path.clone(), &args).with_context(|_| {
                DecoderErrorKind::ExtractingMetadataFailed {
                    file_path: file_path_buf.clone(),
                    cmd_path: ffprobe_path.clone(),
                    args: args,
                }
            })?;

        let best_stream_opt: Option<Stream> = metadata
            .streams
            .into_iter()
            .filter(|s| s.codec_type == CodecType::Audio && s.channels.is_some())
            .min_by_key(|s| s.channels.unwrap());

        let best_stream: Stream;
        match best_stream_opt {
            Some(x) => best_stream = x,
            None => {
                return Err(DecoderError::from(DecoderErrorKind::NoAudioStream {
                    path: file_path.as_ref().into(),
                }))
            }
        }

        let ffmpeg_path: PathBuf = std::env::var_os("ALASS_FFMPEG_PATH")
            .unwrap_or(OsString::from("ffmpeg"))
            .into();

        let args: Vec<OsString> = vec![
            // only print errors
            OsString::from("-v"),
            OsString::from("error"),
            // "yes" -> disables user interaction
            OsString::from("-y"),
            // input file
            OsString::from("-i"),
            file_path.as_ref().into(),
            // select stream
            OsString::from("-map"),
            format!("0:{}", best_stream.index).into(),
            // audio codec: 16-bit signed little endian
            OsString::from("-acodec"),
            OsString::from("pcm_s16le"),
            // resample to 8khz
            OsString::from("-ar"),
            OsString::from("8000"),
            // resample to single channel
            OsString::from("-ac"),
            OsString::from("1"),
            // output 16-bit signed little endian stream directly (no wav, etc.)
            OsString::from("-f"),
            OsString::from("s16le"),
            // output to stdout pipe
            OsString::from("-"),
        ];

        let format_opt: Option<Format> = metadata.format;

        // `.mkv` containers do not store duration info in streams, only the format information does contain it
        let duration_str = best_stream
            .duration
            .or_else(|| format_opt.and_then(|format| format.duration))
            .ok_or_else(|| DecoderError::from(DecoderErrorKind::NoDurationInformation))?;

        let duration = duration_str
            .parse::<f64>()
            .with_context(|_| DecoderErrorKind::FailedToParseDuration { s: duration_str })?;

        let num_samples: i64 = (duration * 8000.0) as i64 / PROGRESS_PRESCALER;

        progress_handler.init(num_samples);

        return Self::extract_audio_stream(receiver, progress_handler, ffmpeg_path.clone(), &args)
            .with_context(|_| DecoderErrorKind::FailedExtractingAudio {
                file_path: file_path_buf.clone(),
                cmd_path: ffmpeg_path.clone(),
                args: args,
            })?
            .into_ok();
    }

    fn extract_audio_stream<T>(
        mut receiver: impl super::AudioReceiver<Output = T>,
        mut progress_handler: impl super::ProgressHandler,
        ffmpeg_path: PathBuf,
        args: &[OsString],
    ) -> Result<T, DecoderError> {
        let mut ffmpeg_process: Child = Command::new(ffmpeg_path.clone())
            .args(args)
            .stdin(Stdio::null())
            .stderr(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .with_context(|_| DecoderErrorKind::FailedSpawningSubprocess {
                path: ffmpeg_path.clone(),
                args: args.to_vec(),
            })?;

        let mut stdout: ChildStdout = ffmpeg_process.stdout.take().unwrap();

        enum ParserState {
            Start,
            SingleByte(u8),
        }

        let mut data: Vec<u8> = std::vec::from_elem(0, 200 * 1024 * 1024);
        let data2_cap = 1024 * 1024;
        let mut data2: Vec<i16> = Vec::with_capacity(data2_cap);
        let mut parser_state: ParserState = ParserState::Start;
        let mut progress_prescaler_counter = 0;

        loop {
            // improves performance by allowing ffmpeg to generate more data in pipe
            // TODO: an async tokio read might also have the same effect (without being as machine dependent)
            //  -> too low: does not do anything (+some otherhead)
            //  -> too high: slows down computaton because ffmpeg has to wait for this process to read
            //std::thread::sleep(Duration::from_nanos(1000));

            let read_bytes = stdout.read(&mut data).with_context(|_| DecoderErrorKind::ReadError)?;
            //println!("{}", read_bytes);

            if read_bytes == 0 {
                match ffmpeg_process
                    .wait()
                    .with_context(|_| DecoderErrorKind::WaitingForProcessFailed {
                        cmd_path: ffmpeg_path.clone(),
                    })?
                    .code()
                {
                    Some(0) => {
                        receiver
                            .push_samples(&data2)
                            .with_context(|_| DecoderErrorKind::AudioSegmentProcessingFailed)?;
                        data2.clear();
                        progress_handler.finish();
                        return Ok(receiver
                            .finish()
                            .with_context(|_| DecoderErrorKind::AudioSegmentProcessingFailed)?);
                    }
                    code @ Some(_) | code @ None => {
                        let error_code_err: DecoderErrorKind = DecoderErrorKind::ProcessErrorCode {
                            cmd_path: ffmpeg_path,
                            code: code,
                        };

                        let mut stderr_data = Vec::new();
                        ffmpeg_process
                            .stderr
                            .unwrap()
                            .read_to_end(&mut stderr_data)
                            .with_context(|_| DecoderErrorKind::ReadError)?;

                        let stderr_str: String = String::from_utf8_lossy(&stderr_data).into();

                        if stderr_str.is_empty() {
                            return Err(error_code_err.into());
                        } else {
                            return Err(DecoderError::from(DecoderErrorKind::ProcessErrorMessage {
                                msg: stderr_str,
                            }))
                            .with_context(|_| error_code_err)
                            .map_err(|x| DecoderError::from(x));
                        }
                    }
                }
            }

            for &byte in &data[0..read_bytes] {
                match parser_state {
                    ParserState::Start => parser_state = ParserState::SingleByte(byte),
                    ParserState::SingleByte(last_byte) => {
                        let two_bytes = [last_byte, byte];
                        let sample = byteorder::LittleEndian::read_i16(&two_bytes);
                        receiver
                            .push_samples(&[sample])
                            .with_context(|_| DecoderErrorKind::AudioSegmentProcessingFailed)?;

                        if progress_prescaler_counter == PROGRESS_PRESCALER {
                            progress_handler.inc();
                            progress_prescaler_counter = 0;
                        }

                        progress_prescaler_counter = progress_prescaler_counter + 1;

                        /*data2.push(sample);
                        if data2.len() == data2_cap {
                            receiver.push_samples(&data2);
                            data2.clear();
                        }*/
                        parser_state = ParserState::Start;
                    }
                }
            }
        }
    }

    fn get_metadata(file_path: PathBuf, ffprobe_path: PathBuf, args: &[OsString]) -> Result<Metadata, DecoderError> {
        let ffprobe_process: Output = Command::new(ffprobe_path.clone())
            .args(args)
            .stdin(Stdio::null())
            .stderr(Stdio::piped())
            .stdout(Stdio::piped())
            .output()
            .with_context(|_| DecoderErrorKind::FailedSpawningSubprocess {
                path: ffprobe_path.clone(),
                args: args.to_vec(),
            })?;

        if !ffprobe_process.status.success() {
            let stderr: String = String::from_utf8_lossy(&ffprobe_process.stderr)
                .to_string()
                .trim_end()
                .to_string();

            let err = DecoderErrorKind::ProcessErrorCode {
                cmd_path: ffprobe_path.clone(),
                code: ffprobe_process.status.code(),
            };

            if stderr.is_empty() {
                return Err(DecoderError::from(err));
            } else {
                return Err(DecoderError::from(DecoderErrorKind::ProcessErrorMessage {
                    msg: stderr,
                }))
                .with_context(|_| err)
                .map_err(|x| DecoderError::from(x));
            }
        }

        let stdout =
            from_utf8(&ffprobe_process.stdout).with_context(|_| DecoderErrorKind::FailedToDecodeVideoStreamInfo)?;

        let metadata: Metadata = serde_json::from_str(stdout)
            .with_context(|_| DecoderErrorKind::DeserializingMetadataFailed { path: file_path })?;

        Ok(metadata)
    }
}
