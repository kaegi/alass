// This file is part of the Rust library and binary `alass`.
//
// Copyright (C) 2017 kaegi
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#![allow(unknown_lints)] // for clippy

// TODO: search for unsafe, panic, unimplemented

extern crate clap;
extern crate encoding_rs;
extern crate pbr;
extern crate subparse;

const PKG_VERSION: Option<&'static str> = option_env!("CARGO_PKG_VERSION");
const PKG_NAME: Option<&'static str> = option_env!("CARGO_PKG_NAME");
const PKG_DESCRIPTION: Option<&'static str> = option_env!("CARGO_PKG_DESCRIPTION");

// Alg* stands for algorithm (the internal alass algorithm types)

use crate::subparse::SubtitleFileInterface;

use alass_core::{align, Statistics, TimeDelta as AlgTimeDelta, TimePoint as AlgTimePoint, TimeSpan as AlgTimeSpan};
use clap::{App, Arg};
use encoding_rs::Encoding;
use encoding_rs::UTF_8;
use failure::ResultExt;
use pbr::ProgressBar;
use std::cmp::{max, min};
use std::ffi::OsStr;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::result::Result;
use std::str::FromStr;

mod video_decoder;

mod errors;
use errors::*;

// subparse
use subparse::timetypes::*;
use subparse::{get_subtitle_format_err, parse_bytes, SubtitleEntry, SubtitleFile, SubtitleFormat};

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
enum VideoFileFormat {
    /// we don't need to differentiate between video file formats in current code
    NotImplemented,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
enum InputFileFormat {
    Subtitle(subparse::SubtitleFormat),
    Video(VideoFileFormat),
}

impl InputFileFormat {
    fn subtitle_format(&self) -> Option<subparse::SubtitleFormat> {
        match self {
            InputFileFormat::Subtitle(f) => Some(*f),
            _ => None,
        }
    }
}

struct ProgressInfo {
    prescaler: i64,
    counter: i64,
    progress_bar: Option<ProgressBar<std::io::Stdout>>,
}

impl ProgressInfo {
    fn new(prescaler: i64) -> ProgressInfo {
        ProgressInfo {
            prescaler,
            counter: 0,
            progress_bar: None,
        }
    }
}

impl alass_core::ProgressHandler for ProgressInfo {
    fn init(&mut self, steps: i64) {
        self.progress_bar = Some(ProgressBar::new(steps as u64));
    }
    fn inc(&mut self) {
        self.counter = self.counter + 1;
        if self.counter == self.prescaler {
            self.progress_bar.as_mut().unwrap().inc();
            self.counter = 0;
        }
    }
    fn finish(&mut self) {
        self.progress_bar.as_mut().unwrap().finish_println("\n");
    }
}

impl video_decoder::ProgressHandler for ProgressInfo {
    fn init(&mut self, steps: i64) {
        self.progress_bar = Some(ProgressBar::new((steps / self.prescaler) as u64));
    }
    fn inc(&mut self) {
        self.counter = self.counter + 1;
        if self.counter == self.prescaler {
            self.progress_bar.as_mut().unwrap().inc();
            self.counter = 0;
        }
    }
    fn finish(&mut self) {
        self.progress_bar.as_mut().unwrap().finish_println("\n");
    }
}

fn read_file_to_bytes(path: &Path) -> std::result::Result<Vec<u8>, FileOperationError> {
    let mut file = File::open(path).with_context(|_| FileOperationErrorKind::FileOpen {
        path: path.to_path_buf(),
    })?;
    let mut v = Vec::new();
    file.read_to_end(&mut v)
        .with_context(|_| FileOperationErrorKind::FileRead {
            path: path.to_path_buf(),
        })?;
    Ok(v)
}

fn write_data_to_file(path: &Path, d: Vec<u8>) -> std::result::Result<(), FileOperationError> {
    let mut file = File::create(path).with_context(|_| FileOperationErrorKind::FileOpen {
        path: path.to_path_buf(),
    })?;
    file.write_all(&d).with_context(|_| FileOperationErrorKind::FileWrite {
        path: path.to_path_buf(),
    })?;
    Ok(())
}

fn timing_to_alg_timepoint(t: TimePoint, interval: i64) -> AlgTimePoint {
    assert!(interval > 0);
    AlgTimePoint::from(t.msecs() / interval)
}

fn alg_delta_to_delta(t: AlgTimeDelta, interval: i64) -> TimeDelta {
    assert!(interval > 0);
    let time_int: i64 = t.into();
    TimeDelta::from_msecs(time_int * interval)
}

fn timings_to_alg_timespans(v: &[TimeSpan], interval: i64) -> Vec<AlgTimeSpan> {
    v.iter()
        .cloned()
        .map(|timespan| {
            AlgTimeSpan::new_safe(
                timing_to_alg_timepoint(timespan.start, interval),
                timing_to_alg_timepoint(timespan.end, interval),
            )
        })
        .collect()
}
fn alg_deltas_to_timing_deltas(v: &[AlgTimeDelta], interval: i64) -> Vec<TimeDelta> {
    v.iter().cloned().map(|x| alg_delta_to_delta(x, interval)).collect()
}

/// Groups consecutive timespans with the same delta together.
fn get_subtitle_delta_groups(mut v: Vec<(AlgTimeDelta, TimeSpan)>) -> Vec<(AlgTimeDelta, Vec<TimeSpan>)> {
    v.sort_by_key(|t| min((t.1).start, (t.1).end));

    let mut result: Vec<(AlgTimeDelta, Vec<TimeSpan>)> = Vec::new();

    for (delta, original_timespan) in v {
        let mut new_block = false;

        if let Some(last_tuple_ref) = result.last_mut() {
            if delta == last_tuple_ref.0 {
                last_tuple_ref.1.push(original_timespan);
            } else {
                new_block = true;
            }
        } else {
            new_block = true;
        }

        if new_block {
            result.push((delta, vec![original_timespan]));
        }
    }

    result
}

/// Does reading, parsing and nice error handling for a f64 clap parameter.
fn unpack_clap_number_f64(
    matches: &clap::ArgMatches,
    parameter_name: &'static str,
) -> Result<f64, InputArgumentsError> {
    let paramter_value_str: &str = matches.value_of(parameter_name).unwrap();
    f64::from_str(paramter_value_str)
        .with_context(|_| {
            InputArgumentsErrorKind::ArgumentParseError {
                argument_name: parameter_name.to_string(),
                value: paramter_value_str.to_string(),
            }
            .into()
        })
        .map_err(|e| InputArgumentsError::from(e))
}

/// Does reading, parsing and nice error handling for a f64 clap parameter.
fn unpack_clap_number_i64(
    matches: &clap::ArgMatches,
    parameter_name: &'static str,
) -> Result<i64, InputArgumentsError> {
    let paramter_value_str: &str = matches.value_of(parameter_name).unwrap();
    i64::from_str(paramter_value_str)
        .with_context(|_| {
            InputArgumentsErrorKind::ArgumentParseError {
                argument_name: parameter_name.to_string(),
                value: paramter_value_str.to_string(),
            }
            .into()
        })
        .map_err(|e| InputArgumentsError::from(e))
}

fn get_encoding(opt: Option<&str>) -> &'static Encoding {
    match opt {
        None => UTF_8,
        Some(label) => {
            match Encoding::for_label(label.as_bytes()) {
                None => {
                    panic!("{} is not a known encoding label; exiting.", label); // TODO: error handling
                }
                Some(encoding) => encoding,
            }
        }
    }
}

struct InputFileHandler {
    subtitle_file: Option<SubtitleFile>,
    file_format: InputFileFormat,
    subparse_timespans: Vec<subparse::timetypes::TimeSpan>,
    aligner_timespans: Vec<alass_core::TimeSpan>,
}

impl InputFileHandler {
    pub fn open(
        file_path: &Path,
        interval: i64,
        sub_encoding: &'static Encoding,
        sub_fps: f64,
    ) -> Result<InputFileHandler, InputFileError> {
        let known_subitle_endings: [&str; 6] = ["srt", "vob", "idx", "ass", "ssa", "sub"];

        let extension: Option<&OsStr> = file_path.extension();

        for subtitle_ending in known_subitle_endings.into_iter() {
            if extension == Some(OsStr::new(subtitle_ending)) {
                return Ok(Self::open_sub_file(file_path, interval, sub_encoding, sub_fps)
                    .with_context(|_| InputFileErrorKind::SubtitleFile(file_path.to_path_buf()))?);
            }
        }

        return Ok(Self::open_video_file(file_path, interval)
            .with_context(|_| InputFileErrorKind::VideoFile(file_path.to_path_buf()))?);
    }

    pub fn open_video_file(file_path: &Path, interval: i64) -> Result<InputFileHandler, InputVideoError> {
        //video_decoder::VideoDecoder::decode(file_path, );
        use webrtc_vad::*;

        struct WebRtcFvad {
            fvad: Vad,
            vad_buffer: Vec<bool>,
        }

        impl video_decoder::AudioReceiver for WebRtcFvad {
            type Output = Vec<bool>;
            type Error = InputVideoError;

            fn push_samples(&mut self, samples: &[i16]) -> Result<(), InputVideoError> {
                // the chunked audio receiver should only provide 10ms of 8000kHz -> 80 samples
                assert!(samples.len() == 80);

                let is_voice = self
                    .fvad
                    .is_voice_segment(samples)
                    .map_err(|_| InputVideoErrorKind::VadAnalysisFailed)?;

                self.vad_buffer.push(is_voice);

                Ok(())
            }

            fn finish(self) -> Result<Vec<bool>, InputVideoError> {
                Ok(self.vad_buffer)
            }
        }

        let vad_processor = WebRtcFvad {
            fvad: Vad::new(8000).map_err(|_| InputVideoErrorKind::VadCreationFailed)?,
            vad_buffer: Vec::new(),
        };

        let chunk_processor = video_decoder::ChunkedAudioReceiver::new(80, vad_processor);

        println!("extracting audio from reference file '{}'...", file_path.display());
        let vad_buffer = video_decoder::VideoDecoder::decode(file_path, chunk_processor, ProgressInfo::new(500))
            .with_context(|_| InputVideoErrorKind::FailedToDecode {
                path: PathBuf::from(file_path),
            })?;

        let mut voice_segments: Vec<(i64, i64)> = Vec::new();
        let mut voice_segment_start: i64 = 0;
        let mut last_was_voice_segment = false;

        for (i, is_voice_segment) in vad_buffer.into_iter().chain(std::iter::once(false)).enumerate() {
            match (last_was_voice_segment, is_voice_segment) {
                (false, false) | (true, true) => {}
                (false, true) => {
                    voice_segment_start = i as i64;
                }
                (true, false) => {
                    voice_segments.push((voice_segment_start, i as i64 - 1));
                }
            }

            last_was_voice_segment = is_voice_segment;
        }

        let min_span_length_ms = 200;

        let subparse_timespans: Vec<subparse::timetypes::TimeSpan> = voice_segments
            .into_iter()
            .filter(|&(start, end)| start + min_span_length_ms / 10 < end)
            .map(|(start, end)| {
                subparse::timetypes::TimeSpan::new(
                    subparse::timetypes::TimePoint::from_msecs(start * 10),
                    subparse::timetypes::TimePoint::from_msecs(end * 10),
                )
            })
            .collect();

        let aligner_timespans: Vec<alass_core::TimeSpan> = timings_to_alg_timespans(&subparse_timespans, interval);

        Ok(InputFileHandler {
            file_format: InputFileFormat::Video(VideoFileFormat::NotImplemented),
            subparse_timespans,
            aligner_timespans,
            subtitle_file: None,
        })
    }

    pub fn open_sub_file(
        file_path: &Path,
        interval: i64,
        sub_encoding: &'static Encoding,
        sub_fps: f64,
    ) -> Result<InputFileHandler, InputSubtitleError> {
        let sub_data = read_file_to_bytes(file_path.as_ref())
            .with_context(|_| InputSubtitleErrorKind::ReadingSubtitleFileFailed(file_path.to_path_buf()))?;

        let file_format = get_subtitle_format_err(file_path.extension(), &sub_data)
            .with_context(|_| InputSubtitleErrorKind::UnknownSubtitleFormat(file_path.to_path_buf()))?;

        let parsed_subtitle_data: SubtitleFile = parse_bytes(file_format, &sub_data, sub_encoding, sub_fps)
            .with_context(|_| InputSubtitleErrorKind::ParsingSubtitleFailed(file_path.to_path_buf()))?;

        let subparse_timespans: Vec<subparse::timetypes::TimeSpan> = parsed_subtitle_data
            .get_subtitle_entries()
            .with_context(|_| InputSubtitleErrorKind::RetreivingSubtitleLinesFailed(file_path.to_path_buf()))?
            .into_iter()
            .map(|subentry| subentry.timespan)
            .map(|timespan: subparse::timetypes::TimeSpan| {
                TimeSpan::new(min(timespan.start, timespan.end), max(timespan.start, timespan.end))
            })
            .collect();

        let aligner_timespans: Vec<alass_core::TimeSpan> = timings_to_alg_timespans(&subparse_timespans, interval);

        Ok(InputFileHandler {
            file_format: InputFileFormat::Subtitle(file_format),
            subparse_timespans,
            aligner_timespans,
            subtitle_file: Some(parsed_subtitle_data),
        })
    }

    pub fn into_subtitle_file(self) -> Option<SubtitleFile> {
        self.subtitle_file
    }

    pub fn subparse_timespans(&self) -> &[subparse::timetypes::TimeSpan] {
        self.subparse_timespans.as_slice()
    }

    pub fn aligner_timespans(&self) -> &[alass_core::TimeSpan] {
        self.aligner_timespans.as_slice()
    }

    pub fn file_format(&self) -> InputFileFormat {
        self.file_format
    }
}

// //////////////////////////////////////////////////////////////////////////////////////////////////

struct Arguments {
    reference_file_path: PathBuf,
    incorrect_file_path: PathBuf,
    output_file_path: PathBuf,

    statistics_folder_path_opt: Option<PathBuf>,
    statistics_required_tags: Vec<String>,

    interval: i64,

    split_penalty: f64,

    sub_fps_inc: f64,
    sub_fps_ref: f64,

    allow_negative_timestamps: bool,
    encoding_ref: &'static Encoding,
    encoding_inc: &'static Encoding,

    no_split_mode: bool,
    speed_optimization: Option<f64>,
}

fn parse_args() -> Result<Arguments, InputArgumentsError> {
    let matches = App::new(PKG_NAME.unwrap_or("unkown (not compiled with cargo)"))
        .version(PKG_VERSION.unwrap_or("unknown (not compiled with cargo)"))
        .about(PKG_DESCRIPTION.unwrap_or("unknown (not compiled with cargo)"))
        .arg(Arg::with_name("reference-file")
            .help("Path to the reference subtitle or video file")
            .required(true))
        .arg(Arg::with_name("incorrect-sub-file")
            .help("Path to the incorrect subtitle file")
            .required(true))
        .arg(Arg::with_name("output-file-path")
            .help("Path to corrected subtitle file")
            .required(true))
        .arg(Arg::with_name("split-penalty")
            .short("p")
            .long("split-penalty")
            .value_name("floating point number from 0 to 100")
            .help("Determines how eager the algorithm is to avoid splitting of the subtitles. 100 means that all lines will be shifted by the same offset, while 0 will produce MANY segments with different offsets. Values from 0.1 to 20 are the most useful.")
            .default_value("4"))
        .arg(Arg::with_name("interval")
            .short("i")
            .long("interval")
            .value_name("integer in milliseconds")
            .help("The smallest recognized time interval, smaller numbers make the alignment more accurate, greater numbers make aligning faster.")
            .default_value("1"))
        .arg(Arg::with_name("allow-negative-timestamps")
            .short("n")
            .long("allow-negative-timestamps")
            .help("Negative timestamps can lead to problems with the output file, so by default 0 will be written instead. This option allows you to disable this behavior."))
        .arg(Arg::with_name("sub-fps-ref")
            .long("sub-fps-ref")
            .value_name("floating-point number in frames-per-second")
            .default_value("30")
            .help("Specifies the frames-per-second for the accompanying video of MicroDVD `.sub` files (MicroDVD `.sub` files store timing information as frame numbers). Only affects the reference subtitle file."))
        .arg(Arg::with_name("sub-fps-inc")
            .long("sub-fps-inc")
            .value_name("floating-point number in frames-per-second")
            .default_value("30")
            .help("Specifies the frames-per-second for the accompanying video of MicroDVD `.sub` files (MicroDVD `.sub` files store timing information as frame numbers). Only affects the incorrect subtitle file."))
        .arg(Arg::with_name("encoding-ref")
            .long("encoding-ref")
            .value_name("encoding")
            .help("Charset encoding of the reference subtitle file."))
        .arg(Arg::with_name("encoding-inc")
            .long("encoding-inc")
            .value_name("encoding")
            .help("Charset encoding of the incorrect subtitle file."))
        .arg(Arg::with_name("statistics-path")
            .long("statistics-path")
            .short("s")
            .value_name("path")
            .help("enable statistics and put files in the specified folder")
            .required(false)
            )
        .arg(Arg::with_name("speed-optimization")
            .long("speed-optimization")
            .short("O")
            .value_name("path")
            .default_value("2")
            .help("(greatly) speeds up synchronization by sacrificing some accuracy; set to 0 to disable speed optimization")
            .required(false)
            )
        .arg(Arg::with_name("statistics-required-tag")
            .long("statistics-required-tag")
            .short("t")
            .value_name("tag")
            .help("only output statistics containing this tag (you can find the tags in statistics file)")
            .required(false)
            )
        .arg(Arg::with_name("no-split")
            .help("synchronize subtitles without looking for splits/breaks - this mode is much faster")
            .short("l")
            .long("no-split")
        )
        .after_help("This program works with .srt, .ass/.ssa, .idx and .sub files. The corrected file will have the same format as the incorrect file.")
        .get_matches();

    let reference_file_path: PathBuf = matches.value_of("reference-file").unwrap().into();
    let incorrect_file_path: PathBuf = matches.value_of("incorrect-sub-file").unwrap().into();
    let output_file_path: PathBuf = matches.value_of("output-file-path").unwrap().into();

    let statistics_folder_path_opt: Option<PathBuf> = matches.value_of("statistics-path").map(|v| PathBuf::from(v));
    let statistics_required_tags: Vec<String> = matches
        .values_of("statistics-required-tag")
        .map(|iter| iter.map(|s| s.to_string()).collect::<Vec<_>>())
        .unwrap_or_else(|| Vec::new());

    let interval: i64 = unpack_clap_number_i64(&matches, "interval")?;
    if interval < 1 {
        return Err(InputArgumentsErrorKind::ExpectedPositiveNumber {
            argument_name: "interval".to_string(),
            value: interval,
        }
        .into());
    }

    let split_penalty: f64 = unpack_clap_number_f64(&matches, "split-penalty")?;
    if split_penalty < 0.0 || split_penalty > 100.0 {
        return Err(InputArgumentsErrorKind::ValueNotInRange {
            argument_name: "interval".to_string(),
            value: split_penalty,
            min: 0.0,
            max: 100.0,
        }
        .into());
    }

    let speed_optimization: f64 = unpack_clap_number_f64(&matches, "speed-optimization")?;
    if split_penalty < 0.0 {
        return Err(InputArgumentsErrorKind::ExpectedNonNegativeNumber {
            argument_name: "speed-optimization".to_string(),
            value: speed_optimization,
        }
        .into());
    }

    let no_split_mode: bool = matches.is_present("no-split");

    Ok(Arguments {
        reference_file_path,
        incorrect_file_path,
        output_file_path,
        statistics_folder_path_opt,
        statistics_required_tags,
        interval,
        split_penalty,
        sub_fps_ref: unpack_clap_number_f64(&matches, "sub-fps-ref")?,
        sub_fps_inc: unpack_clap_number_f64(&matches, "sub-fps-inc")?,
        allow_negative_timestamps: matches.is_present("allow-negative-timestamps"),
        encoding_ref: get_encoding(matches.value_of("encoding-ref")),
        encoding_inc: get_encoding(matches.value_of("encoding-inc")),
        no_split_mode,
        speed_optimization: if speed_optimization <= 0. {
            None
        } else {
            Some(speed_optimization)
        },
    })
}

// //////////////////////////////////////////////////////////////////////////////////////////////////

fn run() -> Result<(), failure::Error> {
    let args = parse_args()?;

    let ref_file = InputFileHandler::open(
        &args.reference_file_path,
        args.interval,
        args.encoding_ref,
        args.sub_fps_ref,
    )?;

    if args.incorrect_file_path.eq(OsStr::new("_")) {
        // DEBUG MODE FOR REFERENCE FILE WAS ACTIVATED

        println!("input file path was given as '_'");
        println!("the output file is a .srt file only containing timing information from the reference file");
        println!("this can be used as a debugging tool");
        println!();

        let lines: Vec<(subparse::timetypes::TimeSpan, String)> = ref_file
            .subparse_timespans()
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, time_span)| (time_span, format!("line {}", i)))
            .collect();

        let debug_file =
            subparse::SrtFile::create(lines).with_context(|_| TopLevelErrorKind::FailedToInstantiateSubtitleFile)?;

        write_data_to_file(
            &args.output_file_path,
            debug_file.to_data().unwrap(), // error handling
        )?;

        return Ok(());
    }

    let inc_file = InputFileHandler::open_sub_file(
        args.incorrect_file_path.as_path(),
        args.interval,
        args.encoding_inc,
        args.sub_fps_inc,
    )?;

    let output_file_format;

    match inc_file.file_format() {
        InputFileFormat::Subtitle(f) => output_file_format = f,
        InputFileFormat::Video(_) => unreachable!(),
    }

    // this program internally stores the files in a non-destructable way (so
    // formatting is preserved) but has no abilty to convert between formats
    if !subparse::is_valid_extension_for_subtitle_format(args.output_file_path.extension(), output_file_format) {
        return Err(TopLevelErrorKind::FileFormatMismatch {
            input_file_path: args.incorrect_file_path,
            output_file_path: args.output_file_path,
            input_file_format: inc_file.file_format().subtitle_format().unwrap(),
        }
        .into_error()
        .into());
    }

    let statistics_module_opt: Option<Statistics>;
    if let Some(statistics_folder_path) = args.statistics_folder_path_opt {
        statistics_module_opt = Some(Statistics::new(statistics_folder_path, args.statistics_required_tags));
    } else {
        statistics_module_opt = None;
    }

    println!(
        "synchronizing '{}' to reference file '{}'...",
        args.incorrect_file_path.display(),
        args.reference_file_path.display()
    );
    let alg_deltas;
    if args.no_split_mode {
        let alg_delta = alass_core::align_nosplit(
            inc_file.aligner_timespans().to_vec(),
            ref_file.aligner_timespans().to_vec(),
            Some(Box::new(ProgressInfo::new(1))),
            statistics_module_opt,
        );
        alg_deltas = std::vec::from_elem(alg_delta, inc_file.aligner_timespans().len());
    } else {
        alg_deltas = align(
            inc_file.aligner_timespans().to_vec(),
            ref_file.aligner_timespans().to_vec(),
            args.split_penalty / 100.0,
            args.speed_optimization,
            Some(Box::new(ProgressInfo::new(1))),
            statistics_module_opt,
        );
    }
    let deltas = alg_deltas_to_timing_deltas(&alg_deltas, args.interval);

    // group subtitles lines which have the same offset
    let shift_groups: Vec<(AlgTimeDelta, Vec<TimeSpan>)> = get_subtitle_delta_groups(
        alg_deltas
            .iter()
            .cloned()
            .zip(inc_file.subparse_timespans().iter().cloned())
            .collect(),
    );

    for (shift_group_delta, shift_group_lines) in shift_groups {
        // computes the first and last timestamp for all lines with that delta
        // -> that way we can provide the user with an information like
        //     "100 subtitles with 10min length"
        let min = shift_group_lines
            .iter()
            .map(|subline| subline.start)
            .min()
            .expect("a subtitle group should have at least one subtitle line");
        let max = shift_group_lines
            .iter()
            .map(|subline| subline.start)
            .max()
            .expect("a subtitle group should have at least one subtitle line");

        println!(
            "shifted block of {} subtitles with length {} by {}",
            shift_group_lines.len(),
            max - min,
            alg_delta_to_delta(shift_group_delta, args.interval)
        );
    }

    println!();

    if ref_file.subparse_timespans().is_empty() {
        println!("warn: reference file has no subtitle lines");
        println!();
    }
    if inc_file.subparse_timespans().is_empty() {
        println!("warn: file with incorrect subtitles has no lines");
        println!();
    }

    let mut corrected_timespans: Vec<subparse::timetypes::TimeSpan> = inc_file
        .subparse_timespans()
        .iter()
        .zip(deltas.iter())
        .map(|(&timespan, &delta)| timespan + delta)
        .collect();

    if corrected_timespans.iter().any(|ts| ts.start.is_negative()) {
        println!("warn: some subtitles now have negative timings, which can cause invalid subtitle files");
        if args.allow_negative_timestamps {
            println!(
                "warn: negative timestamps will be written to file, because you passed '-n' or '--allow-negative-timestamps'",
            );
        } else {
            println!(
                "warn: negative subtitles will therefore moved to the start of the subtitle file by default; pass '-n' or '--allow-negative-timestamps' to disable this behavior",
            );

            for corrected_timespan in &mut corrected_timespans {
                if corrected_timespan.start.is_negative() {
                    let offset = subparse::timetypes::TimePoint::from_secs(0) - corrected_timespan.start;
                    corrected_timespan.start = corrected_timespan.start + offset;
                    corrected_timespan.end = corrected_timespan.end + offset;
                }
            }
        }
        println!();
    }

    // .idx only has start timepoints (the subtitle is shown until the next subtitle starts) - so retiming with gaps might
    // produce errors
    if output_file_format == SubtitleFormat::VobSubIdx {
        println!("warn: writing to an '.idx' file can lead to unexpected results due to restrictions of this format");
    }

    // incorrect file -> correct file
    let shifted_timespans: Vec<SubtitleEntry> = corrected_timespans
        .into_iter()
        .map(|timespan| SubtitleEntry::from(timespan))
        .collect();

    // write corrected files
    let mut correct_file = inc_file
        .into_subtitle_file()
        .expect("incorrect input file can only be a subtitle")
        .clone();
    correct_file
        .update_subtitle_entries(&shifted_timespans)
        .with_context(|_| TopLevelErrorKind::FailedToUpdateSubtitle)?;

    write_data_to_file(
        &args.output_file_path,
        correct_file
            .to_data()
            .with_context(|_| TopLevelErrorKind::FailedToGenerateSubtitleData)?,
    )?;

    Ok(())
}

// //////////////////////////////////////////////////////////////////////////////////////////////////

fn main() {
    match run() {
        Ok(_) => std::process::exit(0),
        Err(error) => {
            let show_bt_opt = std::env::vars()
                .find(|(key, _)| key == "RUST_BACKTRACE")
                .map(|(_, value)| value);
            let show_bt = show_bt_opt != None && show_bt_opt != Some("0".to_string());

            println!("error: {}", error);
            if show_bt {
                println!("stack trace: {}", error.backtrace());
            }

            for cause in error.as_fail().iter_causes() {
                println!("caused by: {}", cause);
                if show_bt {
                    if let Some(backtrace) = cause.backtrace() {
                        println!("stack trace: {}", backtrace);
                    }
                }
            }

            if !show_bt {
                println!("");
                println!("not: run with environment variable 'RUST_BACKTRACE=1' for detailed stack traces");
            }

            std::process::exit(1)
        }
    }
}
