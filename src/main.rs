// This file is part of the Rust library and binary `aligner`.
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

#[macro_use]
extern crate error_chain;
extern crate aligner;
extern crate clap;
extern crate pbr;
pub extern crate combine;

const PKG_VERSION: Option<&'static str> = option_env!("CARGO_PKG_VERSION");
const PKG_NAME: Option<&'static str> = option_env!("CARGO_PKG_NAME");
const PKG_DESCRIPTION: Option<&'static str> = option_env!("CARGO_PKG_DESCRIPTION");

// Alg* stands for algorithm (the internal aligner algorithm types)
use aligner::{ProgressHandler, TimeDelta as AlgTimeDelta, TimePoint as AlgTimePoint, TimeSpan as AlgTimeSpan, align};
use clap::{App, Arg};
use pbr::ProgressBar;
use std::fs::File;
use std::io::{Read, Write};
use std::cmp::{max, min};
use std::str::FromStr;

mod binary;

pub use binary::errors;
pub use binary::errors::*;
pub use binary::errors::ErrorKind::*;
use binary::formats::{get_subtitle_format_with_error, parse_file};
pub use binary::formats::SubtitleFormat;
pub use binary::formats::MutableTimedData;
pub use binary::formats::ParseSubtitle;
pub use binary::formats::{TimeDelta, TimePoint, TimeSpan};

#[derive(Default)]
struct ProgressInfo {
    progress_bar: Option<ProgressBar<std::io::Stdout>>,
}

impl ProgressHandler for ProgressInfo {
    fn init(&mut self, steps: i64) {
        self.progress_bar = Some(ProgressBar::new(steps as u64));
    }
    fn inc(&mut self) {
        self.progress_bar.as_mut().unwrap().inc();
    }
    fn finish(&mut self) {
        self.progress_bar.as_mut().unwrap().finish_println("\n");
    }
}

fn read_file_to_string(path: &str) -> Result<String> {
    let mut file = File::open(path).map_err(|e| Error::from(Io(e))).chain_err(|| FileOperation(path.to_string()))?;
    let mut s = String::new();
    file.read_to_string(&mut s).map_err(|e| Error::from(Io(e))).chain_err(|| FileOperation(path.to_string()))?;
    Ok(s)
}

fn write_string_to_file(path: &str, s: String) -> Result<()> {
    let mut file = File::create(path).map_err(|e| Error::from(Io(e))).chain_err(|| FileOperation(path.to_string()))?;
    file.write_all(&s.bytes().collect::<Vec<_>>())
        .map_err(|e| Error::from(Io(e)))
        .chain_err(|| FileOperation(path.to_string()))?;
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
     .map(|(begin, end)| {
         AlgTimeSpan::new_safe(timing_to_alg_timepoint(begin, interval),
                               timing_to_alg_timepoint(end, interval))
     })
     .collect()
}
fn alg_deltas_to_timing_deltas(v: &[AlgTimeDelta], interval: i64) -> Vec<TimeDelta> {
    v.iter().cloned().map(|x| alg_delta_to_delta(x, interval)).collect()
}

/// Groups consecutive timespans with the same delta together.
fn get_subtitle_delta_groups(mut v: Vec<(AlgTimeDelta, TimeSpan)>) -> Vec<(AlgTimeDelta, Vec<TimeSpan>)> {
    v.sort_by_key(|t| min((t.1).0, (t.1).1));

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


/// Will return an array where the start time of an subtitle is always less than the end time (will switch incorrect ones).
fn corrected_timings(v: Vec<TimeSpan>) -> Vec<TimeSpan> {
    v.into_iter().map(|(a, b)| (min(a, b), max(a, b))).collect()
}

/// Every delta that where `start + delta`, is negative will be adjusted so that `start + delta` is zero. This avoids
/// invalid files for formats that don't support negative timestamps.
fn get_truncated_deltas(timespans: &[TimeSpan], deltas: Vec<TimeDelta>) -> Vec<TimeDelta> {
    deltas.into_iter()
          .zip(timespans.iter().cloned())
          .map(|(delta, (start, _))| if (delta + start).is_negative() { -start } else { delta })
          .collect()
}

/// Prints warning.
fn pwarning<'a, T: Into<std::borrow::Cow<'a, str>>>(s: T) {
    println!("WW: {}", s.into());
}

/// Prints info.
fn pinfo<'a, T: Into<std::borrow::Cow<'a, str>>>(s: T) {
    println!("II: {}", s.into());
}

/// Prints error.
fn perror<'a, T: Into<std::borrow::Cow<'a, str>>>(s: T) {
    println!("EE: {}", s.into());
}

fn run() -> Result<()> {
    let matches = App::new(PKG_NAME.unwrap_or("unkown (not compiled with cargo)"))
        .version(PKG_VERSION.unwrap_or("unknown (not compiled with cargo)"))
        .about(PKG_DESCRIPTION.unwrap_or("unknown (not compiled with cargo)"))
        .arg(Arg::with_name("reference-sub-file")
            .help("Path to the reference subtitle file")
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
            .help("The smallest recognized time interval, smaller numbers make alignment more accurate, greater numbers make it faster.")
            .default_value("1"))
        .arg(Arg::with_name("allow-negative-timestamps")
            .short("n")
            .long("allow-negative-timestamps")
            .help("Negative timestamps can lead to problems with the output file, so by default 0 will be written instead. This option allows you to disable this behavior."))
        .after_help("This program works with .srt, .ass/.ssa and .idx files. The corrected file will have the same format as the incorrect file.")
        .get_matches();

    let incorrect_file_path = matches.value_of("incorrect-sub-file").unwrap();
    let reference_file_path = matches.value_of("reference-sub-file").unwrap();
    let output_file_path = matches.value_of("output-file-path").unwrap();

    let interval_str: &str = matches.value_of("interval").unwrap();
    let interval: i64 = FromStr::from_str(interval_str).chain_err(|| ArgumentParseError("interval", interval_str.to_string()))?;
    if interval < 1 {
        return Err(Error::from(ExpectedPositiveNumber(interval))).chain_err(|| Error::from(InvalidArgument("interval")));
    }

    let split_penalty_str: &str = matches.value_of("split-penalty").unwrap();
    let split_penalty: f64 = FromStr::from_str(split_penalty_str).chain_err(|| ArgumentParseError("split-penalty", split_penalty_str.to_string()))?;
    if split_penalty < 0.0 || split_penalty > 100.0 {
        return Err(Error::from(ValueNotInRange(split_penalty, 0.0, 100.0))).chain_err(|| Error::from(InvalidArgument("split-penalty")));
    }

    let allow_negative_timestamps = matches.is_present("allow-negative-timestamps");


    let reference_sub_string = read_file_to_string(reference_file_path)?;
    let incorrect_sub_string = read_file_to_string(incorrect_file_path)?;

    let incorrect_file_format = get_subtitle_format_with_error(incorrect_file_path)?;
    let reference_file_format = get_subtitle_format_with_error(reference_file_path)?;
    let output_file_format = get_subtitle_format_with_error(output_file_path)?;

    // this program internally stores the files in a non-destructable way (so
    // formatting is preserved) but has no abilty to convert between formats
    if incorrect_file_format != output_file_format {
        return Err(DifferentOutputFormat(incorrect_file_path.to_string(),
                                         output_file_path.to_string())
                .into());
    }

    let timed_reference_file = parse_file(reference_file_format, reference_sub_string.clone())
        .chain_err(|| FileOperation(reference_file_path.to_string()))?;
    let timed_incorrect_file = parse_file(incorrect_file_format, incorrect_sub_string.clone())
        .chain_err(|| FileOperation(incorrect_file_path.to_string()))?;

    let timings_reference = corrected_timings(timed_reference_file.get_timespans()?);
    let timings_incorrect = corrected_timings(timed_incorrect_file.get_timespans()?);

    let alg_reference_timespans = timings_to_alg_timespans(&timings_reference, interval);
    let alg_incorrect_timespans = timings_to_alg_timespans(&timings_incorrect, interval);

    let alg_deltas = align(alg_incorrect_timespans.clone(),
                           alg_reference_timespans,
                           split_penalty / 100.0,
                           Some(Box::new(ProgressInfo::default())));
    let mut deltas = alg_deltas_to_timing_deltas(&alg_deltas, interval);

    // list of original subtitles lines which have the same timings
    let shift_groups: Vec<(AlgTimeDelta, Vec<TimeSpan>)> = get_subtitle_delta_groups(alg_deltas.iter()
                                                                                               .cloned()
                                                                                               .zip(timings_incorrect.iter().cloned())
                                                                                               .collect());

    for (shift_group_delta, shift_group_lines) in shift_groups {
        // computes the first and last timestamp for all lines with that delta
        // -> that way we can provide the user with an information like
        //     "100 subtitles with 10min length"
        let min_max_opt = shift_group_lines.iter().fold(None, |last_opt, subline| {
            let new_min = subline.0;
            let new_max = subline.1;
            if let Some((last_min, last_max)) = last_opt { Some((min(last_min, new_min), max(last_max, new_max))) } else { Some((new_min, new_max)) }
        });

        let (min, max) = match min_max_opt {
            Some(v) => v,
            None => unreachable!(),
        };

        pinfo(format!("shifted block of {} subtitles with length {} by {}",
                      shift_group_lines.len(),
                      max - min,
                      alg_delta_to_delta(shift_group_delta, interval)));
    }


    if timings_reference.is_empty() {
        println!("");
        pwarning("reference file has no subtitle lines");
    }
    if timings_incorrect.is_empty() {
        println!("");
        pwarning("file with incorrect subtitles has no lines");
    }

    let writing_negative_timespans = deltas.iter().zip(timings_incorrect.iter()).any(|(&delta, &(start, _))| (delta + start).is_negative());
    if writing_negative_timespans {
        println!("");
        pwarning("some subtitles now have negative timings, which can cause invalid subtitle files");
        if allow_negative_timestamps {
            pwarning("negative timestamps will be written to file, because you passed '-n' or '--allow-negative-timestamps'");
        } else {
            pwarning("negative subtitles will therefore be set to zero by default; pass '-n' or '--allow-negative-timestamps' to disable this behavior");
            deltas = get_truncated_deltas(&timings_incorrect, deltas);
        }
    }

    // .idx only has start timepoints (the subtitle is shown until the next subtitle starts) - so retiming with gaps might
    // produce errors
    if output_file_format == SubtitleFormat::VobSubIdx {
        println!("");
        pwarning("writing to an '.idx' file can lead to unexpected results due to restrictions of this format");
    }

    // incorrect file -> correct file
    let correct_file = timed_incorrect_file.shift_by_deltas(&mut deltas.into_iter())?;

    write_string_to_file(output_file_path, correct_file.to_data_string()?)?;

    Ok(())
}

fn main() {
    match run() {
        Ok(_) => std::process::exit(0),
        Err(e) => {
            perror(format!("error: {}", e));

            for e in e.iter().skip(1) {
                perror(format!("caused by: {}", e));
            }

            // The backtrace is not always generated. Try to this with `RUST_BACKTRACE=1`.
            if let Some(backtrace) = e.backtrace() {
                perror(format!("backtrace: {:?}", backtrace));
            } else {
                perror("note: run program with `env RUST_BACKTRACE=1` for a backtrace");
            }
            std::process::exit(1);
        }
    }
}
