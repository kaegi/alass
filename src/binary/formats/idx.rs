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



use {MutableTimedData, ParseSubtitle, TimeDelta, TimePoint, TimeSpan};
use errors::Result as ProgramResult; // the crate wide error type (we use a custom error type here)
use binary::formats::common::*;

use std::iter::once;

use combine::char::*;
use combine::combinator::*;
use combine::primitives::Parser;

// see https://docs.rs/error-chain/0.8.1/error_chain/
error_chain! {
    errors {
        IdxLineParseError(line_num: usize, msg: String) {
            display("parsing the line `{}` failed because of `{}`", line_num, msg)
        }
    }
}

// ////////////////////////////////////////////////////////////////////////////////////////////////
// .idx file parts

#[derive(Debug, Clone)]
enum IdxFilePart {
    /// Spaces, field information, comments, unimportant fields, ...
    Filler(String),

    /// Represents a parsed time string like "00:42:20:204".
    Timestamp(TimePoint),
}


// ////////////////////////////////////////////////////////////////////////////////////////////////
// .idx file

/// Represents a reconstructable .idx file.
///
/// All (for this project) unimportant information are saved into `IdxFilePart::Filler(...)`, so
/// a timespan-altered file still has the same meta-information.
#[derive(Debug, Clone)]
pub struct IdxFile {
    v: Vec<IdxFilePart>,
}

impl IdxFile {
    fn new(v: Vec<IdxFilePart>) -> IdxFile {
        // cleans up multiple fillers after another
        let new_file_parts = dedup_string_parts(v, |part: &mut IdxFilePart| {
            match *part {
                IdxFilePart::Filler(ref mut text) => Some(text),
                _ => None,
            }
        });
        IdxFile { v: new_file_parts }
    }
}

impl MutableTimedData for IdxFile {
    /// Timings can only be changed by `shifted_by_deltas`.
    fn get_timespans(&self) -> ProgramResult<Vec<TimeSpan>> {
        let timings: Vec<_> = self.v
                                  .iter()
                                  .filter_map(|file_part| match *file_part {
                                      IdxFilePart::Filler(_) => None,
                                      IdxFilePart::Timestamp(t) => Some(t),
                                  })
                                  .collect();

        Ok(match timings.last() {
            Some(&last_timing) => {
                // .idx files do not store timespans. Every subtitle is shown until the next subtitle
                // starts. Mpv shows the last subtile for exactly one minute.
                let next_timings = timings.iter().cloned().skip(1).chain(once(last_timing + TimeDelta::from_minutes(1)));
                timings.iter().cloned().zip(next_timings).collect()
            }
            None => {
                // no timings
                Vec::new()
            }
        })
    }

    /// Shift the timespans from the `get_timespans()` vector by the given
    /// duration.
    ///
    /// The length of the given iterator should always match the length of
    /// `get_timespans()`.
    fn shift_by_deltas(&self, i: &mut Iterator<Item = TimeDelta>) -> ProgramResult<Box<MutableTimedData>> {
        let new_parts = self.v
                            .iter()
                            .map(|file_part| {
                                match *file_part {
                                    IdxFilePart::Filler(ref t) => IdxFilePart::Filler(t.to_string()),
                                    IdxFilePart::Timestamp(ref t) => IdxFilePart::Timestamp(*t + i.next().expect("iterator to few elements")),
                                }
                            })
                            .collect();
        assert_eq!(i.next(), None); // too many elements
        Ok(Box::new(IdxFile::new(new_parts)))
    }

    /// Returns a string in the respective format (.ssa, .srt, etc.) with the
    /// corrected time spans.
    fn to_data_string(&self) -> ProgramResult<String> {


        // timing to string like "00:03:28:308"
        let fn_timing_to_string = |t: TimePoint| {
            let p = if t.0 < 0 { -t } else { t };
            format!("{}{:02}:{:02}:{:02}:{:03}",
                    if t.0 < 0 { "-" } else { "" },
                    p.hours(),
                    p.mins_comp(),
                    p.secs_comp(),
                    p.msecs_comp())
        };

        let fn_file_part_to_string = |part: &IdxFilePart| {
            use self::IdxFilePart::*;
            match *part {
                Filler(ref t) => t.clone(),
                Timestamp(t) => fn_timing_to_string(t),
            }
        };

        let result: String = self.v
                                 .iter()
                                 .map(fn_file_part_to_string)
                                 .collect();

        Ok(result)
    }
}

// ////////////////////////////////////////////////////////////////////////////////////////////////
// .idx parser

pub struct IdxParser;

impl ParseSubtitle for IdxParser {
    type Result = IdxFile;

    fn parse(s: String) -> ProgramResult<IdxFile> {
        match IdxParser.parse_inner(&s) {
            Ok(v) => Ok(v),
            Err(e) => Err(e.into()),
        }
    }
}


impl IdxParser {
    fn parse_inner(&mut self, i: &str) -> Result<IdxFile> {
        // remove utf-8 BOM
        let mut result = Vec::new();
        let (bom, s) = split_bom(&i);
        result.push(IdxFilePart::Filler(bom.to_string()));

        let lines = get_lines_non_destructive(s).map_err(|(line_num, err_str)| ErrorKind::IdxLineParseError(line_num, err_str))?;
        for (line_num, (line, newl)) in lines.into_iter().enumerate() {
            let mut file_parts = self.parse_line(line_num, line)?;
            result.append(&mut file_parts);
            result.push(IdxFilePart::Filler(newl));
        }

        Ok(IdxFile::new(result))
    }

    fn parse_line(&mut self, line_num: usize, s: String) -> Result<Vec<IdxFilePart>> {
        if !s.trim_left().starts_with("timestamp:") {
            return Ok(vec![IdxFilePart::Filler(s)]);
        }

        (many(ws()), string("timestamp:"), many(ws()), many(or(digit(), token(':'))), many(try(any())), eof())
            .map(|(ws1, s1, ws2, timestamp_str, s2, _): (String, &str, String, String, String, ())| -> Result<Vec<IdxFilePart>> {
                let mut result = Vec::<IdxFilePart>::new();
                result.push(IdxFilePart::Filler(ws1));
                result.push(IdxFilePart::Filler(s1.to_string()));
                result.push(IdxFilePart::Filler(ws2));
                result.push(IdxFilePart::Timestamp(self.parse_timestamp(line_num, timestamp_str.as_str())?));
                result.push(IdxFilePart::Filler(s2.to_string()));
                Ok(result)
            })
            .parse(s.as_str())
            .map_err(|e| ErrorKind::IdxLineParseError(line_num, parse_error_to_string(e)))?
            .0
    }

    /// Parse an .idx timestamp like `00:41:36:961`.
    fn parse_timestamp(&mut self, line_num: usize, s: &str) -> Result<TimePoint> {
        (parser(number_i64), token(':'), parser(number_i64), token(':'), parser(number_i64), token(':'), parser(number_i64), eof())
            .map(|(hours, _, mins, _, secs, _, msecs, _)| TimePoint::from_components(hours, mins, secs, msecs))
            .parse(s) // <- return type is ParseResult<(Timing, &str)>
            .map(|(file_part, _)| file_part)
            .map_err(|e| ErrorKind::IdxLineParseError(line_num, parse_error_to_string(e)).into())
    }
}
