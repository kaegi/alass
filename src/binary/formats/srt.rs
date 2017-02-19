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


use std;
use {MutableTimedData, ParseSubtitle, TimeDelta, TimePoint, TimeSpan};
use errors::Result as ProgramResult;
use binary::formats::common::*;

use combine::char::{char, string};
use combine::combinator::{many, parser as p, eof};
use combine::primitives::{ParseError, ParseResult, Parser, Stream};

pub struct SrtParser;

// see https://docs.rs/error-chain/0.8.1/error_chain/
// this error type might be overkill, but that way it stays consistent with
// the other parsers
error_chain! {
    errors {
        LineError(line_num: usize, msg: String) {
            display("parse error at line `{}` because of `{}`", line_num, msg)
        }
        SrtParseError(msg: String) {
            description(msg)
        }
    }
}

/// This makes creating a vector with file parts much nicer, shorter and more readable.
trait ExtendWithSrtFilePart {
    fn space(self, s: String) -> Self;
    fn index(self, i: i64) -> Self;
    fn dialog(self, s: String) -> Self;
    fn begin(self, t: TimePoint) -> Self;
    fn end(self, t: TimePoint) -> Self;
}

impl ExtendWithSrtFilePart for Vec<SrtFilePart> {
    fn space(mut self, s: String) -> Self {
        self.push(SrtFilePart::Space(s));
        self
    }
    fn index(mut self, i: i64) -> Self {
        self.push(SrtFilePart::Index(i));
        self
    }
    fn dialog(mut self, s: String) -> Self {
        self.push(SrtFilePart::Dialog(s));
        self
    }
    fn begin(mut self, t: TimePoint) -> Self {
        self.push(SrtFilePart::TimespanBegin(t));
        self
    }
    fn end(mut self, t: TimePoint) -> Self {
        self.push(SrtFilePart::TimespanEnd(t));
        self
    }
}


/// The parsing works as a finite state machine. These are the states in it.
enum SrtParserState {
    // emptyline or index follows
    Emptyline,

    /// timing line follows
    Index,

    /// dialog or emptyline follows
    Timing,

    /// emptyline follows
    Dialog,
}

impl ParseSubtitle for SrtParser {
    type Result = SubRipFile;

    fn parse(s: String) -> ProgramResult<SubRipFile> {
        let file_opt = Self::parse_file(s.as_str());
        match file_opt {
            Ok(file) => Ok(SubRipFile::new(file)),
            Err(err) => Err(err.into()),
        }

    }
}

impl SrtParser {
    fn parse_file(mut s: &str) -> Result<Vec<SrtFilePart>> {
        // remove utf-8 bom
        s = skip_bom(s);

        let mut result: Vec<SrtFilePart> = Vec::new();
        let mut state: SrtParserState = SrtParserState::Emptyline; // expect emptyline or index
        let lines_with_newl: Vec<(String, String)> = get_lines_non_destructive(s)
            .map_err(|(line_num, err_str)| ErrorKind::LineError(line_num, err_str))?;

        for (line_num, (line, newl)) in lines_with_newl.into_iter().enumerate() {
            state = match state {
                SrtParserState::Emptyline => Self::next_state_from_emptyline(&mut result, line_num, line)?,
                SrtParserState::Index => Self::next_state_from_index(&mut result, line_num, line)?,
                SrtParserState::Timing | SrtParserState::Dialog => Self::next_state_from_timing_or_dialog(&mut result, line_num, line)?,
            };

            // we also want to preserve the line break
            result.push(SrtFilePart::Space(newl));
        }

        Ok(result)
    }

    fn next_state_from_emptyline(result: &mut Vec<SrtFilePart>, line_num: usize, line: String) -> Result<SrtParserState> {
        if line.trim().is_empty() {
            result.push(SrtFilePart::Space(line));
            Ok(SrtParserState::Emptyline)
        } else {
            result.append(&mut Self::parse_index_line(line_num, line.as_str())?);
            Ok(SrtParserState::Index)
        }
    }


    fn next_state_from_index(result: &mut Vec<SrtFilePart>, line_num: usize, line: String) -> Result<SrtParserState> {
        result.append(&mut Self::parse_timestamp_line(line_num, line.as_str())?);
        Ok(SrtParserState::Timing)
    }

    fn next_state_from_timing_or_dialog(result: &mut Vec<SrtFilePart>, _: usize, line: String) -> Result<SrtParserState> {
        if line.trim().is_empty() {
            result.push(SrtFilePart::Space(line));
            Ok(SrtParserState::Emptyline)
        } else {
            result.push(SrtFilePart::Dialog(line));
            Ok(SrtParserState::Dialog)
        }
    }

    /// Matches a line with a single index.
    fn parse_index_line(line_num: usize, s: &str) -> Result<Vec<SrtFilePart>> {
        Self::handle_error((many(ws()), p(number_i64), many(ws()), eof())
                               .map(|(ws1, num, ws2, ()): (_, _, _, ())| Vec::new().space(ws1).index(num).space(ws2))
                               .expected("SubRip index")
                               .parse(s),
                           line_num)
    }

    /// Convert a result/error from the combine library to the srt parser error.
    fn handle_error<T>(r: std::result::Result<(T, &str), ParseError<&str>>, line_num: usize) -> Result<T> {
        r.map(|(v, _)| v)
         .map_err(|e| ErrorKind::LineError(line_num, parse_error_to_string(e)).into())
    }


    /// Matches a `SubRip` timestamp like "00:24:45,670"
    fn parse_timestamp<I>(input: I) -> ParseResult<TimePoint, I>
        where I: Stream<Item = char>
    {
        (p(number_i64), char(':'), p(number_i64), char(':'), p(number_i64), char(','), p(number_i64))
            .map(|t| TimePoint::from_components(t.0, t.2, t.4, t.6))
            .expected("SubRip timestamp")
            .parse_stream(input)
    }


    /// Matches a `SubRip` timespan like "00:24:45,670 --> 00:24:45,680".
    fn parse_timespan<I>(input: I) -> ParseResult<Vec<SrtFilePart>, I>
        where I: Stream<Item = char>
    {
        (many(ws()), p(Self::parse_timestamp), many(ws()), string("-->"), many(ws()), p(Self::parse_timestamp), many(ws()), eof())
            .map(|t| Vec::new().space(t.0).begin(t.1).space(t.2).space(t.3.to_string()).space(t.4).end(t.5).space(t.6))
            .expected("SubRip timespan")
            .parse_stream(input)
    }

    /// Matches a `SubRip` timespan line like "00:24:45,670 --> 00:24:45,680".
    fn parse_timestamp_line(line_num: usize, s: &str) -> Result<Vec<SrtFilePart>> {
        Self::handle_error(p(Self::parse_timespan).parse(s), line_num)
    }
}

impl MutableTimedData for SubRipFile {
    /// Durations can only be changed by `shifted_by_deltas`.
    fn get_timespans(&self) -> ProgramResult<Vec<TimeSpan>> {
        #[derive(Debug, Clone, Copy, PartialEq,Eq)]
        enum SubRipState {
            FindBegin,
            FindEnd(TimePoint /* timing of the timespan start */),
        }

        let mut next_state = SubRipState::FindBegin;
        let result = {
            // satisfy the borrow checker, so next_state is released
            let closure = &mut |part: &SrtFilePart| {
                use self::SrtFilePart::*;
                match *part {
                    Space(_) | Index(_) | Dialog(_) => {}
                    TimespanBegin(begin) => {
                        assert!(next_state == SubRipState::FindBegin);
                        next_state = SubRipState::FindEnd(begin);
                    }
                    TimespanEnd(end) => {
                        if let SubRipState::FindEnd(begin) = next_state {
                            next_state = SubRipState::FindBegin;
                            return Some((begin, end));
                        } else {
                            // the parser shouldn't be able to construct such a case
                            panic!("expected end of SubRip timespan");
                        }
                    }
                }

                None
            };
            self.v.iter().filter_map(closure).collect()
        };

        // every timespan should now consist of a beginning and a end
        assert_eq!(next_state, SubRipState::FindBegin);
        Ok(result)
    }

    /// Shift the timespans from the `get_timespans()` vector by the given
    /// duration.
    ///
    /// The length of the given iterator should always match the length of
    /// `get_timespans()`.
    fn shift_by_deltas(&self, mut i: &mut Iterator<Item = TimeDelta>) -> ProgramResult<Box<MutableTimedData>> {
        let mut current_delta = i.next();
        let shift_closure = |part: &SrtFilePart| {
            use self::SrtFilePart::*;
            match *part {
                ref p @ Space(_) |
                ref p @ Dialog(_) |
                ref p @ Index(_) => p.clone(),
                TimespanBegin(t) => TimespanBegin(t + current_delta.unwrap()),
                TimespanEnd(t) => {
                    let new_result = TimespanEnd(t + current_delta.unwrap());
                    current_delta = i.next();
                    new_result
                }
            }
        };

        let result: Vec<SrtFilePart> = self.v
                                           .iter()
                                           .map(shift_closure)
                                           .collect();
        Ok(Box::new(SubRipFile::new(result)))
    }

    /// Returns a string in the respective format (.ssa, .srt, etc.) with the
    /// corrected time spans.
    fn to_data_string(&self) -> ProgramResult<String> {
        let closure = &mut |part: &SrtFilePart| {
            use self::SrtFilePart::*;
            match *part {
                Space(ref t) | Dialog(ref t) => t.clone(),
                Index(i) => i.to_string(),
                TimespanBegin(t) | TimespanEnd(t) => {
                    format!("{:02}:{:02}:{:02},{:03}",
                            t.hours(),
                            t.mins_comp(),
                            t.secs_comp(),
                            t.msecs_comp())
                }
            }
        };

        Ok(self.v.iter().map(closure).collect::<String>())
    }
}

/// The whole .srt file will be split into semantic segments (index, text,
/// timepan information) and this enum provides the information which
/// information a segment holds.
#[derive(Debug, Clone)]
enum SrtFilePart {
    /// Spaces, empty lines, etc.
    Space(String),

    /// The beginnig timestamp of a timespan
    TimespanBegin(TimePoint),

    /// The ending timestamp of a timespan
    TimespanEnd(TimePoint),

    /// The index, which determines the order of all subtitle blocks
    Index(i64),

    /// The dialog text
    Dialog(String),
}

#[derive(Debug, Clone)]
pub struct SubRipFile {
    v: Vec<SrtFilePart>,
}


impl SubRipFile {
    fn new(v: Vec<SrtFilePart>) -> SubRipFile {
        // cleans up multiple fillers after another
        let new_file_parts = dedup_string_parts(v, |part: &mut SrtFilePart| {
            match *part {
                SrtFilePart::Space(ref mut text) => Some(text),
                _ => None,
            }
        });

        SubRipFile { v: new_file_parts }
    }
}

// TODO: parser tests
