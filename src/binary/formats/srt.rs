// This file is part of the Rust library and binary `aligner`.
//
// Copyright (C) 2017 kaegi
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.


use {MutableTimedData, ParseSubtitle, TimeDelta, TimePoint, TimeSpan};
use errors::Result as ProgramResult;
use binary::formats::common::*;

use combine::char::{char, space as multispace, string};
use combine::combinator::{eof, many, many1, optional, parser as p, try};
use combine::primitives::{ParseResult, Parser, State, Stream};

pub struct SrtParser;

// see https://docs.rs/error-chain/0.8.1/error_chain/
// this error type might be overkill, but that way it stays consistent with
// the other parsers
error_chain! {
    errors {
        SrtParseError(msg: String) {
            description(msg)
        }
    }
}


impl ParseSubtitle for SrtParser {
    type Result = SubRipFile;

    fn parse(s: String) -> ProgramResult<SubRipFile> {
        let file_opt = p(srt_file).parse(State::new(&s[..]));
        match file_opt {
            Ok(file) => Ok(file.0),
            Err(err) => {
                let err2 = Error::from(ErrorKind::SrtParseError(parse_error_to_string(err)));
                Err(err2.into())
            }
        }

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
        let mut result = Vec::new();
        {
            // satisfy the borrow checker, so next_state and result are released

            let closure = &mut |part: &SubRipAnnotations| {
                use self::SubRipAnnotations::*;
                match *part {
                    Space(_) | Index(_) | Dialog(_) => {}
                    TimespanBegin(begin) => {
                        assert!(next_state == SubRipState::FindBegin);
                        next_state = SubRipState::FindEnd(begin);
                    }
                    TimespanEnd(end) => {
                        if let SubRipState::FindEnd(begin) = next_state {
                            result.push((begin, end));
                            next_state = SubRipState::FindBegin;
                        } else {
                            // the parser shouldn't be able to construct such a case
                            panic!("expected end of SubRip timespan");
                        }
                    }
                }
            };
            self.visit(closure);
        }

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
        // Option is necessary because the closure will change to FnOnce we call
        // "SubRipFile::index(SubRipFile, i64) -> SubRipFile" (moves result); we use
        // take
        let mut result_opt = Some(SubRipFile::new());
        let mut current_delta = i.next();
        self.visit(&mut |part: &SubRipAnnotations| {
            use self::SubRipAnnotations::*;
            let result = result_opt.take().unwrap();
            let new_result = match *part {
                Space(ref t) => result.space(t.clone()),
                Dialog(ref t) => result.dialog(&[t.clone()]),
                Index(i) => result.index(i),
                TimespanBegin(t) => result.begin(t + current_delta.unwrap()),
                TimespanEnd(t) => {
                    let new_result = result.end(t + current_delta.unwrap());
                    current_delta = i.next();
                    new_result
                }
            };
            result_opt = Some(new_result);
        });
        Ok(Box::new(result_opt.unwrap()))
    }

    /// Returns a string in the respective format (.ssa, .srt, etc.) with the
    /// corrected time spans.
    fn to_data_string(&self) -> ProgramResult<String> {
        let mut result = Vec::<String>::new();
        self.visit(&mut |part: &SubRipAnnotations| {
            use self::SubRipAnnotations::*;
            match *part {
                Space(ref t) | Dialog(ref t) => {
                    result.push(t.clone());
                }
                Index(i) => {
                    result.push(i.to_string());
                }
                TimespanBegin(t) | TimespanEnd(t) => {
                    result.push(format!("{:02}:{:02}:{:02},{:03}",
                                        t.hours(),
                                        t.mins_comp(),
                                        t.secs_comp(),
                                        t.msecs_comp()));
                }
            }
        });

        Ok(result.into_iter().collect())
    }
}

/// The whole .srt file will be split into semantic segments (index, text,
/// timepan information) and this enum provides the information which
/// information a segment holds.
#[derive(Debug, Clone)]
enum SubRipAnnotations {
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
    v: Vec<SubRipAnnotations>,
}

impl SubRipFile {
    fn new() -> SubRipFile {
        SubRipFile { v: Vec::new() }
    }

    fn begin(mut self, i: TimePoint) -> SubRipFile {
        self.v.push(SubRipAnnotations::TimespanBegin(i));
        self
    }

    fn end(mut self, i: TimePoint) -> SubRipFile {
        self.v.push(SubRipAnnotations::TimespanEnd(i));
        self
    }

    fn index(mut self, i: i64) -> SubRipFile {
        self.v.push(SubRipAnnotations::Index(i));
        self
    }

    fn dialog(mut self, i: &[String]) -> SubRipFile {
        let dialog: String = i.into_iter().flat_map(|i| i.chars()).collect();
        self.v.push(SubRipAnnotations::Dialog(dialog));
        self
    }

    fn space(mut self, i: String) -> SubRipFile {
        let mut changed = false;
        if let Some(&mut SubRipAnnotations::Space(ref mut last_str)) = self.v.last_mut() {
            *last_str = (*last_str).clone() + i.as_str();
            changed = true;
        }
        if !changed {
            self.v.push(SubRipAnnotations::Space(i));
        }
        self
    }

    fn spaces(self, i: &[String]) -> SubRipFile {
        let space: String = i.into_iter().flat_map(|i| i.chars()).collect();
        self.space(space)
    }

    fn append(mut self, mut i: SubRipFile) -> SubRipFile {
        for t in i.v.drain(..) {
            match t {
                SubRipAnnotations::Space(s) => {
                    self = self.space(s);
                }
                s => {
                    self.v.push(s);
                }
            }
        }
        self
    }

    /// "Visitor pattern": Go through each annotation and leave the computation
    /// to the caller.
    fn visit<F: FnMut(&SubRipAnnotations)>(&self, mut f: F) {
        for a in &self.v {
            f(a);
        }
    }
}

/// Matches a `SubRip` timestamp like "00:24:45,670"
fn srt_timestamp<I>(input: I) -> ParseResult<TimePoint, I>
    where I: Stream<Item = char>
{
    (p(number_i64), char(':'), p(number_i64), char(':'), p(number_i64), char(','), p(number_i64))
        .map(|t| TimePoint::from_components(t.0, t.2, t.4, t.6))
        .expected("SubRip timestamp")
        .parse_stream(input)
}


/// Matches a `SubRip` timespan like "00:24:45,670 --> 00:24:45,680".
fn srt_timespan<I>(input: I) -> ParseResult<SubRipFile, I>
    where I: Stream<Item = char>
{
    (p(srt_timestamp), many(ws()), string("-->"), many(ws()), p(srt_timestamp))
        .map(|t| SubRipFile::new().begin(t.0).spaces(&[t.1, t.2.to_string(), t.3]).end(t.4))
        .expected("SubRip timespan")
        .parse_stream(input)
}

/// Matches a line with a single index.
fn srt_timespan_line<I>(input: I) -> ParseResult<SubRipFile, I>
    where I: Stream<Item = char>
{
    (many(ws()), p(srt_timespan), many(ws()), newl())
        .map(|t: (String, _, String, &str)| SubRipFile::new().space(t.0).append(t.1).spaces(&[t.2, t.3.to_string()]))
        .parse_stream(input)
}

/// Matches a line with a single index.
fn srt_index_line<I>(input: I) -> ParseResult<SubRipFile, I>
    where I: Stream<Item = char>
{
    (many(ws()), p(number_i64), many(ws()), newl())
        .map(|(ws1, num, ws2, nl): (_, _, _, &str)| SubRipFile::new().space(ws1).index(num).spaces(&[ws2, nl.to_string()]))
        .parse_stream(input)
}


/// Matches a block with index, timspan text and emptylines.
fn srt_block<I>(input: I) -> ParseResult<SubRipFile, I>
    where I: Stream<Item = char>
{
    (p(srt_index_line), p(srt_timespan_line), many1(p(non_emptyline)), many1(p(emptyline)))
        .map(|(idx, ts, dialog, l): (_, _, Vec<_>, Vec<_>)| SubRipFile::new().append(idx).append(ts).dialog(&dialog).spaces(&l))
        .parse_stream(input)
}

/// Matches a srt file (without BOMs etc.)
fn srt_file<I>(input: I) -> ParseResult<SubRipFile, I>
    where I: Stream<Item = char>
{
    (optional(boms()), many(multispace()), many1(try(p(srt_block))), many(ws()), eof())
        .map(|(bom_opt, s1, blocks, s2, _): (Option<&str>, String, Vec<_>, String, _)| {
            let mut s = SubRipFile::new();
            if let Some(bom) = bom_opt {
                s = s.space(bom.to_string());
            }
            s = s.space(s1);
            for r in blocks {
                s = s.append(r);
            }
            s = s.space(s2);
            s
        })
        .parse_stream(input)
}
