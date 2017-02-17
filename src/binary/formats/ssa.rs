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

use std::iter::once;

use combine::char::*;
use combine::combinator::*;
use combine::primitives::Parser;

// see https://docs.rs/error-chain/0.8.1/error_chain/
error_chain! {
    errors {
        SsaFieldsInfoNotFound {
            description(".ssa/.ass file did not have a line beginning with `Format: ` in a `[Events]` section")
        }
        SsaMissingField(line_num: usize, f: &'static str) {
            display("the '{}' field is missing in the field info in line {}", f, line_num)
        }
        SsaDuplicateField(line_num: usize, f: &'static str) {
            display("the '{}' field is twice in the field info in line {}", f, line_num)
        }
        SsaTextFieldNotLast(line_num: usize) {
            display("the field info in line {} has to have `Text` as its last field", line_num)
        }
        SsaIncorrectNumberOfFields(line_num: usize) {
            display("the dialog at line {} has incorrect number of fields", line_num)
        }
        SsaWrongTimepointFormat(line_num: usize, string: String) {
            display("the timepoint `{}` in line {} has wrong format", string, line_num)
        }
        SsaDialogLineParseError(line_num: usize, msg: String) {
            display("parsing the line `{}` failed because of `{}`", line_num, msg)
        }
        SsaLineParseError(line_num: usize, msg: String) {
            display("parsing the line `{}` failed because of `{}`", line_num, msg)
        }
    }
}

// ////////////////////////////////////////////////////////////////////////////////////////////////
// SSA field info

struct SsaFieldsInfo {
    start_field_idx: usize,
    end_field_idx: usize,
    text_field_idx: usize,
    num_fields: usize,
}

impl SsaFieldsInfo {
    /// Parses a format line like "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text".
    fn new_from_fields_info_line(line_num: usize, s: String) -> Result<SsaFieldsInfo> {
        assert!(s.starts_with("Format:"));
        let field_info = &s["Format:".len()..];
        let mut start_field_idx: Option<usize> = None;
        let mut end_field_idx: Option<usize> = None;
        let mut text_field_idx: Option<usize> = None;

        // filter "Start" and "End" and "Text"
        let split_iter = field_info.split(',');
        let num_fields = split_iter.clone().count();
        for (i, field_name) in split_iter.enumerate() {
            let trimmed = field_name.trim();
            if trimmed == "Start" {
                if start_field_idx.is_some() {
                    return Err(ErrorKind::SsaDuplicateField(line_num, "Start"))?;
                }
                start_field_idx = Some(i);
            } else if trimmed == "End" {
                if end_field_idx.is_some() {
                    return Err(ErrorKind::SsaDuplicateField(line_num, "End"))?;
                }
                end_field_idx = Some(i);
            } else if trimmed == "Text" {
                if text_field_idx.is_some() {
                    return Err(ErrorKind::SsaDuplicateField(line_num, "Text"))?;
                }
                text_field_idx = Some(i);
            }
        }

        let text_field_idx2 = text_field_idx.ok_or_else(|| Error::from(ErrorKind::SsaMissingField(line_num, "Text")))?;
        if text_field_idx2 != num_fields - 1 {
            return Err(ErrorKind::SsaTextFieldNotLast(line_num))?;
        }

        Ok(SsaFieldsInfo {
            start_field_idx: start_field_idx.ok_or_else(|| Error::from(ErrorKind::SsaMissingField(line_num, "Start")))?,
            end_field_idx: end_field_idx.ok_or_else(|| Error::from(ErrorKind::SsaMissingField(line_num, "End")))?,
            text_field_idx: text_field_idx2,
            num_fields: num_fields,
        })
    }
}


// ////////////////////////////////////////////////////////////////////////////////////////////////
// SSA parser

pub struct SsaParser;

impl ParseSubtitle for SsaParser {
    type Result = SsaFile;

    fn parse(s: String) -> ProgramResult<SsaFile> {
        match Self::parse_inner(s) {
            Ok(v) => Ok(v),
            Err(e) => Err(e.into()),
        }
    }
}

impl SsaParser {
    /// Parses a whole `.ssa` file from string.
    fn parse_inner(s: String) -> Result<SsaFile> {
        // first we need to find and parse the format line, which then dictates how to parse the file
        let (line_num, field_info_line) = Self::get_format_info(&s)?;
        let fields_info = SsaFieldsInfo::new_from_fields_info_line(line_num, field_info_line)?;

        // parse the dialog lines with the given format
        let file_parts = Self::parse_dialog_lines(&fields_info, &s)?;
        Ok(SsaFile::new(file_parts))
    }

    /// Searches and parses a format line like "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text".
    fn get_format_info(s: &str) -> Result<(usize, String)> {
        let mut section_opt = None;
        for (line_num, line) in s.lines().enumerate() {
            // parse section headers like `[Events]`
            let trimmed_line = line.trim();
            if trimmed_line.starts_with('[') && trimmed_line.ends_with(']') {
                section_opt = Some(&trimmed_line[1..trimmed_line.len() - 1]);
            }

            // most sections have a format line, but we only want the one for the subtitle events
            if section_opt != Some("Events") {
                continue;
            }
            if !line.trim().starts_with("Format:") {
                continue;
            }
            return Ok((line_num, line.to_string()));
        }

        Err(ErrorKind::SsaFieldsInfoNotFound.into())
    }

    /// Filters file for lines like this and parses them:
    ///
    /// ```text
    /// "Dialogue: 1,0:22:43.52,0:22:46.22,ED-Romaji,,0,0,0,,{\fad(150,150)\blur0.5\bord1}some text"
    /// ```
    fn parse_dialog_lines(fields_info: &SsaFieldsInfo, s: &str) -> Result<Vec<SsaFilePart>> {
        let mut result = Vec::new();
        let mut section_opt: Option<String> = None;
        let lines_with_newl: Vec<(String, String)> = get_lines_non_destructive(s)
            .map_err(|(line_num, err_str)| ErrorKind::SsaLineParseError(line_num, err_str))?;

        for (line_num, (line, newl)) in lines_with_newl.into_iter().enumerate() {
            let trimmed_line = line.trim().to_string();

            // parse section headers like `[Events]`
            if trimmed_line.starts_with('[') && trimmed_line.ends_with(']') {
                section_opt = Some(trimmed_line[1..trimmed_line.len() - 1].to_string());
                result.push(SsaFilePart::Filler(line));
                result.push(SsaFilePart::Filler("\n".to_string()));
                continue;
            }

            if section_opt.is_none() || section_opt.iter().any(|s| s != "Events") || !trimmed_line.starts_with("Dialogue:") {
                result.push(SsaFilePart::Filler(line));
                result.push(SsaFilePart::Filler("\n".to_string()));
                continue;
            }

            result.append(&mut Self::parse_dialog_line(line_num, line.as_str(), fields_info)?);
            result.push(SsaFilePart::Filler(newl));
        }

        Ok(result)
    }

    /// Parse lines like:
    ///
    /// ```text
    /// "Dialogue: 1,0:22:43.52,0:22:46.22,ED-Romaji,,0,0,0,,{\fad(150,150)\blur0.5\bord1}some text"
    /// ```
    fn parse_dialog_line(line_num: usize, line: &str, fields_info: &SsaFieldsInfo) -> Result<Vec<SsaFilePart>> {
        let parts_res = (many(ws()),
                         string("Dialogue:"),
                         many(ws()),
                         count(fields_info.num_fields - 1,
                               (many(none_of(once(','))), token(','))),
                         many(try(any())))
                .map(|(ws1, dl, ws2, v, text): (String, &str, String, Vec<(String, char)>, String)| -> Result<Vec<SsaFilePart>> {
                    let mut result: Vec<SsaFilePart> = Vec::new();
                    result.push(SsaFilePart::Filler(ws1));
                    result.push(SsaFilePart::Filler(dl.to_string()));
                    result.push(SsaFilePart::Filler(ws2.to_string()));
                    result.append(&mut Self::parse_fields(line_num, fields_info, v)?);
                    result.push(SsaFilePart::Text(text));
                    Ok(result)
                })
                .parse(line);

        match parts_res {
            // Ok() means that parsing succeded, but the "map" function might created an SSA error
            Ok((parts, _)) => Ok(parts?),
            Err(e) => Err(ErrorKind::SsaDialogLineParseError(line_num, parse_error_to_string(e)).into()),
        }
    }


    /// Parses an array of fields with the "fields info".
    ///
    /// The fields (comma seperated information) as an array like `vec!["1", "0:22:43.52", "0:22:46.22", "ED-Romaji", "", "0", "0", "0", "", "{\fad(150,150)\blur0.5\bord1}some text"]`.
    fn parse_fields(line_num: usize, fields_info: &SsaFieldsInfo, v: Vec<(String, char)>) -> Result<Vec<SsaFilePart>> {
        let extract_file_parts_closure = |(i, (field, sep_char)): (_, (String, char))| -> Result<Vec<SsaFilePart>> {
            let (begin, field, end) = trim_non_destructive(&field);

            let part = if i == fields_info.start_field_idx {
                SsaFilePart::TimespanStart(Self::parse_timepoint(line_num, &field)?)
            } else if i == fields_info.end_field_idx {
                SsaFilePart::TimespanEnd(Self::parse_timepoint(line_num, &field)?)
            } else if i == fields_info.text_field_idx {
                SsaFilePart::Text(field.to_string())
            } else {
                SsaFilePart::Filler(field.to_string())
            };

            Ok(vec![SsaFilePart::Filler(begin), part, SsaFilePart::Filler(end), SsaFilePart::Filler(sep_char.to_string())])
        };

        let result = v.into_iter()
                      .enumerate()
                      .map(extract_file_parts_closure)
                      .collect::<Result<Vec<Vec<SsaFilePart>>>>()?
                      .into_iter()
                      .flat_map(|part| part)
                      .collect();
        Ok(result)
    }

    /// Something like "0:19:41.99"
    fn parse_timepoint(line: usize, s: &str) -> Result<TimePoint> {
        let parse_res = (parser(number_i64),
                         token(':'),
                         parser(number_i64),
                         token(':'),
                         parser(number_i64),
                         or(token('.'), token(':')),
                         parser(number_i64),
                         eof())
                .map(|(h, _, mm, _, ss, _, ms, _)| TimePoint::from_components(h, mm, ss, ms * 10))
                .parse(s);
        match parse_res {
            Ok(res) => Ok(res.0),
            Err(e) => Err(ErrorKind::SsaWrongTimepointFormat(line, parse_error_to_string(e)).into()),
        }
    }
}

// ////////////////////////////////////////////////////////////////////////////////////////////////
// SSA file parts

#[derive(Debug, Clone)]
enum SsaFilePart {
    /// Spaces, field information, comments, unimportant fields, ...
    Filler(String),

    /// Timespan start of a dialogue line
    TimespanStart(TimePoint),

    /// Timespan end of a dialogue line
    TimespanEnd(TimePoint),

    /// Dialog lines
    Text(String),
}


// ////////////////////////////////////////////////////////////////////////////////////////////////
// SSA file

/// Represents a reconstructable SSA file.
///
/// All unimportant information (for this project) are saved into `SsaFilePart::Filler(...)`, so
/// a timespan-altered file still has the same field etc.
#[derive(Debug, Clone)]
pub struct SsaFile {
    v: Vec<SsaFilePart>,
}

impl SsaFile {
    fn new(v: Vec<SsaFilePart>) -> SsaFile {
        // cleans up multiple fillers after another
        let new_file_parts = dedup_string_parts(v, |part: &mut SsaFilePart| {
            match *part {
                SsaFilePart::Filler(ref mut text) => Some(text),
                _ => None,
            }
        });

        SsaFile { v: new_file_parts }
    }
}

impl MutableTimedData for SsaFile {
    /// Durations can only be changed by `shifted_by_deltas`.
    fn get_timespans(&self) -> ProgramResult<Vec<TimeSpan>> {
        #[derive(Debug, Clone, Copy, PartialEq,Eq)]
        enum SsaState {
            Plain,
            HasStart(TimePoint),
            HasEnd(TimePoint),
        }

        let mut state = SsaState::Plain;

        // the extra block satisfies the borrow checker
        let timings: Vec<_> = {
            let filter_map_closure = |part: &SsaFilePart| {
                use self::SsaFilePart::*;
                match *part {
                    TimespanStart(ref start) => {
                        match state {
                            SsaState::Plain => {
                                state = SsaState::HasStart(*start);
                                None
                            }
                            SsaState::HasStart(_) => panic!("parser should have ensured that no two consecutive SSA start times exist"),
                            SsaState::HasEnd(end) => {
                                state = SsaState::Plain;
                                Some((*start, end))
                            }
                        }
                    }
                    TimespanEnd(ref end) => {
                        match state {
                            SsaState::Plain => {
                                state = SsaState::HasEnd(*end);
                                None
                            }
                            SsaState::HasEnd(_) => panic!("parser should have ensured that no two consecutive SSA end times exist"),
                            SsaState::HasStart(start) => {
                                state = SsaState::Plain;
                                Some((start, *end))
                            }
                        }
                    }
                    Filler(_) | Text(_) => None,
                }
            };

            self.v
                .iter()
                .filter_map(filter_map_closure)
                .collect()
        };

        // every timespan should now consist of a beginning and a end (this should be ensured by parser)
        assert_eq!(state, SsaState::Plain);

        Ok(timings)
    }

    /// Shift the timespans from the `get_timespans()` vector by the given
    /// duration.
    ///
    /// The length of the given iterator should always match the length of
    /// `get_timespans()`.
    fn shift_by_deltas(&self, i: &mut Iterator<Item = TimeDelta>) -> ProgramResult<Box<MutableTimedData>> {
        let mut to_shift: (Option<TimeDelta>, // need to shift start with current timing
                           Option<TimeDelta> /* need to shift end with current timing */) = (None, None);

        // this block indent satisfies the borrow checker
        let new_file_parts: Vec<_> = {

            let map_closure = |part: &SsaFilePart| {
                use self::SsaFilePart::*;
                match *part {
                    ref p @ Filler(_) |
                    ref p @ Text(_) => p.clone(),
                    TimespanStart(start) => {
                        if to_shift == (None, None) {
                            let timing = i.next().expect("too few iterator items");
                            to_shift = (Some(timing), Some(timing));
                        }
                        let shift_by = to_shift.0.expect("parser should have ensured that no two consecutive SSA start times exist");
                        to_shift.0 = None;
                        SsaFilePart::TimespanStart(start + shift_by)
                    }
                    TimespanEnd(end) => {
                        if to_shift == (None, None) {
                            let timing = i.next().expect("too few iterator items");
                            to_shift = (Some(timing), Some(timing));
                        }
                        let shift_by = to_shift.1.expect("parser should have ensured that no two consecutive SSA start times exist");
                        to_shift.1 = None;
                        SsaFilePart::TimespanStart(end + shift_by)
                    }
                }
            };

            self.v
                .iter()
                .map(map_closure)
                .collect()
        };

        // every timespan should now consist of a beginning and a end (this should be ensured by parser)
        assert_eq!(to_shift, (None, None));
        assert_eq!(i.next(), None); // iterator should be completely empty by now

        Ok(Box::new(SsaFile::new(new_file_parts)))
    }

    /// Returns a string in the respective format (.ssa, .srt, etc.) with the
    /// corrected time spans.
    fn to_data_string(&self) -> ProgramResult<String> {
        // timing to string like "0:00:22.21"
        let fn_timing_to_string = |t: TimePoint| {
            let p = if t.0 < 0 { -t } else { t };
            format!("{}{}:{:02}:{:02}.{:02}",
                    if t.0 < 0 { "-" } else { "" },
                    p.hours(),
                    p.mins_comp(),
                    p.secs_comp(),
                    p.csecs_comp())
        };

        let fn_file_part_to_string = |part: &SsaFilePart| {
            use self::SsaFilePart::*;
            match *part {
                Filler(ref t) | Text(ref t) => t.clone(),
                TimespanStart(start) => fn_timing_to_string(start),
                TimespanEnd(end) => fn_timing_to_string(end),
            }
        };

        let result: String = self.v
                                 .iter()
                                 .map(fn_file_part_to_string)
                                 .collect();

        Ok(result)
    }
}
