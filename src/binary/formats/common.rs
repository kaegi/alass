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



use std::str::FromStr;
use std::fmt::Display;
use std::result::Result as StdResult;

use combine::char::*;
use combine::combinator::*;
use combine::primitives::{ParseError, ParseResult, Parser, Stream};

type CustomCharParser<I> = Expected<Satisfy<I, fn(char) -> bool>>;
type NewlineParser<I> = Or<With<Token<I>, Value<I, &'static str>>, With<(Token<I>, Token<I>), Value<I, &'static str>>>;

/// Parses whitespaces and tabs.
#[inline]
pub fn ws<I>() -> CustomCharParser<I>
    where I: Stream<Item = char>
{
    fn f(c: char) -> bool {
        c == ' ' || c == '\t'
    }
    satisfy(f as fn(_) -> _).expected("tab or space")
}

/// Parses newline and carriage return.
#[inline]
pub fn newl<I>() -> NewlineParser<I>
    where I: Stream<Item = char>
{
    let nl: _ = token('\n').with(value("\n"));
    let y: _ = (token('\r'), token('\n')).with(value("\r\n"));
    or(nl, y)
}

/// Parses everything but newline and carriage returns.
#[inline]
pub fn no_newl<I>() -> CustomCharParser<I>
    where I: Stream<Item = char>
{
    fn f(c: char) -> bool {
        c != '\r' && c != '\n'
    }
    satisfy(f as fn(_) -> _).expected("non-line break character (neither \\n nor \\r)")
}

/// Parses everything but spaces, tabs, newlines and carriage returns.
#[inline]
pub fn no_multispace<I>() -> CustomCharParser<I>
    where I: Stream<Item = char>
{
    fn f(c: char) -> bool {
        c != '\r' && c != '\n' && c != '\t' && c != ' '
    }
    satisfy(f as fn(_) -> _).expected("non-whitespace characters (everything but a space, '\\t', '\\n' or \\r)")
}


/// Matches a srt file (without BOMs etc.)
pub fn boms<I>() -> Or<Str<I>, Str<I>>
    where I: Stream<Item = char>
{
    or(string("\u{FEFF}"), string("\u{EFBB}\u{BF}"))
}

/// Matches a block with index, timespan and text (and empty lines).
pub fn emptyline<I>(input: I) -> ParseResult<String, I>
    where I: Stream<Item = char>
{
    (many(ws()), newl())
        .map(|t: (_, &str)| [t.0, t.1.to_string()].into_iter().flat_map(|s| s.chars()).collect())
        .parse_stream(input)
}

/// Matches a block with index, timespan and text (and empty lines).
pub fn non_emptyline<I>(input: I) -> ParseResult<String, I>
    where I: Stream<Item = char>
{
    (many(ws()), no_multispace(), many(no_newl()), newl())
        .map(|t| [t.0, (t.1 as char).to_string(), t.2, t.3.to_string()].into_iter().flat_map(|s| s.chars()).collect())
        .parse_stream(input)
}


/// Matches a positive or negative intger number.
pub fn number_i64<I>(input: I) -> ParseResult<i64, I>
    where I: Stream<Item = char>
{
    (optional(char('-')), many1(digit()))
        .map(|(a, c): (Option<_>, String)| {
            // we provide a string that only contains digits: this unwrap should never fail
            let i: i64 = FromStr::from_str(&c).unwrap();
            match a {
                Some(_) => -i,
                None => i,
            }
        })
        .expected("positive or negative number")
        .parse_stream(input)
}

/// Create a single-line-error string from a combine parser error.
pub fn parse_error_to_string<I, R, P>(p: ParseError<I>) -> String
    where I: Stream<Item = char, Range = R, Position = P>,
          R: PartialEq + Clone + Display,
          P: Ord + Display
{
    p.to_string().trim().lines().fold("".to_string(),
                                      |a, b| if a.is_empty() { b.to_string() } else { a + "; " + b })
}


/// This function does a very common task for non-destructive parsers: merging mergable consecutive file parts.
///
/// Each file has some "filler"-parts in it (unimportant information) which only get stored to reconstruct the
/// original file. Two consecutive filler parts (their strings) can be merged. This function abstracts over the
/// specific file part type.
pub fn dedup_string_parts<T, F>(v: Vec<T>, mut extract_fn: F) -> Vec<T>
    where F: FnMut(&mut T) -> Option<&mut String>
{

    let mut result = Vec::new();
    for mut part in v {
        let mut push_part = true;
        if let Some(last_part) = result.last_mut() {
            if let Some(exchangeable_text) = extract_fn(last_part) {
                if let Some(new_text) = extract_fn(&mut part) {
                    exchangeable_text.push_str(new_text);
                    push_part = false;
                }
            }
        }

        if push_part {
            result.push(part);
        }
    }

    result
}

// used in `get_lines_non_destructive()`
type SplittedLine = (String /* string */, String /* newline string like \n or \r\n */);
type ErrorWithLineInfo = (usize /* error line number */, String /* error string */);

/// Iterates over all lines in `s` and calls the `process_line` closure for every line and line ending.
/// This ensures that we can reconstuct the file with correct line endings.
pub fn get_lines_non_destructive(mut s: &str) -> StdResult<Vec<SplittedLine>, ErrorWithLineInfo> {
    // go through each line
    let mut line_num = 0;
    let mut result = Vec::new();
    loop {
        // we need to go this way, because we also want to restore line breaks. A "{String}.lines()" call would discard them.
        let line_parse_result = (many(no_newl()), or(newl(), with(eof(), string("")))).parse(s);
        match line_parse_result {
            Ok(((line, newl), rest)) => {
                result.push((line, newl.to_string()));
                if rest.is_empty() {
                    return Ok(result);
                }
                s = rest;
            }
            Err(err) => {
                return Err((line_num, parse_error_to_string(err)));
            }
        }

        line_num += 1;
    }
}

#[test]
fn get_lines_non_destructive_test0() {
    let lines = ["aaabb", "aaabb\r\nbcccc\n\r\n ", "aaabb\r\nbcccc"];
    for &full_line in lines.into_iter() {
        let split_line = get_lines_non_destructive(full_line).unwrap();
        let joined: String = split_line.into_iter().flat_map(|(s1, s2)| vec![s1, s2].into_iter()).collect();
        assert_eq!(full_line, joined);
    }
}


/// Trim a string left and right, but also preserve the white-space characters. The
/// seconds element in the returned tuple contains the non-whitespace string.
pub fn trim_non_destructive(s: &str) -> (String, String, String) {
    let (begin, rest) = trim_left(s);
    let (end, rest2) = trim_left(&rest.chars().rev().collect::<String>());
    (begin, rest2.chars().rev().collect(), end.chars().rev().collect())
}

/// Splits a string in whitespace string and the rest "   hello " -> ("   ", "hello ").
fn trim_left(s: &str) -> (String, String) {
    (many(ws()), many(try(any())), eof()).map(|t| (t.0, t.1)).parse(s).expect("the trim parser should accept any input").0
}
