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


use errors::*;
use std::fmt::{Debug, Display, Formatter, Result as FmtResult};
use std::ops::{Add, AddAssign, Neg, Sub};

pub mod srt;
pub mod ssa;
pub mod idx;
pub mod common;


pub trait ParseSubtitle {
    type Result;

    fn parse(s: String) -> Result<Self::Result>;
}

/// Represents a timepoint (e.g. start timepoint of a subtitle line).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Timing(i64 /* number of milliseconds */);

impl Debug for Timing {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "Timing({})", self.to_string())
    }
}

impl Display for Timing {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let t = if self.0 < 0 { -*self } else { *self };
        write!(f,
               "{}{}:{:02}:{:02}.{:03}",
               if self.0 < 0 { "-" } else { "" },
               t.hours(),
               t.mins_comp(),
               t.secs_comp(),
               t.msecs_comp())
    }
}

impl Timing {
    pub fn from_components(hours: i64, mins: i64, secs: i64, ms: i64) -> Timing {
        Timing(ms + 1000 * (secs + 60 * (mins + 60 * hours)))
    }

    pub fn from_msecs(ms: i64) -> Timing {
        Timing(ms)
    }

    pub fn from_minutes(mins: i64) -> Timing {
        Timing(mins * 1000 * 60)
    }

    pub fn msecs(&self) -> i64 {
        self.0
    }

    pub fn hours(&self) -> i64 {
        self.0 / (60 * 60 * 1000)
    }

    pub fn mins_comp(&self) -> i64 {
        (self.0 / (60 * 1000)) % 60
    }

    pub fn secs_comp(&self) -> i64 {
        (self.0 / 1000) % 60
    }

    pub fn csecs_comp(&self) -> i64 {
        ((self.0 / 10) % 100)
    }

    pub fn msecs_comp(&self) -> i64 {
        (self.0 % 1000)
    }

    pub fn is_negative(&self) -> bool {
        self.0 < 0
    }
}

impl Add for Timing {
    type Output = Timing;
    fn add(self, rhs: Timing) -> Timing {
        Timing(self.0 + rhs.0)
    }
}

impl Sub for Timing {
    type Output = Timing;
    fn sub(self, rhs: Timing) -> Timing {
        Timing(self.0 - rhs.0)
    }
}

impl AddAssign for Timing {
    fn add_assign(&mut self, r: Timing) {
        self.0 += r.0;
    }
}

impl Neg for Timing {
    type Output = Timing;
    fn neg(self) -> Timing {
        Timing(-self.0)
    }
}

pub type TimePoint = Timing;
pub type TimeDelta = Timing;
pub type TimeSpan = (Timing, Timing);


pub trait MutableTimedData {
    /// Timings can only be changed by `shifted_by_deltas`.
    fn get_timespans(&self) -> Result<Vec<TimeSpan>>;

    /// Shift the timespans from the `get_timespans()` vector by the given
    /// duration.
    ///
    /// The length of the given iterator should always match the length of
    /// `get_timespans()`.
    fn shift_by_deltas(&self, i: &mut Iterator<Item = Timing>) -> Result<Box<MutableTimedData>>;

    /// Returns a string in the respective format (.ssa, .srt, etc.) with the
    /// corrected time spans.
    fn to_data_string(&self) -> Result<String>;
}


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubtitleFormat {
    /// .srt file
    SubRip,

    /// .ssa/.ass file
    SubStationAlpha,

    /// .idx file
    VobSubIdx,
}

pub fn get_subtitle_format(path: &str) -> Option<SubtitleFormat> {

    if path.ends_with(".srt") {
        Some(SubtitleFormat::SubRip)
    } else if path.ends_with(".ssa") || path.ends_with(".ass") {
        Some(SubtitleFormat::SubStationAlpha)
    } else if path.ends_with(".idx") {
        Some(SubtitleFormat::VobSubIdx)
    } else {
        None
    }
}


pub fn parse_file(format: SubtitleFormat, content: String) -> Result<Box<MutableTimedData>> {
    match format {
        SubtitleFormat::SubRip => Ok(Box::new(srt::SrtParser::parse(content)?)),
        SubtitleFormat::SubStationAlpha => Ok(Box::new(ssa::SsaParser::parse(content)?)),
        SubtitleFormat::VobSubIdx => Ok(Box::new(idx::IdxParser::parse(content)?)),
    }
}


pub fn get_subtitle_format_with_error(path: &str) -> Result<SubtitleFormat> {
    match get_subtitle_format(path) {
        Some(format) => Ok(format),
        None => Err(Error::from(ErrorKind::UnknownFileFormat)).chain_err(|| ErrorKind::FileOperation(path.to_string())),
    }
}
