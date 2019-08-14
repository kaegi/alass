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

use std;
use std::cmp::{max, min, Ordering};
use std::ops::*;

/// Implements conversion to integer variables for TimeDelta and TimePoint.
macro_rules! impl_from {
    ($f:ty, $t:ty) => {
        impl From<$f> for $t {
            fn from(t: $f) -> $t {
                t.0 as $t
            }
        }
    };
}

/// This struct represents a time difference between two `TimePoints`.
/// Internally its an integer type.
#[derive(Copy, Clone, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct TimeDelta(i64);

impl TimeDelta {
    /// No difference in time.
    pub fn zero() -> TimeDelta {
        TimeDelta(Default::default())
    }

    /// Smallest positive time difference the library can work with.
    pub fn one() -> TimeDelta {
        TimeDelta(1)
    }

    /// Create time delta as "TimeDelta::one() * v".
    pub fn from_i64(v: i64) -> TimeDelta {
        TimeDelta(v)
    }

    /// Return time difference as f64.
    pub fn as_f64(&self) -> f64 {
        self.0 as f64
    }

    /// Return time difference as f64.
    pub fn as_f32(&self) -> f32 {
        self.0 as f32
    }

    /// Return time difference as i64.
    pub fn as_i64(&self) -> i64 {
        self.0 as i64
    }
}

impl_from!(TimeDelta, i32);
impl_from!(TimeDelta, u32);
impl_from!(TimeDelta, i64);
impl_from!(TimeDelta, u64);

impl std::iter::Sum for TimeDelta {
    fn sum<I: Iterator<Item = TimeDelta>>(iter: I) -> TimeDelta {
        TimeDelta(iter.map(|d| d.0).sum())
    }
}

impl std::fmt::Display for TimePoint {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::fmt::Display for TimeDelta {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Add for TimeDelta {
    type Output = TimeDelta;
    fn add(self, rhs: TimeDelta) -> TimeDelta {
        TimeDelta(self.0 + rhs.0)
    }
}

impl AddAssign<TimeDelta> for TimeDelta {
    fn add_assign(&mut self, rhs: TimeDelta) {
        self.0 += rhs.0;
    }
}

impl Sub<TimeDelta> for TimeDelta {
    type Output = TimeDelta;
    fn sub(self, rhs: TimeDelta) -> TimeDelta {
        TimeDelta(self.0 - rhs.0)
    }
}

impl SubAssign<TimeDelta> for TimeDelta {
    fn sub_assign(&mut self, rhs: TimeDelta) {
        self.0 -= rhs.0;
    }
}

impl Mul<i64> for TimeDelta {
    type Output = TimeDelta;
    fn mul(self, rhs: i64) -> TimeDelta {
        TimeDelta(self.0 * rhs)
    }
}

impl MulAssign<i64> for TimeDelta {
    fn mul_assign(&mut self, rhs: i64) {
        self.0 *= rhs;
    }
}

impl Mul<TimeDelta> for i64 {
    type Output = TimeDelta;
    fn mul(self, rhs: TimeDelta) -> TimeDelta {
        TimeDelta(self * rhs.0)
    }
}

impl Neg for TimeDelta {
    type Output = TimeDelta;
    fn neg(self) -> TimeDelta {
        TimeDelta(-self.0)
    }
}

// //////////////////////////////////////////////////////////////////////////////////////////////////
// struct TimeSpan

/// Represents a timepoint in your own metric.
///
/// A timepoint is internally represented by an integer (because the align
/// algorithm needs discrete
/// time steps). You will have to choose your own metric: for example 1i64 means
/// 2ms. The internal algorithm does not use any non-user given `TimePoint`s
/// (so its interpretation is
/// up to you).
///
/// This is the reason this library works with `TimePoint` and `TimeDelta`: to
/// enforce
/// an absolute and delta relationship an a own metric.
///
/// The only way to create a new `TimePoint` is with `TimePoint::from({i64})`.
///
/// ```
/// use alass_core::TimePoint;
///
/// let p = TimePoint::from(10);
///
/// // to get that i64 again
/// let i1: i64 = p.into();
/// let i2 = i64::from(p);
/// ```
///
#[derive(Copy, Clone, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct TimePoint(i64);

impl TimePoint {
    /// Returns a f32 for the given time point.
    pub fn as_f32(self) -> f32 {
        self.0 as f32
    }

    /// Returns a i64 for the given time point.
    pub fn as_i64(self) -> i64 {
        self.0 as i64
    }
}

impl From<i64> for TimePoint {
    fn from(f: i64) -> TimePoint {
        TimePoint(f)
    }
}
impl_from!(TimePoint, i64);

impl Sub for TimePoint {
    type Output = TimeDelta;
    fn sub(self, rhs: TimePoint) -> TimeDelta {
        TimeDelta(self.0 - rhs.0)
    }
}

impl Add<TimeDelta> for TimePoint {
    type Output = TimePoint;
    fn add(self, rhs: TimeDelta) -> TimePoint {
        TimePoint(self.0 + rhs.0)
    }
}

impl AddAssign<TimeDelta> for TimePoint {
    fn add_assign(&mut self, rhs: TimeDelta) {
        self.0 += rhs.0;
    }
}

impl Sub<TimeDelta> for TimePoint {
    type Output = TimePoint;
    fn sub(self, rhs: TimeDelta) -> TimePoint {
        TimePoint(self.0 - rhs.0)
    }
}

impl SubAssign<TimeDelta> for TimePoint {
    fn sub_assign(&mut self, rhs: TimeDelta) {
        self.0 -= rhs.0;
    }
}

// //////////////////////////////////////////////////////////////////////////////////////////////////
// struct TimeSpan

/// Represents a time span from "start" (included) to "end" (excluded).
///
/// The constructors will ensure "start <= end", this condition will hold at
/// any given time.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TimeSpan {
    /// The first time point of the time span (inclusive)
    pub start: TimePoint,

    /// The last time point of the time span (excluded)
    pub end: TimePoint,
}

impl TimeSpan {
    /// Create a new TimeSpan with `start` and `end`.
    ///
    /// # Examples
    /// ```rust
    /// use alass_core::{TimeSpan, TimePoint};
    ///
    /// let t0 = TimePoint::from(0);
    /// let t10 = TimePoint::from(10);
    ///
    /// let ts = TimeSpan::new(t0, t10);
    /// ```
    ///
    /// # Panics
    ///
    ///
    /// This function asserts that `start` is less or equal `end`.
    ///
    /// ```rust,should_panic
    /// use alass_core::{TimeSpan, TimePoint};
    ///
    /// let t0 = TimePoint::from(0);
    /// let t10 = TimePoint::from(10);
    ///
    /// // this will case a panic
    /// let ts = TimeSpan::new(t10, t0);
    /// ```
    #[inline]
    pub fn new(start: TimePoint, end: TimePoint) -> TimeSpan {
        assert!(start <= end);
        TimeSpan { start: start, end: end }
    }

    /// Create a new TimeSpan with `start` and `end`. This function will not
    /// panic on `end < start`, but
    /// swap the values before calling `TimeSpan::new()`.
    ///
    /// # Examples
    /// ```rust
    /// use alass_core::{TimeSpan, TimePoint};
    ///
    /// let t0 = TimePoint::from(0);
    /// let t10 = TimePoint::from(10);
    ///
    /// let ts = TimeSpan::new_safe(t10, t0);
    /// assert!(ts.start() == t0 && ts.end() == t10);
    /// ```
    pub fn new_safe(start: TimePoint, end: TimePoint) -> TimeSpan {
        if end < start {
            TimeSpan::new(end, start)
        } else {
            TimeSpan::new(start, end)
        }
    }

    /// Mutates a `TimeSpan`s end.
    ///
    /// # Panics
    ///
    /// Will panic if `new_end` is less than current `start`.
    pub fn new_copy_with_end(self, new_end: TimePoint) -> TimeSpan {
        TimeSpan::new(self.start, new_end)
    }

    /// Returns the length of the `TimeSpan`.
    ///
    /// `len()` is zero, if and only if `start` is `end`.
    pub fn len(self) -> TimeDelta {
        self.end - self.start
    }

    /// Returns true if `start == end`.
    pub fn is_empty(self) -> bool {
        self.end == self.start
    }

    /// Returns the start point of the `TimeSpan`.
    pub fn start(self) -> TimePoint {
        self.start
    }

    /// Returns the end point of the `TimeSpan`.
    pub fn end(self) -> TimePoint {
        self.end
    }

    /// Returns one (of the possibly two) points in the center of the `TimeSpan`.
    pub fn half(self) -> TimePoint {
        TimePoint::from((self.start.as_i64() + self.end.as_i64()) / 2)
    }

    /// Returns true if `self` contains `TimeSpan` `other`.
    ///
    /// # Examples
    /// ```
    /// use alass_core::{TimeSpan, TimePoint};
    /// ```
    pub fn contains(self, other: TimeSpan) -> bool {
        other.start >= self.start && other.end <= self.end
    }

    /// Returns the smallest difference between two `TimeSpan`s.
    ///
    /// ```
    /// use alass_core::{TimeSpan, TimePoint, TimeDelta};
    ///
    /// let p = TimePoint::from(0);
    /// let d = TimeDelta::one();
    ///
    /// let ts1 = TimeSpan::new(p, p + 10 * d);
    /// let ts4 = TimeSpan::new(p + 20 * d, p + 100 * d);
    ///
    /// assert!(TimeSpan::fast_distance_to(ts1, ts1) == 0 * d);
    /// assert!(TimeSpan::fast_distance_to(ts1, ts4) == 10 * d);
    /// assert!(TimeSpan::fast_distance_to(ts4, ts1) == 10 * d);
    /// assert!(TimeSpan::fast_distance_to(ts4, ts4) == 0 * d);
    /// ```
    pub fn fast_distance_to(self, other: TimeSpan) -> TimeDelta {
        // self < other
        if self.end < other.start {
            other.start - self.end
        }
        // self > other
        else if self.start > other.end {
            self.start - other.end
        }
        // self and other overlap
        else {
            TimeDelta::zero()
        }
    }

    /// Returns the smallest difference between two `TimeSpan`s.
    pub fn get_overlapping_length(self, other: TimeSpan) -> TimeDelta {
        let start_max = max(self.start, other.start);
        let end_min = min(self.end, other.end);
        max(TimeDelta::zero(), end_min - start_max)
    }

    /// Compares two `TimeSpan`s by their start timepoint.
    pub fn cmp_start(self, other: TimeSpan) -> Ordering {
        self.start.cmp(&other.start)
    }

    /// Compares two `TimeSpan`s by their end timepoint.
    pub fn cmp_end(self, other: TimeSpan) -> Ordering {
        self.end.cmp(&other.end)
    }
}

impl Add<TimeDelta> for TimeSpan {
    type Output = TimeSpan;
    fn add(self, rhs: TimeDelta) -> TimeSpan {
        TimeSpan::new(self.start + rhs, self.end + rhs)
    }
}
