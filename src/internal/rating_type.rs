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


use internal::TimeDelta;
use std;
use std::cmp::max;
use std::ops::*;

// these objects determine the precision/length of the rating (i32/i64) - lower
// values take less space and time, higher values have higher precision
type RatingIntern = i64;
const RATING_PRECISION: RatingIntern = (1 << 32);

/// Use an integer for internal rating, because we add MANY small values which
/// lead to precision issues for floats
#[derive(Copy, Clone, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub struct Rating(RatingIntern);
impl std::fmt::Display for Rating {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0 as f64 / RATING_PRECISION as f64)
    }
}


impl Rating {
    pub fn zero() -> Rating {
        Rating(0)
    }
    pub fn from_overlapping_spans(a: TimeDelta, b: TimeDelta) -> Rating {
        let max: RatingIntern = max(a, b).into();
        // if min == 0 || max == 0 { return Self::zero() }

        // the score is "score := min/max" so it ges a score of 1 if both spans are equally long.
        // The score is added in all overlapping segments, so if we don't devide by a length value, we
        // end up with "score * num_overlapping_segments" which is dependent on the choosen resolution
        // and will priorize longer time spans.
        //
        // By deviding this value by "min", which is maximum number of overlapping segments, we can
        // "normalize" the score and get the final rating. But now we have "score/min == (min/max)/min == 1/max".
        //
        // The total score is now "overlapping_percentage_in_min / 100 * score".
        let x = RATING_PRECISION / max;

        Rating(x as RatingIntern)
    }

    pub fn nosplit_bonus(unnormalized: f64) -> Rating {
        Rating((RATING_PRECISION as f64 * unnormalized) as RatingIntern)
    }
}

// There is no absolute `Rating` so, there should be no way to construct one.
// This is pretty nifty for testing though.
#[cfg(test)]
impl From<i64> for Rating {
    fn from(f: i64) -> Rating {
        Rating(f as RatingIntern)
    }
}


impl Add for Rating {
    type Output = Rating;
    fn add(self, c: Rating) -> Rating {
        Rating(self.0 + c.0)
    }
}
impl Sub for Rating {
    type Output = Rating;
    fn sub(self, c: Rating) -> Rating {
        Rating(self.0 - c.0)
    }
}
impl AddAssign for Rating {
    fn add_assign(&mut self, c: Rating) {
        self.0 += c.0;
    }
}
impl SubAssign for Rating {
    fn sub_assign(&mut self, c: Rating) {
        self.0 -= c.0;
    }
}
impl std::iter::Sum for Rating {
    fn sum<I>(iter: I) -> Rating
        where I: Iterator<Item = Rating>
    {
        Rating(iter.map(|c| c.0).sum())
    }
}
impl Mul<u64> for Rating {
    type Output = Rating;
    fn mul(self, rhs: u64) -> Rating {
        Rating(self.0 * rhs as RatingIntern)
    }
}
impl Mul<i64> for Rating {
    type Output = Rating;
    fn mul(self, rhs: i64) -> Rating {
        Rating(self.0 * rhs as RatingIntern)
    }
}
impl Div<Rating> for Rating {
    type Output = i64;
    fn div(self, rhs: Rating) -> i64 {
        (self.0 / rhs.0) as i64
    }
}

impl Neg for Rating {
    type Output = Rating;

    fn neg(self) -> Rating {
        Rating(-self.0)
    }
}
