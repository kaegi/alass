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

use crate::TimeDelta;
use std::cmp::{max, min};

// these objects determine the precision/length of the rating (i32/i64) - lower
// values take less space and time, higher values have higher precision
pub type Rating = i64;
pub type RatingDelta = i64;
pub type RatingDeltaDelta = i64;
pub const RATING_PRECISION: Rating = (1 << 32);

pub trait RatingExt {
    #[inline]
    fn add_mul(r: Rating, rd: RatingDelta, td: TimeDelta) -> Rating {
        r + rd * td.as_i64()
    }

    #[inline]
    fn compute(a: TimeDelta, b: TimeDelta) -> Rating {
        let min: i64 = min(a, b).as_i64();
        let max: i64 = max(a, b).as_i64();
        (min * RATING_PRECISION) / max
    }

    #[inline]
    fn nosplit_bonus(unnormalized: f64) -> Rating {
        (RATING_PRECISION as f64 * unnormalized) as Rating
    }

    #[inline]
    fn zero() -> Rating {
        0
    }

    #[inline]
    fn as_f32(self) -> f32;

    #[inline]
    fn as_f64(self) -> f32;

    #[inline]
    fn as_readable_f32(self) -> f32;
}
impl RatingExt for Rating {
    #[inline]
    fn as_f32(self) -> f32 {
        self as f32
    }

    #[inline]
    fn as_f64(self) -> f32 {
        self as f32
    }

    #[inline]
    fn as_readable_f32(self) -> f32 {
        self as f32 / RATING_PRECISION as f32
    }
}

pub trait RatingDeltaExt {
    #[inline]
    fn compute_rating_delta(a: TimeDelta, b: TimeDelta) -> RatingDelta {
        Rating::compute(a, b) / min(a, b).as_i64()
    }

    #[inline]
    fn from_i64(v: i64) -> RatingDelta {
        v
    }
}
impl RatingDeltaExt for RatingDelta {}
