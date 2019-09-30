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

pub use rating_i64::*;

/*mod rating_f64 {
    use crate::{TimeDelta, TimeSpan};
    use ordered_float::NotNan;
    use std::cmp::{max, min};

    // these objects determine the precision/length of the rating (i32/i64) - lower
    // values take less space and time, higher values have higher precision
    pub type Rating = NotNan<f64>;
    pub type RatingDelta = NotNan<f64>;
    pub type RatingDeltaDelta = NotNan<f64>;

    pub trait RatingExt {
        #[inline]
        fn add_mul(r: Rating, rd: RatingDelta, td: TimeDelta) -> Rating {
            r + rd * td.as_f64()
        }

        #[inline]
        fn add_mul_usize(r: Rating, rd: RatingDelta, td: usize) -> Rating {
            r + rd * td as f64
        }

        #[inline]
        fn compute(a: TimeDelta, b: TimeDelta) -> Rating {
            let min: f64 = min(a, b).as_f64();
            let max: f64 = max(a, b).as_f64();
            NotNan::from(min / max)
        }

        /*#[inline]
        fn compute(a: TimeDelta, b: TimeDelta) -> i64 {
            let min: i64 = min(a, b).as_i64();
            let max: i64 = max(a, b).as_i64();
           (min * RATING_PRECISION) / max
        }*/

        #[inline]
        fn compute2(a: TimeDelta, b: TimeDelta) -> Rating {
            //Self::compute(a, b)
            Self::compute(a, b)
        }

        #[inline]
        fn from_timespans(a: TimeSpan, b: TimeSpan) -> Rating {
            let overlap = TimeSpan::get_overlapping_length(a, b).as_f64();
            let max_rating = Rating::compute2(a.len(), b.len());
            let length_normalization_factor = min(a.len(), b.len()).as_f64();

            NotNan::from(max_rating * overlap / length_normalization_factor)
        }

        #[inline]
        fn nosplit_bonus(unnormalized: f64) -> Rating {
            NotNan::from(unnormalized)
        }

        #[inline]
        fn convert_from_f64(v: f64) -> Rating {
            NotNan::from(v)
        }

        #[inline]
        fn zero() -> Rating {
            NotNan::from(0.)
        }

        #[inline]
        fn is_zero(self) -> bool;

        #[inline]
        fn as_f32(self) -> f32;

        #[inline]
        fn as_f64(self) -> f64;

        #[inline]
        fn as_readable_f32(self) -> f32;

    }
    impl RatingExt for Rating {
        #[inline]
        fn is_zero(self) -> bool {
            const EPSILON: f64 = 0.0000001f64;
            self.into_inner() > -EPSILON && self.into_inner() < EPSILON
        }

        #[inline]
        fn as_f32(self) -> f32 {
            self.into_inner() as f32
        }

        #[inline]
        fn as_f64(self) -> f64 {
            self.into_inner()
        }

        #[inline]
        fn as_readable_f32(self) -> f32 {
            self.into_inner() as f32
        }
    }

    pub trait RatingDeltaExt {
        #[inline]
        fn compute_rating_delta(a: TimeDelta, b: TimeDelta) -> RatingDelta {
            let min: NotNan<f64> = NotNan::from(min(a, b).as_f64());
            Rating::compute(a, b) / min
            //Rating::compute(a, b) / min(a, b).as_i64()
        }
    }
    impl RatingDeltaExt for RatingDelta {}

}*/

mod rating_i64 {
    use crate::{TimeDelta, TimeSpan};
    use std::cmp::min;

    // these objects determine the precision/length of the rating (i32/i64) - lower
    // values take less space and time, higher values have higher precision
    pub type Rating = i64;
    pub type RatingDelta = i64;
    pub type RatingDeltaDelta = i64;

    const RATING_PRECISION: i64 = (1 << 32);

    pub trait RatingExt {
        #[inline]
        fn add_mul(r: Rating, rd: RatingDelta, td: TimeDelta) -> Rating {
            r + rd * td.as_i64()
        }

        #[inline]
        fn add_mul_usize(r: Rating, rd: RatingDelta, td: usize) -> Rating {
            r + rd * td as i64
        }

        #[inline]
        fn from_timespans(a: TimeSpan, b: TimeSpan, score_fn: impl Fn(TimeDelta, TimeDelta) -> f64 + Copy) -> Rating {
            let overlap = TimeSpan::get_overlapping_length(a, b).as_f64();
            let max_rating = score_fn(a.len(), b.len());
            let length_normalization_factor = min(a.len(), b.len()).as_f64();

            Rating::convert_from_f64(max_rating * overlap / length_normalization_factor)
        }

        #[inline]
        fn convert_from_f64(v: f64) -> Rating {
            (v * RATING_PRECISION as f64) as i64
        }

        #[inline]
        fn zero() -> Rating {
            0
        }

        #[inline]
        fn is_zero(self) -> bool;

        #[inline]
        fn div_by_delta_to_i64(r: Rating, other: RatingDelta) -> i64 {
            r / other
        }

        #[inline]
        fn div_by_i64_to_delta(r: Rating, other: i64) -> RatingDelta {
            r / other
        }

        #[inline]
        fn as_readable_f32(self) -> f32;

        #[inline]
        fn as_readable_f64(self) -> f64;
    }

    impl RatingExt for Rating {
        #[inline]
        fn is_zero(self) -> bool {
            self == 0
        }

        #[inline]
        fn as_readable_f32(self) -> f32 {
            self as f32 / RATING_PRECISION as f32
        }

        #[inline]
        fn as_readable_f64(self) -> f64 {
            self as f64 / RATING_PRECISION as f64
        }
    }

    pub trait RatingDeltaExt {
        #[inline]
        fn compute_rating_delta(
            a: TimeDelta,
            b: TimeDelta,
            score_fn: impl Fn(TimeDelta, TimeDelta) -> f64 + Copy,
        ) -> RatingDelta {
            let min: f64 = min(a, b).as_f64();
            RatingDelta::convert_from_f64(score_fn(a, b) / min)
            //Rating::compute(a, b) / min(a, b).as_i64()
        }
    }
    impl RatingDeltaExt for RatingDelta {}
}
