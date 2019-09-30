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

#![deny(
    //missing_docs,
    missing_debug_implementations,
    //missing_copy_implementations,
    trivial_casts,
    //unsafe_code,
    unstable_features,
    unused_import_braces,
    unused_qualifications
)]
#![allow(unknown_lints)] // for clippy

//! `alass` takes two timespan arrays (e.g. from two subtitle files) and
//! tries to align the `incorrect` subtitles
//! to the `reference` subtitle. It automatically fixes offsets and
//! introduces/removes breaks between subtitles in the `incorrect`
//! subtitle to achive the best alignment.

#[cfg(test)]
extern crate rand;

mod alass;
mod rating_type;
#[allow(dead_code)]
mod segments;
mod time_types;
mod timespan_ops;

use crate::alass::Aligner;
pub use crate::alass::NoProgressHandler;
pub use crate::alass::ProgressHandler;
use crate::rating_type::{Rating, RatingDelta, RatingExt};
pub use crate::time_types::{TimeDelta, TimePoint, TimeSpan};
use crate::timespan_ops::prepare_time_spans;
use std::cmp::{max, min};

fn denormalize_split_penalty(ref_list_len: usize, in_list_len: usize, split_penalty_normalized: f64) -> RatingDelta {
    RatingDelta::convert_from_f64(min(ref_list_len, in_list_len) as f64 * split_penalty_normalized / 1000.0)
}

pub type Score = f64;

/// This score is 1 for equally length spans and lower the more the spans are unequal in length (use this scoring if you're not sure what to take).
pub fn standard_scoring(a: TimeDelta, b: TimeDelta) -> Score {
    let min: f64 = min(a, b).as_f64();
    let max: f64 = max(a, b).as_f64();
    min / max
}

/// Calculate score based only on the overlapping length of the
/// intervals (better when comparing scaled subtitles; used for FPS correction).
pub fn overlap_scoring(a: TimeDelta, b: TimeDelta) -> Score {
    let min: f64 = min(a, b).as_f64();
    min * 0.00001
}

/// Matches an `incorrect` subtitle list to a `reference` subtitle list with only a single constant shift (no split).
///
/// Returns the delta for every time span in list.
///
/// This function takes usually less than 300ms on 2h30min subtitle data.
///
/// Use `standard_scoring` as score function if no fine tuning is required.
pub fn align_nosplit(
    reference: &[TimeSpan],
    list: &[TimeSpan],
    score_fn: impl Fn(TimeDelta, TimeDelta) -> f64 + Copy,
    mut progress_handler: impl ProgressHandler,
) -> (TimeDelta, Score) {
    progress_handler.init(1);

    let (ref_nonoverlapping, _) = prepare_time_spans(reference);
    let (list_nonoverlapping, _) = prepare_time_spans(list);

    if list_nonoverlapping.is_empty() || ref_nonoverlapping.is_empty() {
        return (TimeDelta::zero(), 0.);
    }

    // get deltas for non-overlapping timespans
    let (delta, score) = Aligner::align_constant_delta(&ref_nonoverlapping, &list_nonoverlapping, score_fn);
    progress_handler.inc();
    progress_handler.finish();

    return (delta, score.as_readable_f64());
}

/// Matches an `incorrect` subtitle list to a `reference` subtitle list.
///
/// Returns the delta for every time span in list.
///
/// The `split_penalty_normalized` is a value between
/// 0 and 1000. Providing 0 will make the algorithm indifferent of splitting lines (resulting in MANY
/// different deltas), so this is not recommended. Providing 1000 will assure that no split will occur,
/// so only one/the best offset is applied to ALL lines. The most common useful values are in the
/// 4 to 20 range (optimum 7+-1).
///
/// Especially for larger subtitles (e.g. 1 hour in millisecond resolution and 1000 subtitle lines) this
/// process might take some seconds. To provide user feedback one can pass a `ProgressHandler` to
/// this function.
///
/// If you want to increase the speed of the alignment process, you can use the `speed_optimization`
/// parameter. This value can be between `0` and `+inf`, altough after `10` the accuracy
/// will have greatly degraded. It is recommended to supply a value around `3`.
///
/// Use `standard_scoring` as score function if no fine tuning is required.
pub fn align(
    reference: &[TimeSpan],
    list: &[TimeSpan],
    split_penalty: f64,
    speed_optimization: Option<f64>,
    score_fn: impl Fn(TimeDelta, TimeDelta) -> f64 + Copy,
    progress_handler: impl ProgressHandler,
) -> (Vec<TimeDelta>, f64) {
    let (list_nonoverlapping, list_indices) = prepare_time_spans(&list);
    let (ref_nonoverlapping, _) = prepare_time_spans(&reference);

    if list_nonoverlapping.is_empty() || ref_nonoverlapping.is_empty() {
        return (vec![TimeDelta::zero(); list.len()], 0.);
    }

    let nosplit_bonus = denormalize_split_penalty(ref_nonoverlapping.len(), list_nonoverlapping.len(), split_penalty);

    // get deltas for non-overlapping timespans
    let (deltas, score) = Aligner::align_with_splits(
        &ref_nonoverlapping,
        &list_nonoverlapping,
        nosplit_bonus,
        speed_optimization,
        score_fn,
        progress_handler,
    );

    // get deltas for overlapping timspan-list
    (
        list_indices.into_iter().map(|i| deltas[i]).collect(),
        score.as_readable_f64(),
    )
}

/// Calculate the split score (see thesis in repository of source code).
pub fn get_split_rating(
    ref_spans: &[TimeSpan],
    in_spans: &[TimeSpan],
    offets: &[TimeDelta],
    split_penalty: f64,
    score_fn: impl Fn(TimeDelta, TimeDelta) -> f64 + Copy,
) -> Score {
    let mut total_rating = get_nosplit_rating_iter(ref_spans.iter().cloned(), in_spans.iter().cloned(), score_fn);

    let nosplit_bonus = denormalize_split_penalty(ref_spans.len(), in_spans.len(), split_penalty);

    total_rating = Rating::add_mul_usize(
        total_rating,
        -nosplit_bonus,
        offets
            .iter()
            .cloned()
            .zip(offets.iter().skip(1).cloned())
            .filter(|(o1, o2)| o1 != o2)
            .count(),
    );

    total_rating.as_readable_f64()
}

/// Calculate the no-split score (see thesis in repository of source code).
pub fn get_nosplit_score(
    ref_spans: impl Iterator<Item = TimeSpan>,
    in_spans: impl Iterator<Item = TimeSpan>,
    score_fn: impl Fn(TimeDelta, TimeDelta) -> f64 + Copy,
) -> Score {
    get_nosplit_rating_iter(ref_spans, in_spans, score_fn).as_readable_f64()
}

fn get_nosplit_rating_iter(
    mut ref_spans: impl Iterator<Item = TimeSpan>,
    mut in_spans: impl Iterator<Item = TimeSpan>,
    score_fn: impl Fn(TimeDelta, TimeDelta) -> f64 + Copy,
) -> Rating {
    let mut total_rating = Rating::zero();

    let mut ref_span;
    let mut in_span;

    let ref_span_opt = ref_spans.next();
    match ref_span_opt {
        None => return total_rating,
        Some(v) => ref_span = v,
    }

    let in_span_opt = in_spans.next();
    match in_span_opt {
        None => return total_rating,
        Some(v) => in_span = v,
    }

    loop {
        let rating = Rating::from_timespans(ref_span, in_span, score_fn);
        total_rating += rating;

        if ref_span.end() <= in_span.end() {
            let ref_span_opt = ref_spans.next();
            match ref_span_opt {
                None => return total_rating,
                Some(v) => ref_span = v,
            }
        } else {
            let in_span_opt = in_spans.next();
            match in_span_opt {
                None => return total_rating,
                Some(v) => in_span = v,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{prepare_time_spans, TimePoint};
    use rand;
    use rand::RngCore;

    /// Some special time span sequences.
    fn predefined_time_spans() -> Vec<Vec<TimeSpan>> {
        let t0 = TimePoint::from(0);
        let t1000 = TimePoint::from(1000);
        let t2000 = TimePoint::from(2000);
        vec![
            vec![],
            vec![TimeSpan::new(t0, t0)],
            vec![TimeSpan::new(t0, t1000)],
            vec![TimeSpan::new(t0, t1000), TimeSpan::new(t1000, t1000)],
            vec![
                TimeSpan::new(t0, t1000),
                TimeSpan::new(t1000, t1000),
                TimeSpan::new(t1000, t2000),
            ],
            vec![TimeSpan::new(t1000, t1000), TimeSpan::new(t1000, t1000)],
        ]
    }

    /// Generate random time span sequences
    fn generate_random_time_spans() -> Vec<TimeSpan> {
        let mut rng = rand::thread_rng();

        let len: usize = (rng.next_u32() % 400) as usize;
        let mut v = Vec::with_capacity(len);
        let mut current_pos = 0i64;
        for _ in 0..len {
            current_pos += (rng.next_u32() % 200) as i64 - 50;
            let current_len = (rng.next_u32() % 400) as i64;
            v.push(TimeSpan::new(
                TimePoint::from(current_pos),
                TimePoint::from(current_pos + current_len),
            ));
        }

        v
    }

    /// All test time span sequences (some are predefined some are random).
    pub fn get_test_time_spans() -> Vec<Vec<TimeSpan>> {
        (0..1000)
            .map(|_| generate_random_time_spans())
            .chain(predefined_time_spans().into_iter())
            .collect()
    }

    /// All test time span sequences (some are predefined some are random).
    pub fn get_random_prepared_test_time_spans() -> Vec<TimeSpan> {
        prepare_time_spans(&generate_random_time_spans()).0
    }
}
