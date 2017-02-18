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


#![deny(missing_docs,
        missing_debug_implementations, missing_copy_implementations,
        trivial_casts,
        unsafe_code,
        unstable_features,
        unused_import_braces, unused_qualifications)]
#![allow(unknown_lints)] // for clippy

//! `aligner` takes two timespan arrays (e.g. from two subtitle files) and
//! tries to align the `incorrect` subtitles
//! to the `reference` subtitle. It automatically fixes offsets and
//! introduces/removes breaks between subtitles in the `incorrect`
//! subtitle to achive the best alignment.

#[cfg(test)]
extern crate rand;
extern crate arrayvec;

// for internal use (in sub-modules)
mod internal;

// for external use (in other crates)
pub use internal::{ProgressHandler, TimeDelta, TimePoint, TimeSpan};

// for use in this module (in lib.rs)
use std::vec::from_elem;
use internal::{Aligner, prepare_time_spans};


/// Matches an `incorrect` subtitle list to a `reference` subtitle list.
///
/// Returns the delta for every time span in list.
///
/// The `split_penalty_normalized` is a value between
/// 0 and 1. Providing 0 will make the algorithm indifferent of splitting lines (resulting in MANY
/// different deltas), so this is not recommended. Providing 1 will assure that no split will occur,
/// so only one/the best offset is applied to ALL lines. The most common useful values are in the
/// 0.2 to 0.01 range.
///
/// Especially for larger subtitles(e.g. 1 hour in millisecond resolution and 1000 subtitle lines) this
/// process might take some seconds. To provide user feedback one can pass a `ProgressHandler` to
/// this function.
pub fn align(list: Vec<TimeSpan>,
             reference: Vec<TimeSpan>,
             split_penalty_normalized: f64,
             progress_handler: Option<Box<ProgressHandler>>)
             -> Vec<TimeDelta> {
    let (list_nonoverlapping, list_indices) = prepare_time_spans(list.clone());
    let (ref_nonoverlapping, _) = prepare_time_spans(reference.clone());

    if list_nonoverlapping.is_empty() || ref_nonoverlapping.is_empty() {
        return from_elem(TimeDelta::zero(), list.len());
    }

    let list_len = list_nonoverlapping.len();
    let aligner_opt = Aligner::new(list_nonoverlapping,
                                   ref_nonoverlapping,
                                   split_penalty_normalized,
                                   progress_handler);

    // get deltas for non-overlapping timespans
    let deltas = match aligner_opt {
        Some(mut aligner) => aligner.align_all_spans(),
        None => (0..list_len).map(|_| TimeDelta::zero()).collect(),
    };

    // get deltas for overlapping timspan-list
    list_indices.into_iter().map(|i| deltas[i]).collect()
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand;
    use rand::Rng;
    use internal::{TimePoint, prepare_time_spans};

    /// Some special time span sequences.
    fn predefined_time_spans() -> Vec<Vec<TimeSpan>> {
        let t0 = TimePoint::from(0);
        let t1000 = TimePoint::from(1000);
        let t2000 = TimePoint::from(2000);
        vec![vec![],
             vec![TimeSpan::new(t0, t0)],
             vec![TimeSpan::new(t0, t1000)],
             vec![TimeSpan::new(t0, t1000), TimeSpan::new(t1000, t1000)],
             vec![TimeSpan::new(t0, t1000), TimeSpan::new(t1000, t1000), TimeSpan::new(t1000, t2000)],
             vec![TimeSpan::new(t1000, t1000), TimeSpan::new(t1000, t1000)]]
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
            v.push(TimeSpan::new(TimePoint::from(current_pos),
                                 TimePoint::from(current_pos + current_len)));
        }

        v
    }

    /// All test time span sequences (some are predefined some are random).
    pub fn get_test_time_spans() -> Vec<Vec<TimeSpan>> {
        (0..100)
            .map(|_| generate_random_time_spans())
            .chain(predefined_time_spans().into_iter())
            .collect()
    }

    /// All test time span sequences (some are predefined some are random).
    pub fn get_random_prepared_test_time_spans() -> Vec<TimeSpan> {
        prepare_time_spans(generate_random_time_spans()).0
    }
}
