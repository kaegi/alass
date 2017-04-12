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


use internal::{TimeDelta, TimeSpan};
use std;
use std::cmp::max;

fn prepare_spans_sorted(overlapping: Vec<TimeSpan>) -> (Vec<TimeSpan>, Vec<usize>) {
    if overlapping.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // the constructor of TimeSpan ensures "start <= end"

    // sort the spans by starting time but save the permuation through enumeration
    let mut sorted_overlapping: Vec<(usize, TimeSpan)> = overlapping.iter().cloned().enumerate().collect();
    sorted_overlapping.sort_by(|a, b| TimeSpan::cmp_start(a.1, b.1));

    // create a mapping from "original vector index -> sorted vector index"
    let mut mapping = std::vec::from_elem(0usize, overlapping.len());
    for (i2, &(i, _)) in sorted_overlapping.iter().enumerate() {
        mapping[i] = i2;
    }

    (sorted_overlapping.into_iter()
                       .map(|(_, ts)| ts)
                       .collect(),
     mapping)
}

/// Returns a smaller list of non-overlapping time spans and a vector with
/// original length which contains a mapping from "index in overlapping vector
/// -> index in non-overlapping-vector"
/// Requires that all spans are sorted by start time and the vector is not
/// empty.
fn prepare_spans_non_overlapping(v: Vec<TimeSpan>) -> (Vec<TimeSpan>, Vec<usize>) {
    if v.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // condense overlapping time spans and create a mapping "sorted vector index ->
    // non-overlapping vector index"
    let mut result: Vec<TimeSpan> = Vec::with_capacity(v.len());
    let mut mapping: Vec<usize> = Vec::with_capacity(v.len());
    let mut current_end = v[0].start(); // this does not overlap with first time span
    for ts in v {
        if ts.start() < current_end {
            // timespans overlap -> only extend current timespan (if anything at all)
            let last_element_index = result.len() - 1;
            current_end = max(current_end, ts.end());
            result[last_element_index] = result[last_element_index].new_copy_with_end(current_end);
        } else {
            // time span does not overlap
            result.push(ts);
            current_end = ts.end();
        }

        // the currennt time span is now inside the last new timespan
        mapping.push(result.len() - 1);
    }

    (result, mapping)
}

/// `v` should only contain non-overlapping sorted timespans, sorted by
/// starting time.
/// Returns a list of time-spans without spans of zero-length. The zero-length
/// time spans
/// are grouped together with next or previous time spans.
fn prepare_spans_nonzero(v: Vec<TimeSpan>) -> (Vec<TimeSpan>, Vec<usize>) {
    // list of non-zero spans
    let non_zero_spans: Vec<TimeSpan> = v.iter()
                                         .cloned()
                                         .filter(|&ts| ts.len() > TimeDelta::zero())
                                         .collect();
    if non_zero_spans.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let mut new_index = 0;
    let mut indices = Vec::with_capacity(v.len());
    for ts in v {
        if ts.len() != TimeDelta::zero() {
            // this timespan is in the non_zero_spans vector -> go to the right index
            indices.push(new_index);
            new_index += 1;
            continue;
        }

        let prev_nonzero_ts_opt = if new_index == 0 {
            None
        } else {
            Some(non_zero_spans[new_index - 1])
        };
        let next_nonzero_ts_opt = if new_index == non_zero_spans.len() {
            None
        } else {
            Some(non_zero_spans[new_index])
        };

        let merge_with_prev = match (prev_nonzero_ts_opt, next_nonzero_ts_opt) {
            (None, None) => panic!("No previous or next span in non-empty non_zero_span vector"),
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (Some(p), Some(n)) => ts.fast_distance_to(p) <= ts.fast_distance_to(n),
        };


        indices.push(if merge_with_prev {
                         new_index - 1
                     } else {
                         new_index
                     });
    }

    (non_zero_spans, indices)
}

pub fn prepare_time_spans(v: Vec<TimeSpan>) -> (Vec<TimeSpan>, Vec<usize>) {
    if v.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let operations = [prepare_spans_sorted,
                      prepare_spans_non_overlapping,
                      prepare_spans_nonzero];
    let mut mapping: Vec<usize> = (0..v.len()).collect();
    let mut result = v;
    for &operation in &operations {
        let (new_result, new_mapping) = (operation)(result);
        if new_result.is_empty() {
            return (Vec::new(), Vec::new());
        }
        mapping = mapping.iter().map(|&i| new_mapping[i]).collect();
        result = new_result;
    }

    (result, mapping)
}


#[cfg(test)]
mod tests {
    use super::*;
    use internal::prepare_time_spans;
    use tests::get_test_time_spans;

    #[test]
    fn test_prepare_time_spans() {
        for time_spans in get_test_time_spans() {
            let (non_overlapping, indices) = prepare_time_spans(time_spans.clone());

            assert!(non_overlapping.len() <= time_spans.len());

            // function will condense non-zero timespans into one -> vector of zero-length
            // timespans will turn into empty vector
            let full_length: i64 = time_spans.iter()
                                             .cloned()
                                             .map(|time_spans| i64::from(time_spans.len()))
                                             .sum();
            if full_length == 0 {
                assert!(non_overlapping.is_empty());
                continue;
            }

            if time_spans.len() == 0 {
                continue;
            }
            assert!(non_overlapping.len() > 0);

            // test whether some spans overlap (they shouldn't)
            non_overlapping.iter()
                           .cloned()
                           .zip(non_overlapping.iter().cloned().skip(1))
                           .inspect(|&(last, current)| {
                                        assert!(last.start() <= last.end());
                                        assert!(last.end() <= current.start());
                                        assert!(current.start() <= current.end());
                                    })
                           .count();

            // test mapping from "overlapping -> non-overlapping"
            assert!(time_spans.len() == indices.len());
            for (i, span) in time_spans.iter().cloned().enumerate() {
                assert!(non_overlapping[indices[i]].contains(span) || span.len() == TimeDelta::zero());
            }


            // -----------------------------------------------------------
            // apply `prepare_time_spans()` a second time which should now be a noop
            let (prepared_timespans2, indices2) = prepare_time_spans(non_overlapping.clone());
            assert_eq!(non_overlapping, prepared_timespans2);
            assert_eq!(indices2, (0..indices2.len()).collect::<Vec<_>>());

        }
    }
}
