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

use crate::rating_type::{Rating, RatingDelta, RatingDeltaDelta, RatingDeltaExt, RATING_PRECISION};
use crate::segments::{
    add_rating_iterators, combined_maximum_of_dual_iterators, zero_rating_iterator, DifferentialRatingBufferBuilder,
    PositionBuffer, RatingBuffer, RatingIterator, RatingSegment, SeparateDualBuffer,
};
use crate::statistics::Statistics;
use crate::time_types::{TimeDelta, TimePoint, TimeSpan};

use arrayvec::ArrayVec;
use std::cell::RefCell;
use std::cmp::min;
use std::iter::once;
use std::rc::Rc;

/// Use this trait if you want more detailed information about the progress of the align operation
/// (which might take some seconds).
pub trait ProgressHandler {
    /// Will be called one time before `inc()` is called. `steps` is the
    /// number of times `inc()` will be called.
    ///
    /// The number of steps is around the number of lines in the "incorrect" subtitle.
    /// Be aware that this number can be zero!
    #[allow(unused_variables)]
    fn init(&mut self, steps: i64) {}

    /// We made (small) progress!
    fn inc(&mut self) {}

    /// Will be called after the last `inc()`, when `inc()` was called `steps` times.
    fn finish(&mut self) {}
}

/// The "main" structure which holds the infomation needed to align the subtitles to each other.
pub struct Aligner {
    /// List of incorrect subtitles which are aligned with this library. This
    /// list will always be non-empty.
    list: Vec<TimeSpan>,

    /// The fixed reference subtitles. This list will always be non-empty.
    reference: Vec<TimeSpan>,

    /// Contains the range in which the incorrect subtitles can be moved.
    buffer_timespan: TimeSpan,

    #[allow(dead_code)]
    statistics: Option<Rc<RefCell<Statistics>>>,
}

impl Aligner {
    /// In each list no time span should intersect any other and both list are
    /// sorted by starting times.
    pub fn new(list: Vec<TimeSpan>, reference: Vec<TimeSpan>, statistics_opt: Option<Statistics>) -> Aligner {
        assert!(list.len() > 0);
        assert!(reference.len() > 0);

        /*println!("{} reference lines", reference.len());
        println!("{} incorrect lines", list.len());*/

        let incorrect_start: TimePoint = (*list.first().unwrap()).start();
        let incorrect_end: TimePoint = (*list.last().unwrap()).end();

        let reference_start: TimePoint = (*reference.first().unwrap()).start();
        let reference_end: TimePoint = (*reference.last().unwrap()).end();

        // this is the timespan length which can contain all incorrect subtitles
        let list_timespan = incorrect_end - incorrect_start;

        let start = reference_start - list_timespan - TimeDelta::one();
        let end = reference_end + list_timespan + TimeDelta::one();

        // It might be possible that all corrected subtiles fit in the reference list
        // timeframe. It they don't
        // we need to provide extra space, so that the produting corrected subtitles
        // still fit into the
        // whole [start, end] timeframe. Because `list_timespan` is the length of the
        // whole incorrect subtitle file,
        // we can just extend the reference timeframe by `list_timespan` on both ends.
        let min_offset: TimeDelta = start - incorrect_start;
        let max_offset: TimeDelta = end - incorrect_start;

        assert!(min_offset <= max_offset);

        Aligner {
            list: list,
            reference: reference,
            buffer_timespan: TimeSpan::new(start, end),
            statistics: statistics_opt.map(|s| Rc::new(RefCell::new(s))),
        }
    }

    pub fn get_start(&self) -> TimePoint {
        self.buffer_timespan.start()
    }

    pub fn get_end(&self) -> TimePoint {
        self.buffer_timespan.end()
    }

    #[allow(dead_code)]
    pub fn align_constant_delta(&self) -> TimeDelta {
        #[derive(PartialEq, Eq, Clone)]
        struct DeltaCorrect {
            rating: RatingDeltaDelta,
            time: TimeDelta,
        }

        impl DeltaCorrect {
            fn new(rating: RatingDeltaDelta, time: TimeDelta) -> DeltaCorrect {
                DeltaCorrect {
                    rating: rating,
                    time: time,
                }
            }
        }

        type OrderedDeltaCorrect = Vec<DeltaCorrect>;

        let mut delta_corrects: Vec<OrderedDeltaCorrect> = Vec::new();

        for incorrect_ts in &self.list {
            let mut rise_ordered_delta_corrects: OrderedDeltaCorrect = OrderedDeltaCorrect::new();
            let mut up_ordered_delta_corrects: OrderedDeltaCorrect = OrderedDeltaCorrect::new();
            let mut fall_ordered_delta_corrects: OrderedDeltaCorrect = OrderedDeltaCorrect::new();
            let mut down_ordered_delta_corrects: OrderedDeltaCorrect = OrderedDeltaCorrect::new();

            for reference_ts in &self.reference {
                let rise_time;
                let up_time;
                let fall_time;
                let down_time;

                if incorrect_ts.len() < reference_ts.len() {
                    rise_time = reference_ts.start() - incorrect_ts.len();
                    up_time = reference_ts.start();
                    fall_time = reference_ts.end() - incorrect_ts.len();
                    down_time = reference_ts.end();
                } else {
                    rise_time = reference_ts.start() - incorrect_ts.len();
                    up_time = reference_ts.end() - incorrect_ts.len();
                    fall_time = reference_ts.start();
                    down_time = reference_ts.end();
                }

                let rating_delta_delta: RatingDeltaDelta =
                    RatingDelta::compute_rating_delta(incorrect_ts.len(), reference_ts.len());

                rise_ordered_delta_corrects
                    .push(DeltaCorrect::new(rating_delta_delta, rise_time - incorrect_ts.start()));
                up_ordered_delta_corrects.push(DeltaCorrect::new(-rating_delta_delta, up_time - incorrect_ts.start()));
                fall_ordered_delta_corrects
                    .push(DeltaCorrect::new(-rating_delta_delta, fall_time - incorrect_ts.start()));
                down_ordered_delta_corrects
                    .push(DeltaCorrect::new(rating_delta_delta, down_time - incorrect_ts.start()));
            }

            delta_corrects.push(rise_ordered_delta_corrects);
            delta_corrects.push(up_ordered_delta_corrects);
            delta_corrects.push(fall_ordered_delta_corrects);
            delta_corrects.push(down_ordered_delta_corrects);
        }

        // test if all delta correct arrays are sorted (should be true)
        /*for dc in &delta_corrects {
            for (a, b) in dc.iter().zip(dc.iter().skip(1)) {
                assert!(a.time < b.time);
            }
        }
        println!("b");*/

        // we now have "4 * len(incorrect_list)" sorted arrays with each "4 * len(reference_list)" elements
        //  -> sort with max heap (pop from end)

        // in heap sort implementation, the delta corrects are sorted descending by time, in the "simple" sort ascending
        let mut all_delta_corrects: Vec<DeltaCorrect>;
        let sorted_delta_corrects_iter; // : impl Iter<DeltaCorrect>
        let first_delta_correct: DeltaCorrect;

        #[cfg(not(feature = "nosplit-heap-sort"))]
        {
            all_delta_corrects = delta_corrects.into_iter().flat_map(|dc| dc).collect();
            all_delta_corrects.sort_unstable_by_key(|dc| dc.time);

            first_delta_correct = all_delta_corrects
                .first()
                .cloned()
                .expect("delta corrects should have at least one element");

            sorted_delta_corrects_iter = all_delta_corrects.iter();
        }

        #[cfg(feature = "nosplit-heap-sort")]
        {
            use std::cmp::Ordering;
            use std::collections::BinaryHeap;

            #[derive(PartialEq, Eq)]
            struct MaxHeapInfo {
                heap_id: usize,
                data: DeltaCorrect,
            }

            impl Ord for MaxHeapInfo {
                fn cmp(&self, other: &MaxHeapInfo) -> Ordering {
                    TimeDelta::cmp(&self.data.time, &other.data.time)
                }
            }

            impl PartialOrd for MaxHeapInfo {
                fn partial_cmp(&self, other: &MaxHeapInfo) -> Option<Ordering> {
                    Some(self.cmp(other))
                }
            }

            let mut heap = BinaryHeap::new();

            for (heap_id, data) in delta_corrects.iter_mut().enumerate() {
                let last_elem: DeltaCorrect = data
                    .pop()
                    .expect("at least one element should be in every delta correct list");
                heap.push(MaxHeapInfo {
                    heap_id: heap_id,
                    data: last_elem,
                });
            }

            all_delta_corrects = Vec::with_capacity(4 * self.list.len() * self.reference.len());

            loop {
                let max_heap_elem: MaxHeapInfo;

                match heap.pop() {
                    Some(x) => max_heap_elem = x,

                    // are all vectors empty?
                    None => break,
                }

                all_delta_corrects.push(max_heap_elem.data);

                if let Some(new_delta_correct) = delta_corrects[max_heap_elem.heap_id].pop() {
                    heap.push(MaxHeapInfo {
                        heap_id: max_heap_elem.heap_id,
                        data: new_delta_correct,
                    });
                }
            }

            assert!(all_delta_corrects.len() == 4 * self.list.len() * self.reference.len());
            sorted_delta_corrects_iter = all_delta_corrects.iter().rev();

            first_delta_correct = all_delta_corrects
                .last()
                .cloned()
                .expect("delta corrects should have at least one element");
        }

        // compute maximum rating
        let mut delta: i64 = 0;
        let mut rating: i64 = 0;
        let mut maximum: (i64, TimeDelta) = (0, first_delta_correct.time);
        for (delta_correct, next_delta_correct) in sorted_delta_corrects_iter
            .clone()
            .zip(sorted_delta_corrects_iter.skip(1))
        {
            //println!("rating: {}", rating);
            delta += delta_correct.rating;
            rating += delta * (next_delta_correct.time - delta_correct.time).as_i64();
            if rating > maximum.0 {
                maximum = (rating, next_delta_correct.time);
            }
        }

        assert!(rating == 0);

        return maximum.1;
    }

    #[cfg(feature = "statistics")]
    pub fn do_statistics(&self, f: impl Fn(&mut Statistics) -> std::io::Result<()>) {
        if let Some(statistics) = &self.statistics {
            f(&mut statistics.borrow_mut()).expect("failed to write statistics");
        }
    }

    pub fn align_with_splits(
        &self,
        mut progress_handler_opt: Option<Box<dyn ProgressHandler>>,
        nopsplit_bonus_normalized: f64,
        speed_optimization_opt: Option<f64>,
    ) -> Vec<TimeDelta> {
        // For each segment the full rating can only be 1. So the maximum rating
        // without the nosplit bonus is `min(list.len(), reference.len())`. So to get
        // from the normalized rating `[0, 1]` to a unnormalized rating (where only
        // values between `[0, max_rating]` are interesting) we multiply by
        // `min(list.len(), reference.len())`.

        if let Some(progress_handler) = progress_handler_opt.as_mut() {
            progress_handler.init(self.list.len() as i64);
        }

        let nopsplit_bonus_unnormalized: RatingDelta = ((min(self.list.len(), self.reference.len()) as f64
            * nopsplit_bonus_normalized)
            * RATING_PRECISION as f64) as RatingDelta;

        let mut last_rating_buffer: Option<RatingBuffer> =
            Some(zero_rating_iterator(self.get_start(), self.get_end()).save());

        assert!(self.list.len() > 0);

        /*: impl Iterator<Item=Option<TimeDelta>>*/
        let nosplit_delta_iter = self
            .list
            .iter()
            .zip(self.list.iter().skip(1))
            .map(|(incorrect_timespan, next_timespan)| Some(next_timespan.start - incorrect_timespan.start))
            .chain(once(None));

        // these buffers save the start position of a line dependent on the position of the next line,
        //  -> this allows to compute the final corrected line positions
        let mut position_buffers: Vec<PositionBuffer> = Vec::new();

        for (line_nr, (&incorrect_span, nosplit_delta_opt)) in self.list.iter().zip(nosplit_delta_iter).enumerate() {
            assert!(incorrect_span.len() > TimeDelta::zero()); // otherwise shift_simple/extend_to creates a zero-length segment

            let pline_tag = format!("line:{}", line_nr);
            let nline_tag = format!("line:-{}", self.list.len() - 1 - line_nr);
            let _line_tags: Vec<&str> = vec![pline_tag.as_str(), nline_tag.as_str()];

            let single_span_ratings;
            let _single_span_ratings = self.single_span_ratings(incorrect_span.len());

            #[cfg(not(feature = "statistics"))]
            {
                single_span_ratings = _single_span_ratings;
            }

            #[cfg(feature = "statistics")]
            {
                let single_span_ratings_buffer = _single_span_ratings.save();

                self.do_statistics(|s| {
                    s.save_rating_buffer(
                        "[1] INDIVIDUAL span ratings for start position",
                        &_line_tags
                            .clone()
                            .into_iter()
                            .chain(once("individual"))
                            .collect::<Vec<_>>(),
                        &single_span_ratings_buffer,
                    )
                });

                single_span_ratings = single_span_ratings_buffer.into_iter();
            }

            let added_buffer: RatingBuffer;
            if let Some(speed_optimization) = speed_optimization_opt {
                let progress_factor = line_nr as f64 / self.list.len() as f64;
                let epsilon = (RATING_PRECISION as f64 * speed_optimization * 0.1 * progress_factor) as i64;

                added_buffer = add_rating_iterators(last_rating_buffer.unwrap().into_iter(), single_span_ratings)
                    .discard_start_times()
                    .save_aggressively_simplified(epsilon);
            } else {
                added_buffer = add_rating_iterators(last_rating_buffer.unwrap().into_iter(), single_span_ratings)
                    .discard_start_times()
                    .save();
                //.save_simplified(); // this seems to not change runtime very much (TODO: test with benchmark)
            }

            #[cfg(feature = "statistics")]
            self.do_statistics(|s| {
                s.save_rating_buffer(
                    "[2] TOTAL span ratings for start position (last rating + new span rating)",
                    &_line_tags
                        .clone()
                        .into_iter()
                        .chain(once("individual"))
                        .collect::<Vec<_>>(),
                    &buffer,
                )
            });

            if let Some(nosplit_delta) = nosplit_delta_opt {
                assert!(nosplit_delta > TimeDelta::zero()); // otherwise shift_simple/extend_to creates a zero-length segment

                let best_split_positions;
                let _best_split_positions = added_buffer
                    .iter()
                    .shift_simple(incorrect_span.len())
                    .clamp_end(self.get_end())
                    .annotate_with_segment_start_times()
                    .annotate_with_position_info(|segment_start_point| segment_start_point - incorrect_span.len())
                    .left_to_right_maximum()
                    .discard_start_times()
                    .simplify()
                    .discard_start_times();

                let nosplit_positions;
                let _nosplit_positions = added_buffer
                    .iter()
                    .shift_simple(nosplit_delta)
                    .clamp_end(self.get_end())
                    .annotate_with_segment_start_times()
                    .discard_start_times()
                    .add_rating(nopsplit_bonus_unnormalized)
                    .annotate_with_segment_start_times()
                    .annotate_with_position_info(|segment_start_point| segment_start_point - nosplit_delta)
                    .discard_start_times();
                //.simplify()
                //.discard_start_times();

                let combined_maximum_buffer: SeparateDualBuffer;

                #[cfg(feature = "statistics")]
                {
                    let nosplit_positions_buffer = _nosplit_positions.save();

                    self.do_statistics(|s| {
                        s.save_rating_buffer(
                            "[3a] NOsplit ratings (positions are FIXED to end)",
                            &_line_tags
                                .clone()
                                .into_iter()
                                .chain(once("nosplit"))
                                .collect::<Vec<_>>(),
                            &nosplit_positions_buffer.iter().only_ratings().save(),
                        )
                    });

                    self.do_statistics(|s| {
                        s.save_position_buffer(
                            "[3b] NOsplit positions (positions are FIXED to end)",
                            &_line_tags
                                .clone()
                                .into_iter()
                                .chain(once("nosplit"))
                                .collect::<Vec<_>>(),
                            &nosplit_positions_buffer.iter().only_positions().save(),
                        )
                    });

                    let best_split_positions_buffer = _best_split_positions.save();

                    self.do_statistics(|s| {
                        s.save_rating_buffer(
                            "[4a] split ratings (positions computed by LEFT-TO-RIGHT maxmimum)",
                            &_line_tags.clone().into_iter().chain(once("split")).collect::<Vec<_>>(),
                            &best_split_positions_buffer.iter().only_ratings().save(),
                        )
                    });

                    self.do_statistics(|s| {
                        s.save_position_buffer(
                            "[4b] split positions (positions computed by LEFT-TO-RIGHT maxmimum)",
                            &_line_tags.clone().into_iter().chain(once("split")).collect::<Vec<_>>(),
                            &best_split_positions_buffer.iter().only_positions().save(),
                        )
                    });

                    nosplit_positions = nosplit_positions_buffer.into_iter();
                    best_split_positions = best_split_positions_buffer.into_iter();
                }

                #[cfg(not(feature = "statistics"))]
                {
                    nosplit_positions = _nosplit_positions;
                    best_split_positions = _best_split_positions;
                }

                combined_maximum_buffer = combined_maximum_of_dual_iterators(nosplit_positions, best_split_positions)
                    .discard_start_times()
                    .save_separate();

                #[cfg(feature = "statistics")]
                self.do_statistics(|s| {
                    s.save_rating_buffer(
                        "[5] COMBINED ratings for this span (vertical maximum of split and nosplit)",
                        &_line_tags
                            .clone()
                            .into_iter()
                            .chain(once("combined"))
                            .collect::<Vec<_>>(),
                        &combined_maximum_buffer.rating_buffer,
                    )
                });

                /*println!(
                    "Last rating buffer length: {}",
                    combined_maximum_buffer.rating_buffer.buffer.len()
                );*/

                last_rating_buffer = Some(combined_maximum_buffer.rating_buffer);

                position_buffers.push(combined_maximum_buffer.position_buffer);
            } else {
                last_rating_buffer = None;

                let best_position = added_buffer
                    .iter()
                    .annotate_with_segment_start_times()
                    .annotate_with_position_info(|segment_start_point| segment_start_point)
                    .left_to_right_maximum()
                    .discard_start_times()
                    .only_positions();

                position_buffers.push(best_position.save());
            }

            #[cfg(feature = "statistics")]
            self.do_statistics(|s| {
                s.save_position_buffer(
                    "[5] COMBINED positions span (by vertical rating maximum of split and nosplit)",
                    &_line_tags
                        .clone()
                        .into_iter()
                        .chain(once("combined"))
                        .collect::<Vec<_>>(),
                    &position_buffers.last().unwrap(),
                )
            });

            if let Some(progress_handler) = progress_handler_opt.as_mut() {
                progress_handler.inc();
            }
        }

        // ------------------------------------------------------------------------------
        // Extract the best position for each incorrect span from position buffers

        assert!(self.list.len() == position_buffers.len());

        let mut next_segment_position = self.get_end() - TimeDelta::one();
        let mut result_deltas = Vec::new();
        for (incorrect_span, position_buffer) in Iterator::zip(self.list.iter(), position_buffers.iter()).rev() {
            let best_position = position_buffer.get_at(next_segment_position);
            result_deltas.push(best_position - incorrect_span.start);

            next_segment_position = best_position;
        }

        // the deltas were inserted back-to-front
        result_deltas.reverse();

        if let Some(progress_handler) = progress_handler_opt.as_mut() {
            progress_handler.finish();
        }

        result_deltas
    }

    /// Requires "start1 <= start2". Returns the compressed rating vector for
    /// the overlapping ratings of a timespan of length
    /// "length" on all start position from "start1" to "start2".
    ///
    /// This function has O(n) runtime, where n is the number of spans in the
    /// reference list.

    fn single_span_ratings(&self, length: TimeDelta) -> RatingIterator<impl Iterator<Item = RatingSegment>> {
        // If we fix one timespan and let an other timespan variable, we get such a
        // curve for the rating:
        //
        //          / --------- \
        //         /             \
        // -------                --------------------------
        //
        // at first the rating be zero, then rise linearly, then it will be constant
        // for a time and then fall to zero again
        //
        // The next function will return these special changepoints and their
        // "delta"-change (delta-delta).
        // Because the timespans in "self.reference" are sorted and non overlapping,
        // the changepoints of a certain type (first rise, start of constant, ...)
        // will also be sorted. That means we only have to compare the current first
        // changepoints of each type to get the really first
        // changepoint. We then apply this changepoint-delta to the current total delta
        // and add the segment with the
        // previous total delta to the buffer. This way we get the segments with the
        // same delta very efficently in O(n).

        let mut builder = DifferentialRatingBufferBuilder::new(self.get_start(), self.get_end());
        let mut timepoints: [Vec<(RatingDelta, TimePoint)>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for &ref_ts in &self.reference {
            let changepoints = Self::get_overlapping_rating_changepoints(length, ref_ts);
            timepoints[0].push(changepoints[0]);
            timepoints[1].push(changepoints[1]);
            timepoints[2].push(changepoints[2]);
            timepoints[3].push(changepoints[3]);
        }

        // this is a vector of 4 iterators, each iterating over the contents of
        // "timepoints[0]" to "timepoints[3]"
        let mut iterators: ArrayVec<[_; 4]> = timepoints
            .into_iter()
            .cloned()
            .map(|v| v.into_iter().peekable())
            .collect();
        loop {
            // unpack the first value of each iterator
            let next_timepoints: ArrayVec<[(usize, (Rating, TimePoint)); 4]> = iterators
                .iter_mut()
                .enumerate()
                .map(|(i, iter)| iter.peek().map(|&v| (i, v)))
                .filter_map(|opt| opt)
                .collect();

            // take the first next timepoint
            let next_changepoint_opt = next_timepoints.into_iter().min_by_key::<TimePoint, _>(|a| (a.1).1);

            // because each original array had the same length, all iterators should end at
            // the same time
            let (next_id, (segment_end_delta_delta, segment_end)) = match next_changepoint_opt {
                Some(next_changepoint) => next_changepoint,
                None => break,
            };

            builder.add_segment(segment_end, segment_end_delta_delta);

            // "next_id" contains the index of the iterator which contains
            // "next_changepoint" -> pop that from the front so we don't have a endless loop
            iterators[next_id].next();
        }

        // the rating values are continuous, so the first value of a segment is the
        // last value of the previous segment.
        // To avoid having each of these segment-break values two times in the buffer,
        // every segments stops one timepoint
        // before the real segment end. The real segment end is then the first value of
        // the next value.
        //
        // The last rating has to be 0, so we extend the last segment with the missing
        // timepoint.
        builder.extend_to_end();

        builder.build().into_rating_iter()
    }

    /// Returns the timepoints at which the rating delta changes if we move one
    /// subtitle compared to
    /// an other.
    ///
    /// If we fix one timespan and let an other timespan variable, we get such a
    /// curve for the rating:
    ///
    /// ```text
    ///
    ///          / --------- \
    ///         /             \
    /// -------                --------------------------
    /// ```
    ///
    /// At first the rating be zero, then rise linearly, then it will be constant
    /// for a time and then fall to zero again. This function computes these 4
    /// special timepoints.
    pub fn get_overlapping_rating_changepoints(
        length: TimeDelta,
        constspan: TimeSpan,
    ) -> [(RatingDeltaDelta, TimePoint); 4] {
        let start_of_rise = constspan.start() - length;
        let end_of_rise = constspan.end() - length;
        let start_of_fall = constspan.start();
        let end_of_fall = constspan.end();

        let timepoints: [TimePoint; 4] = if end_of_rise <= start_of_fall {
            [start_of_rise, end_of_rise, start_of_fall, end_of_fall]
        } else {
            [start_of_rise, start_of_fall, end_of_rise, end_of_fall]
        };

        assert!(timepoints[0] <= timepoints[1]);
        assert!(timepoints[1] <= timepoints[2]);
        assert!(timepoints[2] <= timepoints[3]);

        let rise_delta = RatingDelta::compute_rating_delta(length, constspan.len());

        [
            (rise_delta, timepoints[0]),
            (-rise_delta, timepoints[1]),
            (-rise_delta, timepoints[2]),
            (rise_delta, timepoints[3]),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::rating_type::RatingExt;
    use crate::segments::RatingFullSegment;
    use crate::tests::get_random_prepared_test_time_spans;

    fn get_dummy_aligner() -> Aligner {
        loop {
            let reference_ts = get_random_prepared_test_time_spans();
            let incorrect_ts = get_random_prepared_test_time_spans();

            // this is unlikely
            if reference_ts.is_empty() || incorrect_ts.is_empty() {
                continue;
            }

            // new will return None, if both lists are empty -> highly unlikely
            return Aligner::new(reference_ts, incorrect_ts, None);
        }
    }

    #[test]
    /// Aligns random timespans to each other and calls alass. General test whether any internal
    /// assertions are invalidated.
    fn run_aligner() {
        for _ in 0..20 {
            get_dummy_aligner().align_with_splits(None, 0.1, None);
        }
    }

    #[test]
    fn test_single_span_ratings() {
        for _ in 0..30 {
            let alass = get_dummy_aligner();

            for span in alass.list.clone() {
                let last: RatingFullSegment = alass
                    .single_span_ratings(span.len())
                    .annotate_with_segment_start_times()
                    .into_iter()
                    .last()
                    .unwrap();
                assert_eq!(last.end_rating(), Rating::zero());
                //assert_eq!(dbg!(last.data.delta), RatingDelta::zero());
            }
        }
    }

    /*#[test]
    /// `get_compressed_overlapping_ratings()` is highly optimized -> compare the results of slow and fast
    /// implemntations.
    fn get_compressed_overlapping_ratings() {
        let mut rng = rand::thread_rng();

        for _ in 0..30 {
            let alass = get_dummy_aligner();
            let len: i64 = (rng.next_u32() % 100) as i64;
            let rating_buffer1 = alass.get_compressed_overlapping_ratings(
                alass.get_start(),
                alass.get_end(),
                TimeDelta::one() * len,
            );
            let rating_buffer2 = alass.get_compressed_overlapping_ratings_slow(
                alass.get_start(),
                alass.get_end(),
                TimeDelta::one() * len,
            );
            assert_eq!(
                rating_buffer1.iter().collect::<Vec<_>>(),
                rating_buffer2.iter().collect::<Vec<_>>()
            );
        }
    }*/

}
