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

use crate::rating_type::{Rating, RatingDelta, RatingDeltaDelta, RatingDeltaExt, RatingExt};
use crate::segments::{
    combined_maximum_of_dual_iterators, DifferentialRatingBufferBuilder, OffsetBuffer, RatingBuffer, RatingIterator,
    RatingSegment, SeparateDualBuffer,
};
use crate::time_types::{TimeDelta, TimePoint, TimeSpan};

use std::convert::TryInto;

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

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
/// `impl ProgressHandler` with empty functions
pub struct NoProgressHandler;
impl ProgressHandler for NoProgressHandler {}

/// The "main" structure which holds the infomation needed to align the subtitles to each other.
pub struct Aligner;

impl Aligner {
    pub fn get_offsets_bounds(ref_spans: &[TimeSpan], in_spans: &[TimeSpan]) -> (TimeDelta, TimeDelta) {
        assert!(ref_spans.len() > 0);
        assert!(in_spans.len() > 0);

        let in_start: TimePoint = (*in_spans.first().unwrap()).start();
        let in_end: TimePoint = (*in_spans.last().unwrap()).end();

        let ref_start: TimePoint = (*ref_spans.first().unwrap()).start();
        let ref_end: TimePoint = (*ref_spans.last().unwrap()).end();

        assert!(in_start <= in_end);
        assert!(ref_start <= ref_end);

        (ref_start - in_end, ref_end - in_start)
    }

    pub fn align_constant_delta_bucket_sort(
        ref_spans: &[TimeSpan],
        in_spans: &[TimeSpan],
        score_fn: impl Fn(TimeDelta, TimeDelta) -> f64 + Copy,
    ) -> (TimeDelta, Rating) {
        let (min_offset, max_offset) = Self::get_offsets_bounds(ref_spans, in_spans);

        let len: usize = (max_offset - min_offset).as_i64().try_into().unwrap();

        //let ta = std::time::Instant::now();
        let mut deltas: Vec<RatingDeltaDelta> = vec![RatingDeltaDelta::zero(); len + 1];

        //let tb = std::time::Instant::now();
        for reference_ts in ref_spans {
            for incorrect_ts in in_spans {
                let rating_delta_delta: RatingDeltaDelta =
                    RatingDelta::compute_rating_delta(incorrect_ts.len(), reference_ts.len(), score_fn);

                #[inline(always)]
                fn accum(d: &mut [RatingDeltaDelta], idx: TimeDelta, x: RatingDeltaDelta, sigma_min: TimeDelta) {
                    let idx: usize = (idx - sigma_min).as_i64().try_into().unwrap();
                    d[idx] = d[idx] + x;
                };

                accum(
                    &mut deltas,
                    reference_ts.start() - incorrect_ts.end(),
                    rating_delta_delta,
                    min_offset,
                );
                accum(
                    &mut deltas,
                    reference_ts.end() - incorrect_ts.end(),
                    -rating_delta_delta,
                    min_offset,
                );
                accum(
                    &mut deltas,
                    reference_ts.start() - incorrect_ts.start(),
                    -rating_delta_delta,
                    min_offset,
                );
                accum(
                    &mut deltas,
                    reference_ts.end() - incorrect_ts.start(),
                    rating_delta_delta,
                    min_offset,
                );
            }
        }
        //let tc = std::time::Instant::now();

        // compute maximum rating
        let mut delta: RatingDelta = RatingDelta::zero();
        let mut rating: Rating = Rating::zero();
        let mut maximum: (Rating, TimeDelta) = (Rating::zero(), min_offset);
        //let mut nonzero: i64 = 0;
        for (sigma, jump_value) in deltas.into_iter().enumerate() {
            /*if !RatingDeltaDelta::is_zero(jump_value) {
                nonzero = nonzero + 1;
            }*/
            rating += delta;
            delta += jump_value;
            if rating > maximum.0 {
                maximum = (rating, sigma as i64 * TimeDelta::one() + min_offset);
            }
        }

        /*let td = std::time::Instant::now();
        println!("init {}ms", (tb - ta).as_millis());
        println!("insert {}ms", (tc - tb).as_millis());
        println!("calc {}ms", (td - tc).as_millis());

        println!(
            "{}MB {}% nonzero {}% max",
            len * std::mem::size_of::<RatingDeltaDelta>() / (1024 * 1024),
            nonzero as f64 / len as f64 * 100.0,
            (self.list.len() * self.reference.len() * 4) as f64 / len as f64 * 100.0
        );*/

        assert!(Rating::is_zero(rating));

        return (maximum.1, maximum.0);
    }

    pub fn align_constant_delta(
        ref_spans: &[TimeSpan],
        in_spans: &[TimeSpan],
        score_fn: impl Fn(TimeDelta, TimeDelta) -> f64 + Copy,
    ) -> (TimeDelta, Rating) {
        let (min_offset, max_offset) = Self::get_offsets_bounds(ref_spans, in_spans);

        let num_slots: usize = TryInto::<usize>::try_into((max_offset - min_offset).as_i64()).unwrap();
        let num_entries: usize = in_spans.len() * ref_spans.len() * 4;

        if num_entries as f64 > num_slots as f64 * 0.1 {
            Self::align_constant_delta_bucket_sort(ref_spans, in_spans, score_fn)
        } else {
            Self::align_constant_delta_merge_sort(ref_spans, in_spans, score_fn)
        }
    }

    pub fn align_constant_delta_merge_sort(
        ref_spans: &[TimeSpan],
        in_spans: &[TimeSpan],
        score_fn: impl Fn(TimeDelta, TimeDelta) -> f64 + Copy,
    ) -> (TimeDelta, Rating) {
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

        for incorrect_ts in in_spans {
            let mut rise_ordered_delta_corrects: OrderedDeltaCorrect = OrderedDeltaCorrect::new();
            let mut up_ordered_delta_corrects: OrderedDeltaCorrect = OrderedDeltaCorrect::new();
            let mut fall_ordered_delta_corrects: OrderedDeltaCorrect = OrderedDeltaCorrect::new();
            let mut down_ordered_delta_corrects: OrderedDeltaCorrect = OrderedDeltaCorrect::new();

            for reference_ts in ref_spans {
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
                    RatingDelta::compute_rating_delta(incorrect_ts.len(), reference_ts.len(), score_fn);

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
        let mut delta: RatingDelta = RatingDelta::zero();
        let mut rating: Rating = Rating::zero();
        let mut maximum: (Rating, TimeDelta) = (Rating::zero(), first_delta_correct.time);
        for (delta_correct, next_delta_correct) in sorted_delta_corrects_iter
            .clone()
            .zip(sorted_delta_corrects_iter.skip(1))
        {
            //println!("rating: {}", rating);
            delta = delta + delta_correct.rating;
            rating = Rating::add_mul(rating, delta, next_delta_correct.time - delta_correct.time);
            if rating > maximum.0 {
                maximum = (rating, next_delta_correct.time);
            }
        }

        assert!(Rating::is_zero(rating));

        return (maximum.1, maximum.0);
    }

    #[cfg(feature = "statistics")]
    pub fn do_statistics(&self, f: impl Fn(&mut Statistics) -> std::io::Result<()>) {
        if let Some(statistics) = &self.statistics {
            f(&mut statistics.borrow_mut()).expect("failed to write statistics");
        }
    }

    pub fn align_with_splits(
        ref_spans: &[TimeSpan],
        in_spans: &[TimeSpan],
        split_penalty: RatingDelta,
        speed_optimization_opt: Option<f64>,
        score_fn: impl Fn(TimeDelta, TimeDelta) -> f64 + Copy,
        mut progress_handler: impl ProgressHandler,
    ) -> (Vec<TimeDelta>, Rating) {
        // For each segment the full rating can only be 1. So the maximum rating
        // without the split penalty is `min(list.len(), reference.len())`. So to get
        // from the normalized rating `[0, 1]` to a unnormalized rating (where only
        // values between `[0, max_rating]` are interesting) we multiply by
        // `min(list.len(), reference.len())`.

        assert!(in_spans.len() > 0);
        assert!(ref_spans.len() > 0);

        progress_handler.init(in_spans.len() as i64);

        let speed_optimization = speed_optimization_opt.unwrap_or(0.0);

        let (min_offset, max_offset) = Self::get_offsets_bounds(ref_spans, in_spans);
        let (min_offset, max_offset) = (min_offset - TimeDelta::one(), max_offset + TimeDelta::one());

        // these buffers save the offsets of a subtitle line dependent on the offset of the next line,
        //  -> this allows to compute the final corrected line offsets
        let mut offset_buffers: Vec<OffsetBuffer> = Vec::new();

        let mut culmulative_rating_buffer: RatingBuffer =
            Self::single_span_ratings(ref_spans, in_spans[0], score_fn, min_offset, max_offset).save();

        progress_handler.inc();

        for (line_nr, (&last_incorrect_span, &incorrect_span)) in
            in_spans.iter().zip(in_spans.iter().skip(1)).enumerate()
        {
            assert!(last_incorrect_span.len() > TimeDelta::zero()); // otherwise shift_simple/extend_to creates a zero-length segment
            assert!(incorrect_span.len() > TimeDelta::zero()); // otherwise shift_simple/extend_to creates a zero-length segment

            let span_distance = incorrect_span.start - last_incorrect_span.end;

            assert!(span_distance >= TimeDelta::zero()); // otherwise shift_simple/extend_to creates a zero-length segment
            assert!(culmulative_rating_buffer.first_end_point().unwrap() - min_offset > span_distance);

            let best_split_offsets = culmulative_rating_buffer
                .iter()
                .add_rating(-split_penalty)
                .shift_simple(-span_distance)
                //.clamp_end(self.get_max_offset())
                .extend_to(max_offset)
                .annotate_with_segment_start_points()
                .annotate_with_offset_info(|offset| offset + span_distance)
                .left_to_right_maximum()
                .discard_start_times()
                .simplify()
                .discard_start_times();

            let nosplit_offsets = culmulative_rating_buffer
                .iter()
                .annotate_with_segment_start_points()
                .annotate_with_offset_info(|offset| offset)
                .discard_start_times();
            //.simplify()
            //.discard_start_times();

            let single_span_ratings =
                Self::single_span_ratings(ref_spans, incorrect_span, score_fn, min_offset, max_offset).save();

            let progress_factor = (line_nr + 1) as f64 / in_spans.len() as f64;
            let epsilon = Rating::convert_from_f64(speed_optimization * 0.05 * (progress_factor * 0.8 + 0.2));

            let combined_maximum_buffer: SeparateDualBuffer =
                combined_maximum_of_dual_iterators(nosplit_offsets, best_split_offsets)
                    .discard_start_times()
                    .add_ratings_from(single_span_ratings.iter())
                    .discard_start_times()
                    .save_separate(epsilon);

            culmulative_rating_buffer = combined_maximum_buffer.rating_buffer;

            /*println!(
                "Last rating buffer length: {}",
                combined_maximum_buffer.rating_buffer.buffer.len()
            );*/

            offset_buffers.push(combined_maximum_buffer.offset_buffer);

            progress_handler.inc();
        }

        // ------------------------------------------------------------------------------
        // Extract the best offset for each incorrect span from offset buffers

        assert!(offset_buffers.len() == in_spans.len() - 1);

        let (total_rating, mut span_offset) = culmulative_rating_buffer.maximum();

        let mut result_deltas = Vec::new();
        result_deltas.push(span_offset);

        //let sum: usize = offset_buffers.iter().map(|ob| ob.len()).sum();
        //println!("{} {}MB", sum, (sum * std::mem::size_of::<crate::segments::OffsetSegment>()) as f64 / (1024 * 1024) as f64);

        for offset_buffer in offset_buffers.into_iter().rev() {
            span_offset = offset_buffer.get_offset_at(span_offset);

            // Due to ''aggressive optimization'' of the rating curve in each step
            // the maximum of the nosplit curve might be at the start. Then the
            // annotated offsets are negative. In that case - to avoid a
            // out of bounds error - we simply copy the current result delta
            // to all remaining spans.
            /*if span_offset < self.get_min_offset() {
                span_offset = std::cmp::max(span_offset, self.get_start());
                let error_delta = span_offset - incorrect_span.start;
                for _ in 0..self.list.len() - result_deltas.len() {
                    result_deltas.push(error_delta);
                }
                break;
            }*/

            result_deltas.push(span_offset);
        }

        // the deltas were inserted back-to-front
        result_deltas.reverse();

        progress_handler.finish();

        (result_deltas, total_rating)
    }

    /// Requires "start1 <= start2". Returns the compressed rating vector for
    /// the overlapping ratings of a timespan of length
    /// "length" on all start offset from "start1" to "start2".
    ///
    /// This function has O(n) runtime, where n is the number of spans in the
    /// reference list.

    fn single_span_ratings(
        ref_spans: &[TimeSpan],
        in_span: TimeSpan,
        score_fn: impl Fn(TimeDelta, TimeDelta) -> f64 + Copy,
        min_offset: TimeDelta,
        max_offset: TimeDelta,
    ) -> RatingIterator<impl Iterator<Item = RatingSegment>> {
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

        let mut builder = DifferentialRatingBufferBuilder::new(min_offset, max_offset);
        let len = ref_spans.len();
        let mut timepoints: Vec<Option<(TimeDelta, RatingDeltaDelta)>> = vec![None; 4 * len];
        for (i, &ref_span) in ref_spans.iter().enumerate() {
            let rise_delta = RatingDelta::compute_rating_delta(ref_span.len(), in_span.len(), score_fn);

            timepoints[0 * len + i] = Some((ref_span.start() - in_span.end(), rise_delta));
            timepoints[1 * len + i] = Some((ref_span.end() - in_span.end(), -rise_delta));
            timepoints[2 * len + i] = Some((ref_span.start() - in_span.start(), -rise_delta));
            timepoints[3 * len + i] = Some((ref_span.end() - in_span.start(), rise_delta));
        }

        let timepoints: Vec<(TimeDelta, RatingDeltaDelta)> = timepoints.into_iter().map(|x| x.unwrap()).collect();

        // standard merge sort
        fn merge(
            a: &[(TimeDelta, RatingDeltaDelta)],
            b: &[(TimeDelta, RatingDeltaDelta)],
        ) -> Vec<(TimeDelta, RatingDeltaDelta)> {
            let mut ai = 0;
            let mut bi = 0;
            let mut result = Vec::with_capacity(a.len() + b.len());
            loop {
                if ai == a.len() && bi == b.len() {
                    return result;
                }
                if bi == b.len() {
                    while ai < a.len() {
                        result.push(a[ai]);
                        ai = ai + 1;
                    }
                    return result;
                }
                if ai == a.len() {
                    while bi < b.len() {
                        result.push(b[bi]);
                        bi = bi + 1;
                    }
                    return result;
                }
                if a[ai].0 <= b[bi].0 {
                    result.push(a[ai]);
                    ai = ai + 1;
                } else {
                    result.push(b[bi]);
                    bi = bi + 1;
                }
            }
        }

        let x = merge(&timepoints[len * 0..len * 1], &timepoints[len * 1..len * 2]);
        let y = merge(&timepoints[len * 2..len * 3], &timepoints[len * 3..len * 4]);

        let timepoints = merge(&x, &y);

        for (segment_end, segment_end_delta_delta) in timepoints {
            builder.add_segment(segment_end, segment_end_delta_delta);
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
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::rating_type::RatingExt;
    use crate::segments::RatingFullSegment;
    use crate::tests::get_random_prepared_test_time_spans;

    fn get_dummy_spans() -> Vec<TimeSpan> {
        loop {
            let ts = get_random_prepared_test_time_spans();

            // this is unlikely
            if ts.is_empty() {
                continue;
            }

            // new will return None, if both lists are empty -> highly unlikely
            return ts;
        }
    }

    #[test]
    /// Aligns random timespans to each other and calls alass. General test whether any internal
    /// assertions are invalidated.
    fn run_aligner() {
        for _ in 0..40 {
            let (ref_spans, in_spans) = (get_dummy_spans(), get_dummy_spans());
            Aligner::align_with_splits(
                &ref_spans,
                &in_spans,
                RatingDelta::convert_from_f64(0.001),
                None,
                crate::standard_scoring,
                NoProgressHandler,
            );
        }
    }

    #[test]
    fn test_single_span_ratings() {
        for _ in 0..30 {
            let (ref_spans, in_spans) = (get_dummy_spans(), get_dummy_spans());
            let (min_offset, max_offset) = Aligner::get_offsets_bounds(&ref_spans, &in_spans);
            let (min_offset, max_offset) = (min_offset - TimeDelta::one(), max_offset + TimeDelta::one());

            for in_span in in_spans {
                let last: RatingFullSegment =
                    Aligner::single_span_ratings(&ref_spans, in_span, crate::standard_scoring, min_offset, max_offset)
                        .annotate_with_segment_start_points()
                        .into_iter()
                        .last()
                        .unwrap();
                assert!(Rating::is_zero(last.end_rating()));
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
