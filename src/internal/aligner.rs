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


use arrayvec::ArrayVec;
use internal::{CombinedSegmentIterator, DeltaBufferBuilder, DeltaBufferReader, DeltaSegment, OptionSegment, Rating, RatingBuffer, RatingSegment,
               TimeDelta, TimePoint, TimeSpan, TimepointBuffer, get_best_rating_segments_of_2, get_best_rating_segments_of_3,
               get_overlapping_rating_changepoints};
use std::cmp::min;
use std::iter::{FromIterator, once};

/// The main align algorithm uses a long buffer of different ratings. This
/// structure provides the information to
/// express that a `rating` is at a `specific timepoint` in that buffer.
struct RatingLocation {
    pub rating: Rating,
    pub location: TimePoint,
}


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

    /// The bonus rating for two consecutive subtitles which preserve the
    /// orginal space (which get
    /// shifted by the same `TimeDelta`)/the penalty for introducing splits in
    /// the alignment.
    nosplit_bonus: Rating,

    /// Progress handler provided by the user of this crate.
    progress_handler_opt: Option<Box<ProgressHandler>>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
/// In its core, the algorithm uses a dynamic programming approach and
/// generates a new value/rating in
/// the internal table by comparing already-existing ratings at three different
/// positions in the table.
/// It then chooses the highest rating and does some processing dependent on
/// the case.
///
/// These three choices is where to put the/new current timespan: leave the
/// timespan on the position from the
/// previous iteration, repositon its start to the current timepoint, or
/// reposition it to the non-split
/// position (to get the nosplit bonus).
///
/// This enumeration represents the three different choices the algorithm can
/// select from.
enum Choice {
    /// Reposition start of timespan to current timepoint.
    Reposition,

    /// Leave timespan on same position like in previous iteration.
    Fixed,

    /// Reposition timespan so no split is introduced (gives `nosplit bonus`).
    NosplitReposition,
}


impl Aligner {
    /// In each list no time span should intersect any other and both list are
    /// sorted by starting times.
    pub fn new(list: Vec<TimeSpan>,
               reference: Vec<TimeSpan>,
               nopsplit_bonus_normalized: f64,
               mut progress_handler_opt: Option<Box<ProgressHandler>>)
               -> Option<Aligner> {
        if let Some(ref mut progress_handler) = progress_handler_opt {
            progress_handler.init(list.len() as i64);
        }

        if list.is_empty() || reference.is_empty() {
            if let Some(ref mut progress_handler) = progress_handler_opt {
                progress_handler.finish();
            }
            return None;
        }

        // this is the timespan length which can contain all incorrect subtitles
        let list_timespan = list.last().unwrap().end() - list.first().unwrap().start();

        // It might be possible that all corrected subtiles fit in the reference list
        // timeframe. It they don't
        // we need to provide extra space, so that the produting corrected subtitles
        // still fit into the
        // whole [start, end] timeframe. Because `list_timespan` is the length of the
        // whole incorrect subtitle file,
        // we can just extend the reference timeframe by `list_timespan` on both ends.
        let start = reference.first().unwrap().start() - list_timespan;
        let end = reference.last().unwrap().end() + list_timespan;

        // For each segment the full rating can only be 1. So the maximum rating
        // without the nosplit bonus is `min(list.len(), reference.len())`. So to get
        // from the normalized rating `[0, 1]` to a unnormalized rating (where only
        // values between `[0, max_rating]` are interesting) we multiply by
        // `min(list.len(), reference.len())`.
        let nopsplit_bonus_unnormalized = min(list.len(), reference.len()) as f64 * nopsplit_bonus_normalized;

        // quick check for integrity
        assert!(start < end);

        Some(Aligner {
                 list: list,
                 reference: reference,
                 buffer_timespan: TimeSpan::new(start, end),
                 nosplit_bonus: Rating::nosplit_bonus(nopsplit_bonus_unnormalized),
                 progress_handler_opt: progress_handler_opt,
             })
    }

    pub fn get_start(&self) -> TimePoint {
        self.buffer_timespan.start()
    }

    pub fn get_end(&self) -> TimePoint {
        self.buffer_timespan.end()
    }

    pub fn get_buffer_length(&self) -> u64 {
        u64::from(self.buffer_timespan.len())
    }

    pub fn align_all_spans(&mut self) -> Vec<TimeDelta> {
        let mut all_spanstart_buffers: Vec<TimepointBuffer> = Vec::new();
        let mut last_rating_buffer: RatingBuffer = RatingBuffer::init_with_one_segment(Rating::zero(), Rating::zero(), self.get_buffer_length());

        // iterator that removes the first element and adds a None value to the end ->
        // provedes the "next" span
        for (i, time_span) in self.list.iter().cloned().enumerate() {
            // compute the space between this span and the next span
            let next_span_opt = self.list.get(i + 1);
            let optimal_startdiff_opt = next_span_opt.map(|next_span| next_span.start() - time_span.start());
            let (rating_buffer, span_positions_buffer) = self.align_new_span(last_rating_buffer, time_span, optimal_startdiff_opt);

            // the rating buffer is only needed for the next lane, but the last span
            // position have to be remembered to get the deltas for each subtitle at the end
            last_rating_buffer = rating_buffer;
            all_spanstart_buffers.push(span_positions_buffer);

            // inform user we have done one step
            if let Some(ref mut progress_handler) = self.progress_handler_opt {
                progress_handler.inc();
            }
        }

        // find the index in the last rating buffer (which represents all spans) with
        // maximum rating - which is the last index because the ratings rise monotonous.
        let mut best_end = self.get_end() - TimeDelta::one();

        // because we can read each interval ends at the span start of next span, we
        // just have to go backwards from span start to span start
        let mut time_span_starts = Vec::new();
        for last_span_positions in all_spanstart_buffers.into_iter().rev() {
            let mut reader = DeltaBufferReader::new(&last_span_positions, self.get_start());
            best_end = reader.read_by_timepoint(best_end);
            time_span_starts.push(best_end);
        }

        // inform user we are done with the work
        if let Some(ref mut progress_handler) = self.progress_handler_opt {
            progress_handler.finish();
        }

        time_span_starts = time_span_starts.into_iter().rev().collect();
        self.list
            .iter()
            .zip(time_span_starts.iter())
            .map(|(&original_time_span, &new_start)| new_start - original_time_span.start())
            .collect()
    }

    /// Returns the align rating of n + 1 time spans from the align rating of n
    /// time spans.
    ///
    /// total_timespan_time: the sum of all timespan lengths NOT including the
    /// new span
    fn align_new_span(&self,
                      prev_rating_buffer: RatingBuffer,
                      new_span: TimeSpan,
                      optimal_startdiff_opt: Option<TimeDelta>)
                      -> (RatingBuffer, TimepointBuffer) {

        assert!(prev_rating_buffer.len() == self.get_buffer_length());

        let overlapping_rating = self.get_compressed_overlapping_ratings(self.get_start(), self.get_end(), new_span.len());

        // for an "end" the repositon rating is the best rating where the new span ends
        // at that "end"
        let rating_by_repositioning = RatingBuffer::combined_add(&overlapping_rating, &prev_rating_buffer);

        match optimal_startdiff_opt {
            Some(optimal_startdiff) => self.get_next_lane_with_nosplit(&rating_by_repositioning, optimal_startdiff),
            None => self.get_next_lane_without_nosplit(&rating_by_repositioning),
        }
    }

    /// The algorithm creates a NxU matrix where N is the number of incorrect
    /// subtitles and U is the timespan all subtitles together. On a high
    /// level, each N+1 row (giving the next N+1xU matrix) can be computed by
    /// the NxU matrix and the previous values on the N+1 row. I call this new
    /// matrix-row "lane".
    fn get_next_lane<I1, I2, F>(&self,
                                // will be ignored if `get_maxrat_segments` never return Choice::NosplitReposition
                                optimal_startdiff: TimeDelta,
                                reposition_rating_iter: I1,
                                nosplit_rating_iter: I2,
                                get_maxrat_segments: F)
                                -> (RatingBuffer, TimepointBuffer)
        where I1: Iterator<Item = RatingSegment>,
              I2: Iterator<Item = OptionSegment<RatingSegment>>,

              // The three rating segemnts are the "reposition rating" the "fixed rating" and the "nosplit rating".
              //
              // The last parameter is the "absolute best choice" in the last loop. Because these absolute best (== one maxrat segment),
              // choice is probably the same in the next loop, we can do optimizations with it.
              F: Fn(RatingSegment,
                    RatingSegment,
                    Option<RatingSegment>,
                    Option<Choice>)
                    -> ArrayVec<[(RatingSegment, Choice); 3]>
    {

        let mut rating_builder = DeltaBufferBuilder::<Rating, Rating>::new();
        let mut spanstart_builder = DeltaBufferBuilder::<TimePoint, TimeDelta>::new();
        let mut segstart_timepoint = self.get_start();
        let mut past_max: RatingLocation = RatingLocation {
            rating: Rating::zero(),
            location: segstart_timepoint,
        };
        let mut last_absolute_best_choice: Option<Choice> = None;

        for (reposition_rating_seg, nosplit_reposition_rating) in CombinedSegmentIterator::new(reposition_rating_iter, nosplit_rating_iter) {

            if reposition_rating_seg.is_decreasing() && reposition_rating_seg.first_value() > past_max.rating {
                // the first rating value is the new maximum, but after that, the rating
                // decreases
                //  -> the fixed spanstart should point to the first rating
                past_max.rating = reposition_rating_seg.first_value();
                past_max.location = segstart_timepoint;
            }

            let fixed_rating_seg = DeltaSegment::new(past_max.rating, Rating::zero(), reposition_rating_seg.len());

            let maxrat_segs = get_maxrat_segments(reposition_rating_seg,
                                                  fixed_rating_seg,
                                                  nosplit_reposition_rating,
                                                  last_absolute_best_choice);

            last_absolute_best_choice = if maxrat_segs.len() == 1 {
                Some(maxrat_segs[0].1)
            } else {
                None
            };


            // = segment with maximal rating
            for (maxrat_seg, choice) in maxrat_segs {
                // depending on the best maxrat-choice (choice that leads to the segment with
                // maximum rating),
                // the spanstart has to set differently -> maximum rating comes into being with
                // that spanstart
                let (spanstart_seg_timepoint, spanstart_seg_delta) = match choice {
                    Choice::Reposition => (segstart_timepoint, TimeDelta::one()),
                    Choice::NosplitReposition => (segstart_timepoint - optimal_startdiff, TimeDelta::one()),
                    Choice::Fixed => (past_max.location, TimeDelta::zero()),
                };

                let new_spanstart_segment = DeltaSegment::new(spanstart_seg_timepoint,
                                                              spanstart_seg_delta,
                                                              maxrat_seg.len());

                // add the best rating and the associated spanstarts to the buffer builders
                rating_builder.add_segment(maxrat_seg);
                spanstart_builder.add_segment(new_spanstart_segment);

                segstart_timepoint += TimeDelta::one() * maxrat_seg.len() as i64;
            }

            // the reposition rating is constant or increases (decreasing case has already been handled
            // at the start of the loop), so the next "past_max.rating" might be at the end of of the
            // repositon rating segment
            if past_max.rating < reposition_rating_seg.last_value() {
                // segstart_timepoint points to beginning of next segment, so minus one and we have the
                // end of the current segment
                past_max.rating = reposition_rating_seg.last_value();
                past_max.location = segstart_timepoint - TimeDelta::one();
            }

        }

        (rating_builder.get_buffer(), spanstart_builder.get_buffer())
    }

    fn get_next_lane_without_nosplit(&self, reposition_rating_buffer: &RatingBuffer) -> (RatingBuffer, TimepointBuffer) {
        self.get_next_lane(TimeDelta::zero(), // ignored
                           reposition_rating_buffer.iter_segments().cloned(),
                           once(OptionSegment::NoneSeg::<RatingSegment>(self.get_buffer_length())),

                           // do not compare with nosplit segment
                           Self::get_segments_and_choices_with_nosplit)

    }

    fn get_segments_and_choices_without_nosplit(reposition_rating_seg: RatingSegment,
                                                fixed_rating_seg: RatingSegment,
                                                _: Option<Choice>)
                                                -> ArrayVec<[(RatingSegment, Choice); 3]> {

        let compared_segs = [reposition_rating_seg, fixed_rating_seg];
        let corresponding_choices = [Choice::Reposition, Choice::Fixed];
        get_best_rating_segments_of_2(compared_segs, corresponding_choices)
    }

    fn get_segments_and_choices_with_nosplit(reposition_rating_seg: RatingSegment,
                                             fixed_rating_seg: RatingSegment,
                                             nosplit_rating_seg_opt: Option<RatingSegment>,
                                             last_absolute_best_choice: Option<Choice>)
                                             -> ArrayVec<[(RatingSegment, Choice); 3]> {

        let nosplit_rating_seg = match nosplit_rating_seg_opt {
            Some(x) => x,
            None => {
                return Self::get_segments_and_choices_without_nosplit(reposition_rating_seg,
                                                                      fixed_rating_seg,
                                                                      last_absolute_best_choice);
            }
        };

        // Two consecutive segment parts often have the same result. So if we use the
        // the result from the
        // last loop, we can easily check for a single result instead of having to
        // compute it expensively.
        match last_absolute_best_choice {
            Some(Choice::Reposition) => {
                if reposition_rating_seg.is_greatequal(nosplit_rating_seg) && reposition_rating_seg.is_greatequal(fixed_rating_seg) {
                    return ArrayVec::from_iter([(reposition_rating_seg, Choice::Reposition)]
                                                   .into_iter()
                                                   .cloned());
                }
            }
            Some(Choice::NosplitReposition) => {
                if nosplit_rating_seg.is_greatequal(reposition_rating_seg) && nosplit_rating_seg.is_greatequal(fixed_rating_seg) {
                    return ArrayVec::from_iter([(nosplit_rating_seg, Choice::NosplitReposition)]
                                                   .into_iter()
                                                   .cloned());
                }
            }
            Some(Choice::Fixed) => {
                if fixed_rating_seg.is_greatequal(reposition_rating_seg) && fixed_rating_seg.is_greatequal(nosplit_rating_seg) {
                    return ArrayVec::from_iter([(fixed_rating_seg, Choice::Fixed)].into_iter().cloned());
                }
            }
            None => {}
        }

        let compared_segs = [reposition_rating_seg, nosplit_rating_seg, fixed_rating_seg];
        let corresponding_choices = [Choice::Reposition, Choice::NosplitReposition, Choice::Fixed];
        get_best_rating_segments_of_3(compared_segs, corresponding_choices)
    }

    fn get_next_lane_with_nosplit(&self, reposition_rating_buffer: &RatingBuffer, optimal_startdiff: TimeDelta) -> (RatingBuffer, TimepointBuffer) {

        // Get an iterator where each lookup at position
        // "shifted_bonus_segments_iter[x]" is equivalent
        // to "rating_by_repositioning[x - space_to_next] + bonus_time".
        // This is the rating for a nosplit alignment. To achive that "x -
        // space_to_next" we add a dummy segment to the front.
        let dummy_segment = OptionSegment::NoneSeg(optimal_startdiff.into());
        let bonus_segments_iter =
            reposition_rating_buffer.iter_segments()
                                    .map(|&x| OptionSegment::SomeSeg(x + self.nosplit_bonus));
        let shifted_bonus_segments_iter = once(dummy_segment).chain(bonus_segments_iter);

        self.get_next_lane(optimal_startdiff,
                           reposition_rating_buffer.iter_segments().cloned(),
                           shifted_bonus_segments_iter,
                           Self::get_segments_and_choices_with_nosplit)
    }

    /// Requires "start1 <= start2". Returns the compressed rating vector for
    /// the overlapping ratings of a timespan of length
    /// "length" on all start position from "start1" to "start2".
    ///
    /// This function has O(n) runtime, where n is the number of spans in the
    /// reference list.

    fn get_compressed_overlapping_ratings(&self, start1: TimePoint, start2: TimePoint, length: TimeDelta) -> RatingBuffer {
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
        assert!(start1 <= start2);

        let mut builder = DeltaBufferBuilder::new();
        let mut timepoints: [Vec<(Rating, TimePoint)>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for &ref_ts in &self.reference {
            let changepoints = get_overlapping_rating_changepoints(length, ref_ts);
            timepoints[0].push(changepoints[0]);
            timepoints[1].push(changepoints[1]);
            timepoints[2].push(changepoints[2]);
            timepoints[3].push(changepoints[3]);
        }

        // this is a vector of 4 iterators, each iterating over the contents of
        // "timepoints[0]" to "timepoints[3]"
        let mut iterators: ArrayVec<[_; 4]> = timepoints.into_iter()
                                                        .cloned()
                                                        .map(|v| v.into_iter().peekable())
                                                        .collect();
        let mut first_timepoint: Option<TimePoint> = None;
        let mut last_timepoint: Option<TimePoint> = None;
        let mut current_abs = Rating::zero();
        let mut current_delta = Rating::zero();
        loop {
            // unpack the first value of each iterator
            let next_timepoints: ArrayVec<[(usize, (Rating, TimePoint)); 4]> =
                iterators.iter_mut()
                         .enumerate()
                         .map(|(i, iter)| iter.peek().map(|&v| (i, v)))
                         .filter_map(|opt| opt)
                         .collect();

            // take the first next timepoint
            let next_changepoint_opt = next_timepoints.into_iter()
                                                      .min_by_key::<TimePoint, _>(|a| (a.1).1);

            // because each original array had the same length, all iterators should end at
            // the same time
            let (next_id, (next_rating_delta, next_timepoint)) = match next_changepoint_opt {
                Some(next_changepoint) => next_changepoint,
                None => break,
            };


            if first_timepoint.is_none() {
                first_timepoint = Some(next_timepoint)
            };

            // add the new segment with the current_delta
            if let Some(last_timepoint) = last_timepoint {
                let len: u64 = u64::from(next_timepoint - last_timepoint);
                builder.add_segment(DeltaSegment::new(current_abs, current_delta, len));
                current_abs += current_delta * len as i64;
            }

            current_delta += next_rating_delta;
            last_timepoint = Some(next_timepoint);

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
        assert_eq!(current_abs, Rating::zero());
        assert_eq!(current_delta, Rating::zero());
        builder.add_segment(DeltaSegment::new(current_abs, current_delta, 1));

        match (first_timepoint, last_timepoint) {
            (Some(first_timepoint), Some(_)) => {
                builder.get_buffer()
                       .with_new_borders(i64::from(start1 - first_timepoint),
                                         i64::from(start2 - start1))
            }
            _ => unreachable!(), // lists in aligner should be non-empty
        }
    }


    /// Computes the same buffer as the fast variant (hopefully) and is shorter and easier to debug, but slower.
    #[cfg(test)]
    fn get_compressed_overlapping_ratings_slow(&self, start1: TimePoint, start2: TimePoint, length: TimeDelta) -> RatingBuffer {
        let istart1: i64 = start1.into();
        let istart2: i64 = start2.into();
        let mut rating_buffer_builder = DeltaBufferBuilder::new();

        for istart in istart1..istart2 {
            let start = TimePoint::from(istart);
            let span = TimeSpan::new(start, start + length);

            // summation of ratings of current time span with each reference time span
            let rating: Rating = self.reference
                                     .iter()
                                     .map(|ref_ts| {
                                              let num_overlapping_segments: i64 =
                                                  ref_ts.get_overlapping_length(span).into();
                                              let single_segment_rating = Rating::from_overlapping_spans(span.len(), ref_ts.len());
                                              single_segment_rating * num_overlapping_segments
                                          })
                                     .sum();

            rating_buffer_builder.add_segment(RatingSegment::new(rating, Rating::zero(), 1));
        }

        rating_buffer_builder.get_buffer()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand;
    use rand::Rng;
    use tests::get_random_prepared_test_time_spans;

    fn get_dummy_aligner() -> Aligner {
        loop {
            let reference_ts = get_random_prepared_test_time_spans();
            let incorrect_ts = get_random_prepared_test_time_spans();

            // new will return None, if both lists are empty -> highly unlikely
            if let Some(a) = Aligner::new(reference_ts, incorrect_ts, 0.03, None) {
                return a;
            }
        }
    }

    #[test]
    /// Aligns random timespans to each other and calls aligner. General test whether any internal
    /// assertions are invalidated.
    fn run_aligner() {
        for _ in 0..20 {
            get_dummy_aligner().align_all_spans();
        }
    }

    #[test]
    /// `get_compressed_overlapping_ratings()` is highly optimized -> compare the results of slow and fast
    /// implemntations.
    fn get_compressed_overlapping_ratings() {
        let mut rng = rand::thread_rng();

        for _ in 0..30 {
            let aligner = get_dummy_aligner();
            let len: i64 = (rng.next_u32() % 100) as i64;
            let rating_buffer1 = aligner.get_compressed_overlapping_ratings(aligner.get_start(),
                                                                            aligner.get_end(),
                                                                            TimeDelta::one() * len);
            let rating_buffer2 = aligner.get_compressed_overlapping_ratings_slow(aligner.get_start(),
                                                                                 aligner.get_end(),
                                                                                 TimeDelta::one() * len);
            assert_eq!(rating_buffer1.iter().collect::<Vec<_>>(),
                       rating_buffer2.iter().collect::<Vec<_>>());
        }
    }

}
