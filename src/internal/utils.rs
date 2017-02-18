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


use internal::{DeltaSegment, Rating, RatingSegment, TimeDelta, TimePoint, TimeSpan};
use arrayvec::ArrayVec;
use std::iter::FromIterator;

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
pub fn get_overlapping_rating_changepoints(length: TimeDelta, constspan: TimeSpan) -> [(Rating /* Delta */, TimePoint); 4] {

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

    let rise_delta = Rating::from_overlapping_spans(length, constspan.len());

    [(rise_delta, timepoints[0]), (-rise_delta, timepoints[1]), (-rise_delta, timepoints[2]), (rise_delta, timepoints[3])]
}

/// Creates a new vector with `prev` as first element and the sorted elements
/// of `a`.
pub fn sort_arrayvec3(prev: u64, a: ArrayVec<[u64; 3]>) -> ArrayVec<[u64; 4]> {
    #[allow(collapsible_if)]
    match a.len() {
        0 => ArrayVec::from_iter([prev].into_iter().cloned()),
        1 => ArrayVec::from_iter([prev].into_iter().cloned().chain(a.into_iter())),
        2 => {
            if a[0] <= a[1] {
                ArrayVec::from_iter([prev].into_iter().cloned().chain(a.into_iter()))
            } else {
                ArrayVec::from_iter([prev].into_iter().cloned().chain(a.into_iter().rev()))
            }
        }
        3 => {
            if a[0] <= a[1] {
                if a[1] <= a[2] {
                    ArrayVec::from([prev, a[0], a[1], a[2]])
                } else if a[0] <= a[2] {
                    ArrayVec::from([prev, a[0], a[2], a[1]])
                } else {
                    ArrayVec::from([prev, a[2], a[0], a[1]])
                }
            } else {
                // here: a[1] <= a[0]

                if a[0] <= a[2] {
                    ArrayVec::from([prev, a[1], a[0], a[2]])
                } else if a[1] <= a[2] {
                    ArrayVec::from([prev, a[1], a[2], a[0]])
                } else {
                    ArrayVec::from([prev, a[2], a[1], a[0]])
                }
            }
        }
        _ => panic!("ArrayVec<[T; 3]>::len() is greater than 3"),
    }
}

/// Removes any duplicate elements from the vector.
///
/// Requires a sorted non-empty array vector.
fn dedup_arrayvec4(a: ArrayVec<[u64; 4]>) -> ArrayVec<[u64; 4]> {
    let mut last = a[0];
    let mut result = ArrayVec::new();
    result.push(last);
    for elem in a {
        if last != elem {
            result.push(elem);
            last = elem;
        }
    }
    result
}

pub fn subseg_by_max_start<ID>(segs: [DeltaSegment<Rating, Rating>; 2], ids: [ID; 2], start: u64, end: u64) -> (RatingSegment, ID)
    where ID: Copy + Eq
{
    if (segs[0].value_at_index(start as i64), segs[0].delta()) >= (segs[1].value_at_index(start as i64), segs[1].delta()) {
        (segs[0].subseg(start, end), ids[0])
    } else {
        (segs[1].subseg(start, end), ids[1])
    }
}


/// Given two segments, it computes the subsegments which have the best rating
/// in their span.
///
/// The parameter `ids` is there to identify which segment the subsegment came
/// from. The length
/// of all given spans have to be same, the sum of all returned segments is the
/// orginal length.
pub fn get_best_rating_segments_of_2<ID>(segs: [DeltaSegment<Rating, Rating>; 2], ids: [ID; 2]) -> ArrayVec<[(DeltaSegment<Rating, Rating>, ID); 3]>
    where ID: Copy + Eq
{
    let len = segs[0].len();
    assert!(len == segs[1].len());

    match get_switch_point(segs[0], segs[1]) {
        None => ArrayVec::from_iter([subseg_by_max_start(segs, ids, 0, len)].iter().cloned()),
        Some(point) => ArrayVec::from_iter([subseg_by_max_start(segs, ids, 0, point), subseg_by_max_start(segs, ids, point, len)].iter().cloned()),
    }
}


/// Given three segments, it computes the subsegments which have the best
/// rating in their span.
///
/// The parameter `ids` is there to identify which segment the subsegment came
/// from. The length
/// of all given spans have to be same, the sum of all returned segments is the
/// orginal length.
pub fn get_best_rating_segments_of_3<ID>(segs: [DeltaSegment<Rating, Rating>; 3], ids: [ID; 3]) -> ArrayVec<[(DeltaSegment<Rating, Rating>, ID); 3]>
    where ID: Copy + Eq
{
    let seg1 = segs[0];
    let seg2 = segs[1];
    let seg3 = segs[2];

    let id1 = ids[0];
    let id2 = ids[1];
    let id3 = ids[2];


    let len = seg1.len();
    assert!(len == seg2.len());
    assert!(len == seg3.len());

    // get best ratings segments of first 2 segments
    let mut switch_points: ArrayVec<[u64; 3]> = ArrayVec::new();
    if let Some(split_point) = get_switch_point(seg1, seg2) {
        switch_points.push(split_point);
    }
    if let Some(split_point) = get_switch_point(seg1, seg3) {
        switch_points.push(split_point);
    }
    if let Some(split_point) = get_switch_point(seg2, seg3) {
        switch_points.push(split_point);
    }

    let switch_points = sort_arrayvec3(0, switch_points);
    let switch_points = dedup_arrayvec4(switch_points);

    let mut segments: ArrayVec<[(RatingSegment, ID); 3]> = ArrayVec::new();

    let next_points = switch_points.iter().cloned().skip(1).chain(Some(len).into_iter());
    for (switch_point, next_point) in switch_points.iter().cloned().zip(next_points) {
        // get the best segment for the current index
        //  -> first compare their rating, in case they are the same, compare the deltas
        let value1 = (seg1.value_at_index(switch_point as i64), seg1.delta());
        let value2 = (seg2.value_at_index(switch_point as i64), seg2.delta());
        let value3 = (seg3.value_at_index(switch_point as i64), seg3.delta());

        #[allow(collapsible_if)]
        let (current_best_segment, current_id) = if value1 < value2 {
            if value2 < value3 { (seg3, id3) } else { (seg2, id2) }
        } else {
            if value1 < value3 { (seg3, id3) } else { (seg1, id1) }
        };

        if let Some(last_ref) = segments.last_mut() {
            if last_ref.1 == current_id {
                last_ref.0 = RatingSegment::with_new_length((*last_ref).0, last_ref.0.len() + next_point - switch_point);
                continue;
            }
        }

        segments.push((current_best_segment.subseg(switch_point, next_point), current_id));
    }

    assert!(segments.len() < 4);

    segments
}

/// Requires 'seg1.len() == seg2.len()'. Returns the split point index if it is
/// within "(0, len)" (borders excluded), otherwise `None`.
/// The switch point is the first index where one segment overtakes the other.
#[allow(if_same_then_else)]
fn get_switch_point(seg1: DeltaSegment<Rating, Rating>, seg2: DeltaSegment<Rating, Rating>) -> Option<u64> {
    let len = seg1.len();
    assert!(len == seg2.len());
    let (f1, f2, l1, l2) = (seg1.first_value(), seg2.first_value(), seg1.last_value(), seg2.last_value());

    if f1 <= f2 && l1 <= l2 {
        // segment1 is always smaller than segment2
        None
    } else if f2 <= f1 && l2 <= l1 {
        // segment2 is always smaller than segment1
        None
    } else {
        let (d1, d2) = (seg1.delta(), seg2.delta());

        // solve "t1 + d1 * x = t2 + d2 * x" for x
        //  =>  "x = (t1 - t2) / (d2 - d1)"
        //
        // because this is a interger division (giving us x_int): x_int <= x < x_int + 1
        //
        // switch point is then "(x_int + 1)" which is the first index where the second
        // segment is better than the
        // original
        let switch_point: u64 = ((f1 - f2) / (d2 - d1) + 1) as u64;
        assert!(0 < switch_point);
        assert!(switch_point <= len);

        Some(switch_point)
    }

}

#[cfg(test)]
mod tests {
    use rand;
    use rand::Rng;
    use std::ops::Range;
    use internal::*;
    use arrayvec::ArrayVec;

    fn get_random_rating_segment(s: Range<i64>, d: Range<i64>, len: Range<u64>) -> RatingSegment {
        let mut rng = rand::thread_rng();
        let vs = rng.gen_range(s.start, s.end + 1);
        let vd = rng.gen_range(d.start, d.end + 1);
        let vlen = rng.gen_range(len.start, len.end + 1);
        DeltaSegment::new(Rating::from(vs), Rating::from(vd), vlen)
    }

    // Test `get_best_rating_segments_of_3` by validating it with `validate_best_segments`.
    #[test]
    fn test_get_best_segments3() {
        // genrate test data
        let gen_segment = || get_random_rating_segment(-100..100, -100..100, 100..100);
        let data_vec: Vec<_> = (0..2000).map(|_| [gen_segment(), gen_segment(), gen_segment()]).collect();

        for test_segs in data_vec {
            let ids = [0, 1, 2];
            let best_segments: ArrayVec<[(RatingSegment, i32); 3]> = get_best_rating_segments_of_3(test_segs, ids);
            validate_best_segments(test_segs.into_iter()
                                            .map(|&seg| RatingBuffer::from(seg))
                                            .collect(),
                                   best_segments);
        }
    }

    // Test `get_best_rating_segments_of_2` by validating it with `validate_best_segments`.
    #[test]
    fn test_get_best_segments2() {

        // genrate test data
        let gen_segment = || get_random_rating_segment(-100..100, -100..100, 100..100);
        let data_vec: Vec<_> = (0..2000).map(|_| [gen_segment(), gen_segment()]).collect();

        for test_segs in data_vec {
            let ids = [0, 1];
            let best_segments: ArrayVec<[(RatingSegment, i32); 3]> = get_best_rating_segments_of_2(test_segs, ids);
            println!("Test segments: {:?}", test_segs);
            println!("Best segments: {:?}", best_segments);
            println!();
            validate_best_segments(test_segs.into_iter()
                                            .map(|&seg| RatingBuffer::from(seg))
                                            .collect(),
                                   best_segments);
        }
    }

    /// Checks whether the current segment from `best_segs` always holds the maximum value at that position from all buffers in `segs`.
    fn validate_best_segments(segs: Vec<RatingBuffer>, best_segs: ArrayVec<[(RatingSegment, i32); 3]>) {
        assert!(segs.len() > 0);
        let len = segs[0].len();
        for seg in &segs {
            assert!(seg.len() == len);
        }
        assert_eq!(len,
                   best_segs.iter().map(|&(ref rating_buffer, _)| rating_buffer.len()).sum());

        // a vector of iterators (each representing a rating buffer)
        let mut iters: Vec<_> = segs.iter().map(|seg_ref| seg_ref.iter()).collect();

        for (best_seg, id) in best_segs {
            // go through all values in this supposedly "best" segment
            for best_value_by_best_segment in RatingBuffer::from(best_seg).iter() {
                // compute the maxium value by comparing all raings at the current position
                let separate_ratings: Vec<Rating> = iters.iter_mut().map(|iter| iter.next().unwrap()).collect();
                let real_max: Rating = separate_ratings.iter().cloned().max().unwrap();

                // assert that the maximum rating really is the maximum rating
                assert_eq!(real_max, best_value_by_best_segment);

                // require that the
                assert_eq!(real_max, separate_ratings[id as usize]);
            }
        }
    }

}
