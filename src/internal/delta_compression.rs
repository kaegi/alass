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


use internal::{Rating, TimeDelta, TimePoint};

use std::fmt::Display;
use std::iter::Peekable;
use std::ops::{Mul, Sub};
use std::ops::Add;
use std::slice::Iter;

pub type TimepointBuffer = DeltaBuffer<TimePoint, TimeDelta>;
pub type RatingBuffer = DeltaBuffer<Rating, Rating>;

// pub type TimepointSegment = DeltaSegment<TimePoint, TimeDelta>;
pub type RatingSegment = DeltaSegment<Rating, Rating>;

// //////////////////////////////////////////////////////////////////////////////////////////////////
// ZERO TRAIT

pub trait Zero {
    fn zero() -> Self;
}

impl Zero for TimeDelta {
    fn zero() -> TimeDelta {
        TimeDelta::zero()
    }
}

impl Zero for Rating {
    fn zero() -> Rating {
        Rating::zero()
    }
}


// /////////////////////////////////////////////////////////////////////////////////////////////////
// DELTA SEGMENT
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
/// T is the actual value, D is the delta type
pub struct DeltaSegment<T, D> {
    start: T,
    delta: D,
    len: u64,
}

impl<T, D> DeltaSegment<T, D> {
    pub fn new(start: T, delta: D, len: u64) -> DeltaSegment<T, D> {
        DeltaSegment {
            delta: delta,
            start: start,
            len: len,
        }
    }

    pub fn with_new_length(seg: DeltaSegment<T, D>, len: u64) -> DeltaSegment<T, D> {
        Self::new(seg.start, seg.delta, len)
    }


    pub fn delta(&self) -> D
    where
        D: Copy,
    {
        self.delta
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_decreasing(&self) -> bool
    where
        D: Copy + Zero + PartialOrd,
    {
        self.delta() < D::zero()
    }
}

impl<T, D> DeltaSegment<T, D>
where
    T: Add<D, Output = T> + Copy,
    D: Mul<i64, Output = D> + Copy,
{
    pub fn first_value(&self) -> T {
        self.start
    }

    pub fn last_value(&self) -> T {
        self.value_at_index(self.len as i64 - 1)
    }

    pub fn value_at_index(&self, i: i64) -> T {
        self.start + self.delta * i
    }

    /// Create a new delta segment containing all elements in `(from, to)`
    /// where `from` is included and `to` excluded.
    pub fn subseg(&self, from: u64, to: u64) -> DeltaSegment<T, D> {
        assert!(from < to);
        assert!(to <= self.len());
        DeltaSegment {
            start: self.value_at_index(from as i64),
            delta: self.delta(),
            len: to - from,
        }
    }

    pub fn is_greatequal(&self, other: DeltaSegment<T, D>) -> bool
    where
        T: Ord,
    {
        self.first_value() >= other.first_value() && self.last_value() >= other.last_value()
    }

    pub fn split_to_end(&self, from: u64) -> DeltaSegment<T, D> {
        assert!(from <= self.len());
        DeltaSegment::new(
            self.value_at_index(from as i64),
            self.delta(),
            self.len() - from as u64,
        )
    }

    pub fn split_from_begin_to(&self, to: u64) -> DeltaSegment<T, D> {
        assert!(to <= self.len());
        DeltaSegment::new(self.first_value(), self.delta(), to)
    }
}

impl<T, D> Add<D> for DeltaSegment<T, D>
where
    T: Add<D, Output = T> + Copy,
    D: Mul<i64, Output = D> + Copy,
{
    type Output = DeltaSegment<T, D>;
    fn add(self, rhs: D) -> DeltaSegment<T, D> {
        DeltaSegment::new(self.first_value() + rhs, self.delta(), self.len())
    }
}

// //////////////////////////////////////////////////////////////////////////////////////////////////
// DELTA COMPRESSION BUFFER
#[derive(Clone, Debug)]
pub struct DeltaBuffer<T, D> {
    data: Vec<DeltaSegment<T, D>>,
    length: u64,
}

impl<T, D> From<DeltaSegment<T, D>> for DeltaBuffer<T, D>
where
    T: Add<D, Output = T> + Sub<T, Output = D> + Eq + Copy,
    D: Mul<i64, Output = D> + Copy + Eq,
{
    fn from(seg: DeltaSegment<T, D>) -> DeltaBuffer<T, D> {
        DeltaBuffer::init_with_one_segment(seg.first_value(), seg.delta(), seg.len())
    }
}

impl<T, D, I: Iterator<Item = DeltaSegment<T, D>>> From<I> for DeltaBuffer<T, D>
where
    T: Add<D, Output = T> + Sub<T, Output = D> + Eq + Copy,
    D: Mul<i64, Output = D> + Copy + Eq,
{
    fn from(i: I) -> DeltaBuffer<T, D> {
        let mut builder = DeltaBufferBuilder::new();
        for seg in i {
            builder.add_segment(seg);
        }
        builder.get_buffer()
    }
}

impl<T, D> DeltaBuffer<T, D>
where
    T: Add<D, Output = T> + Sub<T, Output = D> + Eq + Copy,
    D: Mul<i64, Output = D> + Copy + Eq,
{
    pub fn new() -> DeltaBuffer<T, D> {
        DeltaBuffer {
            data: Vec::new(),
            length: 0,
        }
    }

    pub fn init_with_one_segment(start: T, delta: D, len: u64) -> DeltaBuffer<T, D> {
        DeltaBuffer {
            data: vec![DeltaSegment::new(start, delta, len)],
            length: len,
        }
    }

    pub fn len(&self) -> u64 {
        self.length
    }


    /// This function will return a new buffer from "new_start" to "new_end"
    /// from the current data, which get
    /// filled left and right with the first/last value of the entire buffer.
    pub fn with_new_borders(&self, new_start: i64, new_length: i64) -> DeltaBuffer<T, D>
    where
        D: Zero,
    {
        // XXX: do not use intermediate buffer? (measure performance impact first)
        assert!(new_length >= 0);
        assert!(!self.data.is_empty());
        let first_value = self.first_value().unwrap();
        let last_value = self.last_value().unwrap();

        let mut buffer = if new_start < 0 {
            self.extended_front(DeltaSegment::new(first_value, D::zero(), -new_start as u64))
        } else {
            self.truncated_front(new_start as u64)
        };


        buffer = buffer.fixed_length(DeltaSegment::new(last_value, D::zero(), new_length as u64));

        buffer
    }

    pub fn first_value(&self) -> Option<T> {
        self.data.first().map(|&first_segment| {
            first_segment.first_value()
        })
    }

    pub fn last_value(&self) -> Option<T> {
        self.data.last().map(
            |&last_segment| last_segment.last_value(),
        )
    }

    pub fn extended_front(&self, seg: DeltaSegment<T, D>) -> DeltaBuffer<T, D> {
        let mut builder = DeltaBufferBuilder::new();
        builder.add_segment(seg);
        builder.add_buffer(self);
        builder.get_buffer()
    }

    pub fn truncated_front(&self, num_entries: u64) -> DeltaBuffer<T, D> {
        let mut builder = DeltaBufferBuilder::new();
        builder.add_buffer_from(num_entries, self);
        builder.get_buffer()
    }

    /// The default value will be used to create a vector with segments from an
    /// empty self.data.
    /// For "new_length >= length" this will just create a copy.
    pub fn truncated(&self, new_length: u64) -> DeltaBuffer<T, D> {
        let mut builder = DeltaBufferBuilder::new();
        builder.add_buffer_until(new_length, self);
        builder.get_buffer()
    }

    /// Returns the buffer with an additional segment.
    pub fn extended_with(&self, seg: DeltaSegment<T, D>) -> DeltaBuffer<T, D> {
        let mut builder = DeltaBufferBuilder::new();
        builder.add_buffer(self);
        builder.add_segment(seg);
        builder.get_buffer()
    }

    /// Set the length exactly. Truncate segments or extend the missing part
    /// with new segment
    /// with given values.
    pub fn fixed_length(&self, seg: DeltaSegment<T, D>) -> DeltaBuffer<T, D> {
        if self.len() > seg.len() {
            self.truncated(seg.len())
        } else if self.len() < seg.len() {
            self.extended_with(seg.split_to_end(self.len()))
        } else {
            self.clone()
        }
    }

    #[cfg(test)]
    pub fn iter(&self) -> DeltaBufferIter<T, D>
    where
        T: Add<D, Output = T> + Copy,
        D: Mul<i64, Output = D> + Copy,
    {
        DeltaBufferIter { reader: DeltaBufferReader::new(self, TimePoint::from(0)) }
    }

    pub fn iter_segments(&self) -> Iter<DeltaSegment<T, D>> {
        self.data.iter()
    }

    /// Both Buffers have to have same total length.
    pub fn combine_fast<F>(&self, other: &DeltaBuffer<T, D>, mut f: F) -> DeltaBuffer<T, D>
    where
        F: FnMut(T, D, T, D) -> (T, D),
    {
        let mut builder = DeltaBufferBuilder::new();
        for (seg1, seg2) in CombinedSegmentIterator::new(
            self.iter_segments().cloned(),
            other.iter_segments().cloned(),
        )
        {
            let seglen = seg1.len();
            assert!(seg2.len() == seglen);

            let (new_t, new_d) = f(
                seg1.first_value(),
                seg1.delta(),
                seg2.first_value(),
                seg2.delta(),
            );

            // this will handle optimizations where we join two segments
            builder.add_segment(DeltaSegment::new(new_t, new_d, seglen));
        }

        builder.get_buffer()
    }

    pub fn combined_add(&self, other: &DeltaBuffer<T, D>) -> DeltaBuffer<T, D>
    where
        T: Add<T, Output = T>,
        D: Add<D, Output = D>,
    {
        self.combine_fast(other, |t1: T, d1: D, t2: T, d2: D| (t1 + t2, d1 + d2))
    }

    #[allow(dead_code)]
    pub fn write_to_file(&self, path: String) -> ::std::io::Result<()>
    where
        D: Display,
        T: Display,
    {
        use std::fs::File;
        use std::io::prelude::*;
        let mut f = File::create(path)?;
        for segments in &self.data {
            f.write_all(
                format!(
                    "{}, {}, {}\n",
                    segments.first_value(),
                    segments.delta(),
                    segments.len()
                )
                .as_bytes(),
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use internal::Rating;
    use rand;
    use rand::Rng;
    use std::cmp::min;
    use std::convert::From;

    fn random_buffer() -> RatingBuffer {
        let mut rng = rand::thread_rng();

        let mut builder = DeltaBufferBuilder::new();
        let len = rng.next_u32() % 10;
        let mut fulllen = 0;
        for _ in 0..len {
            let seglen = rng.next_u32() as u64 % 100 + 1;
            builder.add_segment(DeltaSegment::new(
                Rating::from(rng.next_u32() as i64 % 2000 - 1000),
                Rating::from(rng.next_u32() as i64 % 2000 - 1000),
                seglen,
            ));
            fulllen += seglen;
        }
        let buffer = builder.get_buffer();

        assert_eq!(buffer.len(), fulllen);
        assert_eq!(buffer.iter().count() as u64, fulllen);

        buffer
    }

    #[test]
    fn combine_add() {
        for _ in 0..100 {
            let buffer1 = random_buffer();
            let buffer2 = random_buffer();
            let buffer3 = buffer1.combined_add(&buffer2);
            assert!(buffer3.len() == min(buffer1.len(), buffer2.len()));
            for (v1, (v2, v3)) in buffer1.iter().zip(buffer2.iter().zip(buffer3.iter())) {
                assert!(v1 + v2 == v3);
            }
        }
    }

    fn test_truncation(buffer: &RatingBuffer, len: i64) {
        if len < 0 {
            return;
        }
        let len = len as u64;
        let new_buffer = buffer.truncated(len);
        let new_len = min(len, buffer.len());

        assert_eq!(new_buffer.len(), new_len);
        assert_eq!(new_buffer.iter().count() as u64, new_len);

        for (v1, v2) in buffer.iter().zip(new_buffer.iter()) {
            assert_eq!(v1, v2);
        }
    }

    #[test]
    fn truncate() {
        for _ in 0..100 {
            let buffer = random_buffer();
            test_truncation(&buffer, 0);
            test_truncation(&buffer, buffer.len() as i64 / 2);
            test_truncation(&buffer, buffer.len() as i64 - 1);
            test_truncation(&buffer, buffer.len() as i64);
            test_truncation(&buffer, buffer.len() as i64 + 1);
            test_truncation(&buffer, buffer.len() as i64 * 2);
        }
    }
}

// /////////////////////////////////////////////////////////////////////////////////////////////////
// DELTA COMPRESSION BUILDER

pub struct DeltaBufferBuilder<T, D> {
    inner: DeltaBuffer<T, D>,
}

impl<T, D> DeltaBufferBuilder<T, D>
where
    T: Add<D, Output = T> + Sub<T, Output = D> + Eq + Copy,
    D: Mul<i64, Output = D> + Copy + Eq,
{
    pub fn new() -> DeltaBufferBuilder<T, D> {
        DeltaBufferBuilder { inner: DeltaBuffer::new() }
    }

    pub fn add_segment(&mut self, seg: DeltaSegment<T, D>) {
        self.add_segment_intern(seg.start, seg.delta, seg.len);
    }

    fn add_segment_intern(&mut self, t: T, d: D, len: u64) {
        if len == 0 {
            return;
        }

        self.inner.length += len;

        if let Some(last_segment) = self.inner.data.last_mut() {
            let anticipated_timepoint = last_segment.start + last_segment.delta * last_segment.len as i64;
            if (last_segment.delta == d || len == 1) && t == anticipated_timepoint {
                last_segment.len += len;
                return; // extend only
            } else if last_segment.len == 1 {
                let new_delta = t - last_segment.start;
                if len == 1 || new_delta == d {
                    last_segment.delta = new_delta;
                    last_segment.len += len;
                    return; // extend only
                }
            }
        }

        self.inner.data.push(DeltaSegment {
            delta: d,
            start: t,
            len: len,
        });
    }

    pub fn add_buffer_from(&mut self, index: u64, buffer: &DeltaBuffer<T, D>) {
        let mut index = index as i64;
        for &segment in &buffer.data {
            if index <= 0 {
                self.add_segment(segment)
            } else if index > 0 && index < segment.len as i64 {
                self.add_segment(segment.split_to_end(index as u64));
            } else {
                // index >= segment.len
            }

            // change start delta so it is relative to next segment
            index -= segment.len as i64;
        }
    }


    pub fn add_buffer(&mut self, buffer: &DeltaBuffer<T, D>) {
        for &segment in &buffer.data {
            self.add_segment(segment)
        }
    }

    pub fn add_buffer_until(&mut self, index: u64, buffer: &DeltaBuffer<T, D>) {
        let mut index: i64 = index as i64;
        for &segment in &buffer.data {
            if index <= 0 {
                return;
            } else if index > 0 && index < segment.len as i64 {
                self.add_segment(segment.split_from_begin_to(index as u64));
            } else {
                // index >= segment.len
                self.add_segment(segment);
            }

            // change start delta so it is relative to next segment
            index -= segment.len as i64;
        }
    }

    pub fn get_buffer(self) -> DeltaBuffer<T, D> {
        self.inner
    }
}

// /////////////////////////////////////////////////////////////////////////////////////////////////
// DELTA COMPRESSION READER

pub struct DeltaBufferReader<'a, T: 'a, D: 'a> {
    iter: Peekable<Iter<'a, DeltaSegment<T, D>>>,
    last_query: TimePoint,
    query_rest: u64,
}

impl<'a, T, D> DeltaBufferReader<'a, T, D>
where
    T: Add<D, Output = T> + Copy,
    D: Mul<i64, Output = D> + Copy,
{
    pub fn new(buffer: &DeltaBuffer<T, D>, first_timepoint: TimePoint) -> DeltaBufferReader<T, D> {
        let iter = buffer.data.iter();
        DeltaBufferReader {
            iter: iter.peekable(),
            last_query: first_timepoint,
            query_rest: 0,
        }
    }

    pub fn read_by_timepoint(&mut self, t: TimePoint) -> T {
        let delta = t - self.last_query;
        self.read_by_delta(delta)
    }

    pub fn read_by_delta(&mut self, d: TimeDelta) -> T {
        assert!(d >= TimeDelta::zero());
        self.read_by_delta_safe(d).unwrap_or_else(|| {
            panic!(
                "DeltaBuffer::read_by_delta(): out of bounds access (delta is {})",
                d
            )
        })
    }

    fn read_by_delta_safe(&mut self, d: TimeDelta) -> Option<T> {
        self.last_query += d;
        self.query_rest += u64::from(d);
        loop {
            {
                let segment = match self.iter.peek() {
                    Some(segment) => segment,
                    None => return None,
                };

                if self.query_rest < segment.len {
                    return Some(segment.value_at_index(self.query_rest as i64));
                }

                self.query_rest -= segment.len;
            }
            self.iter.next();
        }
    }

    #[cfg(test)]
    pub fn read_current_safe(&mut self) -> Option<T> {
        let query_rest = self.query_rest;
        self.iter.peek().map(|segment| {
            segment.start + segment.delta * query_rest as i64
        })
    }
}

// /////////////////////////////////////////////////////////////////////////////////////////////////
// BUFFER ITERATOR

#[cfg(test)]
pub struct DeltaBufferIter<'a, T: 'a, D: 'a> {
    reader: DeltaBufferReader<'a, T, D>,
}

#[cfg(test)]
impl<'a, T, D> Iterator for DeltaBufferIter<'a, T, D>
where
    T: Add<D, Output = T> + Copy,
    D: Mul<i64, Output = D> + Copy,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        let result = self.reader.read_current_safe();
        self.reader.read_by_delta_safe(TimeDelta::one());
        result
    }
}

// /////////////////////////////////////////////////////////////////////////////////////////////////
// COMBINED SEGMENT ITERATOR

pub trait Segment {
    type Item;
    fn len(self) -> u64;
    fn split_from(self, start_index: u64, len: u64) -> Self::Item;
}

impl<T, D> Segment for DeltaSegment<T, D>
where
    T: Add<D, Output = T> + Copy,
    D: Mul<i64, Output = D> + Copy,
{
    type Item = DeltaSegment<T, D>;

    #[inline]
    fn len(self) -> u64 {
        DeltaSegment::len(&self)
    }

    #[inline]
    fn split_from(self, start_index: u64, len: u64) -> DeltaSegment<T, D> {
        DeltaSegment::new(self.value_at_index(start_index as i64), self.delta(), len)
    }
}

/// Iterator that steps through two buffers simultanously. Each step goes until
/// the next beginning/end
/// of a segment in either buffer.
pub struct CombinedSegmentIterator<I1, I2, K1, K2>
where
    I1: Iterator<Item = K1>,
    I2: Iterator<Item = K2>,
    K1: Segment,
    K2: Segment,
{
    pos1: u64,
    pos2: u64,
    segment_iter_1: Peekable<I1>,
    segment_iter_2: Peekable<I2>,
}

impl<I1, I2, K1, K2> CombinedSegmentIterator<I1, I2, K1, K2>
where
    I1: Iterator<Item = K1>,
    I2: Iterator<Item = K2>,
    K1: Segment + Copy,
    K2: Segment + Copy,
{
    pub fn new(i1: I1, i2: I2) -> CombinedSegmentIterator<I1, I2, K1, K2> {
        CombinedSegmentIterator {
            pos1: 0,
            pos2: 0,
            segment_iter_1: i1.peekable(),
            segment_iter_2: i2.peekable(),
        }
    }
}

impl<I1, I2, K1, K2> Iterator for CombinedSegmentIterator<I1, I2, K1, K2>
where
    I1: Iterator<Item = K1>,
    I2: Iterator<Item = K2>,
    K1: Segment + Copy,
    K2: Segment + Copy,
{
    type Item = (K1::Item, K2::Item);

    #[inline]
    fn next(&mut self) -> Option<(K1::Item, K2::Item)> {
        let (segment1, segment2) = {
            let segment1_opt = self.segment_iter_1.peek();
            let segment2_opt = self.segment_iter_2.peek();

            match (segment1_opt, segment2_opt) {
                (Some(a), Some(b)) => (*a, *b),
                _ => return None,
            }
        };

        let rest1 = <K1 as Segment>::len(segment1) - self.pos1;
        let rest2 = <K2 as Segment>::len(segment2) - self.pos2;
        let orig_pos1 = self.pos1;
        let orig_pos2 = self.pos2;

        let step = if rest1 < rest2 {
            self.segment_iter_1.next();
            self.pos1 = 0;
            self.pos2 += rest1;
            rest1
        } else if rest2 < rest1 {
            self.segment_iter_2.next();
            self.pos1 += rest2;
            self.pos2 = 0;
            rest2
        } else {
            // rest2 == rest1
            self.segment_iter_1.next();
            self.segment_iter_2.next();
            self.pos1 = 0;
            self.pos2 = 0;
            rest1
        };

        let t1 = <K1 as Segment>::split_from(segment1, orig_pos1, step);
        let t2 = <K2 as Segment>::split_from(segment2, orig_pos2, step);

        Some((t1, t2))
    }
}

#[derive(Clone, Copy)]
pub enum OptionSegment<K: Segment> {
    NoneSeg(u64),
    SomeSeg(K),
}

impl<K> Segment for OptionSegment<K>
where
    K: Segment,
{
    type Item = Option<K::Item>;

    #[inline]
    fn len(self) -> u64 {
        match self {
            OptionSegment::NoneSeg(len) => len,
            OptionSegment::SomeSeg(seg) => seg.len(),
        }
    }

    #[inline]
    fn split_from(self, start_index: u64, len: u64) -> Option<K::Item> {
        match self {
            OptionSegment::NoneSeg(_) => None,
            OptionSegment::SomeSeg(seg) => Some(seg.split_from(start_index, len)),
        }
    }
}
