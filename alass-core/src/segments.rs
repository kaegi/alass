use crate::rating_type::{Rating, RatingDelta, RatingDeltaDelta, RatingExt};
use crate::time_types::{TimeDelta, TimePoint, TimeSpan};
use std::cmp::{max, min};
use std::iter::once;
use std::ops::Add;

#[derive(Clone, Copy, Debug)]
pub struct PositionInfo {
    pub position: TimePoint,
    pub drag: bool, // if true, position has a "delta" of 1; if false it has a delta of 0
}

impl PositionInfo {
    #[inline]
    fn constant(position: TimePoint) -> PositionInfo {
        PositionInfo { position, drag: false }
    }

    #[inline]
    pub fn start_position(&self) -> TimePoint {
        self.position
    }

    #[inline]
    pub fn end_position(&self, span_length: TimeDelta) -> TimePoint {
        if self.drag {
            self.position + span_length - TimeDelta::one()
        } else {
            self.position
        }
    }

    #[inline]
    pub fn exclusive_end_position(&self, span_length: TimeDelta) -> TimePoint {
        if self.drag {
            self.position + span_length
        } else {
            self.position
        }
    }

    #[inline]
    fn advanced_position(self, time_delta: TimeDelta) -> TimePoint {
        if self.drag {
            self.position + time_delta
        } else {
            self.position
        }
    }

    #[inline]
    fn advanced(self, time_delta: TimeDelta) -> PositionInfo {
        if self.drag {
            PositionInfo {
                position: self.position + time_delta,
                drag: true,
            }
        } else {
            PositionInfo {
                position: self.position,
                drag: false,
            }
        }
    }

    #[inline]
    fn advance(&mut self, time_delta: TimeDelta) {
        if self.drag {
            self.position += time_delta;
        }
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub struct RatingInfo {
    pub rating: Rating,
    pub delta: RatingDelta,
}

impl RatingInfo {
    #[inline]
    fn constant(rating: Rating) -> RatingInfo {
        RatingInfo {
            rating,
            delta: RatingDelta::zero(),
        }
    }

    #[inline]
    fn advanced(self, len: TimeDelta) -> RatingInfo {
        RatingInfo {
            rating: Rating::add_mul(self.rating, self.delta, len),
            delta: self.delta,
        }
    }

    #[inline]
    fn get_at(self, len: TimeDelta) -> Rating {
        Rating::add_mul(self.rating, self.delta, len)
    }

    #[inline]
    fn advance(&mut self, len: TimeDelta) {
        self.rating = Rating::add_mul(self.rating, self.delta, len);
    }

    #[inline]
    pub fn start_rating(self) -> Rating {
        self.rating
    }

    #[inline]
    pub fn end_rating(self, len: TimeDelta) -> Rating {
        Rating::add_mul(self.rating, self.delta, len - TimeDelta::one())
    }

    #[inline]
    pub fn exclusive_end_rating(self, len: TimeDelta) -> Rating {
        Rating::add_mul(self.rating, self.delta, len)
    }
}

#[derive(Clone, Debug)]
pub struct DualInfo {
    position_info: PositionInfo,
    rating_info: RatingInfo,
}

impl DualInfo {
    #[inline]
    fn advanced(self, len: TimeDelta) -> DualInfo {
        DualInfo {
            rating_info: self.rating_info.advanced(len),
            position_info: self.position_info.advanced(len),
        }
    }
}

impl Add<RatingInfo> for RatingInfo {
    type Output = RatingInfo;

    #[inline]
    fn add(self, rhs: RatingInfo) -> RatingInfo {
        RatingInfo {
            rating: self.rating + rhs.rating,
            delta: self.delta + rhs.delta,
        }
    }
}

#[derive(Debug)]
pub struct RatingBuffer {
    pub start: TimePoint,
    pub buffer: Vec<RatingSegment>,
}

impl RatingBuffer {
    #[inline]
    pub fn into_iter(self) -> RatingIterator<impl RI> {
        RatingIterator::<_> {
            start: self.start,
            iter: self.buffer.into_iter(),
        }
    }

    #[inline]
    pub fn iter(&self) -> RatingIterator<std::iter::Cloned<impl Iterator<Item = &RatingSegment>>> {
        RatingIterator::<_> {
            start: self.start,
            iter: self.buffer.iter().cloned(),
        }
    }

    #[inline]
    pub fn first_end_point(&self) -> Option<TimePoint> {
        self.buffer.first().map(|rating_segment| rating_segment.end_point)
    }

    #[inline]
    pub fn end(&self) -> Option<TimePoint> {
        self.buffer.last().map(|rating_segment| rating_segment.end_point)
    }

    #[inline]
    pub fn start(&self) -> TimePoint {
        self.start
    }

    #[inline]
    pub fn maximum(&self) -> Rating {
        self.buffer
            .iter()
            .fold(
                (Rating::zero(), self.start),
                #[inline]
                |(current_max, segment_start): (Rating, TimePoint), segment: &RatingSegment| {
                    assert!(segment_start < segment.end_point);
                    let start_rating = segment.start_rating();
                    let end_rating = segment.end_rating(segment.end_point - segment_start);

                    let new_max = max(max(current_max, start_rating), end_rating);

                    (new_max, segment.end_point)
                },
            )
            .0
    }
}

pub struct DifferentialRatingBufferBuilder {
    start: TimePoint,
    end: TimePoint,
    buffer: Vec<Segment<RatingDeltaDelta>>,
}

impl DifferentialRatingBufferBuilder {
    #[inline]
    pub fn new(start: TimePoint, end: TimePoint) -> DifferentialRatingBufferBuilder {
        assert!(start < end);

        DifferentialRatingBufferBuilder {
            start: start,
            end: end,
            buffer: Vec::new(),
        }
    }

    #[inline]
    pub fn add_segment(&mut self, segment_end: TimePoint, segment_end_delta_delta: RatingDeltaDelta) {
        if let Some(last_segment) = self.buffer.last_mut() {
            assert!(last_segment.end_point <= segment_end);

            if last_segment.end_point == segment_end {
                last_segment.data += segment_end_delta_delta;
                return;
            }
        } else {
            assert!(self.start < segment_end);
        }

        self.buffer.push(Segment {
            end_point: segment_end,
            data: segment_end_delta_delta,
        });
    }

    #[inline]
    pub fn extend_to_end(&mut self) {
        self.add_segment(self.end, RatingDeltaDelta::zero())
    }

    #[inline]
    pub fn build(self) -> DifferentialRatingBuffer {
        DifferentialRatingBuffer {
            start: self.start,
            buffer: self.buffer,
        }
    }
}

pub struct DifferentialRatingBuffer {
    start: TimePoint,
    buffer: Vec<Segment<RatingDeltaDelta>>,
}

impl DifferentialRatingBuffer {
    #[inline]
    pub fn into_rating_iter(self) -> RatingIterator<impl Iterator<Item = RatingSegment>> {
        struct ScanState {
            rating: Rating,
            delta: RatingDelta,
            last_segment_end: TimePoint,
        }

        let start_state = ScanState {
            rating: Rating::zero(),
            delta: RatingDelta::zero(),
            last_segment_end: self.start,
        };

        let iter = self.buffer.into_iter().scan(
            start_state,
            #[inline]
            |state: &mut ScanState, segment: Segment<RatingDeltaDelta>| {
                let result = Segment {
                    end_point: segment.end_point,
                    data: RatingInfo {
                        rating: state.rating,
                        delta: state.delta,
                    },
                };

                state.rating = Rating::add_mul(state.rating, state.delta, segment.end_point - state.last_segment_end);
                state.delta += segment.data;
                state.last_segment_end = segment.end_point;

                Some(result)
            },
        );

        RatingIterator::<_> {
            start: self.start,
            iter: iter,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Segment<D> {
    pub end_point: TimePoint,
    pub data: D,
}

impl<D> Segment<D> {
    #[inline]
    fn with_start_point(self, start_point: TimePoint) -> FullSegment<D> {
        assert!(start_point < self.end_point);

        FullSegment {
            span: TimeSpan::new(start_point, self.end_point),
            data: self.data,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FullSegment<D> {
    pub span: TimeSpan,
    pub data: D,
}
impl<D> FullSegment<D> {
    #[inline]
    fn discard_start_time(self) -> Segment<D> {
        Segment {
            end_point: self.span.end,
            data: self.data,
        }
    }
}

#[derive(Debug)]
pub struct FullSegmentIterator<D, I>
where
    I: Iterator<Item = FullSegment<D>>,
{
    start: TimePoint,
    iter: I,
}

impl<D, I> FullSegmentIterator<D, I>
where
    I: Iterator<Item = FullSegment<D>>,
{
    pub fn into_iter(self) -> I {
        self.iter
    }
}

#[derive(Debug)]
pub struct SegmentIterator<D, I>
where
    I: Iterator<Item = Segment<D>>,
{
    start: TimePoint,
    iter: I,
}

impl<D, I> SegmentIterator<D, I>
where
    I: Iterator<Item = Segment<D>>,
{
    #[inline]
    pub fn annotate_with_segment_start_times(self) -> FullSegmentIterator<D, impl Iterator<Item = FullSegment<D>>> {
        FullSegmentIterator::<D, _> {
            start: self.start,
            iter: self.iter.scan(
                self.start,
                #[inline]
                |last_segment_end: &mut TimePoint, segment: Segment<D>| {
                    /*println!(
                        "ANNOTATE segment start {}   segment end {}",
                        *last_segment_end, segment.end_point
                    );*/
                    if *last_segment_end >= segment.end_point {
                        // DEBUG TODO: remove
                        println!(
                            "ANNOTATE segment start {}   segment end {}",
                            *last_segment_end, segment.end_point
                        );
                    }

                    assert!(*last_segment_end < segment.end_point);

                    let span = TimeSpan::new(*last_segment_end, segment.end_point);

                    *last_segment_end = segment.end_point;

                    Some(FullSegment {
                        span: span,
                        data: segment.data,
                    })
                },
            ),
        }
    }

    #[inline]
    pub fn into_iter(self) -> I {
        self.iter
    }

    #[inline]
    pub fn shift(self, t: TimeDelta) -> SegmentIterator<D, impl SI<D>> {
        SegmentIterator::<D, _> {
            start: self.start + t,
            iter: self.iter.map(
                #[inline]
                move |mut segment: Segment<D>| {
                    segment.end_point += t;
                    segment
                },
            ),
        }
    }

    #[inline]
    pub fn shift_simple(self, t: TimeDelta) -> SegmentIterator<D, impl SI<D>> {
        SegmentIterator::<D, _> {
            start: self.start,
            iter: self.iter.map(
                #[inline]
                move |mut segment: Segment<D>| {
                    segment.end_point += t;
                    segment
                },
            ),
        }
    }

    #[inline]
    pub fn append(self, end_point: TimePoint, data: D) -> SegmentIterator<D, impl SI<D>> {
        SegmentIterator::<D, _> {
            start: self.start,
            iter: self.iter.chain(once(Segment::<D> {
                end_point: end_point,
                data: data,
            })),
        }
    }
}

impl<D, I: SFI<D>> FullSegmentIterator<D, I> {
    #[inline]
    pub fn discard_start_times(self) -> SegmentIterator<D, impl SI<D>> {
        SegmentIterator::<D, _> {
            start: self.start,
            iter: self.iter.map(|segment: FullSegment<D>| Segment::<D> {
                end_point: segment.span.end,
                data: segment.data,
            }),
        }
    }
}

pub type PositionSegment = Segment<PositionInfo>;
impl PositionSegment {
    #[inline]
    pub fn start_position(&self) -> TimePoint {
        self.data.start_position()
    }

    #[inline]
    pub fn end_position(&self, len: TimeDelta) -> TimePoint {
        self.data.end_position(len)
    }
}

pub type PositionFullSegment = FullSegment<PositionInfo>;
impl PositionFullSegment {
    #[inline]
    pub fn exclusive_end_position(&self) -> TimePoint {
        self.data.exclusive_end_position(self.span.len())
    }
}

pub type RatingSegment = Segment<RatingInfo>;
impl RatingSegment {
    #[inline]
    pub fn advance(&mut self, delta: TimeDelta) {
        self.data.advance(delta);
    }

    #[inline]
    pub fn start_rating(&self) -> Rating {
        self.data.start_rating()
    }

    #[inline]
    pub fn end_rating(&self, len: TimeDelta) -> Rating {
        self.data.end_rating(len)
    }
}

pub type DualSegment = Segment<DualInfo>;
impl DualSegment {
    #[inline]
    fn advance(&mut self, delta: TimeDelta) {
        self.data.rating_info.advance(delta);
        self.data.position_info.advance(delta);
    }

    #[inline]
    fn as_rating_segment(&self) -> RatingSegment {
        RatingSegment {
            end_point: self.end_point,
            data: self.data.rating_info,
        }
    }

    #[inline]
    fn as_position_segment(&self) -> PositionSegment {
        PositionSegment {
            end_point: self.end_point,
            data: self.data.position_info,
        }
    }

    #[inline]
    pub fn start_rating(&self) -> Rating {
        self.data.rating_info.rating
    }

    #[inline]
    pub fn start_position(&self) -> TimePoint {
        self.data.position_info.position
    }
}

pub type RatingFullSegment = FullSegment<RatingInfo>;
pub type DualFullSegment = FullSegment<DualInfo>;
impl DualFullSegment {
    #[inline]
    pub fn start_rating(&self) -> Rating {
        self.data.rating_info.rating
    }

    #[inline]
    pub fn start_position(&self) -> TimePoint {
        self.data.position_info.position
    }

    #[inline]
    pub fn end_rating(&self) -> Rating {
        Rating::add_mul(
            self.data.rating_info.rating,
            self.data.rating_info.delta,
            self.span.len() - TimeDelta::one(),
        )
    }

    #[inline]
    pub fn exclusive_end_position(&self) -> TimePoint {
        self.data.position_info.exclusive_end_position(self.span.len())
    }

    #[inline]
    pub fn exclusive_end_rating(&self) -> Rating {
        self.data.rating_info.exclusive_end_rating(self.span.len())
    }
}

pub trait SI<D>: Iterator<Item = Segment<D>> {}
impl<T, D> SI<D> for T where T: Iterator<Item = Segment<D>> {}

pub trait RI: Iterator<Item = RatingSegment> {}
impl<T> RI for T where T: Iterator<Item = RatingSegment> {}

pub trait PI: Iterator<Item = PositionSegment> {}
impl<T> PI for T where T: Iterator<Item = PositionSegment> {}

pub trait DI: Iterator<Item = DualSegment> {}
impl<T> DI for T where T: Iterator<Item = DualSegment> {}

pub trait SFI<D>: Iterator<Item = FullSegment<D>> {}
impl<T, D> SFI<D> for T where T: Iterator<Item = FullSegment<D>> {}

pub trait RFI: Iterator<Item = RatingFullSegment> {}
impl<T> RFI for T where T: Iterator<Item = RatingFullSegment> {}

pub trait PFI: Iterator<Item = PositionFullSegment> {}
impl<T> PFI for T where T: Iterator<Item = PositionFullSegment> {}

pub trait DFI: Iterator<Item = DualFullSegment> {}
impl<T> DFI for T where T: Iterator<Item = DualFullSegment> {}

pub type DualIterator<I> = SegmentIterator<DualInfo, I>;
impl<I: DI> DualIterator<I> {
    #[inline]
    pub fn save(self) -> DualBuffer {
        DualBuffer {
            start: self.start,
            buffer: self.iter.collect(),
        }
    }

    pub fn save_separate(self) -> SeparateDualBuffer {
        let (rating_buffer, position_buffer): (Vec<RatingSegment>, Vec<PositionSegment>) = into_push_iter(
            self.iter,
            dual_push_iter(
                only_ratings_push_iter(simplifiy_ratings_push_iter(
                    self.start,
                    discard_start_times_push_iter(collect_to_vec_push_iter()),
                )),
                only_positions_push_iter(simplifiy_positions_push_iter(
                    self.start,
                    discard_start_times_push_iter(collect_to_vec_push_iter()),
                )),
            ),
            /*dual_push_iter(
                only_ratings_push_iter(
                    collect_to_vec_push_iter(),
                ),
                only_positions_push_iter(
                    collect_to_vec_push_iter(),
                ),
            ),*/
        );

        SeparateDualBuffer {
            rating_buffer: RatingBuffer {
                start: self.start,
                buffer: rating_buffer,
            },
            position_buffer: PositionBuffer {
                start: self.start,
                buffer: position_buffer,
            },
        }
    }

    pub fn only_positions(self) -> PositionIterator<impl PI> {
        PositionIterator::<_> {
            start: self.start,
            iter: self.iter.map(|dual_segment| dual_segment.as_position_segment()),
        }
    }

    pub fn only_ratings(self) -> RatingIterator<impl RI> {
        RatingIterator::<_> {
            start: self.start,
            iter: self.iter.map(|dual_segment| dual_segment.as_rating_segment()),
        }
    }

    pub fn simplify(mut self) -> DualFullSegmentIterator<impl DFI> {
        DualFullSegmentIterator {
            start: self.start,
            iter: DualSimplifyIterator {
                current_segment: self.iter.next().map(|seg| seg.with_start_point(self.start)),
                iter: self.iter,
            },
        }
    }
}

pub type PositionIterator<I> = SegmentIterator<PositionInfo, I>;
impl<I: PI> PositionIterator<I> {
    #[inline]
    pub fn save(self) -> PositionBuffer {
        PositionBuffer {
            start: self.start,
            buffer: self.iter.collect(),
        }
    }
}

pub type RatingIterator<I> = SegmentIterator<RatingInfo, I>;
impl<I: RI> RatingIterator<I> {
    #[inline]
    pub fn save(self) -> RatingBuffer {
        RatingBuffer {
            start: self.start,
            buffer: self.iter.collect(),
        }
    }

    #[inline]
    pub fn save_simplified(self) -> RatingBuffer {
        RatingBuffer {
            start: self.start,
            buffer: into_push_iter(
                self.iter,
                simplifiy_ratings_push_iter(self.start, discard_start_times_push_iter(collect_to_vec_push_iter())),
            ),
        }
    }

    #[inline]
    pub fn save_aggressively_simplified(self, epsilon: RatingDelta) -> RatingBuffer {
        RatingBuffer {
            start: self.start,
            buffer: into_push_iter(
                self.iter,
                aggressive_simplifiy_ratings_push_iter(
                    self.start,
                    epsilon,
                    discard_start_times_push_iter(collect_to_vec_push_iter()),
                ),
            ),
        }
    }

    #[inline]
    pub fn extend_to(self, end_point: TimePoint) -> RatingIterator<impl RI> {
        self.append(
            end_point,
            RatingInfo {
                rating: Rating::zero(),
                delta: RatingDelta::zero(),
            },
        )
    }

    #[inline]
    pub fn add_rating(self, rating_delta: RatingDelta) -> RatingIterator<impl RI> {
        RatingIterator::<_> {
            start: self.start,
            iter: self.iter.map(move |rating_segment| RatingSegment {
                end_point: rating_segment.end_point,
                data: RatingInfo {
                    rating: rating_segment.data.rating + rating_delta,
                    delta: rating_segment.data.delta,
                },
            }),
        }
    }

    #[inline]
    pub fn clamp_end(self, clamp: TimePoint) -> RatingIterator<impl RI> {
        //println!("CLAMP {}", clamp);
        RatingIterator::<_> {
            start: self.start,
            iter: self.iter.map(move |rating_segment| {
                /*if rating_segment.end_point >= clamp {
                    println!("CLAMPED {} to {} (prev {})", rating_segment.end_point, clamp, last_start);
                }*/
                RatingSegment {
                    end_point: min(rating_segment.end_point, clamp),
                    data: rating_segment.data,
                }
            }),
        }
    }
}

struct LeftToRightMaximumIterator<I>
where
    I: DFI,
{
    input_iter: I,

    current_best_rating: Rating,

    /// The start timepoint of the position, which has the maximum rating
    current_best_timepoint: TimePoint,

    stored_segment: Option<DualFullSegment>,
}

impl<I: DFI> LeftToRightMaximumIterator<I> {
    #[inline]
    fn new(i: I, start: TimePoint) -> LeftToRightMaximumIterator<I> {
        LeftToRightMaximumIterator::<I> {
            input_iter: i,
            current_best_rating: Rating::zero(),
            current_best_timepoint: start,
            stored_segment: None,
        }
    }

    #[inline]
    fn constant_dual_info(&self) -> DualInfo {
        DualInfo {
            rating_info: RatingInfo::constant(self.current_best_rating),
            position_info: PositionInfo::constant(self.current_best_timepoint),
        }
    }

    #[inline]
    fn constant_segment(&self, span: TimeSpan) -> DualFullSegment {
        DualFullSegment {
            span: span,
            data: self.constant_dual_info(),
        }
    }
}

impl<I: DFI> Iterator for LeftToRightMaximumIterator<I> {
    type Item = DualFullSegment;

    #[inline] // XXX: is this really faster?
    fn next(&mut self) -> Option<DualFullSegment> {
        // TODO: rewrite when changing to 2018 edition
        if let Some(stored_segment) = self.stored_segment.take() {
            return Some(stored_segment);
        }

        // TODO: unify segments when possible
        let segment: DualFullSegment;

        match self.input_iter.next() {
            None => return None,
            Some(_segment) => segment = _segment,
        }

        let segment_start_rating = segment.start_rating();
        let segment_end_rating = segment.end_rating();

        let start_position = segment.data.position_info.position;
        let end_position = segment.data.position_info.end_position(segment.span.len());

        if segment_start_rating <= self.current_best_rating && segment_end_rating <= self.current_best_rating {
            return Some(self.constant_segment(segment.span));
        } else if segment_start_rating >= self.current_best_rating {
            if segment_start_rating >= segment_end_rating {
                self.current_best_rating = segment_start_rating;
                self.current_best_timepoint = start_position;

                return Some(self.constant_segment(segment.span));
            } else {
                self.current_best_rating = segment_end_rating;
                self.current_best_timepoint = end_position;

                return Some(segment);
            }
        } else {
            /* implicit:
                segment_start_rating < self.current_best_rating &&
                (
                    segment_start_rating > self.current_best_rating ||
                    segment_end_rating > self.current_best_rating
                )

                which is equivalent to

                segment_start_rating < self.current_best_rating &&
                    segment_end_rating > self.current_best_rating
            */

            assert!(segment_start_rating < self.current_best_rating);
            assert!(segment_end_rating > self.current_best_rating);

            let switch_i64 = (self.current_best_rating - segment_start_rating) / segment.data.rating_info.delta + 1;
            assert!(0 < switch_i64);
            assert!(switch_i64 < segment.span.len().as_i64());

            let switch_timedelta = TimeDelta::from_i64(switch_i64);

            let segment1 = DualFullSegment {
                span: TimeSpan::new(segment.span.start, segment.span.start + switch_timedelta),
                data: self.constant_dual_info(),
            };

            self.current_best_rating = segment_end_rating;
            self.current_best_timepoint = end_position;

            let segment2 = DualFullSegment {
                span: TimeSpan::new(segment.span.start + switch_timedelta, segment.span.end),
                data: segment.data.advanced(switch_timedelta),
            };

            self.stored_segment = Some(segment2);
            return Some(segment1);
        }
    }
}

#[derive(Debug)]
pub struct PositionBuffer {
    start: TimePoint,
    buffer: Vec<PositionSegment>,
}

impl PositionBuffer {
    #[inline]
    pub fn into_iter(self) -> PositionIterator<impl PI> {
        PositionIterator::<_> {
            start: self.start,
            iter: self.buffer.into_iter(),
        }
    }

    #[inline]
    pub fn iter(&self) -> PositionIterator<std::iter::Cloned<impl Iterator<Item = &PositionSegment>>> {
        PositionIterator::<_> {
            start: self.start,
            iter: self.buffer.iter().cloned(),
        }
    }

    #[inline]
    pub fn end_point(&self) -> Option<TimePoint> {
        self.buffer.last().map(|rating_segment| rating_segment.end_point)
    }

    #[inline]
    pub fn end(&self) -> Option<TimePoint> {
        self.buffer.last().map(|rating_segment| rating_segment.end_point)
    }

    #[inline]
    pub fn start(&self) -> TimePoint {
        self.start
    }

    #[inline]
    pub fn end_position(&self) -> TimePoint {
        assert!(self.buffer.len() > 0);

        let segment_start: TimePoint;
        if self.buffer.len() == 1 {
            segment_start = self.start;
        } else {
            segment_start = self.buffer[self.buffer.len() - 2].end_point;
        }

        let last_segment = *self.buffer.last().unwrap();
        last_segment.data.end_position(last_segment.end_point - segment_start)
    }

    #[inline]
    pub fn get_at(&self, t: TimePoint) -> TimePoint {
        assert!(t >= self.start && t < self.end_point().unwrap());
        let mut segment_start = self.start;

        for segment in &self.buffer {
            if t >= segment_start && t < segment.end_point {
                return segment.data.advanced_position(t - segment_start);
            }
            segment_start = segment.end_point;
        }

        unreachable!()
    }

    #[inline]
    pub fn maximum(&self) -> TimePoint {
        let state: (TimePoint, TimePoint) = (self.buffer.first().unwrap().start_position(), self.start);

        self.buffer
            .iter()
            .fold(
                state,
                #[inline]
                |(current_max, segment_start): (TimePoint, TimePoint), segment: &PositionSegment| {
                    assert!(segment_start < segment.end_point);
                    let start_rating: TimePoint = segment.start_position();
                    let end_rating: TimePoint = segment.end_position(segment.end_point - segment_start);

                    let new_max = max(max(current_max, start_rating), end_rating);

                    (new_max, segment.end_point)
                },
            )
            .0
    }

    #[inline]
    pub fn minimum(&self) -> TimePoint {
        let state: (TimePoint, TimePoint) = (self.buffer.first().unwrap().start_position(), self.start);

        self.buffer
            .iter()
            .fold(
                state,
                #[inline]
                |(current_max, segment_start): (TimePoint, TimePoint), segment: &PositionSegment| {
                    assert!(segment_start < segment.end_point);
                    let start_rating: TimePoint = segment.start_position();
                    let end_rating: TimePoint = segment.end_position(segment.end_point - segment_start);

                    let new_max = min(min(current_max, start_rating), end_rating);

                    (new_max, segment.end_point)
                },
            )
            .0
    }
}

#[derive(Debug)]
pub struct DualBuffer {
    pub start: TimePoint,
    pub buffer: Vec<DualSegment>,
}

impl DualBuffer {
    #[inline]
    pub fn into_iter(self) -> DualIterator<impl DI> {
        DualIterator::<_> {
            start: self.start,
            iter: self.buffer.into_iter(),
        }
    }

    #[inline]
    pub fn iter(&self) -> DualIterator<std::iter::Cloned<impl Iterator<Item = &DualSegment>>> {
        DualIterator::<_> {
            start: self.start,
            iter: self.buffer.iter().cloned(),
        }
    }
}

#[derive(Debug)]
pub struct SeparateDualBuffer {
    pub rating_buffer: RatingBuffer,
    pub position_buffer: PositionBuffer,
}

pub type DualFullSegmentIterator<I> = FullSegmentIterator<DualInfo, I>; // TODO: rename to DualFullIterator
impl<I: DFI> DualFullSegmentIterator<I> {
    #[inline]
    pub fn left_to_right_maximum(self) -> DualFullSegmentIterator<impl DFI> {
        DualFullSegmentIterator::<_> {
            start: self.start,
            iter: LeftToRightMaximumIterator::<_>::new(self.iter, self.start),
        }
    }
}

struct CombinedMaximumDualIterator<I1, I2>
where
    I1: DI,
    I2: DI,
{
    stored_segment: Option<DualFullSegment>,
    segment_start: TimePoint,
    dual_seg1: DualSegment,
    dual_seg2: DualSegment,
    input_iter1: I1,
    input_iter2: I2,
    finished: bool,
}

impl<I1: DI, I2: DI> CombinedMaximumDualIterator<I1, I2> {
    #[inline]
    fn generate_maximum_segments(&mut self, len: TimeDelta, segment_end: TimePoint) -> DualFullSegment {
        let start_rating1 = self.dual_seg1.data.rating_info.rating;
        let start_rating2 = self.dual_seg2.data.rating_info.rating;
        let end_rating1 = self.dual_seg1.data.rating_info.end_rating(len);
        let end_rating2 = self.dual_seg2.data.rating_info.end_rating(len);

        let delta1 = self.dual_seg1.data.rating_info.delta;
        let delta2 = self.dual_seg2.data.rating_info.delta;

        if start_rating1 >= start_rating2 && end_rating1 >= end_rating2 {
            // first segment is better

            DualFullSegment {
                span: TimeSpan::new(self.segment_start, segment_end),
                data: self.dual_seg1.data.clone(),
            }
        } else if start_rating1 <= start_rating2 && end_rating1 <= end_rating2 {
            // second segment is better

            DualFullSegment {
                span: TimeSpan::new(self.segment_start, segment_end),
                data: self.dual_seg2.data.clone(),
            }
        } else {
            // segments switch somewhere in this segment

            // spoint is the first position where the second better segment is better
            let spoint = Self::get_switch_point(start_rating1, start_rating2, delta1, delta2);
            assert!(0 < spoint);
            assert!(spoint < len.as_i64());

            let spoint_delta = TimeDelta::from_i64(spoint);

            let segment1;
            let segment2;

            if start_rating1 > start_rating2 && end_rating1 < end_rating2 {
                // first segment starts above second segment

                segment1 = DualFullSegment {
                    span: TimeSpan::new(self.segment_start, self.segment_start + spoint_delta),
                    data: self.dual_seg1.data.clone(),
                };
                segment2 = DualFullSegment {
                    span: TimeSpan::new(self.segment_start + spoint_delta, segment_end),
                    data: self.dual_seg2.data.clone().advanced(spoint_delta),
                };
            } else {
                // second segment starts above first segment

                segment1 = DualFullSegment {
                    span: TimeSpan::new(self.segment_start, self.segment_start + spoint_delta),
                    data: self.dual_seg2.data.clone(),
                };
                segment2 = DualFullSegment {
                    span: TimeSpan::new(self.segment_start + spoint_delta, segment_end),
                    data: self.dual_seg1.data.clone().advanced(spoint_delta),
                }
            }

            self.stored_segment = Some(segment2);

            segment1
        }
    }

    #[inline]
    fn get_switch_point(start_rating1: Rating, start_rating2: Rating, delta1: RatingDelta, delta2: RatingDelta) -> i64 {
        // start_rating1 + delta1 * x = start_rating2 + delta2 * x
        // delta1 * x - delta2 * x = start_rating2 - start_rating1
        // (delta1 - delta2) * x = start_rating2 - start_rating1
        //
        // solving for x:
        //
        // x = (start_rating2 - start_rating1) / (delta1 - delta2)
        (start_rating2 - start_rating1) / (delta1 - delta2) + 1
    }
}

impl<I1: DI, I2: DI> Iterator for CombinedMaximumDualIterator<I1, I2> {
    type Item = DualFullSegment;

    #[inline]
    fn next(&mut self) -> Option<DualFullSegment> {
        if let Some(stored_segment) = self.stored_segment.take() {
            return Some(stored_segment);
        }

        if self.finished {
            return None;
        }

        assert!(self.segment_start < self.dual_seg1.end_point);
        assert!(self.segment_start < self.dual_seg2.end_point);

        let len;
        let result: DualFullSegment;

        if self.dual_seg1.end_point < self.dual_seg2.end_point {
            len = self.dual_seg1.end_point - self.segment_start;

            /*println!(
                "COMBINED1 {} {}",
                self.dual_seg2.end_point, self.segment_start
            );*/
            result = self.generate_maximum_segments(len, self.dual_seg1.end_point);

            self.segment_start = self.dual_seg1.end_point;

            self.dual_seg1 = self
                .input_iter1
                .next()
                .expect("CombinedMaximumDualIterator: First iterator ended before second");
            self.dual_seg2.advance(len);
        } else if self.dual_seg2.end_point < self.dual_seg1.end_point {
            len = self.dual_seg2.end_point - self.segment_start;

            /*println!(
                "COMBINED2 {} {}",
                self.segment_start, self.dual_seg2.end_point
            );*/
            result = self.generate_maximum_segments(len, self.dual_seg2.end_point);

            self.segment_start = self.dual_seg2.end_point;

            self.dual_seg1.advance(len);
            self.dual_seg2 = self
                .input_iter2
                .next()
                .expect("CombinedMaximumDualIterator: Second iterator ended before first");
        } else {
            match (self.input_iter1.next(), self.input_iter2.next()) {
                (Some(dual_seg1), Some(dual_seg2)) => {
                    len = self.dual_seg1.end_point - self.segment_start;

                    result = self.generate_maximum_segments(len, self.dual_seg1.end_point);

                    self.segment_start = self.dual_seg1.end_point;

                    self.dual_seg1 = dual_seg1;
                    self.dual_seg2 = dual_seg2;
                }
                (Some(_), None) => panic!("CombinedMaximumDualIterator: Second iterator ended before first"),
                (None, Some(_)) => panic!("CombinedMaximumDualIterator: First iterator ended before second"),
                (None, None) => {
                    len = self.dual_seg1.end_point - self.segment_start;

                    //println!("COMBINED END POINT {}", self.dual_seg1.end_point);

                    result = self.generate_maximum_segments(len, self.dual_seg1.end_point);

                    self.finished = true;
                }
            }
        }

        Some(result)
    }
}

#[inline]
pub fn combined_maximum_of_dual_iterators<I1: DI, I2: DI>(
    mut iter1: DualIterator<I1>,
    mut iter2: DualIterator<I2>,
) -> DualFullSegmentIterator<impl DFI> {
    assert!(iter1.start == iter2.start);
    let start = iter1.start;

    let dual_seg1 = iter1
        .iter
        .next()
        .expect("First iterator should have at least one element");
    let dual_seg2 = iter2
        .iter
        .next()
        .expect("Second iterator should have at least one element");

    DualFullSegmentIterator::<_> {
        start: start,
        iter: CombinedMaximumDualIterator::<_, _> {
            stored_segment: None,
            segment_start: start,
            dual_seg1: dual_seg1,
            dual_seg2: dual_seg2,
            input_iter1: iter1.iter,
            input_iter2: iter2.iter,
            finished: false,
        },
    }
}

pub type RatingFullIterator<I> = FullSegmentIterator<RatingInfo, I>;
impl<I: RFI> RatingFullIterator<I> {
    #[inline]
    pub fn annotate_with_position_info(
        self,
        position_generate: impl Fn(/* segment_start */ TimePoint) -> TimePoint,
    ) -> DualFullSegmentIterator<impl DFI> {
        DualFullSegmentIterator::<_> {
            start: self.start,
            iter: self.iter.map(
                #[inline]
                move |rating_full_segment| DualFullSegment {
                    span: rating_full_segment.span,
                    data: DualInfo {
                        rating_info: rating_full_segment.data,
                        position_info: PositionInfo {
                            position: position_generate(rating_full_segment.span.start),
                            drag: true,
                        },
                    },
                },
            ),
        }
    }
}

#[inline]
pub fn zero_rating_iterator(start: TimePoint, end: TimePoint) -> RatingIterator<impl RI> {
    RatingIterator::<_> {
        start: start,
        iter: once(Segment {
            end_point: end,
            data: RatingInfo {
                rating: Rating::zero(),
                delta: RatingDelta::zero(),
            },
        }),
    }
}

impl RatingFullSegment {
    #[inline]
    pub fn start_rating(self) -> Rating {
        self.data.rating
    }

    #[inline]
    pub fn end_rating(self) -> Rating {
        Rating::add_mul(self.data.rating, self.data.delta, self.span.len() - TimeDelta::one())
    }

    #[inline]
    pub fn exclusive_end_rating(&self) -> Rating {
        Rating::add_mul(self.data.rating, self.data.delta, self.span.len())
    }
}

struct RatingAdderIterator<I1, I2>
where
    I1: RI,
    I2: RI,
{
    segment_start: TimePoint,
    dual_seg1: RatingSegment,
    dual_seg2: RatingSegment,
    input_iter1: I1,
    input_iter2: I2,
    finished: bool,
}

impl<I1: RI, I2: RI> RatingAdderIterator<I1, I2> {
    #[inline]
    fn generate_segment(&mut self, segment_end: TimePoint) -> RatingFullSegment {
        let start_rating1 = self.dual_seg1.data.rating;
        let start_rating2 = self.dual_seg2.data.rating;

        let delta1 = self.dual_seg1.data.delta;
        let delta2 = self.dual_seg2.data.delta;

        /*println!(
            "ADDER new segment {} to {}",
            self.segment_start, segment_end
        );*/

        RatingFullSegment {
            span: TimeSpan::new(self.segment_start, segment_end),
            data: RatingInfo {
                rating: start_rating1 + start_rating2,
                delta: delta1 + delta2,
            },
        }
    }
}

impl<I1: RI, I2: RI> Iterator for RatingAdderIterator<I1, I2> {
    type Item = RatingFullSegment;

    #[inline]
    fn next(&mut self) -> Option<RatingFullSegment> {
        if self.finished {
            return None;
        }

        let len;
        let result: RatingFullSegment;

        /*println!(
            "ADDER start: {} ep1: {} ep2: {}",
            self.segment_start, self.dual_seg1.end_point, self.dual_seg2.end_point
        );*/

        if self.dual_seg1.end_point < self.dual_seg2.end_point {
            len = self.dual_seg1.end_point - self.segment_start;

            result = self.generate_segment(self.dual_seg1.end_point);

            self.segment_start = self.dual_seg1.end_point;

            self.dual_seg1 = self
                .input_iter1
                .next()
                .expect("RatingAdderIterator: First iterator ended before second");
            self.dual_seg2.advance(len);
        } else if self.dual_seg2.end_point < self.dual_seg1.end_point {
            len = self.dual_seg2.end_point - self.segment_start;

            result = self.generate_segment(self.dual_seg2.end_point);

            self.segment_start = self.dual_seg2.end_point;

            self.dual_seg1.advance(len);
            self.dual_seg2 = self
                .input_iter2
                .next()
                .expect("RatingAdderIterator: Second iterator ended before first");
        } else {
            // self.dual_seg2.end_point === self.dual_seg1.end_point

            match (self.input_iter1.next(), self.input_iter2.next()) {
                (Some(dual_seg1), Some(dual_seg2)) => {
                    result = self.generate_segment(self.dual_seg1.end_point);

                    self.segment_start = self.dual_seg1.end_point;

                    self.dual_seg1 = dual_seg1;
                    self.dual_seg2 = dual_seg2;
                }
                (Some(new_dual_seg1), None) => {
                    panic!(
                        "RatingAdderIterator: Second iterator ended before first {}",
                        new_dual_seg1.end_point
                    );
                }
                (None, Some(new_dual_seg2)) => {
                    panic!(
                        "RatingAdderIterator: First iterator ended before second {}",
                        new_dual_seg2.end_point
                    );
                }
                (None, None) => {
                    result = self.generate_segment(self.dual_seg1.end_point);

                    self.finished = true;
                }
            }
        }

        Some(result)
    }
}

pub fn add_rating_iterators(
    mut iter1: RatingIterator<impl RI>,
    mut iter2: RatingIterator<impl RI>,
) -> RatingFullIterator<impl RFI> {
    assert!(iter1.start == iter2.start);
    let start = iter1.start;

    //println!("ADDER segment start {}", start);

    let dual_seg1 = iter1
        .iter
        .next()
        .expect("First iterator should have at least one element");
    let dual_seg2 = iter2
        .iter
        .next()
        .expect("Second iterator should have at least one element");

    RatingFullIterator::<_> {
        start: start,
        iter: RatingAdderIterator::<_, _> {
            segment_start: start,
            dual_seg1: dual_seg1,
            dual_seg2: dual_seg2,
            input_iter1: iter1.iter,
            input_iter2: iter2.iter,
            finished: false,
        },
    }
}

trait PushIterator {
    type Item;
    type Output;

    /// False means the called function requests the end of the stream.
    #[inline]
    fn push(&mut self, item: Self::Item);

    #[inline]
    fn finish(self) -> Self::Output;
}

// //////////////////////////////////////////////////////////////////////////////////////////////////

struct AggressiveSimplifySegmentData {
    seg: RatingFullSegment,
    offset_interval: (RatingDelta, RatingDelta),
    pivot: TimePoint,
}

struct AggressiveSimplifyRatingPushIterator<I: PushIterator<Item = RatingFullSegment>> {
    start: TimePoint,
    epsilon: RatingDelta,
    current_segment: Option<AggressiveSimplifySegmentData>,
    iter: I,
}

impl<I: PushIterator<Item = RatingFullSegment>> AggressiveSimplifyRatingPushIterator<I> {
    fn get_min_max_offset_for_target(
        target_rating: Rating,
        target: TimePoint,
        pivot_rating: Rating,
        pivot: TimePoint,
        max_diff: RatingDelta,
    ) -> (RatingDelta, RatingDelta) {
        if target == pivot {
            return (-max_diff, max_diff);
        } // TODO: this works buy actually we want -inf and +inf

        let min_delta = (target_rating - max_diff - pivot_rating) / (target - pivot).as_i64(); // BUG TODO: ROUND UP
        let max_delta = (target_rating + max_diff - pivot_rating) / (target - pivot).as_i64();

        if min_delta <= max_delta {
            (min_delta, max_delta)
        } else {
            (max_delta, min_delta)
        }
    }

    fn get_min_max_offset_for_segment(
        seg: RatingFullSegment,
        pivot_rating: Rating,
        pivot: TimePoint,
        max_diff: RatingDelta,
    ) -> (RatingDelta, RatingDelta) {
        let interval1 =
            Self::get_min_max_offset_for_target(seg.data.start_rating(), seg.span.start, pivot_rating, pivot, max_diff);
        let interval2 = Self::get_min_max_offset_for_target(
            seg.data.end_rating(seg.span.len()),
            seg.span.end - TimeDelta::one(),
            pivot_rating,
            pivot,
            max_diff,
        );

        return Self::intersect_intervals(interval1, interval2);
    }

    fn intersect_intervals(a: (RatingDelta, RatingDelta), b: (RatingDelta, RatingDelta)) -> (RatingDelta, RatingDelta) {
        return (max(a.0, b.0), min(a.1, b.1));
    }

    fn create_segment(&self, seg: RatingFullSegment) -> AggressiveSimplifySegmentData {
        let pivot = seg.span.half();
        let pivot_rating = seg.data.get_at(pivot - seg.span.start);

        AggressiveSimplifySegmentData {
            seg: seg,
            pivot: pivot,
            offset_interval: Self::get_min_max_offset_for_segment(seg, pivot_rating, pivot, self.epsilon),
        }
    }
}

impl<I: PushIterator<Item = RatingFullSegment>> PushIterator for AggressiveSimplifyRatingPushIterator<I> {
    type Item = RatingSegment;
    type Output = I::Output;

    #[inline]
    fn finish(mut self) -> Self::Output {
        if let Some(current_segment) = self.current_segment {
            self.iter.push(current_segment.seg);
        }
        self.iter.finish()
    }

    #[inline]
    fn push(&mut self, next_segment: RatingSegment) {
        let mut current_segment: AggressiveSimplifySegmentData;

        match self.current_segment.take() {
            None => {
                let seg = next_segment.with_start_point(self.start);
                self.current_segment = Some(self.create_segment(seg));
                return;
            }
            Some(v) => current_segment = v,
        }

        let seg = next_segment.with_start_point(current_segment.seg.span.end);

        let pivot_diff: TimeDelta = current_segment.pivot - current_segment.seg.span.start;
        let pivot_rating = current_segment.seg.data.get_at(pivot_diff);

        let interval = Self::get_min_max_offset_for_segment(seg, pivot_rating, current_segment.pivot, self.epsilon);

        let next_interval = Self::intersect_intervals(current_segment.offset_interval, interval);

        if next_interval.0 <= next_interval.1 {
            let new_delta = (next_interval.0 + next_interval.1) / 2;
            let new_start_rating = Rating::add_mul(pivot_rating, new_delta, -pivot_diff);

            //println!("len1 {} len2 {} min {} max {} delta {} new delta {}", current_segment.seg.span.len(), seg.span.len(), next_interval.0.as_readable_f32(), next_interval.1.as_readable_f32(), current_segment.seg.data.delta.as_readable_f32(), new_delta.as_readable_f32());

            current_segment.seg.span.end = next_segment.end_point;
            current_segment.seg.data.delta = new_delta;
            current_segment.seg.data.rating = new_start_rating;
            current_segment.offset_interval = next_interval;
        //println!("Simplified");
        } else {
            //println!("push");
            self.iter.push(current_segment.seg);
            current_segment = self.create_segment(seg);
        }

        self.current_segment = Some(current_segment);
    }
}

fn aggressive_simplifiy_ratings_push_iter<I>(
    start: TimePoint,
    epsilon: RatingDelta,
    iter: I,
) -> impl PushIterator<Item = RatingSegment, Output = I::Output>
where
    I: PushIterator<Item = RatingFullSegment>,
{
    AggressiveSimplifyRatingPushIterator {
        current_segment: None,
        epsilon: epsilon,
        start: start,
        iter: iter,
    }
}

// //////////////////////////////////////////////////////////////////////////////////////////////////

struct SimplifyPositionPushIterator<I: PushIterator<Item = PositionFullSegment>> {
    start: TimePoint,
    current_segment: Option<PositionFullSegment>,
    iter: I,
}

impl<I: PushIterator<Item = PositionFullSegment>> PushIterator for SimplifyPositionPushIterator<I> {
    type Item = PositionSegment;
    type Output = I::Output;

    #[inline]
    fn finish(mut self) -> Self::Output {
        if let Some(current_segment) = self.current_segment {
            self.iter.push(current_segment);
        }
        self.iter.finish()
    }

    #[inline]
    fn push(&mut self, next_segment: PositionSegment) {
        let mut current_segment: PositionFullSegment;

        match self.current_segment {
            None => {
                self.current_segment = Some(next_segment.with_start_point(self.start));
                return;
            }
            Some(v) => current_segment = v,
        }

        if current_segment.data.drag == next_segment.data.drag
            && current_segment.exclusive_end_position() == next_segment.data.start_position()
        {
            current_segment.span.end = next_segment.end_point;
        } else {
            self.iter.push(current_segment);

            current_segment = next_segment.with_start_point(current_segment.span.end);
        }

        self.current_segment = Some(current_segment);
    }
}

#[inline]
fn simplifiy_positions_push_iter<I>(
    start: TimePoint,
    iter: I,
) -> impl PushIterator<Item = PositionSegment, Output = I::Output>
where
    I: PushIterator<Item = PositionFullSegment>,
{
    SimplifyPositionPushIterator {
        current_segment: None,
        start: start,
        iter: iter,
    }
}

// //////////////////////////////////////////////////////////////////////////////////////////////////

struct SimplifyRatingPushIterator<I: PushIterator<Item = RatingFullSegment>> {
    start: TimePoint,
    current_segment: Option<RatingFullSegment>,
    iter: I,
}

impl<I: PushIterator<Item = RatingFullSegment>> PushIterator for SimplifyRatingPushIterator<I> {
    type Item = RatingSegment;
    type Output = I::Output;

    #[inline]
    fn finish(mut self) -> Self::Output {
        if let Some(current_segment) = self.current_segment {
            self.iter.push(current_segment);
        }
        self.iter.finish()
    }

    #[inline]
    fn push(&mut self, next_segment: RatingSegment) {
        let mut current_segment: RatingFullSegment;

        match self.current_segment {
            None => {
                self.current_segment = Some(next_segment.with_start_point(self.start));
                return;
            }
            Some(v) => current_segment = v,
        }

        if current_segment.data.delta == next_segment.data.delta
            && current_segment.exclusive_end_rating() == next_segment.data.start_rating()
        {
            current_segment.span.end = next_segment.end_point;
        } else {
            self.iter.push(current_segment);
            current_segment = next_segment.with_start_point(current_segment.span.end);
        }

        self.current_segment = Some(current_segment);
    }
}

fn simplifiy_ratings_push_iter<I>(
    start: TimePoint,
    iter: I,
) -> impl PushIterator<Item = RatingSegment, Output = I::Output>
where
    I: PushIterator<Item = RatingFullSegment>,
{
    SimplifyRatingPushIterator {
        current_segment: None,
        start: start,
        iter: iter,
    }
}

// //////////////////////////////////////////////////////////////////////////////////////////////////

struct CollectToVecPushIterator<T> {
    v: Vec<T>,
}

impl<T> PushIterator for CollectToVecPushIterator<T> {
    type Item = T;
    type Output = Vec<T>;

    #[inline]
    fn finish(self) -> Vec<T> {
        self.v
    }

    #[inline]
    fn push(&mut self, item: T) {
        self.v.push(item)
    }
}

struct DualPushIterator<T, I1, I2>
where
    T: Clone,
    I1: PushIterator<Item = T>,
    I2: PushIterator<Item = T>,
{
    iter1: I1,
    iter2: I2,
}

impl<T, I1, I2> PushIterator for DualPushIterator<T, I1, I2>
where
    T: Clone,
    I1: PushIterator<Item = T>,
    I2: PushIterator<Item = T>,
{
    type Item = T;
    type Output = (I1::Output, I2::Output);

    #[inline]
    fn finish(self) -> (I1::Output, I2::Output) {
        (self.iter1.finish(), self.iter2.finish())
    }

    #[inline]
    fn push(&mut self, item: T) {
        self.iter1.push(item.clone());
        self.iter2.push(item);
    }
}

fn dual_push_iter<T, I1, I2>(i1: I1, i2: I2) -> DualPushIterator<T, I1, I2>
where
    T: Clone,
    I1: PushIterator<Item = T>,
    I2: PushIterator<Item = T>,
{
    DualPushIterator { iter1: i1, iter2: i2 }
}

struct MapPushIterator<I, F, A, B>
where
    I: PushIterator<Item = B>,
    F: Fn(A) -> B,
{
    iter: I,
    f: F,
    _marker: std::marker::PhantomData<A>,
}

impl<I, F, A, B> PushIterator for MapPushIterator<I, F, A, B>
where
    I: PushIterator<Item = B>,
    F: Fn(A) -> B,
{
    type Item = A;
    type Output = I::Output;

    #[inline]
    fn finish(self) -> I::Output {
        self.iter.finish()
    }

    #[inline]
    fn push(&mut self, item: A) {
        self.iter.push((self.f)(item));
    }
}

fn into_push_iter<T, I, O>(v: I, mut iter: O) -> O::Output
where
    I: Iterator<Item = T>,
    O: PushIterator<Item = T>,
{
    for item in v {
        iter.push(item);
    }

    iter.finish()
}

fn map_push_iterator<I, A, B>(iter: I, f: impl Fn(A) -> B) -> impl PushIterator<Item = A, Output = I::Output>
where
    I: PushIterator<Item = B>,
{
    MapPushIterator {
        iter: iter,
        f: f,
        _marker: Default::default(),
    }
}

fn only_ratings_push_iter<I>(iter: I) -> impl PushIterator<Item = DualSegment, Output = I::Output>
where
    I: PushIterator<Item = RatingSegment>,
{
    map_push_iterator(iter, |dual_segment: DualSegment| dual_segment.as_rating_segment())
}

fn only_positions_push_iter<I>(iter: I) -> impl PushIterator<Item = DualSegment, Output = I::Output>
where
    I: PushIterator<Item = PositionSegment>,
{
    map_push_iterator(iter, |dual_segment: DualSegment| dual_segment.as_position_segment())
}

fn collect_to_vec_push_iter<T>() -> impl PushIterator<Item = T, Output = Vec<T>> {
    CollectToVecPushIterator { v: Vec::new() }
}

fn discard_start_times_push_iter<I, D>(iter: I) -> impl PushIterator<Item = FullSegment<D>, Output = I::Output>
where
    I: PushIterator<Item = Segment<D>>,
{
    map_push_iterator(iter, |full_segment: FullSegment<D>| {
        assert!(full_segment.span.len() > TimeDelta::zero());
        full_segment.discard_start_time()
    })
}

struct DualSimplifyIterator<I>
where
    I: DI,
{
    current_segment: Option<DualFullSegment>,
    iter: I,
}

impl<I: DI> Iterator for DualSimplifyIterator<I> {
    type Item = DualFullSegment;

    #[inline]
    fn next(&mut self) -> Option<DualFullSegment> {
        let mut current_segment: DualFullSegment;

        match self.current_segment.take() {
            None => {
                return None;
            }
            Some(v) => current_segment = v,
        }

        loop {
            match self.iter.next() {
                None => {
                    self.current_segment = None;
                    return Some(current_segment);
                }
                Some(next_segment) => {
                    if next_segment.data.rating_info.delta == current_segment.data.rating_info.delta
                        && next_segment.data.position_info.drag == current_segment.data.position_info.drag
                        && current_segment.exclusive_end_position() == next_segment.start_position()
                        && current_segment.exclusive_end_rating() == next_segment.start_rating()
                    {
                        current_segment.span.end = next_segment.end_point;
                    } else {
                        self.current_segment = Some(next_segment.with_start_point(current_segment.span.end));
                        return Some(current_segment);
                    }
                }
            }
        }
    }
}
