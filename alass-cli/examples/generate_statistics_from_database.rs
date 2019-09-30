// This file is not really meant as an example. This program in conjunction with the `../statistics-helpers/*` create the diagrams .

use alass_cli::*;
use clap::value_t;
use clap::{App, Arg};
use failure::{Backtrace, Context, Fail, ResultExt};
use rmp_serde as rmps;
use std::cmp::Ordering;
use std::cmp::{max, min};
use std::collections::HashSet;
use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic;
use std::sync::atomic::AtomicUsize;

// Raw data
//
// Types of alignment algorithms:
//   unweighted no-split
//   no-split
//   split
//
// Reference Types:
//    video
//    ref sub
//
// Framerate correction
//    none
//    3 (1, 23.976 <-> 25)
//    7 (1, 23.976 <-> 25, 23.976 <-> 24, 24 <-> 25)

use threadpool::ThreadPool;

use std::sync::{Arc, Mutex};

struct Task {
    context: TProgressInfo,
    name: String,
}

struct RunningTasksInfoLocked {
    last_print: std::time::Instant,
    next_id: usize,
    tasks: Vec<(usize, Task)>,
}

struct RunningTasksInfo {
    quiet: bool,
    tasks: Mutex<RunningTasksInfoLocked>,
}

impl RunningTasksInfo {
    fn new(quiet: bool) -> RunningTasksInfo {
        RunningTasksInfo {
            quiet,
            tasks: Mutex::new(RunningTasksInfoLocked {
                last_print: std::time::Instant::now(),
                next_id: 0,
                tasks: Vec::new(),
            }),
        }
    }

    fn run<O>(&self, name: impl ToString, context: TProgressInfo, f: impl FnOnce() -> O) -> O {
        let name = name.to_string();

        let task = Task {
            name: name.clone(),
            context: context.clone(),
        };

        let task_id: usize;
        {
            let mut tasks_lock = self.tasks.lock().unwrap();
            task_id = tasks_lock.next_id;
            tasks_lock.tasks.push((task_id, task));
            tasks_lock
                .tasks
                .sort_by(|(_, task1), (_, task2)| task1.context.cmp(&task2.context));
            tasks_lock.next_id = tasks_lock.next_id + 1;

            if !self.quiet {
                if std::time::Instant::now() - tasks_lock.last_print > std::time::Duration::from_millis(100) {
                    Self::print_tasks(&tasks_lock.tasks);
                } else {
                    println!("=> {}: {} [started]", name, context);
                }

                tasks_lock.last_print = std::time::Instant::now();
            }
        }
        let start_time = std::time::Instant::now();
        let result = f();
        let end_time = std::time::Instant::now();
        {
            let mut tasks_lock = self.tasks.lock().unwrap();
            let remove_idx = tasks_lock
                .tasks
                .iter()
                .position(|(cur_id, _)| *cur_id == task_id)
                .unwrap();
            let task = tasks_lock.tasks.remove(remove_idx);

            if !self.quiet {
                println!(
                    "<= {}: {} [finished in {}ms]",
                    task.1.name,
                    task.1.context,
                    (end_time - start_time).as_millis()
                );

                if std::time::Instant::now() - tasks_lock.last_print > std::time::Duration::from_millis(100) {
                    Self::print_tasks(&tasks_lock.tasks);
                }
                tasks_lock.last_print = std::time::Instant::now();
            }
        }

        result
    }

    fn print_tasks(tasks: &[(usize, Task)]) {
        if tasks.is_empty() {
            return;
        }

        println!("[");
        for (_, task) in tasks {
            println!("\t{}: {}", task.context, task.name);
        }
        println!("]");
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct MovieProgressContext {
    movie_id: SubtitleID,

    movie_nr: usize,
    total_movie_count: usize,
}

impl fmt::Display for MovieProgressContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "M[{}/{}; '{}']",
            self.movie_nr + 1,
            self.total_movie_count,
            self.movie_id,
        )
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct SubtitleProgressContext {
    sub_id: SubtitleID,

    sub_nr: usize,
    movie_sub_count: usize,

    total_sub_nr: usize,
    total_sub_count: usize,
}

impl fmt::Display for SubtitleProgressContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "S[m{}/{} | t{}/{} | '{}']",
            self.sub_nr + 1,
            self.movie_sub_count,
            self.total_sub_nr + 1,
            self.total_sub_count,
            self.sub_id
        )
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum ProgressContext {
    Movie(MovieProgressContext),
    SubtitleForMovie(MovieProgressContext, SubtitleProgressContext),
}

impl Ord for ProgressContext {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (ProgressContext::Movie(m1), ProgressContext::Movie(m2)) => m1.movie_nr.cmp(&m2.movie_nr),
            (ProgressContext::Movie(_m1), ProgressContext::SubtitleForMovie(_m2, _s2)) => Ordering::Less,
            (ProgressContext::SubtitleForMovie(_m1, _s1), ProgressContext::Movie(_m2)) => Ordering::Greater,
            (ProgressContext::SubtitleForMovie(_m1, s1), ProgressContext::SubtitleForMovie(_m2, s2)) => {
                s1.total_sub_nr.cmp(&s2.total_sub_nr)
            }
        }
    }
}

impl PartialOrd for ProgressContext {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for ProgressContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProgressContext::Movie(m) => write!(f, "{}", m),
            ProgressContext::SubtitleForMovie(m, s) => write!(f, "{}->{}", m, s),
        }
    }
}

#[derive(Debug, Copy, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub enum AlignMode {
    NoSplit,
    Split {
        split_penalty: FixedPointNumber,
        optimization: Option<FixedPointNumber>,
    },
}

#[derive(Debug, Copy, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub enum ScalingCorrectMode {
    // only scaling value: 1
    None,
    // using 1, 23.976/25 and 25/23.976
    //Simple,
    // using 1, 23.976/25 and 25/23.976, 24/25 and 25/24, 23.976/24 and 24/23.976
    Advanced,
}

impl ScalingCorrectMode {
    pub fn iter() -> &'static [ScalingCorrectMode] {
        &[
            ScalingCorrectMode::None,
            //ScalingCorrectMode::Simple,
            ScalingCorrectMode::Advanced,
        ]
    }
}

#[derive(Debug, Copy, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub enum ScoringMode {
    Standard,
    Overlap,
}

#[derive(Debug, Copy, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub struct AlignConfig {
    pub align_mode: AlignMode,
    pub ms_per_alg_step: i64,
    pub scaling_correct_mode: ScalingCorrectMode,
    pub scoring_mode: ScoringMode,
}

impl AlignConfig {
    fn with_split_penalty(&self, new_split_penalty: FixedPointNumber) -> AlignConfig {
        AlignConfig {
            align_mode: match self.align_mode {
                AlignMode::NoSplit => AlignMode::NoSplit,
                AlignMode::Split { optimization, .. } => AlignMode::Split {
                    split_penalty: new_split_penalty,
                    optimization,
                },
            },
            ms_per_alg_step: self.ms_per_alg_step,
            scaling_correct_mode: self.scaling_correct_mode,
            scoring_mode: self.scoring_mode,
        }
    }

    fn with_optimization(&self, new_optimization: Option<FixedPointNumber>) -> AlignConfig {
        AlignConfig {
            align_mode: match self.align_mode {
                AlignMode::NoSplit => AlignMode::NoSplit,
                AlignMode::Split { split_penalty, .. } => AlignMode::Split {
                    split_penalty,
                    optimization: new_optimization,
                },
            },
            ms_per_alg_step: self.ms_per_alg_step,
            scaling_correct_mode: self.scaling_correct_mode,
            scoring_mode: self.scoring_mode,
        }
    }

    /*fn with_scaling_correct_mode(&self, new_scaling_correct_mode: ScalingCorrectMode) -> AlignConfig {
        AlignConfig {
            align_mode: self.align_mode,
            ms_per_alg_step: self.ms_per_alg_step,
            scaling_correct_mode: new_scaling_correct_mode,
        }
    }*/
}

mod types {

    #[derive(Clone, Copy, Debug, Hash, serde::Serialize, serde::Deserialize)]
    pub struct Span {
        pub start_ms: i64,
        pub end_ms: i64,
    }

    impl From<&super::database::LineInfo> for Span {
        fn from(l: &super::database::LineInfo) -> Span {
            assert!(l.start_ms <= l.end_ms);
            Span {
                start_ms: l.start_ms,
                end_ms: l.end_ms,
            }
        }
    }

    impl From<subparse::timetypes::TimeSpan> for Span {
        fn from(l: subparse::timetypes::TimeSpan) -> Span {
            assert!(l.start.msecs() <= l.end.msecs());
            Span {
                start_ms: l.start.msecs(),
                end_ms: l.end.msecs(),
            }
        }
    }

    impl Span {
        pub fn plus_delta(self, ms: i64) -> Span {
            Span {
                start_ms: self.start_ms + ms,
                end_ms: self.end_ms + ms,
            }
        }

        pub fn scaled_by(self, f: f64) -> Span {
            Span {
                start_ms: (self.start_ms as f64 * f) as i64,
                end_ms: (self.end_ms as f64 * f) as i64,
            }
        }

        pub fn len_ms(self) -> i64 {
            assert!(self.start_ms <= self.end_ms);
            self.end_ms - self.start_ms
        }

        pub fn to_alass_core_spans(self, ms_per_alg_step: i64) -> alass_core::TimeSpan {
            alass_core::TimeSpan::new(
                alass_core::TimePoint::from(self.start_ms / ms_per_alg_step),
                alass_core::TimePoint::from(self.end_ms / ms_per_alg_step),
            )
        }

        pub fn compute_line_distance(self, b: Span) -> i64 {
            let a = self;

            assert!(a.start_ms <= a.end_ms);
            assert!(b.start_ms <= b.end_ms);

            let start_diff = b.start_ms - a.start_ms;
            let end_diff = b.end_ms - a.end_ms;

            if start_diff > 0 && end_diff > 0 {
                // b is right of a
                std::cmp::min(start_diff, end_diff)
            } else if start_diff < 0 && end_diff < 0 {
                // b is left of a
                std::cmp::min(-start_diff, -end_diff)
            } else {
                // b encloses a, or a enclosed b
                0
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
    pub enum RefConfig {
        Subtitle(SubtitleID),
        Video(MovieID, VADConfig),
    }

    impl RefConfig {
        pub fn iter_from<'a>(
            movie_id: MovieID,
            vad_spans: &'a [Span],
            vad_conf: VADConfig,
            ref_sub_id: SubtitleID,
            ref_spans: &'a [Span],
        ) -> impl Iterator<Item = (RefConfig, &'a [Span])> {
            vec![
                (RefConfig::Video(movie_id, vad_conf), vad_spans),
                (RefConfig::Subtitle(ref_sub_id), ref_spans),
            ]
            .into_iter()
        }

        pub fn as_ref_type(&self) -> super::statistics::SyncReferenceType {
            match self {
                RefConfig::Subtitle(_) => super::statistics::SyncReferenceType::Subtitle,
                RefConfig::Video(_, _) => super::statistics::SyncReferenceType::Video,
            }
        }
    }

    #[derive(Debug, Copy, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
    pub struct VADConfig {
        pub min_span_length_ms: i64,
    }

    impl VADConfig {
        pub fn applied_to(&self, spans: &[Span]) -> Vec<Span> {
            spans
                .iter()
                .cloned()
                .filter(|span| span.end_ms - span.start_ms >= self.min_span_length_ms)
                .collect()
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
    pub struct FixedPointNumber(i64);
    const FIXED_POINT_NUMBER_FACTOR: f64 = 100000000.0;

    impl FixedPointNumber {
        pub fn from_f64(v: f64) -> FixedPointNumber {
            FixedPointNumber((v * FIXED_POINT_NUMBER_FACTOR) as i64)
        }

        pub fn to_f64(self) -> f64 {
            self.0 as f64 / FIXED_POINT_NUMBER_FACTOR
        }

        pub fn to_f32(self) -> f32 {
            self.0 as f32 / FIXED_POINT_NUMBER_FACTOR as f32
        }

        pub fn one() -> FixedPointNumber {
            FixedPointNumber(FIXED_POINT_NUMBER_FACTOR as i64)
        }
    }

    pub type MovieID = String;
    pub type SubtitleID = String;
    pub type LinePair = (usize, usize);
    pub type LinePairs = Vec<LinePair>;
}

use types::*;

// TODO: export how many lines are discarded
// TODO: how long does synchronization take
// TODO: how far off are synchronizations?
// TODO: how far off are synchronizations concerning when optimizing for speed, split penalty, ...

#[derive(Debug, Copy, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub struct LineMatchingConfig {
    certain_match_similarity: FixedPointNumber,
    certain_unmatch_similarity: FixedPointNumber,
}

#[derive(Debug, Copy, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
struct GoodSyncRequirement {
    // specifies that at least `Y*100%` of the subtitle lines should have at most `X ms` offset difference for a good sync
    at_least_proportion_of_all_subs: FixedPointNumber,
    at_most_offset: i64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub struct SyncClassificationConfig {
    required_segments_for_sync_classification: usize,

    good_sync_requirements: Vec<GoodSyncRequirement>,
}

//const CERTAIN_MATCH_SIMILARITY: f64 = 0.9;
//const REQUIRED_SEGMENTS_FOR_SYNC_CLASSIFICATION: usize = 10;
//const GOOD_SYNC_MAXIMUM_LINE_OFFSET_MS: i64 = 150;
//const MAXIMUM_UNSYNC_PERCENTAGE_FOR_GOOD_SYNC: f32 = 5.0f32;

#[derive(Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SyncClassification {
    Synced,
    Unsynced,
    Unknown,
}

pub fn format_time(ms: i64) -> String {
    format!(
        "{:02}:{:02}:{:02},{:03}",
        (ms / (1000 * 60 * 60)),
        (ms / (1000 * 60)) % 60,
        (ms / 1000) % 60,
        ms % 1000
    )
}

/*fn time<T>(f: impl FnOnce() -> T) -> (T, ProgramDuration) {
    let start = std::time::Instant::now();
    let result = f();
    let end = std::time::Instant::now();

    (result, end - start)
}*/

fn enough_lines(sub: &[Span], idxs: impl Iterator<Item = usize>, config: &SyncClassificationConfig) -> bool {
    if sub.len() == 0 {
        return true;
    }

    let start_ms = sub.iter().map(|l| l.start_ms).min().unwrap();
    let end_ms = sub.iter().map(|l| l.end_ms).max().unwrap();

    assert!(start_ms < end_ms);

    if start_ms == end_ms {
        return true;
    }

    let mut line_in_segment: Vec<bool> = vec![false; config.required_segments_for_sync_classification];

    let get_segment_for_ms = |ms: i64| {
        ((ms - start_ms) * config.required_segments_for_sync_classification as i64 / (end_ms - start_ms + 1)) as usize
    };

    for idx in idxs {
        let line = &sub[idx];

        assert!(line.start_ms <= line.end_ms);

        let start_segment = get_segment_for_ms(line.start_ms);
        let end_segment = get_segment_for_ms(line.end_ms);

        for segment_idx in start_segment..=end_segment {
            line_in_segment[segment_idx] = true;
        }
    }

    line_in_segment.into_iter().all(|v| v)
}

fn get_sync_classification(
    ref_sub: &[Span],
    in_sub: &[Span],
    line_pairs: &[(usize, usize)],
    config: &SyncClassificationConfig,
) -> SyncClassification {
    if ref_sub.len() < in_sub.len() / 5 || in_sub.len() < ref_sub.len() / 5 {
        return SyncClassification::Unknown;
    }

    if !enough_lines(ref_sub, line_pairs.iter().map(|&(ref_idx, _)| ref_idx), config)
        || !enough_lines(in_sub, line_pairs.iter().map(|&(_, in_idx)| in_idx), config)
    {
        return SyncClassification::Unknown;
    }

    let good_sync_requirements: &[GoodSyncRequirement] = &config.good_sync_requirements;

    let mut unsync_lines_counts: Vec<usize> = vec![0; good_sync_requirements.len()];

    let max_unsync_counts: Vec<usize> = good_sync_requirements
        .iter()
        .map(|req| line_pairs.len() - (req.at_least_proportion_of_all_subs.to_f64() * line_pairs.len() as f64) as usize)
        .collect();

    for &(ref_idx, in_idx) in line_pairs {
        let ref_span = ref_sub[ref_idx];
        let in_span = in_sub[in_idx];

        let offset = Span::compute_line_distance(ref_span, in_span);
        for ((good_sync_requirement, unsync_lines_counts), max_unsync_count) in good_sync_requirements
            .iter()
            .zip(unsync_lines_counts.iter_mut())
            .zip(max_unsync_counts.iter())
        {
            if offset > good_sync_requirement.at_most_offset {
                *unsync_lines_counts = *unsync_lines_counts + 1;

                if *unsync_lines_counts > *max_unsync_count {
                    return SyncClassification::Unsynced;
                }
            }
        }
    }

    SyncClassification::Synced
}

fn edit_distance(a: &[char], b: &[char]) -> i32 {
    let alen: usize = a.len();
    let blen: usize = b.len();

    if alen == 0 {
        return blen as i32;
    }
    if blen == 0 {
        return alen as i32;
    }

    let mut score: Vec<i32> = vec![0; alen * blen];

    let idx = |ac: usize, bc: usize| -> usize {
        assert!(ac < alen);
        assert!(bc < blen);
        return bc * alen + ac;
    };

    for ac in 0..alen {
        score[idx(ac, 0)] = ac as i32;
    }

    for bc in 1..blen {
        score[idx(0, bc)] = bc as i32;
    }

    for bc in 1..blen {
        for ac in 1..alen {
            if a[ac] == b[bc] {
                score[idx(ac, bc)] = score[idx(ac - 1, bc - 1)];
            } else {
                let s1 = score[idx(ac - 1, bc - 1)];
                let s2 = score[idx(ac - 1, bc - 0)];
                let s3 = score[idx(ac - 0, bc - 1)];
                score[idx(ac, bc)] = min(s1, min(s2, s3)) + 1;
            }
        }
    }

    score[idx(alen - 1, blen - 1)]
}

fn similarity(a: &str, b: &str) -> f32 {
    let ac = &a.chars().collect::<Vec<char>>();
    let bc = &b.chars().collect::<Vec<char>>();

    let len = max(ac.len(), bc.len()) as i32;
    let changes = edit_distance(&ac, &bc);

    let r = (len - changes) as f32 / len as f32;

    //println!("{} =? {} -> {}", a, b, r);
    r
}

fn get_line_pairs(a: &[database::LineInfo], b: &[database::LineInfo], config: &LineMatchingConfig) -> LinePairs {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }

    let alen = a.len();
    let blen = b.len();

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum BestChoice {
        PushedA,
        PushedB,
        Mismatch,
        Match,
    }

    let mut score: Vec<(f32, f32, Option<BestChoice>)> = vec![(0.0f32, 0.0f32, None); alen * blen];

    let idx = |ac: usize, bc: usize| -> usize {
        assert!(ac < alen);
        assert!(bc < blen);
        return bc * alen + ac;
    };

    let origin_similarity = similarity(&a[0].text, &b[0].text);

    if origin_similarity >= config.certain_match_similarity.to_f32() {
        score[idx(0, 0)] = (origin_similarity, origin_similarity, Some(BestChoice::Match));
    } else {
        score[idx(0, 0)] = (0.0, origin_similarity, None);
    }

    for ai in 1..alen {
        score[idx(ai, 0)] = (0.0, similarity(&a[ai].text, &b[0].text), Some(BestChoice::PushedA));
    }

    for bi in 1..blen {
        score[idx(0, bi)] = (0.0, similarity(&a[0].text, &b[bi].text), Some(BestChoice::PushedB));
    }

    for bi in 1..blen {
        let bl = &b[bi];
        for ai in 1..alen {
            let al = &a[ai];

            let similarity = similarity(&al.text, &bl.text);

            if similarity >= config.certain_match_similarity.to_f32() {
                let old_score = score[idx(ai - 1, bi - 1)].0;
                let new_score = &mut score[idx(ai, bi)];
                new_score.0 = old_score + similarity;
                new_score.1 = similarity;
                new_score.2 = Some(BestChoice::Match);
            } else {
                let score_push_a = score[idx(ai - 1, bi - 0)].0;
                let score_push_b = score[idx(ai - 0, bi - 1)].0;
                let score_mismatch = score[idx(ai - 1, bi - 1)].0;

                if score_push_a >= score_push_b && score_push_a >= score_mismatch {
                    score[idx(ai, bi)] = (score_push_a, similarity, Some(BestChoice::PushedA));
                } else if score_push_b >= score_push_a && score_push_b >= score_mismatch {
                    score[idx(ai, bi)] = (score_push_b, similarity, Some(BestChoice::PushedB));
                } else {
                    score[idx(ai, bi)] = (score_mismatch, similarity, Some(BestChoice::Mismatch));
                }
            }
        }
    }

    let mut ai = alen - 1;
    let mut bi = blen - 1;

    let mut result: LinePairs = Vec::with_capacity(max(alen, blen));

    loop {
        match score[idx(ai, bi)].2 {
            Some(BestChoice::Match) => {
                let mut ambigous = false;
                let certain_unmatch_similarity = config.certain_unmatch_similarity.to_f32();
                for ax in 0..alen {
                    if ax == ai {
                        continue;
                    }
                    if score[idx(ax, bi)].1 > certain_unmatch_similarity {
                        ambigous = true;
                        break;
                    }
                }

                if !ambigous {
                    for bx in 0..blen {
                        if bx == bi {
                            continue;
                        }
                        if score[idx(ai, bx)].1 > certain_unmatch_similarity {
                            ambigous = true;
                            break;
                        }
                    }
                }

                if !ambigous {
                    result.push((ai, bi));
                }

                if ai == 0 && bi == 0 {
                    break;
                }

                assert!(ai > 0);
                assert!(bi > 0);

                /*println!(
                    "MATCH {} {}: {} ------- {}\n",
                    similarity(&a[ai].text, &b[bi].text),
                    format_time(a[ai].start_ms),
                    a[ai].text,
                    b[bi].text
                );*/

                ai -= 1;
                bi -= 1;
            }
            Some(BestChoice::Mismatch) => {
                if ai == 0 && bi == 0 {
                    break;
                }

                assert!(ai > 0);
                assert!(bi > 0);
                ai -= 1;
                bi -= 1;

                /*println!(
                    "mismatch {} {}: {} ------- {}\n",
                    similarity(&a[ai].text, &b[bi].text),
                    format_time(a[ai].start_ms),
                    a[ai].text,
                    b[bi].text
                );*/
            }
            Some(BestChoice::PushedA) => {
                /*println!(
                    "push-a {} {}: {} ------- {}\n",
                    similarity(&a[ai].text, &b[bi].text),
                    format_time(a[ai].start_ms),
                    a[ai].text,
                    b[bi].text
                );*/
                assert!(ai > 0);
                ai -= 1;
            }
            Some(BestChoice::PushedB) => {
                /*println!(
                    "push-b {} {}: {} ------- {}\n",
                    similarity(&a[ai].text, &b[bi].text),
                    format_time(a[ai].start_ms),
                    a[ai].text,
                    b[bi].text
                );*/
                assert!(bi > 0);
                bi -= 1;
            }
            None => {
                assert!(ai == 0);
                assert!(bi == 0);

                break;
            }
        }
    }

    result
}

#[derive(Clone, Debug)]
struct RunConfig {
    statistics_folder_path_opt: Option<PathBuf>,
    statistics_required_tags: Vec<String>,

    split_penalties: Vec<f64>,
    optimization_values: Vec<f64>,
    min_span_lengths: Vec<i64>,

    align_config: AlignConfig,
    line_match_config: LineMatchingConfig,
    sync_classification_config: SyncClassificationConfig,
    vad_config: VADConfig,
}

define_error!(TopLevelError, TopLevelErrorKind);

pub enum TopLevelErrorKind {
    ErrorReadingVideoFile { path: PathBuf },
    SerializingCacheFailed {},
}

impl fmt::Display for TopLevelErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TopLevelErrorKind::ErrorReadingVideoFile { path } => {
                write!(f, "error reading video file `{}`", path.display())
            }
            TopLevelErrorKind::SerializingCacheFailed {} => write!(f, "error serializing cache file"),
        }
    }
}

// threaded statistics and cache type
type TStatistics = Arc<Mutex<statistics::Root>>;
type TTransStatistics = Arc<Mutex<statistics::TransientRoot>>;
type TCache = Arc<Mutex<cache::Root>>;
type TSubtitle = Arc<database::Subtitle>;
type TMovie = Arc<database::Movie>;
type TProgressInfo = Arc<ProgressContext>;

fn perform_vad(movie: &database::Movie, cache: TCache) -> Result<Vec<Span>, TopLevelError> {
    let vad_spans: Vec<Span>;

    let vad_spans_opt: Option<Vec<Span>> = { cache.lock().unwrap().vad_spans.get(&movie.id).cloned() };

    match vad_spans_opt {
        Some(v) => {
            vad_spans = v;
        }
        None => {
            let video_file_handler: VideoFileHandler = VideoFileHandler::open_video_file(
                movie.path.as_path(),
                NoProgressInfo {},
                /*ProgressInfo::new(
                    500,
                    Some(format!("extracting audio from movie '{}'...", movie.path.display())),
                ),*/
            )
            .with_context(|_| TopLevelErrorKind::ErrorReadingVideoFile {
                path: movie.path.clone(),
            })?;

            vad_spans = video_file_handler
                .timespans()
                .iter()
                .map(|ts| Span {
                    start_ms: ts.start.msecs(),
                    end_ms: ts.end.msecs(),
                })
                .collect::<Vec<Span>>();

            {
                cache
                    .lock()
                    .unwrap()
                    .vad_spans
                    .insert(movie.id.clone(), vad_spans.clone());
            }
        }
    }

    Ok(vad_spans)
}

fn generate_line_pair_data(
    ref_subtitle: &database::Subtitle,
    in_subtitle: &database::Subtitle,
    cache: TCache,
    line_match_config: &LineMatchingConfig,
) -> LinePairs {
    let line_pairs: LinePairs;

    let cached_line_pairs_opt: Option<LinePairs> = {
        cache
            .lock()
            .unwrap()
            .line_pairs
            .get(&(ref_subtitle.id(), in_subtitle.id(), *line_match_config))
            .cloned()
    };

    match cached_line_pairs_opt {
        Some(v) => {
            line_pairs = v;
        }
        None => {
            line_pairs = get_line_pairs(&ref_subtitle.data, &in_subtitle.data, line_match_config);

            {
                cache.lock().unwrap().line_pairs.insert(
                    (ref_subtitle.id(), in_subtitle.id(), *line_match_config),
                    line_pairs.clone(),
                );
            }
        }
    }

    line_pairs
}

fn align(
    ref_spans: impl Iterator<Item = Span>,
    in_spans: impl Iterator<Item = Span>,
    scaling_factor: FixedPointNumber,
    config: &AlignConfig,
) -> Vec<i64> {
    let ref_alg_spans: Vec<alass_core::TimeSpan> = ref_spans
        .map(|span| span.to_alass_core_spans(config.ms_per_alg_step))
        .collect();

    let in_alg_spans: Vec<alass_core::TimeSpan> = in_spans
        .map(|span| span.to_alass_core_spans(config.ms_per_alg_step))
        .map(|span| span.scaled(scaling_factor.to_f64()))
        .collect();

    let alg_deltas;
    match config.align_mode {
        AlignMode::NoSplit => {
            let num_inc_timespancs = in_alg_spans.len();

            let (alg_delta, _score) = alass_core::align_nosplit(
                &ref_alg_spans,
                &in_alg_spans,
                get_scoring_fn(config.scoring_mode),
                alass_core::NoProgressHandler,
            );
            //println!("align score {}", score);

            alg_deltas = vec![alg_delta; num_inc_timespancs];
        }
        AlignMode::Split {
            split_penalty,
            optimization,
        } => {
            alg_deltas = alass_core::align(
                &ref_alg_spans,
                &in_alg_spans,
                split_penalty.to_f64(),
                optimization.map(FixedPointNumber::to_f64),
                get_scoring_fn(config.scoring_mode),
                alass_core::NoProgressHandler,
            )
            .0;
        }
    }

    alg_deltas_to_timing_deltas(&alg_deltas, config.ms_per_alg_step)
        .into_iter()
        .map(|td| td.msecs())
        .collect()
}

struct CorrectionInfo {
    scaling_factor: FixedPointNumber,
    deltas: Vec<i64>,
}

impl CorrectionInfo {
    fn apply_to(&self, in_spans: &[Span]) -> Vec<Span> {
        assert!(in_spans.len() == self.deltas.len());

        in_spans
            .iter()
            .cloned()
            .map(|span| span.scaled_by(self.scaling_factor.to_f64()))
            .zip(self.deltas.iter().cloned())
            .map(|(span, delta)| span.plus_delta(delta))
            .collect()
    }

    fn count_splits(&self) -> usize {
        let mut r = 0;
        for (d1, d2) in self.deltas.iter().zip(self.deltas.iter().skip(1)) {
            if d1 != d2 {
                r = r + 1;
            }
        }

        r
    }
}

fn assert_nosplit_deltas(deltas: &[i64]) -> i64 {
    assert!(!deltas.is_empty());
    let delta = deltas[0];
    for &delta2 in deltas {
        assert_eq!(delta, delta2);
    }

    delta
}

fn get_scoring_fn(scoring_mode: ScoringMode) -> impl Fn(alass_core::TimeDelta, alass_core::TimeDelta) -> f64 + Copy {
    match scoring_mode {
        ScoringMode::Standard => alass_core::standard_scoring,
        ScoringMode::Overlap => alass_core::overlap_scoring,
    }
}

fn compute_score(
    ref_spans: &[Span],
    in_spans: &[Span],
    ms_per_alg_step: i64,
    offset: i64,
    scaling_factor: FixedPointNumber,
    scoring_mode: ScoringMode,
) -> f64 {
    alass_core::get_nosplit_score(
        ref_spans
            .iter()
            .cloned()
            .map(|span| span.to_alass_core_spans(ms_per_alg_step)),
        in_spans
            .iter()
            .cloned()
            .map(|span| span.to_alass_core_spans(ms_per_alg_step))
            .map(|span| span.scaled(scaling_factor.to_f64()))
            .map(|span| span + alass_core::TimeDelta::from_i64(offset / ms_per_alg_step)),
        get_scoring_fn(scoring_mode),
    )
}

fn guess_scaling_factor(
    ref_conf: RefConfig,
    ref_spans: &[Span],
    in_sub_id: &SubtitleID,
    in_spans: &[Span],
    cache: TCache,
    ms_per_alg_step: i64,
    use_cache: bool,
) -> FixedPointNumber {
    let new_align_conf = AlignConfig {
        ms_per_alg_step,
        align_mode: AlignMode::NoSplit,
        scaling_correct_mode: ScalingCorrectMode::None,
        scoring_mode: ScoringMode::Overlap,
    };

    let ratios = [
        23.976 / 24.,
        23.976 / 25.,
        24. / 23.976,
        24. / 25.,
        25. / 23.976,
        25. / 24.,
    ];
    let corrections = compute_sync_deltas_fixed_scaling_factor(
        ref_conf.clone(),
        ref_spans,
        in_sub_id,
        in_spans,
        cache.clone(),
        FixedPointNumber::one(),
        &new_align_conf,
        use_cache,
    );
    let delta = assert_nosplit_deltas(&corrections.deltas);
    let mut opt_ratio = FixedPointNumber::one();
    let mut opt_score = compute_score(
        ref_spans,
        in_spans,
        ms_per_alg_step,
        delta,
        FixedPointNumber::one(),
        new_align_conf.scoring_mode,
    );
    for ratio in &ratios {
        let scaling_factor = FixedPointNumber::from_f64(*ratio);
        let corrections = compute_sync_deltas_fixed_scaling_factor(
            ref_conf.clone(),
            ref_spans,
            in_sub_id,
            in_spans,
            cache.clone(),
            scaling_factor,
            &new_align_conf,
            use_cache,
        );
        let delta = assert_nosplit_deltas(&corrections.deltas);

        let score = compute_score(
            ref_spans,
            in_spans,
            ms_per_alg_step,
            delta,
            scaling_factor,
            new_align_conf.scoring_mode,
        );
        if score > opt_score {
            opt_score = score;
            opt_ratio = scaling_factor;
        }
    }

    opt_ratio
}

fn compute_sync_deltas_fixed_scaling_factor(
    ref_config: RefConfig,
    ref_spans: &[Span],
    in_sub_id: &SubtitleID,
    in_spans: &[Span],
    cache: TCache,
    scaling_factor: FixedPointNumber,
    conf: &AlignConfig,
    use_cache: bool,
) -> CorrectionInfo {
    let cached_deltas_opt: Option<Vec<i64>> = {
        if use_cache {
            cache
                .lock()
                .unwrap()
                .sync_deltas
                .get(&(ref_config.clone(), in_sub_id.clone(), scaling_factor, conf.align_mode))
                .cloned()
        } else {
            None
        }
    };

    let deltas;
    match cached_deltas_opt {
        Some(v) => {
            deltas = v;
        }
        None => {
            deltas = align(
                ref_spans.iter().cloned(),
                in_spans.iter().cloned(),
                scaling_factor,
                conf,
            );

            {
                cache.lock().unwrap().sync_deltas.insert(
                    (ref_config, in_sub_id.clone(), scaling_factor, conf.align_mode),
                    deltas.clone(),
                );
            }
        }
    }

    CorrectionInfo { deltas, scaling_factor }
}

fn compute_sync_deltas(
    ref_config: RefConfig,
    ref_spans: &[Span],
    in_sub_id: &SubtitleID,
    in_spans: &[Span],
    cache: TCache,
    conf: &AlignConfig,
    use_cache: bool,
) -> CorrectionInfo {
    let scaling_factor: FixedPointNumber; // = FixedPointNumber::one();

    match conf.scaling_correct_mode {
        ScalingCorrectMode::None => scaling_factor = FixedPointNumber::one(),
        ScalingCorrectMode::Advanced => {
            scaling_factor = guess_scaling_factor(
                ref_config.clone(),
                ref_spans,
                in_sub_id,
                in_spans,
                cache.clone(),
                conf.ms_per_alg_step,
                use_cache,
            )
        }
    }

    let cached_deltas_opt: Option<Vec<i64>> = {
        if use_cache {
            cache
                .lock()
                .unwrap()
                .sync_deltas
                .get(&(ref_config.clone(), in_sub_id.clone(), scaling_factor, conf.align_mode))
                .cloned()
        } else {
            None
        }
    };

    let deltas;
    match cached_deltas_opt {
        Some(v) => {
            deltas = v;
        }
        None => {
            deltas = align(
                ref_spans.iter().cloned(),
                in_spans.iter().cloned(),
                scaling_factor,
                conf,
            );

            {
                cache.lock().unwrap().sync_deltas.insert(
                    (ref_config, in_sub_id.clone(), scaling_factor, conf.align_mode),
                    deltas.clone(),
                );
            }
        }
    }

    CorrectionInfo { deltas, scaling_factor }
}

fn update_statistics_with_alignment(
    ref_sub_spans: &[Span],
    in_spans: &[Span],
    corrections: &CorrectionInfo,
    line_pairs: &[LinePair],
    statistics: TStatistics,
    sync_ref_type: statistics::SyncReferenceType,
    update_histogram: bool,
    config: &RunConfig,
) {
    let out_video_spans: Vec<Span> = corrections.apply_to(in_spans);

    // raw_sync_classification == SyncClassification::Unsynced
    if update_histogram {
        update_distance_histogram(
            ref_sub_spans,
            &out_video_spans,
            line_pairs,
            statistics.clone(),
            Some(sync_ref_type),
        );
    }

    let video_sync_classification = get_sync_classification(
        &ref_sub_spans,
        &out_video_spans,
        &line_pairs,
        &config.sync_classification_config,
    );
    statistics
        .lock()
        .unwrap()
        .general
        .get_sync_classification_counter_mut(Some(sync_ref_type))
        .insert(video_sync_classification);
}

fn get_offsets(ref_spans: &[Span], in_spans: &[Span], line_pairs: &[LinePair]) -> Vec<i64> {
    let mut offsets = Vec::<i64>::new();

    for &(ref_line_idx, in_line_idx) in line_pairs {
        let distance_ms = Span::compute_line_distance(ref_spans[ref_line_idx], in_spans[in_line_idx]);

        offsets.push(distance_ms);
    }

    offsets
}

fn update_distance_histogram(
    ref_spans: &[Span],
    in_spans: &[Span],
    line_pairs: &[LinePair],
    statistics: TStatistics,
    ref_type_opt: Option<statistics::SyncReferenceType>,
) {
    let mut statistics = statistics.lock().unwrap();
    let histogram = statistics.get_distance_histogram_mut(ref_type_opt);

    let offsets = get_offsets(ref_spans, in_spans, line_pairs);

    for offset_ms in offsets {
        histogram.insert(offset_ms);
    }
}

/*fn get_offsets_with_deltas(ref_spans: &[Span], in_spans: &[Span], deltas: &[i64], line_pairs: &[LinePair]) -> Vec<i64> {
    line_pairs
        .iter()
        .map(|&(ref_idx, in_idx)| {
            Span::compute_line_distance(ref_spans[ref_idx], in_spans[in_idx].plus_delta(deltas[in_idx]))
        })
        .collect::<Vec<i64>>()
}

fn get_distances_for_line_info_with_deltas(
    ref_lines: &[database::LineInfo],
    in_lines: &[database::LineInfo],
    deltas: &[i64],
    line_pairs: &[LinePair],
) -> Vec<i64> {
    line_pairs
        .iter()
        .map(|&(ref_idx, in_idx)| {
            Span::compute_line_distance(
                Span::from(&ref_lines[ref_idx]),
                Span::from(&in_lines[in_idx]).plus_delta(deltas[in_idx]),
            )
        })
        .collect::<Vec<i64>>()
}*/

fn print_ignore_error_for_movie(e: impl Into<failure::Error>, movie: &database::Movie) {
    println!("<<<< Ignoring error for movie [{}; '{}']", movie.id, movie.name);
    print_error_chain(e.into());
    println!(">>>>");
}

type TDatabase = database::Root;

type TStopRequestPrio = Arc<AtomicUsize>;

fn iterate_movies<'a>(database: &'a TDatabase) -> impl Iterator<Item = (TMovie, TProgressInfo)> + 'a {
    let total_movie_count = database.movies.len();

    database.movies.iter().enumerate().map(move |(movie_nr, movie)| {
        (
            movie.clone(),
            Arc::new(ProgressContext::Movie(MovieProgressContext {
                movie_id: movie.id.clone(),
                movie_nr: movie_nr,
                total_movie_count: total_movie_count,
            })),
        )
    })
}

fn iterate_movie_subs_with_ref_sub(
    database: &TDatabase,
) -> impl Iterator<Item = (TMovie, TSubtitle, TSubtitle, TProgressInfo)> {
    let total_movie_count = database.movies.len();
    let total_sub_count = database.non_ref_sub_count();

    let mut total_sub_nr = 0;

    let mut result: Vec<(TMovie, TSubtitle, TSubtitle, TProgressInfo)> = Vec::new();

    for (movie_nr, movie) in database.movies.iter().enumerate() {
        let ref_subtitle: TSubtitle = movie.reference_subtitle.clone();
        let movie_sub_count = movie.subtitles.len();

        for (subtitle_nr, in_subtitle) in movie.subtitles.iter().cloned().enumerate() {
            let progress_info = ProgressContext::SubtitleForMovie(
                MovieProgressContext {
                    movie_id: movie.id.clone(),
                    movie_nr: movie_nr,
                    total_movie_count: total_movie_count,
                },
                SubtitleProgressContext {
                    sub_id: in_subtitle.id(),
                    sub_nr: subtitle_nr,
                    total_sub_nr: total_sub_nr,
                    movie_sub_count: movie_sub_count,
                    total_sub_count: total_sub_count,
                },
            );

            total_sub_nr = total_sub_nr + 1;

            result.push((
                movie.clone(),
                ref_subtitle.clone(),
                in_subtitle,
                Arc::new(progress_info),
            ))
        }
    }

    //use rand::seq::SliceRandom;
    //use rand::thread_rng;
    //result.shuffle(&mut thread_rng());
    result.into_iter().enumerate().map(|(i, data)| {
        let progress;
        if let ProgressContext::SubtitleForMovie(movie_prog, sub_prog) = &*data.3 {
            let mut sub_prog = sub_prog.clone();
            sub_prog.total_sub_nr = i;
            progress = Arc::new(ProgressContext::SubtitleForMovie(movie_prog.clone(), sub_prog));
        } else {
            progress = data.3;
        }
        (data.0, data.1, data.2, progress)
    })
}

fn compute_sync_offsets(
    ref_sub_spans: &[Span],
    ref_config: RefConfig,
    ref_spans: &[Span],
    in_id: &SubtitleID,
    in_spans: &[Span],
    line_pairs: &[LinePair],
    cache: TCache,
    align_config: &AlignConfig,
    use_cache: bool,
) -> Vec<i64> {
    let correction = compute_sync_deltas(ref_config, ref_spans, in_id, in_spans, cache, align_config, use_cache);

    let out_spans = correction.apply_to(in_spans);

    get_offsets(&ref_sub_spans, &out_spans, &line_pairs)
}

fn run() -> Result<(), TopLevelError> {
    let stop_request_prio: TStopRequestPrio = Arc::new(AtomicUsize::new(0));
    // save cache on ctrl-c
    {
        let stop_request_prio = stop_request_prio.clone();
        ctrlc::set_handler(move || {
            let stop_request_prio_ld = stop_request_prio.load(atomic::Ordering::SeqCst);
            println!(
                "User requested stopping of program... Waiting for processes to finish (stop prio {})...",
                stop_request_prio_ld + 1
            );
            if stop_request_prio_ld + 1 >= 11 {
                println!("FORCE EXITING!");
                std::process::exit(0);
            }
            stop_request_prio.store(stop_request_prio_ld + 1, atomic::Ordering::SeqCst);
        })
        .expect("Error setting Ctrl-C handler");
    }

    let matches = App::new("alass statistics")
        .version(PKG_VERSION.unwrap_or("unknown version (not compiled with cargo)"))
        .author("kaegi")
        .about("Generate statistics of alass for a database JSON file")
        .arg(
            Arg::with_name("database-dir")
                .long("database-dir")
                .value_name("INPUT_DATABASE_DIRECTORY")
                .help("Path to the database directory")
                .multiple(false)
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("statistics-dir")
                .long("statistics-dir")
                .value_name("OUTPUT_STATISTICS_DIRECTORY")
                .help("Path to statistics output directory")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("cache-dir")
                .long("cache-dir")
                .value_name("CACHE_DIRECOTRY")
                .multiple(false)
                .help("Path to the cache directory")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("only-every-nth-sub")
                .long("only-every-nth-sub")
                .value_name("NUMBER")
                .multiple(false)
                .help("Only synchronize every n-th subtitle; speeds up time to plots")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("only-sub")
                .long("only-sub")
                .alias("only-sub-id")
                .alias("only-subtitle")
                .alias("only-subtitle-id")
                .value_name("SUBTITLE_ID")
                .multiple(true)
                .help("Only synchronize the given subtitle")
                .takes_value(true)
        )
        .arg(
            Arg::with_name("only-movie")
                .long("only-movie")
                .alias("only-movie-id")
                .value_name("MOVIE_ID")
                .multiple(true)
                .help("Only synchronize the given movie")
                .takes_value(true)
        )
        .arg(
            Arg::with_name("ignore-movie")
                .long("ignore-movie")
                .alias("ignore-movie-id")
                .value_name("MOVIE_ID")
                .multiple(true)
                .help("Exclude movies by id")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("ignore-sub")
                .long("ignore-sub")
                .alias("ignore-sub-id")
                .value_name("SUB_ID")
                .multiple(true)
                .help("Exclude subtitles by id")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("quiet")
                .long("quiet")
                .short("q")
                .multiple(false)
                .help("Suppress unnecessary output")
        )
        .arg(
            Arg::with_name("clean-cache")
                .long("clean-cache")
                .short("c")
                .multiple(false)
                .help("Clean/Overwrite the cache file")
                .requires("cache-dir"),
        )
        .arg(
            Arg::with_name("clean-cache-line-pairs")
                .long("clean-cache-line-pairs")
                .multiple(false)
                .help("Clean/Overwrite the line pairing/matching in the cache file")
                .requires("cache-dir"),
        )
        .arg(
            Arg::with_name("clean-cache-vad")
                .long("clean-cache-vad")
                .multiple(false)
                .help("Clean/Overwrite the Voice-Activity-Detection spans in the cache file")
                .requires("cache-dir"),
        )
        .arg(
            Arg::with_name("clean-cache-deltas")
                .long("clean-cache-deltas")
                .multiple(false)
                .help("Clean/Overwrite all alignment delta data in the cache file")
                .requires("cache-dir"),
        )
        .arg(
            Arg::with_name("include-synced-subs-in-distance-histogram")
                .long("include-synced-subs-in-distance-histogram")
                .multiple(false)
                .help("Include subtitles that have a classification of being synced without alass in 'distance to reference histogram'")
        )
        .arg(
            Arg::with_name("default-split-penalty")
                .long("default-split-penalty")
                .default_value("0.5")
                .takes_value(true)
                .multiple(false)
        )
        .arg(
            Arg::with_name("only-general-statistics")
                .long("only-general-statistics")
                .help("Option to only generate statistics for default configuration for each subtitle")
        )
        .arg(
            Arg::with_name("transient-statistics")
                .long("transient-statistics")
                .help("Also collect statistics on RAM, runtime performance, ... of the algorithms")
        )
        .arg(
            Arg::with_name("only-transient-statistics")
                .long("only-transient-statistics")
                .help("Only collect statistics on RAM, runtime performance, ... of the algorithms")
        )
        .arg(
            Arg::with_name("split-penalties")
                .long("split-penalties")
                .default_value("0.25,0.5,1,2,3,4,5,6,7,8,9,10,20,30")
                .help("comma separated float values")
                .takes_value(true)
                .multiple(true)
        )
        .arg(
            Arg::with_name("optimization-values")
                .long("optimization-values")
                .default_value("0.5,1,2,3,4,5")
                .help("comma separated float values")
                .takes_value(true)
                .multiple(true)
        )
        .arg(
            Arg::with_name("default-optimization")
                .long("default-optimization")
                .default_value("1.5")
                .takes_value(true)
                .multiple(false)
        )
        .arg(
            Arg::with_name("default-min-span-length")
                .long("default-min-span-length")
                .default_value("300")
                .takes_value(true)
                .multiple(false)
        )
        .arg(
            Arg::with_name("min-span-lengths")
                .long("min-span-lengths")
                .default_value("100,200,300,400,500,600,800,1000")
                .takes_value(true)
                .multiple(false)
        )
        .arg(
            Arg::with_name("default-max-good-sync-offset")
                .long("default-max-good-sync-offset")
                .default_value("200")
                .takes_value(true)
                .multiple(false)
        )
        .arg(
            Arg::with_name("default-required-good-sync-spans-percentage")
                .long("default-required-good-sync-spans-percentage")
                .default_value("95")
                .takes_value(true)
                .multiple(false)
        )
        .arg(
            Arg::with_name("num-threads")
                .long("num-threads")
                .default_value("4")
                .takes_value(true)
                .multiple(false)
        )
        .get_matches();

    let database_path = matches.value_of_os("database-dir").expect("missing database path");
    let output_dir: PathBuf = matches
        .value_of_os("statistics-dir")
        .expect("missing output statistics directory path")
        .into();
    let cache_dir: Option<PathBuf> = matches.value_of_os("cache-dir").map(|v| v.into());
    let clean_cache: bool = matches.is_present("clean-cache");
    let clean_cache_vad: bool = matches.is_present("clean-cache-vad");

    let clean_cache_deltas: bool = matches.is_present("clean-cache-deltas");
    let quiet: bool = matches.is_present("quiet");
    let clean_cache_line_pairs: bool = matches.is_present("clean-cache-line-pairs");

    let only_subtitles_opt: Option<HashSet<SubtitleID>> =
        matches.values_of("only-sub").map(|vs| vs.map(String::from).collect());
    let only_movies_opt: Option<HashSet<SubtitleID>> =
        matches.values_of("only-movie").map(|vs| vs.map(String::from).collect());
    let only_general_statistics: bool = matches.is_present("only-general-statistics");

    let distance_histogram_includes_synced_subtitles = matches.is_present("include-synced-subs-in-distance-histogram");

    // statistics on performance, RAM, ...
    let only_gather_transient_statistics = matches.is_present("only-transient-statistics");
    let gather_transient_statistics = matches.is_present("transient-statistics") || only_gather_transient_statistics;

    let default_split_penalty: f64 = value_t!(matches, "default-split-penalty", f64).unwrap();
    let default_optimization: f64 = value_t!(matches, "default-optimization", f64).unwrap();
    let default_min_span_length_ms: i64 = value_t!(matches, "default-min-span-length", i64).unwrap();
    let max_good_sync_offsets: Vec<i64> = matches
        .value_of("default-max-good-sync-offset")
        .unwrap()
        .split(',')
        .map(|v| v.parse::<i64>().unwrap())
        .collect();
    let required_good_sync_spans_percentages: Vec<f64> = matches
        .value_of("default-required-good-sync-spans-percentage")
        .unwrap()
        .split(',')
        .map(|v| v.parse::<f64>().unwrap())
        .collect();
    let split_penalties: Vec<f64> = matches
        .value_of("split-penalties")
        .unwrap()
        .split(',')
        .map(|v| v.trim().parse::<f64>().unwrap())
        .collect();
    let optimization_values: Vec<f64> = matches
        .value_of("optimization-values")
        .unwrap()
        .split(',')
        .map(|v| v.trim().parse::<f64>().unwrap())
        .collect();
    let min_span_lengths: Vec<i64> = matches
        .value_of("min-span-lengths")
        .unwrap()
        .split(',')
        .map(|v| v.trim().parse::<i64>().unwrap())
        .collect();

    let num_threads: usize = value_t!(matches, "num-threads", usize).unwrap();

    let ignored_movies: HashSet<String> = matches
        .values_of("ignore-movie")
        .map(|v| v.map(|x| x.to_string()).collect::<HashSet<String>>())
        .unwrap_or_else(|| HashSet::new());

    let ignored_subs: HashSet<String> = matches
        .values_of("ignore-sub")
        .map(|v| v.map(|x: &str| x.to_string()).collect::<HashSet<String>>())
        .unwrap_or_else(|| HashSet::new());

    let only_every_nth_sub: Option<usize> = value_t!(matches, "only-every-nth-sub", usize).ok();

    let json_file_path = Path::new(database_path).join("database.json");
    let json_file = File::open(json_file_path).expect("database file not found");
    let file_reader = BufReader::with_capacity(1024, json_file);
    let mut database: database::Root = serde_json::from_reader(file_reader).expect("error while reading json");
    for subtitle in database.all_subtitles_iter_mut() {
        for line in &mut subtitle.data {
            if line.start_ms > line.end_ms {
                std::mem::swap(&mut line.start_ms, &mut line.end_ms);
            }
        }

        subtitle.data.sort_by_key(|line| line.start_ms);
    }

    let mut cache: cache::Root;

    if clean_cache {
        cache = cache::Root::default();
        println!("Cleaning cache as requested by user...");
    } else {
        if let Some(cache_dir) = &cache_dir {
            let cache_file_path = cache_dir.join("cache.dat");
            if cache_file_path.exists() {
                let file = File::open(cache_file_path).expect("cache file not found");
                let file_reader = BufReader::with_capacity(1024, file);
                cache = rmps::from_read(file_reader).expect("error while reading chache file");
            } else {
                cache = cache::Root::default();
                println!("`{}` not found - creating cache file...", cache_file_path.display());
            }
        } else {
            cache = cache::Root::default();
        }
    }

    if clean_cache_line_pairs {
        cache.line_pairs = Default::default();
    }

    if clean_cache_deltas {
        cache.sync_deltas = Default::default();
    }

    if clean_cache_vad {
        cache.vad_spans = Default::default();
    }

    let thread_pool = ThreadPool::new(num_threads);

    let tasks_info = Arc::new(RunningTasksInfo::new(quiet));

    let mut statistics: statistics::Root = statistics::Root::default();
    statistics.general.total_movie_count = database.movies.len();
    statistics.general.total_subtitles_count = database.total_sub_count();
    statistics.general.movie_with_ref_sub_count = database.movies.len() - database.movies_without_reference_sub_count;

    let default_align_conf = AlignConfig {
        align_mode: AlignMode::Split {
            split_penalty: FixedPointNumber::from_f64(default_split_penalty),
            optimization: if default_optimization > 0.0 {
                Some(FixedPointNumber::from_f64(default_optimization))
            } else {
                None
            },
        },
        ms_per_alg_step: 1,
        scaling_correct_mode: ScalingCorrectMode::Advanced,
        scoring_mode: ScoringMode::Standard,
    };

    let default_line_match_conf = LineMatchingConfig {
        certain_unmatch_similarity: FixedPointNumber::from_f64(0.5),
        certain_match_similarity: FixedPointNumber::from_f64(0.8),
    };

    assert!(max_good_sync_offsets.len() == required_good_sync_spans_percentages.len());

    let default_sync_classificiation_conf = SyncClassificationConfig {
        required_segments_for_sync_classification: 10,
        good_sync_requirements: max_good_sync_offsets
            .into_iter()
            .zip(required_good_sync_spans_percentages.into_iter())
            .map(|(max_offset, percentage)| GoodSyncRequirement {
                at_least_proportion_of_all_subs: FixedPointNumber::from_f64(percentage / 100.0),
                at_most_offset: max_offset,
            })
            .collect(),
    };

    let default_vad_conf = VADConfig {
        min_span_length_ms: default_min_span_length_ms,
    };

    let config: RunConfig = RunConfig {
        statistics_folder_path_opt: None,
        statistics_required_tags: Vec::new(),

        split_penalties,
        optimization_values,
        min_span_lengths,

        align_config: default_align_conf,

        line_match_config: default_line_match_conf,

        vad_config: default_vad_conf,

        sync_classification_config: default_sync_classificiation_conf,
    };

    if let Some(only_movies) = only_movies_opt {
        database.movies = database
            .movies
            .into_iter()
            .filter(|movie| only_movies.contains(&movie.id))
            .collect();
    }

    database.movies = database
        .movies
        .into_iter()
        .filter(|movie| !ignored_movies.contains(&movie.id))
        .collect();

    if let Some(only_subtitles) = only_subtitles_opt {
        for movie in &mut database.movies {
            let movie_mut = Arc::get_mut(movie).unwrap();

            let mut i = 0;
            while i != movie_mut.subtitles.len() {
                if !only_subtitles.contains(&movie_mut.subtitles[i].id) {
                    movie_mut.subtitles.remove(i);
                } else {
                    i += 1;
                }
            }
        }
    }

    for movie in &mut database.movies {
        let movie_mut = Arc::get_mut(movie).unwrap();

        let mut i = 0;
        while i != movie_mut.subtitles.len() {
            if ignored_subs.contains(&movie_mut.subtitles[i].id) {
                movie_mut.subtitles.remove(i);
            } else {
                i += 1;
            }
        }
    }

    if let Some(only_every_nth_sub) = only_every_nth_sub {
        let mut nth = 0;

        for movie in &mut database.movies {
            let movie_mut = Arc::get_mut(movie).unwrap();

            let mut i = 0;
            while i != movie_mut.subtitles.len() {
                if nth % only_every_nth_sub != 0 {
                    movie_mut.subtitles.remove(i);
                } else {
                    i += 1;
                }

                nth = nth + 1;
            }
        }
    }

    for subtitle in database.all_subtitles_iter() {
        if stop_request_prio.load(atomic::Ordering::SeqCst) >= 1 {
            break;
        }

        for line in &subtitle.data {
            let len_ms = line.end_ms - line.start_ms;
            statistics.subtitle_span_length_histogram.insert(len_ms.abs());
        }
    }

    let statistics = Arc::new(Mutex::new(statistics));
    let cache = Arc::new(Mutex::new(cache));

    if !only_gather_transient_statistics {
        for (movie, progress_info) in iterate_movies(&database) {
            // Arc clones
            let stop_request_prio: Arc<_> = stop_request_prio.clone();
            let cache: Arc<_> = cache.clone();
            let statistics: Arc<_> = statistics.clone();
            let tasks_info = tasks_info.clone();

            thread_pool.execute(move || {
                if stop_request_prio.load(atomic::Ordering::SeqCst) >= 1 {
                    return;
                }

                let vad_span_opt = tasks_info.run("perform vad", progress_info, || perform_vad(&movie, cache.clone()));

                let vad_spans: Vec<Span>;
                match vad_span_opt {
                    Ok(v) => vad_spans = v,
                    Err(e) => {
                        print_ignore_error_for_movie(e, &movie);
                        return;
                    }
                }

                {
                    let mut statistics = statistics.lock().unwrap();

                    for &vad_span in &vad_spans {
                        let len_ms = vad_span.len_ms();
                        statistics.vad_span_length_histogram.insert(len_ms.abs());
                    }
                }
            });
        }

        thread_pool.join();

        for (movie, ref_subtitle, in_subtitle, progress_info) in iterate_movie_subs_with_ref_sub(&database) {
            if stop_request_prio.load(atomic::Ordering::SeqCst) >= 1 {
                break;
            }

            // cloning data
            let config: RunConfig = config.clone(); // TODO: use Arc

            // cloning Arc<T>
            let statistics: TStatistics = statistics.clone();
            let cache: TCache = cache.clone();
            let stop_request_prio: TStopRequestPrio = stop_request_prio.clone();
            let in_subtitle: TSubtitle = in_subtitle.clone();
            let ref_subtitle: TSubtitle = ref_subtitle.clone();
            let movie: TMovie = movie.clone();
            let tasks_info = tasks_info.clone();

            thread_pool.execute(move || {
                if stop_request_prio.load(atomic::Ordering::SeqCst) >= 1 {
                    return;
                }

                let vad_spans_raw: Vec<Span> = cache.lock().unwrap().get_vad_spans_raw(&movie.id);
                let vad_spans = config.vad_config.applied_to(&vad_spans_raw);

                let line_pairs = tasks_info.run("generate line pairs", progress_info.clone(), || {
                    generate_line_pair_data(&ref_subtitle, &in_subtitle, cache.clone(), &config.line_match_config)
                });

                if stop_request_prio.load(atomic::Ordering::SeqCst) >= 2 {
                    return;
                }

                let ref_sub_spans: Vec<Span> = ref_subtitle.data.iter().map(Span::from).collect();
                let in_spans: Vec<Span> = in_subtitle.data.iter().map(Span::from).collect();

                let raw_sync_classification = get_sync_classification(
                    &ref_sub_spans,
                    &in_spans,
                    &line_pairs,
                    &config.sync_classification_config,
                );

                {
                    statistics
                        .lock()
                        .unwrap()
                        .general
                        .get_sync_classification_counter_mut(None)
                        .insert(raw_sync_classification);
                }

                if raw_sync_classification == SyncClassification::Unknown {
                    // we can't use this subtilte
                    let mut statistics = statistics.lock().unwrap();
                    statistics
                        .general
                        .get_sync_classification_counter_mut(Some(statistics::SyncReferenceType::Video))
                        .insert(raw_sync_classification);

                    statistics
                        .general
                        .get_sync_classification_counter_mut(Some(statistics::SyncReferenceType::Subtitle))
                        .insert(raw_sync_classification);

                    return;
                }

                if guess_scaling_factor(
                    RefConfig::Subtitle(ref_subtitle.id()),
                    &ref_sub_spans,
                    &in_subtitle.id,
                    &in_spans,
                    cache.clone(),
                    config.align_config.ms_per_alg_step,
                    true,
                ) != FixedPointNumber::one()
                {
                    let mut statistics = statistics.lock().unwrap();
                    statistics.general.required_framerate_adjustments =
                        statistics.general.required_framerate_adjustments + 1;
                }

                {
                    let mut statistics = statistics.lock().unwrap();
                    // lines are counted on the reference subtitle as well as on the input subtitle
                    (*statistics).general.used_line_candidates =
                        statistics.general.used_line_candidates + 2 * line_pairs.len();
                    (*statistics).general.total_line_candidates =
                        statistics.general.total_line_candidates + ref_subtitle.data.len() + in_subtitle.data.len();
                }

                if distance_histogram_includes_synced_subtitles
                    || raw_sync_classification == SyncClassification::Unsynced
                {
                    update_distance_histogram(&ref_sub_spans, &in_spans, &line_pairs, statistics.clone(), None);
                }

                if stop_request_prio.load(atomic::Ordering::SeqCst) >= 2 {
                    return;
                }

                for (ref_config, ref_spans) in RefConfig::iter_from(
                    movie.id.clone(),
                    &vad_spans,
                    config.vad_config,
                    ref_subtitle.id(),
                    &ref_sub_spans,
                ) {
                    if stop_request_prio.load(atomic::Ordering::SeqCst) >= 2 {
                        return;
                    }

                    let ref_type = ref_config.as_ref_type();

                    let corrections: CorrectionInfo = tasks_info.run(
                        format!("compute {:?} sync deltas", ref_config.as_ref_type()),
                        progress_info.clone(),
                        || {
                            compute_sync_deltas(
                                ref_config,
                                ref_spans,
                                &in_subtitle.id,
                                &in_spans,
                                cache.clone(),
                                &config.align_config,
                                true,
                            )
                        },
                    );

                    //let synced_spans = corrections.apply_to(&in_spans);

                    /*let video_sync_classification = get_sync_classification(
                        &ref_spans,
                        &synced_spans,
                        &line_pairs,
                        &config.sync_classification_config,
                    );*/

                    update_statistics_with_alignment(
                        &ref_sub_spans,
                        &in_spans,
                        &corrections,
                        &line_pairs,
                        statistics.clone(),
                        ref_type,
                        distance_histogram_includes_synced_subtitles
                            || raw_sync_classification == SyncClassification::Unsynced,
                        &config,
                    );
                }

                if stop_request_prio.load(atomic::Ordering::SeqCst) >= 2 {
                    return;
                }

                if line_pairs.len() > 0 {
                    let sub_sync_correction = compute_sync_deltas(
                        RefConfig::Subtitle(ref_subtitle.id()),
                        &vad_spans,
                        &in_subtitle.id,
                        &in_spans,
                        cache.clone(),
                        &config.align_config,
                        true,
                    );

                    let sub_sync_spans = sub_sync_correction.apply_to(&in_spans);
                    let sub_sync_classification = get_sync_classification(
                        &ref_sub_spans,
                        &sub_sync_spans,
                        &line_pairs,
                        &config.sync_classification_config,
                    );

                    let video_sync_correction = compute_sync_deltas(
                        RefConfig::Video(movie.id.clone(), config.vad_config),
                        &vad_spans,
                        &in_subtitle.id,
                        &in_spans,
                        cache.clone(),
                        &config.align_config,
                        true,
                    );

                    let video_sync_spans = video_sync_correction.apply_to(&in_spans);
                    let video_sync_classification = get_sync_classification(
                        &ref_sub_spans,
                        &video_sync_spans,
                        &line_pairs,
                        &config.sync_classification_config,
                    );

                    let raw_offsets =
                        statistics::BoxPlotData::try_from(get_offsets(&ref_sub_spans, &in_spans, &line_pairs)).unwrap();
                    let video_sync_offsets =
                        statistics::BoxPlotData::try_from(get_offsets(&ref_sub_spans, &video_sync_spans, &line_pairs))
                            .unwrap();
                    let sub_sync_offsets =
                        statistics::BoxPlotData::try_from(get_offsets(&ref_sub_spans, &sub_sync_spans, &line_pairs))
                            .unwrap();

                    let offset_by_subtitle = statistics::OffsetBySubtitle {
                        sub_id: in_subtitle.id(),
                        num_line_pairs: line_pairs.len(),
                        ref_line_count: ref_sub_spans.len(),
                        in_line_count: in_spans.len(),

                        num_video_sync_splits: video_sync_correction.count_splits(),
                        num_sub_sync_splits: sub_sync_correction.count_splits(),

                        raw_sync_classification: raw_sync_classification,
                        video_sync_classification: video_sync_classification,
                        sub_sync_classification: sub_sync_classification,

                        raw_offsets: raw_offsets,
                        video_sync_offsets: video_sync_offsets,
                        sub_sync_offsets: sub_sync_offsets,
                    };

                    statistics.lock().unwrap().offset_by_subtitle.push(offset_by_subtitle);
                }

                if only_general_statistics {
                    return;
                }

                if stop_request_prio.load(atomic::Ordering::SeqCst) >= 2 {
                    return;
                }

                for &split_penalty in &config.split_penalties {
                    let split_penalty_fp = FixedPointNumber::from_f64(split_penalty);

                    let custom_align_config = config.align_config.with_split_penalty(split_penalty_fp);

                    for (ref_config, ref_spans) in RefConfig::iter_from(
                        movie.id.clone(),
                        &vad_spans,
                        config.vad_config,
                        ref_subtitle.id(),
                        &ref_sub_spans,
                    ) {
                        if stop_request_prio.load(atomic::Ordering::SeqCst) >= 3 {
                            return;
                        }

                        let cache = cache.clone();
                        let ref_type = ref_config.as_ref_type();

                        let offsets: Vec<i64> = tasks_info.run(
                            format!("compute sync deltas ({:?}, split penalty {})", ref_type, split_penalty),
                            progress_info.clone(),
                            || {
                                compute_sync_offsets(
                                    &ref_sub_spans,
                                    ref_config,
                                    ref_spans,
                                    &in_subtitle.id,
                                    &in_spans,
                                    &line_pairs,
                                    cache,
                                    &custom_align_config,
                                    true,
                                )
                            },
                        );

                        {
                            let mut statistics = statistics.lock().unwrap();
                            statistics
                                .sync_offset_histogram_by_split_penalty
                                .entry(ref_type)
                                .or_default()
                                .entry(split_penalty_fp)
                                .or_default()
                                .insert_all(&offsets);
                        }
                    }
                }

                for &optimization_value in &config.optimization_values {
                    let optimization_value_fp = FixedPointNumber::from_f64(optimization_value);

                    let custom_align_config = config.align_config.with_optimization(Some(optimization_value_fp));

                    for (ref_config, ref_spans) in RefConfig::iter_from(
                        movie.id.clone(),
                        &vad_spans,
                        config.vad_config,
                        ref_subtitle.id(),
                        &ref_sub_spans,
                    ) {
                        if stop_request_prio.load(atomic::Ordering::SeqCst) >= 3 {
                            return;
                        }
                        let cache = cache.clone();
                        let ref_type = ref_config.as_ref_type();

                        let offsets: Vec<i64> = tasks_info.run(
                            format!(
                                "compute sync deltas ({:?}, optimization {})",
                                ref_config.as_ref_type(),
                                optimization_value
                            ),
                            progress_info.clone(),
                            || {
                                compute_sync_offsets(
                                    &ref_sub_spans,
                                    ref_config,
                                    ref_spans,
                                    &in_subtitle.id,
                                    &in_spans,
                                    &line_pairs,
                                    cache,
                                    &custom_align_config,
                                    true,
                                )
                            },
                        );

                        {
                            let mut statistics = statistics.lock().unwrap();
                            statistics
                                .sync_offset_histogram_by_optimization
                                .entry(ref_type)
                                .or_default()
                                .entry(optimization_value_fp)
                                .or_default()
                                .insert_all(&offsets);
                        }
                    }
                }

                for &min_span_length in &config.min_span_lengths {
                    if stop_request_prio.load(atomic::Ordering::SeqCst) >= 3 {
                        return;
                    }

                    let mut custom_vad_config = config.vad_config.clone();
                    custom_vad_config.min_span_length_ms = min_span_length;
                    let custom_vad_spans = custom_vad_config.applied_to(&vad_spans_raw);

                    let offsets: Vec<i64> = tasks_info.run(
                        format!("try min span lengths {}", min_span_length,),
                        progress_info.clone(),
                        || {
                            compute_sync_offsets(
                                &ref_sub_spans,
                                RefConfig::Video(movie.id.clone(), custom_vad_config),
                                &custom_vad_spans,
                                &in_subtitle.id,
                                &in_spans,
                                &line_pairs,
                                cache.clone(),
                                &config.align_config,
                                true,
                            )
                        },
                    );

                    {
                        let mut statistics = statistics.lock().unwrap();
                        statistics
                            .sync_offset_histogram_by_min_span_length
                            .entry(min_span_length)
                            .or_default()
                            .insert_all(&offsets);
                    }
                }
            });
        }

        thread_pool.join();

        for (movie, ref_subtitle, in_subtitle, progress_info) in iterate_movie_subs_with_ref_sub(&database) {
            if stop_request_prio.load(atomic::Ordering::SeqCst) >= 1 {
                break;
            }

            if only_general_statistics {
                break;
            }

            // cloning data
            let config: RunConfig = config.clone(); // TODO: use Arc

            // cloning Arc<T>
            let statistics: TStatistics = statistics.clone();
            let cache: TCache = cache.clone();
            let stop_request_prio: TStopRequestPrio = stop_request_prio.clone();
            let in_subtitle: TSubtitle = in_subtitle.clone();
            let ref_subtitle: TSubtitle = ref_subtitle.clone();
            let movie: TMovie = movie.clone();
            let tasks_info = tasks_info.clone();

            thread_pool.execute(move || {
                if stop_request_prio.load(atomic::Ordering::SeqCst) >= 1 {
                    return;
                }

                let vad_spans_raw: Vec<Span> = cache.lock().unwrap().get_vad_spans_raw(&movie.id);
                let vad_spans = config.vad_config.applied_to(&vad_spans_raw);

                let line_pairs: LinePairs = {
                    cache
                        .lock()
                        .unwrap()
                        .line_pairs
                        .get(&(ref_subtitle.id(), in_subtitle.id(), config.line_match_config))
                        .cloned()
                        .unwrap()
                };

                if stop_request_prio.load(atomic::Ordering::SeqCst) >= 2 {
                    return;
                }

                let ref_sub_spans: Vec<Span> = ref_subtitle.data.iter().map(Span::from).collect();
                let in_spans: Vec<Span> = in_subtitle.data.iter().map(Span::from).collect();

                for scaling_correct_mode in ScalingCorrectMode::iter() {
                    for algorithm_variant in statistics::AlgorithmVariant::iter() {
                        for (ref_config, ref_spans) in RefConfig::iter_from(
                            movie.id.clone(),
                            &vad_spans,
                            config.vad_config,
                            ref_subtitle.id(),
                            &ref_sub_spans,
                        ) {
                            if stop_request_prio.load(atomic::Ordering::SeqCst) >= 3 {
                                return;
                            }
                            let custom_align_config: AlignConfig = AlignConfig {
                                align_mode: match algorithm_variant {
                                    statistics::AlgorithmVariant::Nosplit => AlignMode::NoSplit,
                                    statistics::AlgorithmVariant::Split => AlignMode::Split {
                                        split_penalty: FixedPointNumber::from_f64(default_split_penalty),
                                        optimization: Some(FixedPointNumber::from_f64(default_optimization)),
                                    },
                                },
                                ms_per_alg_step: config.align_config.ms_per_alg_step,
                                scaling_correct_mode: *scaling_correct_mode,
                                scoring_mode: config.align_config.scoring_mode,
                            };
                            let cache = cache.clone();
                            let ref_type = ref_config.as_ref_type();
                            let offsets = tasks_info.run(
                                format!(
                                    "compute sync deltas ({:?}, {:?}, {:?})",
                                    ref_type, scaling_correct_mode, algorithm_variant
                                ),
                                progress_info.clone(),
                                || {
                                    compute_sync_offsets(
                                        &ref_sub_spans,
                                        ref_config,
                                        ref_spans,
                                        &in_subtitle.id,
                                        &in_spans,
                                        &line_pairs,
                                        cache,
                                        &custom_align_config,
                                        true,
                                    )
                                },
                            );

                            let mut histogram = statistics::Histogram::default();
                            histogram.insert_all(&offsets);

                            {
                                let algorithm_conf = statistics::AlgorithmConf {
                                    sync_ref_type: ref_type,
                                    algorithm_variant: *algorithm_variant,
                                    scaling_correct_mode: *scaling_correct_mode,
                                };

                                let mut statistics = statistics.lock().unwrap();
                                statistics
                                    .all_configurations_offset_histogram
                                    .inner_mut()
                                    .entry(algorithm_conf)
                                    .or_insert_with(|| statistics::Histogram::default())
                                    .merge_from(&histogram);
                            }
                        }
                    }
                }
            });
        }

        thread_pool.join();

        /*{
            let mut statistics = statistics.lock().unwrap();

            statistics.offset_by_subtitle.sort_by_key(|offset_statistics| {
                Reverse((
                    offset_statistics.sub_sync_offsets.perc90,
                    offset_statistics.video_sync_offsets.perc90,
                    offset_statistics.raw_offsets.perc90,
                ))
            });

            for d in statistics.offset_by_subtitle.iter().rev() {
                println!("{:#?}", d);
            }
        }*/

        println!("");
        println!("Done computing!");
        println!("Writing cache...");

        if let Some(cache_dir) = &cache_dir {
            std::fs::create_dir_all(&cache_dir).expect("failed to create cache dir");

            let file = File::create(cache_dir.join("cache.dat")).expect("cache file not found");
            let mut file_write = BufWriter::with_capacity(1024, file);
            rmps::encode::write_named(&mut file_write, &*cache.lock().unwrap())
                .with_context(|_| TopLevelErrorKind::SerializingCacheFailed {})?;
        }

        std::fs::create_dir_all(&output_dir).expect("failed to create statistics dir");

        println!("Writing statistics...");

        {
            let statistics_file =
                File::create(output_dir.join("statistics.json")).expect("statistics file could not be created");
            let file_write = BufWriter::with_capacity(1024, statistics_file);
            serde_json::to_writer_pretty(file_write, &*statistics.lock().unwrap()).expect("writing statistics failed");
        }
    }

    if gather_transient_statistics {
        println!("Gather transient statistics...");

        let transient_statistics: TTransStatistics = Arc::new(Mutex::new(statistics::TransientRoot::default()));

        for (movie, ref_subtitle, in_subtitle, progress_info) in iterate_movie_subs_with_ref_sub(&database) {
            let vad_spans_raw: Vec<Span> = cache.lock().unwrap().get_vad_spans_raw(&movie.id);
            let vad_spans = config.vad_config.applied_to(&vad_spans_raw);

            if stop_request_prio.load(atomic::Ordering::SeqCst) >= 2 {
                break;
            }

            let ref_sub_spans: Vec<Span> = ref_subtitle.data.iter().map(Span::from).collect();
            let in_spans: Vec<Span> = in_subtitle.data.iter().map(Span::from).collect();

            for &optimization_value in &config.optimization_values {
                let optimization_value_fp = FixedPointNumber::from_f64(optimization_value);

                let custom_align_config = config.align_config.with_optimization(Some(optimization_value_fp));

                if stop_request_prio.load(atomic::Ordering::SeqCst) >= 3 {
                    break;
                }
                let cache = cache.clone();

                let duration = tasks_info.run(
                    format!("gather time for (optimization {})", optimization_value),
                    progress_info.clone(),
                    || {
                        let start_time = std::time::Instant::now();
                        compute_sync_deltas(
                            RefConfig::Subtitle(ref_subtitle.id()),
                            &ref_sub_spans,
                            &in_subtitle.id,
                            &in_spans,
                            cache,
                            &custom_align_config,
                            false,
                        );
                        let end_time = std::time::Instant::now();
                        end_time - start_time
                    },
                );

                {
                    let mut transient_statistics = transient_statistics.lock().unwrap();
                    transient_statistics
                        .time_required_by_optimization_value
                        .inner_mut()
                        .entry(optimization_value_fp)
                        .or_default()
                        .push(duration.as_millis().try_into().unwrap());
                }
            }

            for scaling_correct_mode in ScalingCorrectMode::iter() {
                for algorithm_variant in statistics::AlgorithmVariant::iter() {
                    for (ref_config, ref_spans) in RefConfig::iter_from(
                        movie.id.clone(),
                        &vad_spans,
                        config.vad_config,
                        ref_subtitle.id(),
                        &ref_sub_spans,
                    ) {
                        if stop_request_prio.load(atomic::Ordering::SeqCst) >= 3 {
                            break;
                        }
                        let custom_align_config: AlignConfig = AlignConfig {
                            align_mode: match algorithm_variant {
                                statistics::AlgorithmVariant::Nosplit => AlignMode::NoSplit,
                                statistics::AlgorithmVariant::Split => AlignMode::Split {
                                    split_penalty: FixedPointNumber::from_f64(default_split_penalty),
                                    optimization: Some(FixedPointNumber::from_f64(default_optimization)),
                                },
                            },
                            ms_per_alg_step: config.align_config.ms_per_alg_step,
                            scaling_correct_mode: *scaling_correct_mode,
                            scoring_mode: config.align_config.scoring_mode,
                        };

                        let cache = cache.clone();
                        let ref_type = ref_config.as_ref_type();
                        let duration = tasks_info.run(
                            format!(
                                "gather time for algorithm ({:?}, {:?}, {:?})",
                                ref_type, scaling_correct_mode, algorithm_variant
                            ),
                            progress_info.clone(),
                            || {
                                let start_time = std::time::Instant::now();
                                compute_sync_deltas(
                                    ref_config,
                                    ref_spans,
                                    &in_subtitle.id,
                                    &in_spans,
                                    cache,
                                    &custom_align_config,
                                    false,
                                );
                                let end_time = std::time::Instant::now();
                                end_time - start_time
                            },
                        );

                        {
                            let algorithm_conf = statistics::AlgorithmConf {
                                sync_ref_type: ref_type,
                                algorithm_variant: *algorithm_variant,
                                scaling_correct_mode: *scaling_correct_mode,
                            };

                            let mut transient_statistics = transient_statistics.lock().unwrap();
                            transient_statistics
                                .time_required_by_algorithm
                                .inner_mut()
                                .entry(algorithm_conf)
                                .or_default()
                                .push(duration.as_millis().try_into().unwrap());
                        }
                    }
                }
            }
        }

        println!("Writing transient statistics...");

        {
            let statistics_file = File::create(output_dir.join("transient-statistics.json"))
                .expect("transient statistics file could not be created");
            let file_write = BufWriter::with_capacity(1024, statistics_file);
            serde_json::to_writer_pretty(file_write, &*transient_statistics.lock().unwrap())
                .expect("writing transient statistics file failed");
        }
    }

    println!("Done writing!");

    Ok(())
}

fn main() {
    match run() {
        Ok(_) => std::process::exit(0),
        Err(error) => {
            print_error_chain(error.into());
            std::process::exit(1)
        }
    }
}

// interpreted by python script
mod statistics {

    use serde::{Deserialize, Serialize, Serializer};
    use std::collections::HashMap;
    use std::convert::TryFrom;
    use std::hash::Hash;

    use super::types::*;

    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Deserialize, Serialize)]
    pub struct BoxPlotData {
        pub max: i64,
        pub perc99: i64,
        pub perc90: i64,
        pub avg: i64,
        pub median: i64,
        pub perc10: i64,
        pub perc1: i64,
        pub min: i64,
    }

    impl TryFrom<Vec<i64>> for BoxPlotData {
        type Error = ();

        fn try_from(mut v: Vec<i64>) -> Result<BoxPlotData, ()> {
            if v.is_empty() {
                return Err(());
            }

            v.sort_unstable();

            let total: i64 = v.iter().cloned().sum();

            let len = v.len();
            Ok(BoxPlotData {
                min: v[0],
                perc1: v[(len * 1) / 100],
                perc10: v[(len * 10) / 100],
                median: v[len / 2],
                avg: total / len as i64,
                perc90: v[(len * 90) / 100],
                perc99: v[(len * 99) / 100],
                max: v[len - 1],
            })
        }
    }

    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Deserialize, Serialize)]
    pub enum SyncReferenceType {
        Video,
        Subtitle,
    }

    /*impl SyncReferenceType {
        pub fn iter() -> &'static [SyncReferenceType] {
            &[SyncReferenceType::Video, SyncReferenceType::Subtitle]
        }
    }*/

    #[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
    pub enum AlgorithmVariant {
        Nosplit,
        Split,
    }

    impl AlgorithmVariant {
        pub fn iter() -> &'static [AlgorithmVariant] {
            &[AlgorithmVariant::Nosplit, AlgorithmVariant::Split]
        }
    }

    #[derive(Debug, Serialize, Deserialize, Hash, PartialEq, Eq)]
    pub struct AlgorithmConf {
        pub sync_ref_type: SyncReferenceType,
        pub algorithm_variant: AlgorithmVariant,
        pub scaling_correct_mode: super::ScalingCorrectMode,
    }

    #[derive(Debug)]
    pub struct ListOfPairsMap<K: Eq + Hash, V>(HashMap<K, V>);

    impl<K: Eq + Hash, V> std::default::Default for ListOfPairsMap<K, V> {
        fn default() -> ListOfPairsMap<K, V> {
            ListOfPairsMap(HashMap::new())
        }
    }

    impl<K: Eq + Hash, V> ListOfPairsMap<K, V> {
        pub fn inner_mut(&mut self) -> &mut HashMap<K, V> {
            &mut self.0
        }
    }

    impl<K, V> Serialize for ListOfPairsMap<K, V>
    where
        K: Eq + Hash + Serialize,
        V: Serialize,
    {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            #[derive(Serialize)]
            struct Entry<K, V> {
                key: K,
                val: V,
            }

            serializer.collect_seq(self.0.iter().map(|(key, val)| Entry { key, val }))
        }
    }

    #[derive(Debug, Serialize, Default)]
    pub struct TransientRoot {
        pub time_required_by_algorithm: ListOfPairsMap<AlgorithmConf, Vec<i64>>,
        pub time_required_by_optimization_value: ListOfPairsMap<FixedPointNumber, Vec<i64>>,
    }

    #[derive(Debug, Serialize, Default)]
    pub struct Root {
        pub general: GeneralStatistics,

        pub offset_by_subtitle: Vec<OffsetBySubtitle>,

        pub sync_offset_histogram_by_split_penalty:
            HashMap<SyncReferenceType, HashMap<super::FixedPointNumber, Histogram>>,

        pub sync_offset_histogram_by_optimization:
            HashMap<SyncReferenceType, HashMap<super::FixedPointNumber, Histogram>>,

        pub sync_offset_histogram_by_min_span_length: HashMap<i64, Histogram>,

        pub all_configurations_offset_histogram: ListOfPairsMap<AlgorithmConf, Histogram>,

        // ----------------
        pub subtitle_span_length_histogram: Histogram,
        pub vad_span_length_histogram: Histogram,

        // ----------------
        pub raw_distance_histogram: Histogram,
        pub sync_to_video_distance_histogram: Histogram,
        pub sync_to_sub_distance_histogram: Histogram,

        // ----------------
        pub distance_box_plot: Option<BoxPlotData>,

        // grouped by subtitle (so each subtitle has equal part in box plot data)
        pub grouped_distance_box_plot: Option<BoxPlotData>,
    }

    impl Root {
        pub fn get_distance_histogram_mut(&mut self, ref_type_opt: Option<SyncReferenceType>) -> &mut Histogram {
            match ref_type_opt {
                None => &mut self.raw_distance_histogram,
                Some(SyncReferenceType::Subtitle) => &mut self.sync_to_sub_distance_histogram,
                Some(SyncReferenceType::Video) => &mut self.sync_to_video_distance_histogram,
            }
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct OffsetBySubtitle {
        pub sub_id: SubtitleID,

        pub num_line_pairs: usize,
        pub ref_line_count: usize,
        pub in_line_count: usize,

        pub num_video_sync_splits: usize,
        pub num_sub_sync_splits: usize,

        pub raw_sync_classification: super::SyncClassification,
        pub video_sync_classification: super::SyncClassification,
        pub sub_sync_classification: super::SyncClassification,

        pub sub_sync_offsets: BoxPlotData,
        pub video_sync_offsets: BoxPlotData,
        pub raw_offsets: BoxPlotData,
    }

    /*impl OffsetBySubtitle {
        pub fn get_box_plot_by_ref_type(
            &mut self,
            ref_type_opt: Option<SyncReferenceType>,
        ) -> &mut Option<BoxPlotData> {
            match ref_type_opt {
                None => &mut self.raw_offsets,
                Some(SyncReferenceType::Subtitle) => &mut self.sub_sync_offsets,
                Some(SyncReferenceType::Video) => &mut self.video_sync_offsets,
            }
        }
    }*/

    #[derive(Debug, Serialize, Deserialize, Default)]
    pub struct GeneralStatistics {
        pub total_line_candidates: usize,
        pub used_line_candidates: usize,

        pub total_movie_count: usize,
        pub movie_with_ref_sub_count: usize,

        pub total_subtitles_count: usize,

        pub required_framerate_adjustments: usize,

        pub raw_sync_class_counts: SyncClassificationsCount,
        pub sync_to_video_sync_class_counts: SyncClassificationsCount,
        pub sync_to_sub_sync_class_counts: SyncClassificationsCount,
    }

    impl GeneralStatistics {
        pub fn get_sync_classification_counter_mut(
            &mut self,
            ref_type_opt: Option<SyncReferenceType>,
        ) -> &mut SyncClassificationsCount {
            match ref_type_opt {
                None => &mut self.raw_sync_class_counts,
                Some(SyncReferenceType::Video) => &mut self.sync_to_video_sync_class_counts,
                Some(SyncReferenceType::Subtitle) => &mut self.sync_to_sub_sync_class_counts,
            }
        }
    }

    #[derive(Debug, Serialize, Deserialize, Default)]
    pub struct SyncClassificationsCount {
        pub synced: usize,
        pub unsynced: usize,
        pub unknown: usize,
    }

    impl SyncClassificationsCount {
        pub fn insert(&mut self, sc: super::SyncClassification) {
            match sc {
                super::SyncClassification::Synced => {
                    self.synced = self.synced + 1;
                }
                super::SyncClassification::Unsynced => {
                    self.unsynced = self.unsynced + 1;
                }
                super::SyncClassification::Unknown => {
                    self.unknown = self.unknown + 1;
                }
            }
        }
    }

    #[derive(Debug, Serialize, Deserialize, Default)]
    pub struct Histogram {
        pub occurrences: HashMap<i64, usize>,
    }

    impl Histogram {
        pub fn insert(&mut self, data: i64) {
            *self.occurrences.entry(data).or_insert(0) += 1;
        }

        pub fn insert_all(&mut self, data: &[i64]) {
            for &d in data {
                self.insert(d);
            }
        }

        pub fn merge_from(&mut self, other: &Histogram) {
            for (key, value) in other.occurrences.iter() {
                *self.occurrences.entry(*key).or_insert(0) += value;
            }
        }

        /*fn write(&self, w: &mut impl Write) {
            for (data, nr) in self.occurences.iter() {
                writeln!(w, "{},{}", data, nr).expect("failed to write");
            }
        }*/
    }
}

mod cache {
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;

    use super::types::*;

    #[derive(Debug, Serialize, Deserialize, Default)]
    pub struct Root {
        #[serde(default)]
        pub vad_spans: HashMap<MovieID, Vec<super::Span>>,

        /// Reference Subtitle ID x Input Subtitle ID -> List<(Index of ref line, Index of input line)>
        #[serde(default)]
        pub line_pairs: HashMap<(SubtitleID, SubtitleID, super::LineMatchingConfig), LinePairs>,

        #[serde(default)]
        pub sync_deltas: HashMap<(RefConfig, SubtitleID, FixedPointNumber, super::AlignMode), Vec<i64>>,
    }

    impl Root {
        pub fn get_vad_spans_raw(&self, movie_id: &MovieID) -> Vec<super::Span> {
            self.vad_spans.get(movie_id).unwrap().clone()
        }
    }
}

mod database {

    use serde::Deserialize;
    use std::path::PathBuf;
    use std::sync::Arc;

    #[derive(Debug, Deserialize)]
    pub struct Root {
        pub movies: Vec<Arc<Movie>>,
        pub movies_without_reference_sub_count: usize,
    }

    impl Root {
        pub fn non_ref_sub_count(&self) -> usize {
            self.movies.iter().map(|m| m.subtitles.len()).sum()
        }

        pub fn total_sub_count(&self) -> usize {
            self.movies.iter().map(|m| m.subtitles.len() + 1).sum()
        }

        pub fn all_subtitles_iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut Subtitle> {
            let mut result = Vec::<&'a mut Subtitle>::new();
            for movie in &mut self.movies {
                let movie_mut = Arc::get_mut(movie).unwrap();
                result.push(Arc::get_mut(&mut movie_mut.reference_subtitle).unwrap());

                for subtitle in &mut movie_mut.subtitles {
                    result.push(Arc::get_mut(subtitle).unwrap());
                }
            }
            result.into_iter()
        }

        pub fn all_subtitles_iter<'a>(&'a self) -> impl Iterator<Item = &'a Arc<Subtitle>> {
            let mut result = Vec::<&'a Arc<Subtitle>>::new();
            for movie in &self.movies {
                result.push(&movie.reference_subtitle);

                for subtitle in &movie.subtitles {
                    result.push(&subtitle);
                }
            }
            result.into_iter()
        }
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct Movie {
        pub id: String,
        pub name: String,
        pub path: PathBuf,

        pub reference_subtitle: Arc<Subtitle>,
        pub subtitles: Vec<Arc<Subtitle>>,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct Subtitle {
        pub id: String,

        pub opensubtitles_metadata: OpensubtitlesMetadata,
        pub data: Vec<LineInfo>,
    }

    impl Subtitle {
        /*pub fn id_ref(&self) -> &String {
            &self.id
        }*/

        pub fn id(&self) -> String {
            self.id.clone()
        }
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct OpensubtitlesMetadata {
        // there are soooo many fields one could parse
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct LineInfo {
        pub start_ms: i64,
        pub end_ms: i64,
        pub text: String,
    }
}
