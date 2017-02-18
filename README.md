# Introduction

`aligner` is a Rust library and command-line tool that corrects a subtitle given
a second "correct" subtitle. It will figure out offsets and where to
introduce or remove advertisement breaks to get the best alignment possible. It does not use
any language information so it even works for image based subtitles like VobSub.

## Usage

The most basic command is:

```bash
$ aligner reference_subtitle.ssa incorrect_subtitle.srt output.srt
```

You can additionally adjust how much the algorithm tries to avoid introducing or removing a break:

```bash
# split-penalty is a value between 0 and 100
$ aligner reference_subtitle.ssa incorrect_subtitle.srt output.srt --split-penalty 2.6
```

Currently supported are `.srt`, `.ssa`/`.ass` and `.idx` files.

## How to compile the binary

Install [Rust and Cargo](https://www.rust-lang.org/en-US/install.html) then run:

```bash
# this will create ~/.cargo/bin/aligner
$ cargo install aligner
```

## How to use the library
Add this to your `Cargo.toml`:

```toml
[dependencies]
aligner = "~0.1.0"
```

[Documentation](https://docs.rs/aligner)

[Crates.io](https://crates.io/crates/aligner)



## Algorithm

At the core of the algorithm is the _rating_ of an alignment. For each pair of subtitles (one from the reference subtitle on from the incorrect subtitle) the rating is

`overlapping_time(refsub, incsub) / max(length(refsub), length(incsub))`

The maximum of this rating is 1, if and only if `refsub = incsub`. The total rating of an alignment is (for the most part) the sum of all ratings of all possible pairs. By moving the `incsubs` around, we might get a better alignment. As a basic constraint, the order of the `incsubs` will not be changed. So if we have to consecutive subtitles `start(incsubN) <= start(incsubN+1)`, the corrected `'incsubs` will still have `start('incsubN) <= start('incsubN+1)`.

If only this formula was used, the algorithm will probably create different offsets for each subtitle line. To avoid that, we have to use `split_penalty` value. For each consecutive subtitles where  `start(incsubN) - start(incsubN+1) = start('incsubN) - start('incsubN+1)` we add another `split_penalty` to the total rating. That way, with every extra split we lose `split_penalty` of rating.

This algorithm computes the alignment which yields the maximum of all possible ratings. The algorithm is powered by the principle of [dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming).

To simplify the problem, we assume `start(sub)` is the start timestamp in milliseconds of a subtitle line `sub` and `0 <= start(sub)` is true. Let's say `get_rating(t, n)` computes the best rating/alignment for the first `n` incorrect subtitle lines with the additional constraint `0 <= start(sub) <= t` for each of these `n` subtitles.

Of course we can now simply set `get_rating(t, 0) = 0` because if we have no incorrect subtitles to align, we have a rating of zero (independent of `t`).

Now we handle the case `get_rating(0, 1)`. We can simply compute the overlapping rating (where the first incorrect subtitle starts at the "zero timepoint") with every reference subtitle and add up these values. With `get_rating(1, 1)` things get interesting. We can either have `start(sub) = 0` or `start(sub) = 1`. Fortunaly we already have `get_rating(0, 1)`, so we only need the rating where `start(sub) = 1`. This can be computed by adding up all overlapping ratings. Similarly we can compute `get_rating(2, 1)` by taking the maximum of `get_rating(1, 1)` and the rating where `start(sub) = 2`. In this vein we can create `get_rating(t+1, 1)` from `get_rating(t, 1)`. We can also speed up computing the overlapping rating, because the subtitle line will only be shifted by 1ms from `start(sub) = t` to `start(sub) = t + 1`. The subtitle will lose the rating for the segment `[t, t + 1]` and gain the overlapping rating for the segment `[t + length(sub), t + 1 + length(sub)]` on the other side. By creating a lookup-table for reference subtitles for every `t` this process has a runtime of `O(1)`.

When do we stop? Well, the rating won't change anymore if `t` gets big. At the latest when `start(sub)` is be greater than any of the reference subtitle lines, because after that the overlapping rating will always be zero. Let's call `max_t` the timepoint where all incorrect subtitles have been moved behind the reference subtitles. The best total rating is then `get_rating(max_t, number_of_incorrect_subtitles)`.

Now we have all `get_rating(0, 1)` to `get_rating(max_t, 1)`. To compute `get_rating(0, 2)`, which means that `0 <= start(sub0) <= 0` and `0 <= start(sub1) <= 0`. We already have the rating for `sub0` in form of `get_rating(0, 1)`. We only need to add the overlapping rating for `sub1`. To get `get_rating(1, 2)` we can either use `get_rating(0, 2)` (we leave `sub1` where it is), or move `start(sub1)` to 1, which allows `start(sub0)` to be in `0 <= start(sub0) <= start(sub1) = 1`. The best rating for `sub0` for that range has been computed with `get_rating(1, 1)`, we only need to add the overlapping rating for `start(sub1) == 1`. We proceed similarly to get `get_rating(t+1, 2)`: leave the `sub1` like it was for `get_rating(t,2)` or reposition the subtitle to `start(sub1) == t+1` and use `get_rating(t+1,1) + overlapping_rating(sub1,t+1)`.

With the same principle we proceed with `subN`:

-   initialize `get_rating(0, n) = get_rating(0, n - 1) + overlapping_rating(subN, 0)`
-   choose for `get_rating(t+1, n)` the maximum of
    -   `get_rating(t, n)` which means "leaving `subN`" and
    -   `get_rating(t+1, n-1) + overlapping_rating(t+1, subN)` which means repositioning the `subN`


Until now we didn't use the `split_penalty`. We need to add the split penalty when `start(subN) - start(subN+1)` is a specific value (the original distance `diff(N)`). The trick here is seeing that we only need to consider the "repostion choice". The only time `get_rating(_, n-1)` is consulted after the inital phase is when `subN` gets repositioned. `subN` will then start at `t+1` and we consult `get_rating(t+1, n-1)`. So if `subN-1` were positioned at `t+1-diff(N-1)` for `get_rating(t+1-diff(N-1), n-1)` we'd be able to get the `split_penalty`. This is exactly the thing we will do when we are in a phase `n`: We will not only have the "leave choice" or "reposition choice" but also the "nosplit choice". If we compute `get_rating(t, n)`, we can also compare the two other values with `get_rating(t-diffN, n-1) + overlapping_rating(t-diffN, subN) + split_penalty`. The `get_rating(t-diffN, n-1) + overlapping_rating(t-diffN, subN)` is again the best rating where `start(subN) = t-diffN`. We are allowed to add the `split_penalty` because in the next phase `n+1`, `subN+1` will start at `t` when `get_rating(t-diffN, n)` is looked up. So the final rating algorithm is:

-   initialize `get_rating(t, 0)` with 0
-   initialize `get_rating(0, n) = get_rating(0, n - 1) + overlapping_rating(subN, 0)`
-   choose for `get_rating(t+1, n)` the maximum of
    -   `get_rating(t, n)` which means "leaving `subN`" and
    -   `get_rating(t+1, n-1) + overlapping_rating(t+1, subN)` which means repositioning the `subN` and
    -   `get_rating(t+1-diffN, n-1) + overlapping_rating(t+1-diffN, subN) + split_penalty` which means doing a nosplit-repositioning for `subN`


To get the final alignment, we save for each phase `n` and `t+1` where `subN` was positioned (can be `t+1`, `t+1-diffN` or the previous position). If we look up that value for `n = number_of_incorrect_subtitles` and `t = max_t`, we know where the last subtitles `subN` has to be. We then know `start(subN)`. The best alignment of all previous subtitles is then computed with `get_rating(start(subN), n-1)`. So we look up the position for `subN-1` in that table with `n' = n-1` and `t' = start(subN)`. That way we get all corrected positions of all incorrect subtitles and are done!

Though this algorithm works (and was implemented in one of the early versions of `aligner`) it is neither fast nor space-efficient. Let's take a `45 minutes = 2700000 milliseconds < max_t` subtitle file which has about `n = 900 subtitles` (these are realistic values). We build a table of `max_t * n = 2430000000 ` entries. We can discard the ratings of the phase `n-1` after phase `n`, but we always need to store the positions of the subtitle `subN`. Let's assume we need 4 bytes to store them: we then have a table of `2430000000 * 4 bytes = 9720000000 bytes = 9 GB` of data in RAM!!! Even filling the table with zeros might take some noticeable time. But as it turns out we can compress that table in under 2 MB (most of the time; probably the best compression I've ever seen) with `delta encoding`. The empirical foundation is that the choices almost never change from one `t` to `t+1` (about 10 to 1000 times for one phase). If we always take the

-   "leave choice", the position will always be `t+1` for every `t+1` (rise by 1)
-   "nosplit-reposition choice", the position will always be `t+1-diffN` (rise by 1)
-   "reposition choice", the position won't change from `t - 1` (constant)

So if we store values in a `(start, delta, length)` tuple, where the first uncompressed value is `start + 0 * delta`, the second is `start + 1 * delta`, the third is `start + 2 * delta`, ..., the last is `start + length * delta`, we can compress an entire phase into a few bytes. The same thing is applicable to the ratings. Without going into details: if we take the overlapping rating of a incorrect subtitle to a reference subtitle and "move" the incorrect subtitle from the far left to the far right we will have five segments of compressed values (first the rating will be zero, then rise linearly, then be constant, then fall linearly, then be zero again). The comparisons/choices can then be done for rating segments instead of single `t`. This yields a speedup of at least one order of magnitude.
