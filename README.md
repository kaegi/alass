# Introduction

`alass` is a command line tool to synchronize subtitles to movies.

It can automatically correct

 - constant offsets
 - splits due to advertisement breaks, directors cut, ...
 - different framerates

The alignment process is not only fast and
accurate, but also language-agnostic. This means
you can align subtitles to movies in different
languages.

`alass` stands for  "Automatic Language-Agnostic Subtitle Synchronization". The theory and algorithms
are documented in my [bachelor's thesis](documentation/thesis.pdf)
and summarized in my [bachelor's presentation](documentation/slides.pdf).


## Executable for Windows (64-bit)

Get the latest executable from [here](https://github.com/kaegi/alass/releases)! Just download and extract the archive. The file `alass.bat` is the command line tool.

## Executable for Linux (64-bit)

Get the latest executable from [here](https://github.com/kaegi/alass/releases)! To run the executable, `ffmpeg` and
`ffprobe` have to be installed.
You can change their paths with the environment variables
`ALASS_FFMPEG_PATH` (default `ffmpeg`) and `ALASS_FFPROBE_PATH` (default `ffprobe`). 

## Usage

The most basic command is:

```bash
$ alass movie.mp4 incorrect_subtitle.srt output.srt
```

You can also use `alass` to align the incorrect subtitle to a different subtitle:

```bash
$ alass reference_subtitle.ssa incorrect_subtitle.srt output.srt
```

You can additionally adjust how much the algorithm tries to avoid introducing or removing a break:

```bash
# split-penalty is a value between 0 and 1000 (default 7)
$ alass reference_subtitle.ssa incorrect_subtitle.srt output.srt --split-penalty 10
```

Values between 5 and 20 are the most useful. Anything above 20 misses some important splits and anything below 5 introduces many unnecessary splits.

If you only want to shift the subtitle, without introducing splits, you can use `--no-split`:

```bash
# synchronizing the subtitles in this mode is very fast
$ alass movie.mp4 incorrect_subtitle.srt output.srt --no-split
```

Currently supported are `.srt`, `.ssa`/`.ass` and `.idx` files. Every common video format is supported for the reference file.


## Performance and Results

The extraction of the audio from a video takes about 10 to 20 seconds. Computing the alignment usually takes between 5 and 10 seconds.

The alignment is usually perfect -
the percentage of "good subtitles" is about 88% to 98%, depending on how strict you classify a "good subtitle".
Downloading random subtitles
from `OpenSubtitles.org` had an error rate of about 50%
(sample size N=118).
Of all subtitle _lines_ (not subtitle files) in the tested database,
after synchronization

 - 50% were within 50ms of target position
 - 80% were within 100ms of target position
 - 90% were within 400ms of target position
 - 95% were within 800ms of target position

compared to a (possibly not perfect) reference subtitle.

## How to compile the binary

Install [Rust and Cargo](https://www.rust-lang.org/en-US/install.html) then run:

```bash
# this will create the lastest release in ~/.cargo/bin/alass-cli
$ cargo install alass-cli
```


The voice-activity module this project uses is written in C. Therefore a C compiler (`gcc` or `clang`) is needed to compile this project.

To use `alass-cli` with video files, `ffmpeg` and `ffprobe` have to be installed. It is used to extract the raw audio data. You can set the paths used by `alass` using the environment variables `ALASS_FFMPEG_PATH` (default `ffmpeg`) and `ALASS_FFPROBE_PATH` (default `ffprobe`). 

### Building from Source 

If you want to build and run the project from source code:

```bash
$ git clone https://github.com/kaegi/alass
$ cd alass
$ cargo build
$ cargo run -- movie.mp4 input.srt output.srt
```

### Configuration

All parameters are shown for `cargo build` can also be used for `cargo install` and `cargo run`.

#### FFmpeg as a library

You can also link `ffmpeg` as a dynamic library during compile time. The library implementation can extract the audio about 2 to 3 seconds faster. Unfortunately it is harder to compile, the error handling is only very basic and might still have bugs.

You have to remove "`# FFMPEG-LIB`" from every line that starts with it in `alass-cli/Cargo.toml`. Then use:

```bash
# Important: you have to be inside `alass-cli`! Otherwise the parameters get ignored.
$ cargo build --no-default-features --features ffmpeg-library
```


### Alias Setup

*For Linux users:* It is recommended to add the folder path to your system path as well as setup an alias for `alass` to `alass-cli`. Add this to your `~/.bashrc` (or the setup file of your favorite shell):

```bash
export PATH="$PATH:$HOME/.cargo/bin"
alias alass="alass-cli"
```

## Folder structure

This `cargo` workspace contains two projects:

  - `alass-core` which provides the algorithm
  
    It is targeted at *developers* who want to use the same algorithm in their project.

  - `alass-cli` which is the official command line tool

    It is target at *end users* who want to correct their subtitles.

## Library Documentation

[Open README](./alass-core/README.md) from `alass-core`.

## Notes

This program was called `aligner` in the past. This made it nearly impossible to find on a search engine, so `alass` was chosen instead.
