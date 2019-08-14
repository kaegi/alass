# Introduction

`alass` is a command line tool to synchronize subtitles to movies. It will figure out offsets and where to
introduce or remove advertisement breaks to get the best alignment possible. It does not use
any language information so it even works for image based subtitles like VobSub.

`alass` stands for  "Automatic Language-Agnostic Subtitle Synchronization". 

## Executable for Windows

Get the lastest executable from [here](https://github.com/kaegi/alass/releases)! Just download and extract the archive. The file `alass.bat` is the command line tool.

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
# split-penalty is a value between 0 and 100 (default 4)
$ alass reference_subtitle.ssa incorrect_subtitle.srt output.srt --split-penalty 2.6
```

Values between 0.1 and 10 are the most useful. Anything above 10 probably does not split the subtitle and anything below 0.1 introduces many unnecessary splits.

If you only want to shift the subtitle, without introducing splits, you can use `--no-splits`:

```bash
# synchronizing the subtitles in this mode is very fast
$ alass movie.mp4 incorrect_subtitle.srt output.srt --no-splits
```

Currently supported are `.srt`, `.ssa`/`.ass` and `.idx` files. Every common video format is supported for the reference file.


## Performance

The extraction of the audio from a video takes about 10 to 20 seconds. Computing the alignment usually takes between 5 and 10 seconds.

The alignment is usually very good in my test (subtitles are within 0.1s of the target position). Adjusting the split penalty can help in a few cases if aligning does not work out-of-the box. More extensive testing and statistics will be performed in the future.

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

#### Statistics

You can activate the statistics module inside `alass` using:

```bash
# Important: you have to be inside `alass-cli`! Otherwise the parameter is ignored.
$ cargo build --features statistics
$ cargo run -- --statistics-path ./some/folder
```

This will create the statistics file in `./some/folder`. You can use `--statistics-required-tag` to only generate statistics.

The statistics module allows you to understand/debug the algorihm better.

**Warning**: Using this configuration slows down the algorithm by 50% or more _even_ if no statistics files are generated.

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