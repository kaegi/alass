#[cfg(feature = "ffmpeg-library")]
mod ffmpeg_library;

#[cfg(feature = "ffmpeg-library")]
pub use ffmpeg_library::VideoDecoderFFmpegLibrary as VideoDecoder;

#[cfg(feature = "ffmpeg-binary")]
mod ffmpeg_binary;

#[cfg(feature = "ffmpeg-binary")]
pub use ffmpeg_binary::VideoDecoderFFmpegBinary as VideoDecoder;

pub trait AudioReceiver {
    type Output;
    type Error: failure::Fail;

    /// Samples are in 8000kHz mono/single-channel format.
    fn push_samples(&mut self, samples: &[i16]) -> Result<(), Self::Error>;

    fn finish(self) -> Result<Self::Output, Self::Error>;
}

pub struct ChunkedAudioReceiver<R: AudioReceiver> {
    buffer: Vec<i16>,
    filled: usize,
    next: R,
}

impl<R: AudioReceiver> ChunkedAudioReceiver<R> {
    pub fn new(size: usize, next: R) -> ChunkedAudioReceiver<R> {
        ChunkedAudioReceiver {
            buffer: std::vec::from_elem(0, size),
            filled: 0,
            next,
        }
    }
}

impl<R: AudioReceiver> AudioReceiver for ChunkedAudioReceiver<R> {
    type Output = R::Output;
    type Error = R::Error;

    fn push_samples(&mut self, mut samples: &[i16]) -> Result<(), R::Error> {
        assert!(self.buffer.len() > self.filled);

        loop {
            if samples.is_empty() {
                break;
            }

            let sample_count = std::cmp::min(self.buffer.len() - self.filled, samples.len());
            self.buffer[self.filled..self.filled + sample_count].clone_from_slice(&samples[..sample_count]);

            samples = &samples[sample_count..];

            self.filled = self.filled + sample_count;

            if self.filled == self.buffer.len() {
                self.next.push_samples(self.buffer.as_slice())?;
                self.filled = 0;
            }
        }

        Ok(())
    }

    fn finish(self) -> Result<R::Output, R::Error> {
        self.next.finish()
    }
}

/// Use this trait if you want more detailed information about the progress of operations.
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

/*struct NoProgressHandler {}
impl ProgressHandler for NoProgressHandler {}*/
