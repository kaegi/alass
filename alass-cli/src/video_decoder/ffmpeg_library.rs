use failure::{Backtrace, Context, Fail};
use ffmpeg_sys::*;
use std::convert::TryInto;
use std::ffi::{CStr, CString, OsString};
use std::fmt;
use std::path::{Path, PathBuf};
use std::ptr::null_mut;

use crate::define_error;

fn av_err2str(errnum: libc::c_int) -> String {
    let mut err_buffer: [libc::c_char; 256] = [0; 256];
    unsafe {
        av_make_error_string(err_buffer.as_mut_ptr() as *mut i8, err_buffer.len(), errnum);
        CStr::from_ptr(&err_buffer as *const libc::c_char)
            .to_string_lossy()
            .to_string()
    }
}

define_error!(DecoderError, DecoderErrorKind);

#[derive(Debug, Fail)]
pub(crate) enum DecoderErrorKind {}

fn format_cmd(cmd_path: &PathBuf, args: &[OsString]) -> String {
    let args_string: String = args
        .iter()
        .map(|x| format!("{}", x.to_string_lossy()))
        .collect::<Vec<String>>()
        .join(" ");
    format!("{} {}", cmd_path.display(), args_string)
}

impl fmt::Display for DecoderErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unimplemented!()
    }
}

pub struct VideoDecoderFFmpegLibrary {}

impl VideoDecoderFFmpegLibrary {
    /// Samples are pushed in 8kHz mono/single-channel format.
    pub(crate) fn decode<T>(
        file_path: impl AsRef<Path>,
        mut receiver: impl super::AudioReceiver<Output = T>,
        mut progress_handler: impl super::ProgressHandler,
    ) -> Result<T, DecoderError> {
        unsafe {
            let mut format_context: *mut AVFormatContext = avformat_alloc_context();

            let file_path_: String = file_path.as_ref().to_string_lossy().into_owned();

            let result: libc::c_int;

            result = avformat_open_input(
                &mut format_context as *mut *mut AVFormatContext,
                file_path_.as_bytes().as_ptr() as *const i8,
                null_mut(),
                null_mut(),
            );

            if result < 0 {
                // TODO
                panic!(
                    "Failed to open media file '{}': {}",
                    file_path.as_ref().display(),
                    av_err2str(result)
                );
            }

            avformat_find_stream_info(format_context, null_mut());

            let streams: &[*mut AVStream] =
                std::slice::from_raw_parts((*format_context).streams, (*format_context).nb_streams as usize);

            let mut audio_stream_opt: Option<*mut AVStream> = None;

            for &stream in streams {
                let local_codec_parameters: *mut AVCodecParameters = (*stream).codecpar;

                if (*local_codec_parameters).codec_type == AVMediaType::AVMEDIA_TYPE_AUDIO {
                    // choose the audio stream with the least amount of channels (it can be resampled faster)
                    if let Some(saved_audio_stream) = audio_stream_opt {
                        if (*(*saved_audio_stream).codecpar).channels > (*local_codec_parameters).channels {
                            audio_stream_opt = Some(stream);
                        }
                    } else {
                        audio_stream_opt = Some(stream);
                    }
                }
            }

            if audio_stream_opt.is_none() {
                /* TODO */
                panic!("no audio stream found");
            }
            let audio_stream = audio_stream_opt.unwrap();

            let local_codec_parameters: *mut AVCodecParameters = (*audio_stream).codecpar;

            let local_codec: *mut AVCodec = avcodec_find_decoder((*local_codec_parameters).codec_id);
            //let local_codec_name: &CStr = CStr::from_ptr((*local_codec).long_name);

            /*println!(
                "Audio Codec '{}': {} channels, sample rate {}",
                local_codec_name.to_string_lossy(),
                (*local_codec_parameters).channels,
                (*local_codec_parameters).sample_rate
            );*/

            let codec_context: *mut AVCodecContext = avcodec_alloc_context3(local_codec as *const AVCodec);
            avcodec_parameters_to_context(codec_context, local_codec_parameters);
            avcodec_open2(codec_context, local_codec, null_mut());

            let _av_opt_set_int = |swr: *mut SwrContext, name: &str, val: i64, search_flag: libc::c_int| {
                av_opt_set_int(
                    swr as *mut libc::c_void,
                    CString::new(name).unwrap().into_raw(),
                    val,
                    search_flag,
                )
            };

            let _av_opt_set_int = |swr: *mut SwrContext, name: &str, val: i64, search_flag: libc::c_int| {
                av_opt_set_int(
                    swr as *mut libc::c_void,
                    CString::new(name).unwrap().into_raw(),
                    val,
                    search_flag,
                )
            };

            let _av_opt_set_sample_fmt =
                |obj: *mut SwrContext, name: &str, fmt: AVSampleFormat, search_flags: libc::c_int| -> libc::c_int {
                    av_opt_set_sample_fmt(
                        obj as *mut libc::c_void,
                        CString::new(name).unwrap().into_raw(),
                        fmt,
                        search_flags,
                    )
                };

            let in_channel_layout = (*codec_context).channel_layout.try_into().unwrap();
            let in_channel_count: i64 = (*codec_context).channels.try_into().unwrap();
            let in_sample_rate: i64 = (*codec_context).sample_rate.try_into().unwrap();
            let in_sample_format = (*codec_context).sample_fmt;

            let out_channel_count = 1;
            let out_channel_layout = AV_CH_LAYOUT_MONO.try_into().unwrap();
            let out_sample_rate = 8000;
            let out_sample_format = AVSampleFormat::AV_SAMPLE_FMT_S16P;

            // prepare resampler
            let swr: *mut SwrContext = swr_alloc();
            _av_opt_set_int(swr, "in_channel_count", in_channel_count, 0);
            _av_opt_set_int(swr, "in_channel_layout", in_channel_layout, 0);
            _av_opt_set_int(swr, "in_sample_rate", in_sample_rate, 0);
            _av_opt_set_sample_fmt(swr, "in_sample_fmt", in_sample_format, 0);

            _av_opt_set_int(swr, "out_channel_count", out_channel_count, 0);
            _av_opt_set_int(swr, "out_channel_layout", out_channel_layout, 0);
            _av_opt_set_int(swr, "out_sample_rate", out_sample_rate, 0);
            _av_opt_set_sample_fmt(swr, "out_sample_fmt", out_sample_format, 0);

            swr_init(swr);
            if swr_is_initialized(swr) == 0 {
                unimplemented!();
                //pri(stderr, "Resampler has not been properly initialized\n");
                //return -1;
            }

            /* compute the number of converted samples: buffering is avoided
             * ensuring that the output buffer will contain at least all the
             * converted input samples */
            let src_nb_samples = 1024; // this is just a guess...
            let mut max_out_samples: i32 =
                av_rescale_rnd(src_nb_samples, out_sample_rate, in_sample_rate, AVRounding::AV_ROUND_UP) as i32;

            let mut buffer: *mut i16 = null_mut();
            av_samples_alloc(
                &mut buffer as *mut *mut i16 as *mut *mut u8,
                null_mut(),
                out_channel_count as i32,
                max_out_samples,
                out_sample_format,
                0,
            );

            let packet: *mut AVPacket = av_packet_alloc();
            let frame: *mut AVFrame = av_frame_alloc();

            progress_handler.init((*audio_stream).nb_frames);

            while av_read_frame(format_context, packet) >= 0 {
                //println!("read frame {:?}", packet);

                if (*packet).stream_index != (*audio_stream).index {
                    continue;
                }

                progress_handler.inc();

                //println!("stream fits");

                let mut response = avcodec_send_packet(codec_context, packet);
                if response < 0 {
                    panic!("{}", av_err2str(response));
                }

                loop {
                    //println!("begin receive_frame");
                    response = avcodec_receive_frame(codec_context, frame);
                    //println!("end receive_frame");

                    if response == AVERROR(EAGAIN) || response == AVERROR_EOF {
                        break;
                    } else if response < 0 {
                        panic!("Error: {}", av_err2str(response));
                    }

                    //let out_samples = av_rescale_rnd(swr_get_delay(swr, 48000) + in_samples, 44100, 48000, AV_ROUND_UP);
                    let out_sample_count = swr_get_out_samples(swr, (*frame).nb_samples);

                    // Resize output buffer to allow all samples (without buffering) to be stored.
                    if out_sample_count > max_out_samples {
                        max_out_samples = out_sample_count;
                        av_freep(&mut buffer as *mut *mut i16 as *mut libc::c_void);
                        av_samples_alloc(
                            &mut buffer as *mut *mut i16 as *mut *mut u8,
                            null_mut(),
                            out_channel_count as i32,
                            max_out_samples,
                            out_sample_format,
                            0,
                        );
                    }

                    // resample frames
                    let frame_count = swr_convert(
                        swr,
                        &mut buffer as *mut *mut i16 as *mut *mut u8,
                        out_sample_count,
                        (*frame).data.as_mut_ptr() as *mut *const u8,
                        (*frame).nb_samples,
                    );

                    //println!("Samples: {} Predicted: {} Frames: {}", (*frame).nb_samples, out_sample_count, frame_count);
                    let out_slice = std::slice::from_raw_parts_mut(buffer, frame_count as usize);

                    receiver.push_samples(out_slice);

                    /*for v in out_slice {
                        println!("{}", v);
                    }*/

                    //println!("Frame count: {}", frame_count);

                    //println!("freep done");
                }

                av_packet_unref(packet);
            }

            av_freep(&mut buffer as *mut *mut i16 as *mut libc::c_void);

            avformat_free_context(format_context);
            // TODO: cleanup everything
        }

        progress_handler.finish();

        Ok(receiver.finish())
    }
}
