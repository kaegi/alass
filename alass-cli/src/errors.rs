// This file is part of the Rust library and binary `alass`.
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

use failure::{Backtrace, Context, Fail};
use std::fmt;
use std::path::PathBuf;
use subparse::SubtitleFormat;

#[macro_export]
macro_rules! define_error {
    ($error:ident, $errorKind:ident) => {
        #[derive(Debug)]
        pub struct $error {
            inner: Context<$errorKind>,
        }

        impl Fail for $error {
            fn name(&self) -> Option<&str> {
                self.inner.name()
            }

            fn cause(&self) -> Option<&dyn Fail> {
                self.inner.cause()
            }

            fn backtrace(&self) -> Option<&Backtrace> {
                self.inner.backtrace()
            }
        }

        impl fmt::Display for $error {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                fmt::Display::fmt(&self.inner, f)
            }
        }

        #[allow(dead_code)]
        impl $error {
            pub fn kind(&self) -> &$errorKind {
                self.inner.get_context()
            }
        }

        #[allow(dead_code)]
        impl $errorKind {
            pub fn into_error(self) -> $error {
                $error {
                    inner: Context::new(self),
                }
            }
        }

        impl From<$errorKind> for $error {
            fn from(kind: $errorKind) -> $error {
                $error {
                    inner: Context::new(kind),
                }
            }
        }

        impl From<Context<$errorKind>> for $error {
            fn from(inner: Context<$errorKind>) -> $error {
                $error { inner: inner }
            }
        }
    };
}

define_error!(InputFileError, InputFileErrorKind);

#[derive(Clone, Eq, PartialEq, Debug, Fail)]
pub enum InputFileErrorKind {
    VideoFile(PathBuf),
    SubtitleFile(PathBuf),
}

impl fmt::Display for InputFileErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            InputFileErrorKind::VideoFile(p) => write!(f, "processing video file '{}' failed", p.display()),
            InputFileErrorKind::SubtitleFile(p) => write!(f, "processing subtitle file '{}' failed", p.display()),
        }
    }
}

define_error!(FileOperationError, FileOperationErrorKind);

#[derive(Clone, Eq, PartialEq, Debug, Fail)]
pub enum FileOperationErrorKind {
    FileOpen { path: PathBuf },
    FileRead { path: PathBuf },
    FileWrite { path: PathBuf },
}

impl fmt::Display for FileOperationErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FileOperationErrorKind::FileOpen { path } => write!(f, "failed to open file '{}'", path.display()),
            FileOperationErrorKind::FileRead { path } => write!(f, "failed to read file '{}'", path.display()),
            FileOperationErrorKind::FileWrite { path } => write!(f, "failed to read file '{}'", path.display()),
        }
    }
}

define_error!(InputVideoError, InputVideoErrorKind);

#[derive(Clone, Eq, PartialEq, Debug, Fail)]
pub enum InputVideoErrorKind {
    FailedToDecode { path: PathBuf },
    VadAnalysisFailed,
}

impl fmt::Display for InputVideoErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            InputVideoErrorKind::FailedToDecode { path } => {
                write!(f, "failed to extract voice segments from file '{}'", path.display())
            }
            InputVideoErrorKind::VadAnalysisFailed => write!(f, "failed to analyse audio segment for voice activity"),
        }
    }
}

define_error!(InputSubtitleError, InputSubtitleErrorKind);

#[derive(Clone, Eq, PartialEq, Debug, Fail)]
pub enum InputSubtitleErrorKind {
    ReadingSubtitleFileFailed(PathBuf),
    UnknownSubtitleFormat(PathBuf),
    ParsingSubtitleFailed(PathBuf),
    RetreivingSubtitleLinesFailed(PathBuf),
}

impl fmt::Display for InputSubtitleErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            InputSubtitleErrorKind::ReadingSubtitleFileFailed(path) => {
                write!(f, "reading subtitle file '{}' failed", path.display())
            }
            InputSubtitleErrorKind::UnknownSubtitleFormat(path) => {
                write!(f, "unknown subtitle format for file '{}'", path.display())
            }
            InputSubtitleErrorKind::ParsingSubtitleFailed(path) => {
                write!(f, "parsing subtitle file '{}' failed", path.display())
            }
            InputSubtitleErrorKind::RetreivingSubtitleLinesFailed(path) => {
                write!(f, "retreiving subtitle file '{}' failed", path.display())
            }
        }
    }
}

define_error!(InputArgumentsError, InputArgumentsErrorKind);

#[derive(Clone, PartialEq, Debug, Fail)]
pub enum InputArgumentsErrorKind {
    #[fail(
        display = "expected value '{}' to be in range '{}'-'{}', found value '{}'",
        argument_name, min, max, value
    )]
    ValueNotInRange {
        argument_name: String,
        min: f64,
        max: f64,
        value: f64,
    },
    #[fail(display = "expected positive number for '{}', found '{}'", argument_name, value)]
    ExpectedPositiveNumber { argument_name: String, value: i64 },

    #[fail(display = "expected non-negative number for '{}', found '{}'", argument_name, value)]
    ExpectedNonNegativeNumber { argument_name: String, value: f64 },

    #[fail(display = "argument '{}' with value '{}' could not be parsed", argument_name, value)]
    ArgumentParseError { argument_name: String, value: String },
}

define_error!(TopLevelError, TopLevelErrorKind);

pub enum TopLevelErrorKind {
    FileFormatMismatch {
        input_file_path: PathBuf,
        output_file_path: PathBuf,
        input_file_format: SubtitleFormat,
    },
    FailedToUpdateSubtitle,
    FailedToGenerateSubtitleData,
    FailedToInstantiateSubtitleFile,
}

impl fmt::Display for TopLevelErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
         TopLevelErrorKind::FileFormatMismatch { input_file_path, output_file_path, input_file_format } => write!(f, "output file '{}' seems to have a different format than input file '{}' with format '{}' (this program does not perform conversions)", output_file_path.display(), input_file_path.display(), input_file_format.get_name()),
         TopLevelErrorKind::FailedToUpdateSubtitle => write!(f, "failed to change lines in the subtitle"),
         TopLevelErrorKind::FailedToGenerateSubtitleData => write!(f, "failed to generate data for subtitle"),
         TopLevelErrorKind::FailedToInstantiateSubtitleFile => write!(f, "failed to instantiate subtitle file"),
        }
    }
}
