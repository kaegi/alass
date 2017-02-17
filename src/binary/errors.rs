// This file is part of the Rust library and binary `aligner`.
//
// Copyright (C) 2017 kaegi
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.


use binary::formats;

// see https://docs.rs/error-chain/0.8.1/error_chain/
#[cfg_attr(rustfmt, rustfmt_skip)]
error_chain! {
    foreign_links {
        Io(::std::io::Error);
    }

    links {
        SsaError(formats::ssa::Error, formats::ssa::ErrorKind);
        IdxError(formats::idx::Error, formats::idx::ErrorKind);
        SrtError(formats::srt::Error, formats::srt::ErrorKind);
    }

    errors {
        FileOperation(file: String) {
            display("operation on file '{}' failed", file)
        }
        SsaFormattingInfoNotFound {
            description("file did not have a `[Events]` section containing a line beginning with `Format: `")
        }

        UnknownFileFormat {
            description("unknown file format, only SubRip (.srt), SubStationAlpha (.ssa/.ass) and VobSub (.idx) are supported at the moment")
        }
        ArgumentParseError(argument_name: &'static str, s: String) {
            display("command line argument '{}' could not be parsed from string '{}'", argument_name, s)
        }
        InvalidArgument(argument_name: &'static str) {
            display("command line argument '{}' has invalid value", argument_name)
        }
        ExpectedPositiveNumber(i: i64) {
            display("expected positive number, got '{}'", i)
        }
        ValueNotInRange(v: f64, min: f64, max: f64) {
            display("expected value in the range from '{}' to '{}', found value '{}'", min, max, v)
        }
        DifferentOutputFormat(input_file: String, output_file: String) {
            description("the requested output file has a different format than the incorrect subtitle file (this program does not convert)")
            display("the requested output '{}' file has a different format than the incorrect subtitle file '{}' (this program does not convert)", output_file, input_file)
        }
    }
}
