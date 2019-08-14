# alass-core

This Rust library contains the core algorithm for `alass`, the "Automatic Language-Agnostic Subtitle Sychronization" tool. If you want to go to the command line tool instead, please click [here](https://github.com/kaegi/alass).


## How to use the library
Add this to your `Cargo.toml`:

```toml
[dependencies]
alass-core = "1.0"
```

The library only contains one function that takes two sequences of time spans and returns the offsets to get the best possible alignment.

[Documentation](https://docs.rs/alass-core)

[Crates.io](https://crates.io/crates/alass-core)

### Documentaion

For much more information, please see the workspace information [here](https://github.com/kaegi/alass).