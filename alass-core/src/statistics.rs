use crate::rating_type::RatingExt;
use crate::segments::{PositionBuffer, PositionFullSegment, RatingBuffer, RatingFullSegment};
use crate::time_types::TimeDelta;

use std::fs::File;
use std::io::prelude::*;
use std::iter::Iterator;
use std::path::PathBuf;

#[derive(Debug)]
pub struct Statistics {
    number: i64,
    path: PathBuf,
    filter_tags: Vec<String>,
}

#[derive(Clone, Copy)]
struct Color {
    r: u8,
    g: u8,
    b: u8,
}

impl Color {
    fn from_hex(c: u32) -> Color {
        let r = ((c >> 0) & 0xFF) as u8;
        let g = ((c >> 8) & 0xFF) as u8;
        let b = ((c >> 16) & 0xFF) as u8;
        Color { r, g, b }
    }
}

enum GraphicalObject {
    Line {
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        color: Color,
    },
}

impl GraphicalObject {
    fn min_x(&self) -> f32 {
        match self {
            GraphicalObject::Line { x1, x2, .. } => (*x1).min(*x2),
        }
    }

    fn max_x(&self) -> f32 {
        match self {
            GraphicalObject::Line { x1, x2, .. } => (*x1).max(*x2),
        }
    }

    fn min_y(&self) -> f32 {
        match self {
            GraphicalObject::Line { y1, y2, .. } => (*y1).min(*y2),
        }
    }

    fn max_y(&self) -> f32 {
        match self {
            GraphicalObject::Line { y1, y2, .. } => (*y1).max(*y2),
        }
    }
}

impl Statistics {
    pub fn new(path: impl AsRef<std::path::Path>, filter_tags: Vec<String>) -> Statistics {
        Statistics {
            number: 1,
            path: path.as_ref().to_path_buf(),
            filter_tags,
        }
    }

    pub fn prepare_file(&self, filename: String) -> std::io::Result<File> {
        std::fs::create_dir(&self.path).or_else(|error| {
            if error.kind() == std::io::ErrorKind::AlreadyExists {
                return Ok(());
            } else {
                Err(error)
            }
        })?;

        File::create(self.path.join(filename))
    }

    fn write_svg(
        &mut self,
        name: &str,
        tags: &[&str],
        objs_fn: impl Fn() -> Vec<GraphicalObject>,
    ) -> std::io::Result<()> {
        self.number = self.number + 1;

        if !self.passing_tags_filter(tags) {
            return Ok(());
        }

        let mut file = self.prepare_file(format!(
            "{:03}{}.svg",
            self.number,
            name.to_lowercase().replace(' ', "-")
        ))?;

        let objs: Vec<GraphicalObject> = objs_fn();

        if !objs.is_empty() {
            let mut min_x = objs[0].min_x();
            let mut max_x = objs[0].max_x();
            let mut min_y = objs[0].min_y();
            let mut max_y = objs[0].max_y();

            for obj in &objs {
                min_x = min_x.min(obj.min_x());
                min_y = min_y.min(obj.min_y());
                max_x = max_x.max(obj.max_x());
                max_y = max_y.max(obj.max_y());
            }

            let target_width = 8000f32;
            let target_height = 1000f32;

            let height = max_y - min_y + 1.;
            let width = max_x - min_x + 1.;

            let scalex = target_width / width;
            let scaley = -target_height / height * 0.8;

            let movex = -min_x * scalex;
            let movey = -min_y * scaley + target_height - target_height * 0.1;

            file.write_all(format!("<svg height=\"{}\" width=\"{}\">", target_height, target_width).as_bytes())?;

            let line = format!("<text x=\"5\" y=\"20\" font-size=\"25\">Name: {}</text>\n", name);
            file.write_all(line.as_bytes())?;

            let line = format!(
                "<text x=\"5\" y=\"45\" font-size=\"18\">Tags: {}</text>\n",
                tags.join(", ")
            );
            file.write_all(line.as_bytes())?;

            let line = format!("<text x=\"5\" y=\"75\" font-size=\"30\">{}</text>\n", max_y);
            file.write_all(line.as_bytes())?;

            let line = format!(
                "<text x=\"5\" y=\"{}\" font-size=\"30\">{}</text>\n",
                target_height - 5.,
                min_y
            );
            file.write_all(line.as_bytes())?;

            for obj in &objs {
                match obj {
                    GraphicalObject::Line { x1, x2, y1, y2, color } => {
                        let line = format!("<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" style=\"stroke:rgb({},{},{});stroke-width:1\" />\n",
                        x1 * scalex + movex,
                        y1 * scaley + movey,
                        x2 * scalex + movex,
                        y2 * scaley + movey,
                        color.r,color.g,color.b);
                        file.write_all(line.as_bytes())?;
                    }
                }
            }

            file.write_all(b"</svg>")?;
        }

        Ok(())
    }

    pub fn save_rating_buffer(&mut self, name: &str, tags: &[&str], buffer: &RatingBuffer) -> std::io::Result<()> {
        let mut tags2: Vec<&str> = tags.to_vec();
        tags2.push("rating");

        let compute_data = || {
            buffer
                .iter()
                .annotate_with_segment_start_times()
                .into_iter()
                .zip(
                    [Color::from_hex(0xFF00FF), Color::from_hex(0x008888)]
                        .into_iter()
                        .cloned()
                        .cycle(),
                )
                .map(|(segment, color): (RatingFullSegment, Color)| {
                    let start = segment.span.start;
                    let end = segment.span.end;
                    let start_rating = segment.data.start_rating();
                    let end_rating = segment.data.end_rating(end - start);

                    let x1 = start.as_f32();
                    let x2 = (end - TimeDelta::one()).as_f32();
                    let y1 = start_rating.as_readable_f32();
                    let y2 = end_rating.as_readable_f32();

                    return GraphicalObject::Line { x1, x2, y1, y2, color };
                })
                .collect()
        };

        self.write_svg(name, &tags2, compute_data)?;

        Ok(())
    }

    pub fn passing_tags_filter(&self, tags: &[&str]) -> bool {
        for filter_tag in &self.filter_tags {
            if !tags.contains(&filter_tag.as_str()) {
                return false;
            }
        }

        true
    }

    pub fn save_position_buffer(&mut self, name: &str, tags: &[&str], buffer: &PositionBuffer) -> std::io::Result<()> {
        let mut tags2: Vec<&str> = tags.to_vec();
        tags2.push("position");

        let compute_data = || {
            buffer
                .iter()
                .annotate_with_segment_start_times()
                .into_iter()
                .zip(
                    [Color::from_hex(0xFF0000), Color::from_hex(0x00FF00)]
                        .into_iter()
                        .cloned()
                        .cycle(),
                )
                .map(|(segment, color): (PositionFullSegment, Color)| {
                    let start = segment.span.start;
                    let end = segment.span.end;
                    let start_rating = segment.data.start_position();
                    let end_rating = segment.data.end_position(end - start);

                    let x1 = start.as_f32();
                    let x2 = (end - TimeDelta::one()).as_f32();
                    let y1 = start_rating.as_f32();
                    let y2 = end_rating.as_f32();

                    return GraphicalObject::Line { x1, x2, y1, y2, color };
                })
                .collect()
        };

        self.write_svg(name, &tags2, compute_data)?;

        Ok(())
    }
}
