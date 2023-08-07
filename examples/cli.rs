use std::sync::Arc;

use clap::Parser;
use image::Rgba;
use imageproc::{drawing::draw_hollow_rect_mut, rect::Rect};
use ort::Environment;
use yolov8_rs::Config;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// ONNX model to use
    model_path: String,

    /// Output folder to put the labeled images in
    output_folder: String,

    /// Images to detect objects in
    image_paths: Vec<String>,

    /// Thresholds for detection
    #[arg(long, default_value = "0.5")]
    iou: f32,

    /// Confidence threshold
    #[arg(long, default_value = "0.5")]
    confidence: f32,

    /// Pad images instead of resizing
    #[arg(long)]
    pad: bool,
}

fn main() {
    let args = Args::parse();
    let environment = Arc::new(Environment::builder().with_name("cli").build().unwrap());

    let model =
        yolov8_rs::YOLOV8::new(&environment, &args.model_path).expect("Failed to create model");
    let image_paths = args.image_paths.clone();
    let mut images = Vec::with_capacity(image_paths.len());

    for image_path in args.image_paths {
        let image = image::open(&image_path).unwrap();
        images.push(image);
    }

    let results = model
        .run_batch(
            images,
            &Config {
                resize_type: if args.pad {
                    yolov8_rs::ResizeType::Pad
                } else {
                    yolov8_rs::ResizeType::Resize(image::imageops::FilterType::Nearest)
                },
                thresholds: yolov8_rs::Thresholds {
                    iou: args.iou,
                    confidence: args.confidence,
                },
                undo_resize: true,
            },
        )
        .expect("Failed to run model");
    for ((image, boxes), image_path) in results.iter().zip(image_paths.iter()) {
        println!("{}: {:#?}", image_path, boxes);
        let mut image = image.clone();
        for bounding_box in boxes {
            let color = Rgba([0u8, 0u8, 255u8, 255u8]);
            let x1 = (bounding_box.x1 as f32) as u32;
            let x2 = (bounding_box.x2 as f32) as u32;
            let y1 = (bounding_box.y1 as f32) as u32;
            let y2 = (bounding_box.y2 as f32) as u32;
            draw_hollow_rect_mut(
                &mut image,
                Rect::at(x1 as i32, y1 as i32).of_size(x2 - x1, y2 - y1),
                color,
            );
        }
        image
            .save(format!(
                "{}/{}",
                args.output_folder,
                image_path.split("/").last().unwrap()
            ))
            .unwrap();
    }
}
